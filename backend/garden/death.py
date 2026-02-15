"""
Death module: remove plants from user's garden and store death reports.
"""
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from pymongo import MongoClient

from garden.garden import _get_user_garden_collection

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Import watering_to_tags from two_tower so profile values match training vocab
_backend = Path(__file__).resolve().parent.parent
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))
from resources.trained_two_tower.main_fixed import normalize_light, watering_to_tags

# Prevention option text that triggers profile updates (must match PREVENTION_BY_CAUSE)
DROUGHT_TOLERANT_OPTION = "Choose a more drought-tolerant plant"
REDUCE_WATERING_OPTION = "Reduce watering frequency"
WATER_TOLERANT_OPTION = "Recommend a more water-tolerant plant"
LOW_LIGHT_TOLERANT_OPTION = "Recommend a plant that is tolerant to low light"
MATCH_LIGHT_OPTION = "Choose plants that match your light level"

# Canonical watering levels for profile updates
WATERING_LEVEL_DOWN = {"high": "moderate", "moderate": "low", "low": "low"}  # reduce by one level
WATERING_LEVEL_UP = {"low": "moderate", "moderate": "high", "high": "high"}  # increase by one level

# Profile values: low/moderate/high (watering_to_tags accepts these explicitly)
WATERING_PREFERENCE_BY_LEVEL = {
    "high": "high",
    "moderate": "moderate",
    "low": "low",
}

# Light levels: direct_sun, bright_indirect, medium, low (normalize_light canonical buckets)
LIGHT_LEVEL_DOWN = {"direct_sun": "bright_indirect", "bright_indirect": "medium", "medium": "low", "low": "low"}


def _get_user_profiles_collection():
    """Get MongoDB user profiles collection."""
    uri = os.getenv("MONGO_URI", "").strip().strip('"')
    database = os.getenv("MONGO_DATABASE", "HowNotToKillYourPlants")
    collection_name = os.getenv("MONGO_USER_PROFILES_COLLECTION", "UserProfiles")
    if not uri:
        return None
    return MongoClient(uri)[database][collection_name]


def _infer_watering_level(text: Any) -> str:
    """Infer watering level from free-text using same logic as watering_to_tags (two_tower)."""
    tags = watering_to_tags(text)
    if any("watering_level:high" in t for t in tags):
        return "high"
    if any("watering_level:low" in t for t in tags):
        return "low"
    return "moderate"


def _watering_level_to_preference(level: str) -> str:
    """Map canonical level to profile string that watering_to_tags will correctly parse."""
    return WATERING_PREFERENCE_BY_LEVEL.get(level, "moderate")


def _infer_light_level(text: Any) -> str:
    """Infer light level from free-text using normalize_light (two_tower)."""
    return normalize_light(text)


def _apply_light_level_update(coll, doc: dict, direction: str) -> None:
    """Lower light level by one step. Updates light_availability."""
    prefs = dict(doc.get("preferences", {}) or {})
    # Read from light_availability or average_sunlight_type (backwards compat)
    current = prefs.get("light_availability") or prefs.get("average_sunlight_type") or ""
    level = _infer_light_level(current)
    new_level = LIGHT_LEVEL_DOWN.get(level, level)
    # Use dot notation so we only update this field; ensures preferences exists
    coll.update_one(
        {"_id": doc["_id"]},
        {"$set": {"preferences.light_availability": new_level}},
    )


def _apply_hard_filter_update(coll, doc: dict, filter_type: str) -> None:
    """Add filter_type to hard_filter in preferences. Stored in MongoDB for recommender."""
    prefs = dict(doc.get("preferences", {}) or {})
    current = prefs.get("hard_filter") or []
    if not isinstance(current, list):
        current = [current] if current else []
    if filter_type not in current:
        current = list(current) + [filter_type]
    prefs["hard_filter"] = current
    coll.update_one({"_id": doc["_id"]}, {"$set": {"preferences": prefs}})


DEATH_CAUSES = [
    "underwatering",
    "overwatering",
    "light",
    "pests",
    "temperature",
    "humidity",
    "soil/drainage",
    "nutrition",
    "unknown",
]

PREVENTION_BY_CAUSE: Dict[str, List[str]] = {
    "underwatering": [
        "Set up a daily watering reminder",
        "Use a moisture meter",
        DROUGHT_TOLERANT_OPTION,  # reduce watering_preferences by one level
    ],
    "overwatering": [
        "Check soil before watering",
        "Improve pot drainage",
        REDUCE_WATERING_OPTION,  # reduce watering_preferences by one level
        WATER_TOLERANT_OPTION,  # increase watering_preferences by one level
    ],
    "light": [
        "Move to a brighter spot",
        LOW_LIGHT_TOLERANT_OPTION,  # lower light_availability by one step
        MATCH_LIGHT_OPTION,  # hard filter: only plants matching user's light
    ],
    "pests": [
        "Isolate new plants before adding to collection",
        "Inspect regularly for pests",
        "Use neem oil or insecticidal soap as needed",
    ],
    "temperature": [
        "Keep away from drafts and heating vents",
        "Avoid sudden temperature changes",
        "Match plant to your room temperature", #prioritize temperature
    ],
    "humidity": [
        "Use a humidifier or pebble tray",
        "Group plants to raise humidity",
        "Match plant to your room humidity", #prioritize humidity
    ],
    "soil/drainage": [
        "Repot with well-draining soil",
        "Ensure pot has drainage holes",
        "Avoid compacted or waterlogged soil",
    ],
    "nutrition": [
        "Fertilize during growing season only",
        "Use diluted fertilizer",
        "Flush soil occasionally to prevent buildup",
    ],
    "unknown": [
        "Set up a care reminder system",
        "Recommed a plant that is easy to care for",
    ],
}


def _get_death_reports_collection():
    """Get MongoDB DeathReports collection."""
    uri = os.getenv("MONGO_URI", "").strip().strip('"')
    database = os.getenv("MONGO_DATABASE", "HowNotToKillYourPlants")
    collection_name = os.getenv("MONGO_DEATH_REPORTS_COLLECTION", "DeathReports")
    if not uri:
        return None
    return MongoClient(uri)[database][collection_name]


def submit_death_report(
    username: str,
    plant_id: str,
    cause: List[str],
    details: str = "",
    reminder_system: bool = False,
    where_placed: str = "",
    last_watered_date: str | None = None,
    travel_away: bool = False,
) -> Dict[str, Any]:
    """
    Store a death report and return prevention options for selected causes.
    """
    coll = _get_death_reports_collection()
    report = {
        "username": username,
        "plant_id": plant_id,
        "date": datetime.now(timezone.utc).isoformat(),
        "cause": cause,
        "details": (details or "").strip(),
        "care_context": {
            "reminder_system": reminder_system,
            "where_placed": where_placed,
            "last_watered_date": last_watered_date,
            "travel_away": travel_away,
        },
    }
    report_id = ""
    if coll is not None:
        try:
            result = coll.insert_one(report)
            report_id = str(result.inserted_id)
        except Exception:
            pass

    # Build prevention options from selected causes
    prevention_options: List[str] = []
    seen: set[str] = set()
    for c in cause:
        for opt in PREVENTION_BY_CAUSE.get(c, PREVENTION_BY_CAUSE["unknown"]):
            if opt not in seen:
                prevention_options.append(opt)
                seen.add(opt)

    return {
        "report_id": report_id,
        "prevention_options": prevention_options,
    }


def _apply_watering_preference_update(coll, doc: dict, direction: str) -> None:
    """Update watering_preferences. direction: 'down' or 'up'."""
    prefs = dict(doc.get("preferences", {}) or {})
    current = prefs.get("watering_preferences") or ""
    level = _infer_watering_level(current)
    if direction == "down":
        new_level = WATERING_LEVEL_DOWN.get(level, level)
    else:
        new_level = WATERING_LEVEL_UP.get(level, level)
    prefs["watering_preferences"] = _watering_level_to_preference(new_level)
    coll.update_one({"_id": doc["_id"]}, {"$set": {"preferences": prefs}})


def apply_prevention_actions(username: str, selected_preventions: List[str]) -> Dict[str, Any]:
    """
    Apply selected prevention actions to the user's profile.
    - DROUGHT_TOLERANT_OPTION / REDUCE_WATERING_OPTION: reduce watering_preferences by one level
    - WATER_TOLERANT_OPTION: increase watering_preferences by one level
    - LOW_LIGHT_TOLERANT_OPTION: lower light_availability by one level
    - MATCH_LIGHT_OPTION: set hard_filter=["light"] for post-ANN filtering
    Returns dict with applied: list of actions applied, errors: list of error messages.
    """
    if not username or not selected_preventions:
        return {"applied": [], "errors": []}

    coll = _get_user_profiles_collection()
    if coll is None:
        return {"applied": [], "errors": ["MongoDB not configured"]}

    applied: List[str] = []
    errors: List[str] = []

    reduce_watering = [DROUGHT_TOLERANT_OPTION, REDUCE_WATERING_OPTION]
    increase_watering = [WATER_TOLERANT_OPTION]
    lower_light = [LOW_LIGHT_TOLERANT_OPTION]
    hard_filter_options = [MATCH_LIGHT_OPTION]

    for opt in selected_preventions:
        try:
            doc = coll.find_one({"username": {"$regex": f"^{re.escape(username)}$", "$options": "i"}})
            if not doc:
                errors.append("User profile not found")
                continue
            if opt in reduce_watering:
                _apply_watering_preference_update(coll, doc, "down")
                applied.append(opt)
            elif opt in increase_watering:
                _apply_watering_preference_update(coll, doc, "up")
                applied.append(opt)
            elif opt in lower_light:
                _apply_light_level_update(coll, doc, "down")
                applied.append(opt)
            elif opt in hard_filter_options:
                if opt == MATCH_LIGHT_OPTION:
                    _apply_hard_filter_update(coll, doc, "light")
                applied.append(opt)
        except Exception as e:
            errors.append(str(e))

    return {"applied": applied, "errors": errors}


def remove_plant_from_garden(username: str, plant_id: str) -> bool:
    """
    Remove a plant from user's garden. Returns True if removed, False otherwise.
    Uses case-insensitive username match.
    """
    garden_coll = _get_user_garden_collection()
    if garden_coll is None or not username or not plant_id:
        return False
    try:
        result = garden_coll.delete_one(
            {
                "username": {"$regex": f"^{re.escape(username)}$", "$options": "i"},
                "plant_id": plant_id,
            }
        )
        return result.deleted_count > 0
    except Exception:
        return False
