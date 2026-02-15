"""
Garden module: fetch user's plants from User_Garden_Collection.
"""
import os
import re
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

PLANTS_COLLECTION = os.getenv("MONGO_PLANTS_COLLECTION", os.getenv("MONGO_COLLECTION", "PlantsRawCollection"))


def _get_user_garden_collection():
    """Get MongoDB User_Garden_Collection. Handles collection names with accidental trailing newlines."""
    uri = os.getenv("MONGO_URI", "").strip().strip('"')
    database = os.getenv("MONGO_DATABASE", "HowNotToKillYourPlants")
    collection_name = (os.getenv("MONGO_USER_GARDEN_COLLECTION", "User_Garden_Collection") or "").strip()
    if not uri:
        return None
    client = MongoClient(uri)
    db = client[database]
    coll = db[collection_name]
    # If env collection is empty, find collection whose stripped name matches (handles trailing newlines in MongoDB)
    if coll.count_documents({}) == 0:
        for c in db.list_collection_names():
            if c.strip() == collection_name:
                return db[c]
    return coll


def _get_plants_collection():
    """Get MongoDB plants collection for enriching garden entries."""
    uri = os.getenv("MONGO_URI", "").strip().strip('"')
    database = os.getenv("MONGO_DATABASE", "HowNotToKillYourPlants")
    if not uri:
        return None
    return MongoClient(uri)[database][PLANTS_COLLECTION]


def get_user_garden_plants(username: str) -> List[Dict[str, Any]]:
    """
    Fetch all plants for a user from User_Garden_Collection.
    Uses case-insensitive username match.
    Enriches with image_url from Plants collection when plant_id is available.
    Returns list of {latin, plant_id, image_url, name}.
    """
    garden_coll = _get_user_garden_collection()
    if garden_coll is None:
        return []

    # Case-insensitive username lookup
    cursor = garden_coll.find(
        {"username": {"$regex": f"^{re.escape(username)}$", "$options": "i"}}
    )
    entries = list(cursor)

    if not entries:
        return []

    plants_coll = _get_plants_collection()
    result: List[Dict[str, Any]] = []

    for entry in entries:
        latin = (entry.get("latin") or "").strip() or "Unknown"
        plant_id = entry.get("plant_id", "")

        image_url = ""
        if plants_coll is not None and plant_id:
            try:
                from bson.objectid import ObjectId

                doc = plants_coll.find_one({"_id": ObjectId(plant_id)})
                if doc:
                    image_url = doc.get("image_url") or doc.get("img_url") or ""
            except Exception:
                pass

        result.append({
            "latin": latin,
            "name": latin,
            "plant_id": plant_id,
            "image_url": image_url,
        })

    return result


def get_plant_by_id(plant_id: str) -> Dict[str, Any] | None:
    """
    Fetch full plant details by plant_id (MongoDB ObjectId string).
    Returns JSON-serializable dict with all display fields, or None if not found.
    """
    plants_coll = _get_plants_collection()
    if plants_coll is None or not plant_id:
        return None
    try:
        from bson.objectid import ObjectId

        doc = plants_coll.find_one({"_id": ObjectId(plant_id)})
        if not doc:
            return None
        # Build serializable response (ObjectId -> str)
        desc = doc.get("description") or {}
        if isinstance(desc, dict):
            physical = desc.get("physical", "") or ""
        else:
            physical = ""
        return {
            "plant_id": str(doc.get("_id", "")),
            "latin": (doc.get("latin") or "").strip() or "Unknown",
            "common": doc.get("common") or [],
            "climate": doc.get("climate", ""),
            "size_bucket": doc.get("size_bucket", ""),
            "use": doc.get("use") or [],
            "care_level": doc.get("care_level", ""),
            "ideallight": doc.get("ideallight", ""),
            "watering": doc.get("watering", ""),
            "description": {"physical": physical},
            "image_url": doc.get("image_url") or doc.get("img_url") or "",
        }
    except Exception:
        return None


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
