"""
Recommender agent: uses completed profile to recommend plants.
Trained two-tower model: UserTower (profile -> embedding) vs PlantTower (itemTowerEmbeddings in MongoDB).
Uses MongoDB Atlas Vector Search for retrieval.
Vocabs loaded from checkpoint dir (two_tower_vocabs.json) to match trained model.
"""
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from pymongo import MongoClient

from agent.llm import get_ollama_llm

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_backend = Path(__file__).resolve().parent.parent
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

PLANTS_COLLECTION = os.getenv("MONGO_PLANTS_COLLECTION", os.getenv("MONGO_COLLECTION", "PlantsRawCollection"))
EMBEDDING_FIELD = os.getenv("PLANT_EMBEDDING_FIELD", "itemTowerEmbeddings")
VECTOR_INDEX = os.getenv("MONGO_VECTOR_INDEX", "item_emb_vector")
TWO_TOWER_CKPT = os.getenv("TWO_TOWER_CKPT_PATH", str(_backend / "resources" / "trained_two_tower" / "two_tower_model.pt"))
VOYAGE_RERANK_MODEL = os.getenv("VOYAGE_RERANK_MODEL", "rerank-2.5-lite")
VOYAGE_RERANK_ENABLED = os.getenv("VOYAGE_RERANK_ENABLED", "true").lower() in ("true", "1", "yes")

_model_vocabs_config: Optional[Tuple[Any, Dict[str, Dict[str, int]], Dict[str, Any]]] = None


def _load_two_tower() -> Tuple[Any, Dict[str, Dict[str, int]], Dict[str, Any]]:
    """Lazy-load trained two-tower model, vocabs, config."""
    global _model_vocabs_config
    if _model_vocabs_config is not None:
        return _model_vocabs_config
    if not os.path.exists(TWO_TOWER_CKPT):
        raise FileNotFoundError(
            f"Two-tower checkpoint not found at {TWO_TOWER_CKPT}. "
            "Run training (main_fixed.py) first."
        )
    from resources.trained_two_tower.main_fixed import load_two_tower_for_inference
    _model_vocabs_config = load_two_tower_for_inference(TWO_TOWER_CKPT)
    return _model_vocabs_config


def _get_plants_collection():
    uri = os.getenv("MONGO_URI", "").strip().strip('"')
    database = os.getenv("MONGO_DATABASE", "HowNotToKillYourPlants")
    if not uri:
        return None
    return MongoClient(uri)[database][PLANTS_COLLECTION]


def _get_user_profiles_collection():
    uri = os.getenv("MONGO_URI", "").strip().strip('"')
    database = os.getenv("MONGO_DATABASE", "HowNotToKillYourPlants")
    collection_name = os.getenv("MONGO_USER_PROFILES_COLLECTION", "UserProfiles")
    if not uri:
        return None
    return MongoClient(uri)[database][collection_name]


def _fetch_profile_from_db(username: str) -> Optional[Dict[str, Any]]:
    """Fetch latest profile from MongoDB (includes hard_filter from death report)."""
    coll = _get_user_profiles_collection()
    if coll is None or not username:
        return None
    doc = coll.find_one({"username": {"$regex": f"^{re.escape(username)}$", "$options": "i"}})
    if not doc:
        return None
    return doc.get("preferences", {}) or {}


def _get_latin_name(p: dict) -> str:
    """Return Latin (scientific) name for display. Use only Latin, not common names."""
    return (p.get("latin") or "").strip() or "Unknown"


def _find_plant_by_name(coll, name: str) -> Optional[Dict[str, Any]]:
    """Find a plant by common or latin name (case-insensitive, partial match)."""
    name = (name or "").strip()
    if not name:
        return None
    pattern = re.compile(re.escape(name), re.I)
    for field in ("common", "latin"):
        doc = coll.find_one({field: pattern})
        if doc:
            return doc
    return None


@tool
def compare_plants(plant_a: str, plant_b: str) -> str:
    """Compare two plants by name. Use when the user asks to compare plants (e.g. 'compare Croton and Monster cactus').
    Returns care requirements, watering, light, size, and other details for both plants."""
    coll = _get_plants_collection()
    if coll is None:
        return "MongoDB not configured. Cannot compare plants."
    p1 = _find_plant_by_name(coll, plant_a)
    p2 = _find_plant_by_name(coll, plant_b)
    if not p1:
        return f"Plant '{plant_a}' not found in the database."
    if not p2:
        return f"Plant '{plant_b}' not found in the database."
    a_str = _format_plant(p1)
    b_str = _format_plant(p2)
    return f"**{_get_latin_name(p1)}**\n{a_str}\n\n**{_get_latin_name(p2)}**\n{b_str}"


def compare_plants_with_llm(
    plant_a: str,
    plant_b: str,
    user_query: str,
    profile_answers: Dict[str, Any],
) -> str:
    """Fetch both plants, then use LLM to generate a natural comparison. Includes user query and profile in prompt."""
    coll = _get_plants_collection()
    if coll is None:
        return "MongoDB not configured. Cannot compare plants."
    p1 = _find_plant_by_name(coll, plant_a)
    p2 = _find_plant_by_name(coll, plant_b)
    if not p1:
        return f"Plant '{plant_a}' not found in the database."
    if not p2:
        return f"Plant '{plant_b}' not found in the database."
    a_str = _format_plant(p1)
    b_str = _format_plant(p2)
    name_a = _get_latin_name(p1)
    name_b = _get_latin_name(p2)

    profile_str = ", ".join(f"{k}: {v}" for k, v in (profile_answers or {}).items() if v)
    if not profile_str:
        profile_str = "not provided"

    system = f"""You are a plant care expert. Write a natural, helpful comparison of these two plants.

User's question: {user_query or "Compare these plants"}

User's profile/preferences: {profile_str}

Plant data (use this to write your comparison):

**{name_a}**
{a_str}

**{name_b}**
{b_str}

Write a clear comparison that addresses the user's question and relates to their profile when relevant. Use markdown for readability. Be concise but informative."""

    llm = get_ollama_llm(temperature=0.3)
    try:
        resp = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=user_query or "Compare these two plants."),
        ])
        return getattr(resp, "content", None) or f"**{name_a}**\n{a_str}\n\n**{name_b}**\n{b_str}"
    except Exception as e:
        return f"I had trouble generating the comparison. Here's the raw info:\n\n**{name_a}**\n{a_str}\n\n**{name_b}**\n{b_str}\n\n(Error: {e})"


def _encode_user_from_profile(profile_answers: Dict[str, Any]) -> List[float]:
    """Encode user profile using trained UserTower. Uses vocabs from checkpoint (two_tower_vocabs.json)."""
    model, vocabs, _ = _load_two_tower()
    from resources.trained_two_tower.main_fixed import encode_user_embedding
    return encode_user_embedding(model, profile_answers, vocabs)


def _format_plant(p: dict) -> str:
    latin = _get_latin_name(p)
    climate = p.get("climate", "")
    size = p.get("size_bucket", "")
    use = ", ".join(p.get("use", []) or [])
    care = p.get("care_level", "")
    light = (p.get("ideallight", "") or "")[:50]
    watering = (p.get("watering", "") or "")[:80]
    desc = p.get("description") or {}
    physical = (desc.get("physical", "") or "")[:100] if isinstance(desc, dict) else ""
    return (
        f"- **{latin}** (climate: {climate}, size: {size}, use: {use}, care: {care})\n"
        f"  Light: {light}\n  Watering: {watering}\n  {physical}"
    )


def _rerank_plants(
    plants: List[Dict[str, Any]], query: str, top_k: int = 5
) -> List[Dict[str, Any]]:
    """Rerank plants using Voyage AI reranker. Returns top_k plants in relevance order."""
    if not plants or len(plants) <= 1:
        return plants[:top_k]
    api_key = os.getenv("VOYAGE_API_KEY", "").strip()
    if not api_key or not VOYAGE_RERANK_ENABLED:
        return plants[:top_k]
    try:
        import voyageai
        vo = voyageai.Client(api_key=api_key)
        documents = [_format_plant(p) for p in plants]
        result = vo.rerank(query=query, documents=documents, model=VOYAGE_RERANK_MODEL, top_k=min(top_k, len(plants)))
        reordered = [plants[r.index] for r in result.results]
        return reordered
    except Exception as e:
        print(f"[Reranker] Voyage rerank failed, using original order: {e}", flush=True)
        return plants[:top_k]


def _vector_search(coll, user_embedding: list[float], top_k: int = 5) -> list[dict]:
    """Run MongoDB Atlas Vector Search. Returns top-k plants."""
    num_candidates = min(10000, max(top_k * 20, 100))
    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX,
                "path": EMBEDDING_FIELD,
                "queryVector": user_embedding,
                "limit": top_k,
                "numCandidates": num_candidates,
            }
        },
        {"$limit": top_k},
    ]
    return list(coll.aggregate(pipeline))


TOP_K_RETRIEVAL = 25

# Priority boost weights for features before reranking (higher = stronger influence)
PRIORITY_BOOST_WEIGHTS = {
    "watering_preferences": 2.5,
    "climate": 1,
    "max_plant_size_preference": 1.0,
}


def _apply_priority_boost(
    plants: List[Dict[str, Any]], profile_answers: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Re-sort ANN results by boosting plants that match priority profile features.
    Applied before hard filter and reranker so key constraints (light, watering, etc.) rise to the top."""
    if not plants or not profile_answers:
        return plants
    from resources.trained_two_tower.main_fixed import normalize_light, watering_to_tags

    user_light = (profile_answers.get("light_availability") or "").strip()
    user_light_norm = normalize_light(user_light) if user_light else None
    user_watering = (profile_answers.get("watering_preferences") or "").strip().lower()
    user_watering_tags = set(watering_to_tags(user_watering)) if user_watering else set()
    user_climate = (profile_answers.get("climate") or "").strip().lower()
    user_size = (profile_answers.get("max_plant_size_preference") or "").strip().lower()

    def boost_score(plant: Dict[str, Any], rank: int) -> float:
        score = 0.0
        if user_light_norm:
            ideal = normalize_light(plant.get("ideallight"))
            tolerated = normalize_light(plant.get("toleratedlight"))
            if user_light_norm in (ideal, tolerated):
                score += PRIORITY_BOOST_WEIGHTS.get("light_availability", 2.0)
        if user_watering_tags:
            plant_watering = plant.get("watering") or ""
            plant_tags = set(watering_to_tags(plant_watering))
            if user_watering_tags & plant_tags:
                score += PRIORITY_BOOST_WEIGHTS.get("watering_preferences", 1.5)
        if user_climate:
            plant_climate = (plant.get("climate") or "").strip().lower()
            if plant_climate and (user_climate == plant_climate or user_climate in plant_climate):
                score += PRIORITY_BOOST_WEIGHTS.get("climate", 1.2)
        if user_size:
            plant_size = (plant.get("size_bucket") or "").strip().lower()
            size_ok = (
                plant_size == user_size
                or (user_size == "large" and plant_size in ("small", "medium", "large"))
                or (user_size == "medium" and plant_size in ("small", "medium"))
            )
            if size_ok:
                score += PRIORITY_BOOST_WEIGHTS.get("max_plant_size_preference", 1.0)
        return (score, -rank)

    scored = [(p, boost_score(p, i)) for i, p in enumerate(plants)]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [p for p, _ in scored]


def _apply_hard_filter_light(plants: List[Dict[str, Any]], user_light: str) -> List[Dict[str, Any]]:
    """Keep only plants whose ideal or tolerated light matches user_light."""
    from resources.trained_two_tower.main_fixed import normalize_light
    if not user_light or not plants:
        return plants
    user_norm = normalize_light(user_light)
    filtered = []
    for p in plants:
        ideal = normalize_light(p.get("ideallight"))
        tolerated = normalize_light(p.get("toleratedlight"))
        if user_norm in (ideal, tolerated):
            filtered.append(p)
    return filtered


def _two_tower_retrieve(
    profile_answers: Dict[str, Any], top_k: int = 5
) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
    """Retrieve plants using trained UserTower embedding vs itemTowerEmbeddings in MongoDB.
    Returns (plants_str, plants_list) on success, or (error_str, None) on failure."""
    print("[Two-tower] _two_tower_retrieve ENTERED", flush=True)
    coll = _get_plants_collection()
    if coll is None:
        return "MongoDB not configured. MONGO_URI not set.", None

    try:
        user_vec = _encode_user_from_profile(profile_answers)
    except FileNotFoundError as e:
        return str(e), None
    except Exception as e:
        return f"Failed to encode user profile: {e}", None

    # Fetch more candidates for reranking and hard filters (always 40 for ANN)
    fetch_k = max(top_k, TOP_K_RETRIEVAL)
    try:
        top = _vector_search(coll, user_vec, top_k=TOP_K_RETRIEVAL)
    except Exception as e:
        err = str(e).lower()
        if "index" in err or "vector" in err or "vectorsearch" in err:
            return (
                f"MongoDB vector search failed. Ensure a vector index '{VECTOR_INDEX}' exists on "
                f"field '{EMBEDDING_FIELD}'. Error: {e}",
                None,
            )
        raise
    top = _apply_priority_boost(top, profile_answers)
    names = [_get_latin_name(p) for p in top]
    print(f"Two-tower recommended plants (top_k={TOP_K_RETRIEVAL}, count={len(top)}): {names}")
    plants_str = "\n\n".join(_format_plant(p) for p in top) if top else "No matching plants found."
    return plants_str, top if top else None


@tool
def two_tower_inference(profile_answers: str, top_k: int = 5) -> str:
    """Two-tower retrieval: encode user profile with trained UserTower and find similar plants.
    profile_answers: JSON string of profile dict (experience_level, climate, use, etc.).
    Returns formatted plant names and details."""
    import json
    try:
        pa = json.loads(profile_answers) if isinstance(profile_answers, str) else profile_answers
    except Exception:
        pa = {}
    plants_str, _ = _two_tower_retrieve(pa, top_k=top_k)
    return plants_str


_SIGNOFF_PATTERNS = [
    r"\s*<[^>]*[Ll]et me know[^>]*>\s*$",
    r"\s*[Ll]et me know if you have any questions[^.!]*[.!]?\s*$",
    r"\s*[Ff]eel free to ask[^.!]*[.!]?\s*$",
    r"\s*[Ii]f you have any questions[^.!]*[.!]?\s*$",
    r"\s*[Hh]appy (?:planting|gardening)[^.!]*[.!]?\s*$",
]


def _strip_signoff(expl: str) -> str:
    """Remove LLM sign-off phrases from end of explanation."""
    s = expl.strip()
    for pat in _SIGNOFF_PATTERNS:
        s = re.sub(pat, "", s)
    return s.strip()


def _extract_greeting(content: str) -> str:
    """Extract greeting/intro text before the first numbered plant. Ensures LLM greeting is not lost."""
    if not content or not content.strip():
        return ""
    # Match everything before first "1. **" or "1. " (start of numbered plant list)
    match = re.search(r"^(.*?)(?=\d+\.\s+(?:\*\*)?)", content, re.DOTALL)
    if match:
        greeting = match.group(1).strip()
        # Remove debug prefix if present
        if greeting.startswith("[DEBUG]"):
            idx = greeting.find("\n\n")
            if idx >= 0:
                greeting = greeting[idx + 2 :].strip()
        return greeting
    return ""


def _parse_llm_recommendations(content: str, num_plants: int, plant_names: Optional[List[str]] = None) -> List[str]:
    """Extract per-plant explanation blocks from LLM output. Handles numbered (1. **Name**) and unnumbered (**Name**) formats."""
    text = content
    if text.startswith("[DEBUG]"):
        idx = text.find("\n\n")
        text = text[idx + 2 :] if idx >= 0 else text
    explanations: List[str] = []
    fallback = "Fits your profile. Add to your garden to get personalized care tips."

    # Try split by numbered items first: "1. **" or "2. "
    parts = re.split(r"(?=\d+\.\s+(?:\*\*)?)", text)
    if len(parts) >= num_plants + 1:
        for part in parts[1 : num_plants + 1]:
            part = part.strip()
            if not part:
                explanations.append(fallback)
                continue
            expl = re.sub(r"^\d+\.\s+(?:\*\*)?[^*\n]+(?:\*\*)?\s*:?\s*", "", part, count=1).strip()
            expl = _strip_signoff(expl)
            if not expl or len(expl) < 10 or expl.lower() in {n.lower() for n in (plant_names or [])}:
                expl = fallback
            explanations.append(expl)
    else:
        # Fallback: split by **PlantName** pattern (unnumbered format)
        if plant_names:
            for name in plant_names[:num_plants]:
                escaped = re.escape(name)
                match = re.search(rf"\*\*{escaped}\*\*\s*:?\s*([\s\S]*?)(?=\*\*[^*]+\*\*|\Z)", text, re.I)
                if match:
                    expl = _strip_signoff(match.group(1).strip())
                    if expl and len(expl) >= 10:
                        explanations.append(expl)
                        continue
                explanations.append(fallback)
        while len(explanations) < num_plants:
            explanations.append(fallback)

    while len(explanations) < num_plants:
        explanations.append(fallback)
    return explanations[:num_plants]


def recommender_node(state: dict) -> dict:
    """Generate plant recommendations using trained two-tower model (UserTower vs itemTowerEmbeddings)."""
    print("[Recommender] recommender_node ENTERED", flush=True)
    profile_answers = dict(state.get("profile_answers") or {})
    username = state.get("username") or ""

    # Reload from MongoDB to get latest (including hard_filter from death report)
    if username:
        db_prefs = _fetch_profile_from_db(username)
        if db_prefs:
            profile_answers.update(db_prefs)

    try:
        plants_str, plants = _two_tower_retrieve(profile_answers, top_k=TOP_K_RETRIEVAL)
    except Exception as e:
        return {"messages": [AIMessage(content=f"I had trouble fetching recommendations. ({e})")]}

    if "not configured" in plants_str or "vector search failed" in plants_str or "No matching" in plants_str or "not found" in plants_str.lower():
        return {"messages": [AIMessage(content=plants_str)]}
    if not plants:
        return {"messages": [AIMessage(content=plants_str or "No plants retrieved from the database.")]}

    # Apply hard filter from profile (stored in MongoDB after death report selection)
    hard_filter = profile_answers.get("hard_filter") or []
    if isinstance(hard_filter, str):
        hard_filter = [hard_filter] if hard_filter else []
    if "light" in hard_filter:
        user_light = profile_answers.get("light_availability") or ""
        plants = _apply_hard_filter_light(plants, user_light)
        if not plants:
            return {"messages": [AIMessage(content="No plants match your light level. Try adjusting your light preference or removing the 'match light' filter.")]}

    user_features = ", ".join(f"{k}: {v}" for k, v in profile_answers.items() if v and k != "hard_filter")
    messages = state.get("messages", [])
    last_user = ""
    for m in reversed(messages):
        role = getattr(m, "type", None) or getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None)
        if role in ("human", "user"):
            last_user = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else "") or ""
            break
    prompt_text = last_user.strip() if last_user else "Based on my profile, what plants do you recommend?"
    if not prompt_text or "what kind" in prompt_text.lower() or "interested" in prompt_text.lower():
        prompt_text = "Based on my profile, what plants do you recommend?"

    # Voyage AI reranker: refine order by relevance to query (profile + intent)
    rerank_query = f"{prompt_text}. User profile: {user_features}"
    plants = _rerank_plants(plants, rerank_query, top_k=5)
    plants_str = "\n\n".join(_format_plant(p) for p in plants) if plants else "No matching plants found."

    greeting_instruction = (
        f"Start with a brief personalized greeting (e.g. 'Hi {username}! Based on your preferences, here are 5 plants I think you'd like:') "
        if username else "Start with a brief greeting before the plant list. "
    )
    system = (
        f"You are a plant care expert. The user's profile: {user_features}. "
        "Here are plants retrieved by the trained two-tower model (UserTower vs itemTowerEmbeddings):\n\n"
        f"{plants_str}\n\n"
        f"{greeting_instruction}"
        "Then present 5 of the retrieved plants. For EACH plant, write 1-2 sentences explaining WHY it fits their profile, then one care tip. "
        "Format: 1. **Plant name**: [Your explanation - do NOT repeat the plant name. Explain why it fits their light, watering, etc.] * Care tip: [one tip] "
        "Use only the Latin (scientific) names from the plant data above. "
        "Do NOT add sign-off phrases at the end."
    )
    llm = get_ollama_llm(temperature=0.5)
    try:
        resp = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=prompt_text),
        ])
        content = getattr(resp, "content", None) or "I'd be happy to recommend plants. Could you tell me more?"
    except Exception as e:
        content = f"I had trouble formatting recommendations. ({e})"

    # Debug: log LLM output for explanation parsing
    print(f"[Recommender] LLM content (first 600 chars): {repr(content[:600])}", flush=True)
    plant_names = [_get_latin_name(p) for p in plants]
    explanations = _parse_llm_recommendations(content, len(plants), plant_names)
    print(f"[Recommender] Parsed explanations: {explanations}", flush=True)

    recommendations: List[Dict[str, Any]] = []
    last_recommendations: List[Dict[str, Any]] = []
    if plants:
        for i, (p, expl) in enumerate(zip(plants, explanations)):
            latin = _get_latin_name(p)
            common_list = p.get("common") or []
            if isinstance(common_list, str):
                common_list = [common_list] if common_list else []
            img = p.get("image_url") or p.get("img_url") or ""
            plant_id = str(p.get("_id", "")) if p.get("_id") else ""
            recommendations.append({
                "name": latin, "image_url": img, "explanation": expl, "plant_id": plant_id,
            })
            last_recommendations.append({
                "rank": i + 1, "name": latin, "plant_id": plant_id, "latin": latin,
                "common": common_list,  # for matching user input like "croton"
            })

    greeting = _extract_greeting(content)
    # If LLM skipped the greeting, prepend a fallback
    if not greeting and username:
        greeting = f"Hi {username}! Based on your preferences, here are 5 plants I think you'd like:"
    elif not greeting:
        greeting = "Based on your preferences, here are 5 plants I think you'd like:"

    extracted = {
        "node": "recommender",
        "two_tower_called": True,
        "profile_answers_keys": list(profile_answers.keys()),
        "plants_count": len(plants) if plants else 0,
        "recommendations_count": len(recommendations),
    }
    debug_prefix = f"[DEBUG] {json.dumps(extracted, indent=2)}\n\n"
    return {
        "messages": [AIMessage(content=debug_prefix + content)],
        "recommendations": recommendations,
        "greeting": greeting,
        "last_recommendations": last_recommendations,
    }
