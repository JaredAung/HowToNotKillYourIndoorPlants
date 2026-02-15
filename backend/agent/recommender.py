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


def _encode_user_from_profile(profile_answers: Dict[str, Any]) -> List[float]:
    """Encode user profile using trained UserTower. Uses vocabs from checkpoint (two_tower_vocabs.json)."""
    model, vocabs, _ = _load_two_tower()
    from resources.trained_two_tower.main_fixed import encode_user_embedding
    return encode_user_embedding(model, profile_answers, vocabs)


def _format_plant(p: dict) -> str:
    common = ", ".join(p.get("common", []) or [p.get("latin", "Unknown")])
    climate = p.get("climate", "")
    size = p.get("size_bucket", "")
    use = ", ".join(p.get("use", []) or [])
    care = p.get("care_level", "")
    light = (p.get("ideallight", "") or "")[:50]
    watering = (p.get("watering", "") or "")[:80]
    desc = p.get("description") or {}
    physical = (desc.get("physical", "") or "")[:100] if isinstance(desc, dict) else ""
    return (
        f"- **{common}** (climate: {climate}, size: {size}, use: {use}, care: {care})\n"
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

    # Fetch more candidates when reranker enabled for better reranking
    fetch_k = 12 if (VOYAGE_RERANK_ENABLED and os.getenv("VOYAGE_API_KEY", "").strip()) else top_k
    try:
        top = _vector_search(coll, user_vec, top_k=fetch_k)
    except Exception as e:
        err = str(e).lower()
        if "index" in err or "vector" in err or "vectorsearch" in err:
            return (
                f"MongoDB vector search failed. Ensure a vector index '{VECTOR_INDEX}' exists on "
                f"field '{EMBEDDING_FIELD}'. Error: {e}",
                None,
            )
        raise
    names = [", ".join(p.get("common", []) or [p.get("latin", "Unknown")]) for p in top]
    print(f"Two-tower recommended plants (top_k={top_k}, count={len(top)}): {names}")
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


def _parse_llm_recommendations(content: str, num_plants: int) -> List[str]:
    """Extract per-plant explanation blocks from LLM output. Handles:
    - 1. **Plant**: explanation on same line
    - 1. **Plant**\nexplanation on next line(s)
    """
    explanations: List[str] = []
    # Match N. **PlantName** or N. PlantName, then capture explanation (same line after : or on following lines) until next plant
    pattern = r"\d+\.\s+(?:\*\*)?[^*\n]+(?:\*\*)?\s*(?::\s*)?([\s\S]*?)(?=\d+\.\s+(?:\*\*)?|\Z)"
    matches = re.findall(pattern, content)
    for m in matches[:num_plants]:
        expl = _strip_signoff(m.strip())
        explanations.append(expl if expl else "")
    while len(explanations) < num_plants:
        explanations.append("")
    return explanations[:num_plants]


def recommender_node(state: dict) -> dict:
    """Generate plant recommendations using trained two-tower model (UserTower vs itemTowerEmbeddings)."""
    print("[Recommender] recommender_node ENTERED", flush=True)
    profile_answers = state.get("profile_answers") or {}
    username = state.get("username") or ""

    try:
        plants_str, plants = _two_tower_retrieve(profile_answers, top_k=3)
    except Exception as e:
        return {"messages": [AIMessage(content=f"I had trouble fetching recommendations. ({e})")]}

    if "not configured" in plants_str or "vector search failed" in plants_str or "No matching" in plants_str or "not found" in plants_str.lower():
        return {"messages": [AIMessage(content=plants_str)]}
    if not plants:
        return {"messages": [AIMessage(content=plants_str or "No plants retrieved from the database.")]}

    user_features = ", ".join(f"{k}: {v}" for k, v in profile_answers.items() if v)
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
    plants = _rerank_plants(plants, rerank_query, top_k=3)
    plants_str = "\n\n".join(_format_plant(p) for p in plants) if plants else "No matching plants found."

    greeting_instruction = (
        f"Start with a brief personalized greeting (e.g. 'Hi {username}! Based on your preferences, here are 3 plants I think you'd like:') "
        if username else "Start with a brief greeting before the plant list. "
    )
    system = (
        f"You are a plant care expert. The user's profile: {user_features}. "
        "Here are plants retrieved by the trained two-tower model (UserTower vs itemTowerEmbeddings):\n\n"
        f"{plants_str}\n\n"
        f"{greeting_instruction}"
        "Then present 3 of the retrieved plants with why each fits their profile and one care tip. "
        "Use numbered format: 1. **Plant name**: explanation... * Care tip: ... "
        "Do NOT add sign-off phrases like 'Let me know if you have any questions' at the end."
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

    recommendations: List[Dict[str, Any]] = []
    if plants:
        explanations = _parse_llm_recommendations(content, len(plants))
        for p, expl in zip(plants, explanations):
            name = ", ".join(p.get("common", []) or [p.get("latin", "Unknown")])
            img = p.get("image_url") or p.get("img_url") or ""
            recommendations.append({"name": name, "image_url": img, "explanation": expl})

    greeting = _extract_greeting(content)
    # If LLM skipped the greeting, prepend a fallback
    if not greeting and username:
        greeting = f"Hi {username}! Based on your preferences, here are 3 plants I think you'd like:"
    elif not greeting:
        greeting = "Based on your preferences, here are 3 plants I think you'd like:"

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
    }
