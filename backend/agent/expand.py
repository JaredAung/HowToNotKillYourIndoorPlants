"""
Expand agent: retrieves detailed info for a single plant when user asks
(e.g. "tell me more about the croton", "details about the first one").
"""
import re
from typing import Any, Dict, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

from agent.llm import get_ollama_llm
from agent.recommender import _find_plant_by_name, _format_plant, _get_latin_name, _get_plants_collection


@tool
def get_plant_details(plant_name: str) -> str:
    """Retrieve detailed care info for a single plant by name.
    Use when the user asks for more info about one plant (e.g. 'tell me more about the croton').
    Returns care requirements, light, watering, size, and other details."""
    coll = _get_plants_collection()
    if coll is None:
        return "MongoDB not configured. Cannot retrieve plant details."
    p = _find_plant_by_name(coll, plant_name)
    if not p:
        return f"Plant '{plant_name}' not found in the database."
    name = _get_latin_name(p)
    details = _format_plant(p)
    return f"**{name}**\n{details}"


def explain_plant_with_llm(
    plant_name: str,
    user_query: str,
    profile_answers: Dict[str, Any],
) -> str:
    """Fetch plant from DB, then use LLM to generate a natural explanation. Includes user query and profile in prompt."""
    coll = _get_plants_collection()
    if coll is None:
        return "MongoDB not configured. Cannot retrieve plant details."
    p = _find_plant_by_name(coll, plant_name)
    if not p:
        return f"Plant '{plant_name}' not found in the database."
    name = _get_latin_name(p)
    details = _format_plant(p)

    profile_str = ", ".join(f"{k}: {v}" for k, v in (profile_answers or {}).items() if v)
    if not profile_str:
        profile_str = "not provided"

    system = f"""You are a plant care expert. Write a natural, helpful explanation of this plant.

User's question: {user_query or "Tell me more about this plant"}

User's profile/preferences: {profile_str}

Plant data (use this to write your explanation):

**{name}**
{details}

Write a clear explanation that addresses the user's question and relates to their profile when relevant. Use markdown for readability. Be concise but informative."""

    llm = get_ollama_llm(temperature=0.3)
    try:
        resp = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=user_query or f"Tell me more about {name}."),
        ])
        return getattr(resp, "content", None) or f"**{name}**\n{details}"
    except Exception as e:
        return f"I had trouble generating the explanation. Here's the raw info:\n\n**{name}**\n{details}\n\n(Error: {e})"


def _rec_matches_user_input(rec: dict, t: str) -> bool:
    """True if user message contains Latin name or any common name."""
    latin = (rec.get("name") or rec.get("latin") or "").lower()
    if latin and latin in t:
        return True
    common_list = rec.get("common") or []
    if isinstance(common_list, str):
        common_list = [common_list] if common_list else []
    for c in common_list:
        if c and str(c).lower() in t:
            return True
    return False


def resolve_single_plant(user_msg: str, last_recs: list) -> Optional[str]:
    """Resolve user reference to a single plant for expand intent.
    Handles: 'the first one', 'croton', 'details about #2', etc.
    Returns Latin name (for display) or None."""
    if not last_recs:
        return None
    names = [r.get("name", "") for r in last_recs if r.get("name")]
    if not names:
        return None
    t = (user_msg or "").lower()
    n = len(names)

    # By name: "tell me more about the croton", "details about rose grape" - match Latin or common
    for rec in last_recs:
        if _rec_matches_user_input(rec, t):
            return rec.get("name") or rec.get("latin", "")

    # Positional: "first one", "second", "#2", "the last one"
    if re.search(r"\b(first|1st)\s*(one|plant)?\b", t) or "first one" in t:
        return names[0]
    if re.search(r"\b(second|2nd)\s*(one|plant)?\b", t) or "second" in t or "#2" in t:
        return names[1] if n >= 2 else names[0]
    if re.search(r"\b(third|3rd)\s*(one|plant)?\b", t) or "third" in t or "#3" in t:
        return names[2] if n >= 3 else names[-1]
    if "last" in t and n >= 1:
        return names[-1]
    m = re.search(r"#\s*([123])\b", t)
    if m:
        idx = int(m.group(1))
        if 1 <= idx <= n:
            return names[idx - 1]
    return None


def has_single_plant_expand_selection(text: str, last_recs: list) -> bool:
    """True if user has expand intent and we can resolve one plant from last_recommendations."""
    return resolve_single_plant(text, last_recs) is not None


def explain_plant_node(state: dict) -> dict:
    """Fetch detailed info for the selected plant, use LLM to generate natural explanation."""
    selected_plants = state.get("selected_plants") or []
    if not selected_plants:
        return {
            "messages": [AIMessage(content="Which plant would you like to know more about?")],
            "recommendations": [],
            "greeting": None,
        }
    plant_name = selected_plants[0]
    user_query = _get_last_user_message(state)
    profile_answers = state.get("profile_answers") or {}
    result = explain_plant_with_llm(plant_name, user_query, profile_answers)
    return {
        "messages": [AIMessage(content=str(result))],
        "recommendations": [],
        "greeting": None,
    }


def _get_last_user_message(state: dict) -> str:
    """Extract last user message content from state."""
    for m in reversed(state.get("messages", [])):
        role = getattr(m, "type", None) or getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None)
        if role in ("human", "user"):
            return getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else "") or ""
    return ""
