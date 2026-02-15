"""
LangChain-Ollama agents for plant care assistance.
Graph orchestration: deterministic router + profile builder, recommender.
Structured state (last_recommendations, selected_plants) for reliable reference resolution.
"""
import os
import re
import sys
from pathlib import Path
from typing import Annotated, NotRequired, TypedDict, Optional, Tuple

# Allow running as python agent/manager.py from backend/
_backend = Path(__file__).resolve().parent.parent
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from pymongo import MongoClient

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from agent.llm import get_ollama_llm
from agent.profile_builder import (
    FREEFORM_SENTINEL,
    get_profile_questions,
    profile_init,
    profile_collector,
)
from agent.recommender import recommender_node, compare_plants


def _get_user_profiles_collection():
    """Get MongoDB user profiles collection."""
    uri = os.getenv("MONGO_URI", "").strip().strip('"')
    database = os.getenv("MONGO_DATABASE", "HowNotToKillYourPlants")
    collection_name = os.getenv("MONGO_USER_PROFILES_COLLECTION", "UserProfiles")
    if not uri:
        return None
    client = MongoClient(uri)
    return client[database][collection_name]


@tool
def check_user_profile_exists(username: str) -> str:
    """Check if a user profile already exists in the database. Call this first before building a new profile. Returns 'exists' with profile summary if found, or 'not_found' if no profile exists."""
    coll = _get_user_profiles_collection()
    if coll is None:
        return "MongoDB not configured. MONGO_URI not set."
    doc = coll.find_one({"username": username})
    if doc:
        prefs = doc.get("preferences", {})
        summary = ", ".join(f"{k}: {v}" for k, v in prefs.items() if v) if prefs else "no preferences stored"
        return f"exists. username={username}. preferences: {summary or 'none'}"
    return "not_found"


class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    pending_questions: list[str]
    profile_answers: dict
    username: Optional[str]
    active_agent: Optional[str]
    handoff_reason: Optional[str]
    recommendations: NotRequired[list]  # [{name, image_url, explanation}] from recommender
    greeting: NotRequired[str]  # intro text before plant list, from recommender
    last_recommendations: NotRequired[list]  # [{rank, name, plant_id?}] for selection resolution
    selected_plants: NotRequired[list]  # [name, name] resolved from "first 2", "the low light one", etc.


def _get_last_user_message(state: GraphState) -> str:
    """Extract last user message content."""
    for m in reversed(state.get("messages", [])):
        role = getattr(m, "type", None) or getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None)
        if role in ("human", "user"):
            return getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else "") or ""
    return ""


def _has_selection_reference(text: str) -> bool:
    """Detect if user references selection (first 2, top 2, #2, second one, the vine one, etc.)."""
    t = (text or "").lower()
    # Index-based
    if re.search(r"\b(first|top|1st)\s*(2|two|2nd|3|three)\b", t):
        return True
    if re.search(r"\b(1|2|3)\s*(and|&)\s*(1|2|3)\b", t):
        return True
    if re.search(r"#\s*(1|2|3)\b", t):
        return True
    if re.search(r"\b(second|third|last)\s*(one|plant)?\b", t):
        return True
    if re.search(r"\b(first|second|third)\s*(one|two|three|plant)?\b", t):
        return True
    if "first two" in t or "first 2" in t or "top two" in t or "top 2" in t:
        return True
    # Descriptive (needs LLM)
    if re.search(r"\b(the|that)\s+(tall|short|vine|low.?light|pet.?safe|easy)\s+one\b", t):
        return True
    return False


def _has_compare_or_expand_intent(text: str) -> Tuple[bool, str]:
    """Return (has_intent, action) where action is 'compare' or 'expand' or ''."""
    t = (text or "").lower()
    if "compare" in t:
        return True, "compare"
    if any(x in t for x in ["recommend", "tell me more", "expand", "details about", "more about"]):
        return True, "expand"
    return False, ""


def _route_at_start(state: GraphState) -> str:
    """Route: pending -> profile_collector; selection ref + last_recs -> resolve_selection; filled -> recommender; else inject_context."""
    pending = state.get("pending_questions") or []
    profile_answers = state.get("profile_answers") or {}
    filled = any(v for v in profile_answers.values() if v and str(v).strip())
    last_recs = state.get("last_recommendations") or []
    user_msg = _get_last_user_message(state)
    has_sel = _has_selection_reference(user_msg)
    has_intent, action = _has_compare_or_expand_intent(user_msg)

    if pending:
        dest = "profile_collector"
    elif filled and len(last_recs) >= 2 and has_sel and has_intent:
        dest = "resolve_selection"
    elif filled:
        dest = "recommender"
    else:
        dest = "inject_context"
    print(f"[Route] pending={len(pending)}, filled={filled}, has_sel={has_sel} -> {dest}", flush=True)
    return dest


def inject_context(state: GraphState) -> dict:
    """Inject context for reasoning agent: username and/or completed profile from state."""
    username = state.get("username") or ""
    profile_answers = state.get("profile_answers") or {}
    filled = {k: v for k, v in profile_answers.items() if v is not None and str(v).strip()}

    parts = []
    if username:
        parts.append(f"The current user is '{username}'.")
    if filled:
        prefs = ", ".join(f"{k}: {v}" for k, v in filled.items())
        parts.append(
            f"The user has a completed profile with preferences: {prefs}. "
            "Use these for plant recommendations. Do NOT call check_user_profile_exists."
        )
    elif username:
        parts.append(
            f"Use check_user_profile_exists(\"{username}\") to check their profile. "
            "Do NOT treat the user's latest message as a new username."
        )

    if parts:
        return {"messages": [SystemMessage(content="System: " + " ".join(parts))]}
    return {}


def check_profile_node(state: GraphState) -> dict:
    """Directly call check_user_profile_exists and inject ToolMessage. Bypasses LLM for reliability."""
    username = state.get("username") or ""
    if not username:
        return {}
    result = check_user_profile_exists.invoke({"username": username})
    return {"messages": [ToolMessage(content=str(result), tool_call_id="check_profile_direct")]}


def _is_tool_message(m) -> bool:
    """Detect ToolMessage (object or dict). ToolMessage has type='tool'; name is often unset."""
    if isinstance(m, dict):
        return m.get("type") == "tool" or m.get("role") == "tool"
    return getattr(m, "type", None) == "tool" or "ToolMessage" in type(m).__name__


def _route_after_reasoning(state: GraphState) -> str:
    """Profile not found -> profile_init; profile exists in DB -> load_profile; profile in state -> end."""
    profile_answers = state.get("profile_answers") or {}
    if any(v for v in profile_answers.values() if v and str(v).strip()):
        return "end"

    for m in reversed(state.get("messages", [])):
        if not _is_tool_message(m):
            continue
        content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else "") or ""
        content_str = str(content)
        if "not_found" in content_str:
            return "profile_init"
        if "exists" in content_str:
            return "load_profile"
    return "end"


def _extract_username_from_tool_messages(messages: list) -> Optional[str]:
    """Extract username from check_user_profile_exists tool call in messages."""
    for m in reversed(messages):
        tool_calls = getattr(m, "tool_calls", None) or []
        for tc in tool_calls:
            name = getattr(tc, "name", None) or (tc.get("name") if isinstance(tc, dict) else None)
            if name == "check_user_profile_exists":
                args = getattr(tc, "args", None) or (tc.get("args") if isinstance(tc, dict) else {}) or {}
                if isinstance(args, dict) and "username" in args:
                    return args.get("username")
    return None


def _resolve_selection_rules(user_msg: str, last_recs: list) -> Optional[list]:
    """Rule-based selection resolution. Returns [name, name] or None if ambiguous."""
    t = (user_msg or "").lower()
    n = len(last_recs)
    if n < 2:
        return None
    names = [r.get("name", "") for r in last_recs if r.get("name")]
    if len(names) < 2:
        return None

    # "first 2", "top 2", "first two", "1 and 2"
    if re.search(r"\b(first|top|1st)\s*(2|two)\b", t) or "first two" in t or "first 2" in t or "top two" in t or "top 2" in t:
        return names[:2]
    # "2 and 3", "1 and 3"
    m = re.search(r"\b([123])\s*(and|&)\s*([123])\b", t)
    if m:
        i, j = int(m.group(1)), int(m.group(3))
        if 1 <= i <= n and 1 <= j <= n:
            return [names[i - 1], names[j - 1]]
    # "#2", "second", "second one"
    if re.search(r"#\s*([123])\b", t):
        m = re.search(r"#\s*([123])\b", t)
        if m:
            idx = int(m.group(1))
            if 1 <= idx <= n:
                return [names[idx - 1]]
    if "second" in t or "2nd" in t:
        if n >= 2:
            return [names[1]]
    if "third" in t or "3rd" in t:
        if n >= 3:
            return [names[2]]
    if "last" in t and n >= 1:
        return [names[-1]]
    return None


def resolve_selection_node(state: GraphState) -> dict:
    """Resolve user reference ('first 2', 'the low light one') to selected plant names. Rule-based first, LLM fallback."""
    last_recs = state.get("last_recommendations") or []
    user_msg = _get_last_user_message(state)
    if len(last_recs) < 2:
        return {"messages": [AIMessage(content="I don't have recent recommendations to select from. Would you like new recommendations?")], "selected_plants": []}

    names = [r.get("name", "") for r in last_recs if r.get("name")]
    selected = _resolve_selection_rules(user_msg, last_recs)

    if selected is None:
        # LLM fallback for descriptive refs ("the low light one", "that vine one")
        llm = get_ollama_llm(temperature=0)
        numbered = "\n".join(f"{i}. {n}" for i, n in enumerate(names, 1))
        prompt = f"""The user was shown these plants:
{numbered}

User said: "{user_msg}"

Which plant(s) do they mean? Reply with ONLY the plant name(s), one per line, nothing else. If unclear, reply "unclear"."""
        try:
            resp = llm.invoke([HumanMessage(content=prompt)])
            content = (getattr(resp, "content", None) or "").strip()
            if "unclear" in content.lower():
                return {"messages": [AIMessage(content="Which plants would you like me to compare? Please say the names or 'first 2', 'second one', etc.")], "selected_plants": []}
            lines = [l.strip() for l in content.splitlines() if l.strip()]
            selected = [l for l in lines if l in names][:2]
            if not selected:
                selected = names[:2]
        except Exception:
            selected = names[:2]

    if len(selected) < 2 and _has_compare_or_expand_intent(user_msg)[1] == "compare":
        selected = names[:2]
    return {"selected_plants": selected[:2]}


def compare_selected_node(state: GraphState) -> dict:
    """Compare the two selected plants using compare_plants tool."""
    selected = state.get("selected_plants") or []
    if len(selected) < 2:
        return {"messages": [AIMessage(content="I need two plants to compare. Which would you like?")], "recommendations": [], "greeting": None}
    result = compare_plants.invoke({"plant_a": selected[0], "plant_b": selected[1]})
    return {
        "messages": [AIMessage(content=str(result))],
        "recommendations": [],
        "greeting": None,
    }


def _route_after_resolve_selection(state: GraphState) -> str:
    """After resolve_selection: compare if 2 selected, else end."""
    selected = state.get("selected_plants") or []
    user_msg = _get_last_user_message(state)
    _, action = _has_compare_or_expand_intent(user_msg)
    if len(selected) >= 2 and action == "compare":
        return "compare_selected"
    return "end"


def load_profile_from_db(state: GraphState) -> dict:
    """Load profile from MongoDB and populate profile_answers for recommender."""
    username = state.get("username") or _extract_username_from_tool_messages(state.get("messages", []))
    if not username:
        return {}
    coll = _get_user_profiles_collection()
    if coll is None:
        return {}
    doc = coll.find_one({"username": username})
    if not doc:
        return {}
    prefs = doc.get("preferences", {}) or {}
    return {"profile_answers": prefs, "username": username}


def _route_after_profile_collector(state: GraphState) -> str:
    """When profile complete, go to recommender; else end (wait for user)."""
    pending = state.get("pending_questions") or []
    profile_answers = state.get("profile_answers") or {}
    filled = any(v for v in profile_answers.values() if v and str(v).strip())
    if not pending and filled:
        return "recommender"
    return "end"


graph = StateGraph(GraphState)
graph.add_node("inject_context", inject_context)
graph.add_node("check_profile", check_profile_node)
graph.add_node("profile_init", profile_init)
graph.add_node("profile_collector", profile_collector)
graph.add_node("recommender", recommender_node)
graph.add_node("resolve_selection", resolve_selection_node)
graph.add_node("compare_selected", compare_selected_node)
graph.add_conditional_edges(
    START,
    _route_at_start,
    {
        "profile_collector": "profile_collector",
        "recommender": "recommender",
        "inject_context": "inject_context",
        "resolve_selection": "resolve_selection",
    },
)
graph.add_conditional_edges(
    "resolve_selection",
    _route_after_resolve_selection,
    {"compare_selected": "compare_selected", "end": END},
)
graph.add_edge("compare_selected", END)
graph.add_edge("inject_context", "check_profile")
graph.add_node("load_profile", load_profile_from_db)
graph.add_conditional_edges(
    "check_profile",
    _route_after_reasoning,
    {"profile_init": "profile_init", "load_profile": "load_profile", "end": END},
)
graph.add_edge("load_profile", "recommender")
graph.add_edge("profile_init", "profile_collector")
graph.add_conditional_edges(
    "profile_collector",
    _route_after_profile_collector,
    {"recommender": "recommender", "end": END},
)
graph.add_edge("recommender", END)

app = graph.compile()


def _get_last_ai_content(messages: list) -> str | None:
    """Extract last AIMessage content from message list (supports objects and dicts)."""
    for m in reversed(messages):
        content = getattr(m, "content", None) if not isinstance(m, dict) else m.get("content")
        if not content:
            continue
        if isinstance(m, dict):
            if m.get("role") in ("assistant", "ai"):
                return str(content)
        elif "AIMessage" in type(m).__name__:
            return str(content)
    return None


def main():
    """Interactive chat with the agent. Type 'end' to exit."""
    print("Plant care assistant. Type your message and press Enter. Type 'end' to quit.\n")
    state = {"messages": []}

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() == "end":
            print("Goodbye.")
            break

        state["messages"].append({"role": "user", "content": user_input})
        result = app.invoke(state)
        state = result

        last_ai = _get_last_ai_content(result.get("messages", []))
        if last_ai:
            print(f"\nAssistant: {last_ai}\n")
        else:
            print("\nAssistant: (no response)\n")


if __name__ == "__main__":
    main()
