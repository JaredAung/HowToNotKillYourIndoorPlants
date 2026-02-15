"""
LangChain-Ollama agents for plant care assistance.
Graph orchestration: reasoning agent + profile builder, connected by edges.
"""
import os
import sys
from pathlib import Path
from typing import Annotated, NotRequired, TypedDict, Optional

# Allow running as python agent/manager.py from backend/
_backend = Path(__file__).resolve().parent.parent
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langchain.agents import create_agent
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
from agent.recommender import recommender_node


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


def _route_at_start(state: GraphState) -> str:
    """Route: pending -> profile_collector; profile complete -> recommender; else reasoning."""
    pending = state.get("pending_questions") or []
    profile_answers = state.get("profile_answers") or {}
    filled = any(v for v in profile_answers.values() if v and str(v).strip())

    if pending:
        return "profile_collector"
    if filled:
        return "recommender"
    return "inject_context"


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


def get_reasoning_agent():
    """Create and return a ReAct reasoning agent. Only checks if profile exists."""
    llm = get_ollama_llm()
    tools = [check_user_profile_exists]
    return create_agent(
        model=llm,
        tools=tools,
        system_prompt="""You are a helpful plant recommendation assistant. Use check_user_profile_exists with the username to see if they have a saved profile. If profile EXISTS, use their preferences to recommend plants. If profile does NOT exist (returns 'not_found'), say you'll pass them to the user builder to collect their preferences. Do not generate questions yourself—the user builder will handle that.""",
    )


def _route_after_reasoning(state: GraphState) -> str:
    """Profile not found -> profile_init; profile exists in DB -> load_profile; profile in state -> end."""
    profile_answers = state.get("profile_answers") or {}
    if any(v for v in profile_answers.values() if v and str(v).strip()):
        return "end"

    for m in reversed(state.get("messages", [])):
        if getattr(m, "name", None) == "check_user_profile_exists":
            content = getattr(m, "content", "") or ""
            if "not_found" in str(content):
                return "profile_init"
            if "exists" in str(content):
                return "load_profile"
            return "end"
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


reasoning_agent = get_reasoning_agent()

def _route_after_profile_collector(state: GraphState) -> str:
    """When profile complete, go to recommender; else end (wait for user)."""
    pending = state.get("pending_questions") or []
    profile_answers = state.get("profile_answers") or {}
    filled = any(v for v in profile_answers.values() if v and str(v).strip())
    if not pending and filled:
        return "recommender"
    return "end"


graph = StateGraph(GraphState)
graph.add_node("reasoning_agent", reasoning_agent)
graph.add_node("inject_context", inject_context)
graph.add_node("profile_init", profile_init)
graph.add_node("profile_collector", profile_collector)
graph.add_node("recommender", recommender_node)
graph.add_conditional_edges(
    START,
    _route_at_start,
    {"profile_collector": "profile_collector", "recommender": "recommender", "inject_context": "inject_context"},
)
graph.add_edge("inject_context", "reasoning_agent")
graph.add_node("load_profile", load_profile_from_db)
graph.add_conditional_edges(
    "reasoning_agent",
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
    """Extract last AIMessage content from message list."""
    for m in reversed(messages):
        if hasattr(m, "content") and m.content and "AIMessage" in type(m).__name__:
            return m.content
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
