"""
LangChain-Ollama agents for plant care assistance.
Graph orchestration: reasoning agent + profile builder, connected by edges.
"""
import os
import sys
from pathlib import Path
from typing import Annotated, TypedDict, Optional

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


def _route_at_start(state: GraphState) -> str:
    """Pure conditional: if pending_questions non-empty, collect; else reasoning. No LLM."""
    pending = state.get("pending_questions") or []
    if pending:
        return "profile_collector"
    return "inject_context"


def inject_context(state: GraphState) -> dict:
    """Inject username context so reasoning agent uses stored user, not the latest message as username."""
    username = state.get("username") or ""
    if username:
        ctx = (
            f"System: The current user is '{username}'. Use check_user_profile_exists(\"{username}\") "
            "when checking their profile. Do NOT treat the user's latest message as a new username."
        )
        return {"messages": [SystemMessage(content=ctx)]}
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
    """If profile not found, go to profile_init; else end."""
    for m in reversed(state.get("messages", [])):
        if getattr(m, "name", None) == "check_user_profile_exists":
            content = getattr(m, "content", "") or ""
            if "not_found" in str(content):
                return "profile_init"
            return "end"
    return "end"


reasoning_agent = get_reasoning_agent()

graph = StateGraph(GraphState)
graph.add_node("reasoning_agent", reasoning_agent)
graph.add_node("inject_context", inject_context)
graph.add_node("profile_init", profile_init)
graph.add_node("profile_collector", profile_collector)
graph.add_conditional_edges(START, _route_at_start, {"profile_collector": "profile_collector", "inject_context": "inject_context"})
graph.add_edge("inject_context", "reasoning_agent")
graph.add_conditional_edges("reasoning_agent", _route_after_reasoning, {"profile_init": "profile_init", "end": END})
graph.add_edge("profile_init", "profile_collector")
graph.add_edge("profile_collector", END)

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
