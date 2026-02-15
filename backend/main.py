"""
FastAPI backend for How To Not Kill Your Indoor Plants.
Chat API (sync and streaming) with LangGraph agent.
"""
import json
import uuid
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from agent.manager import app as graph_app, get_profile_questions, _get_last_ai_content

app = FastAPI(title="How To Not Kill Your Indoor Plants API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session state: session_id -> {messages, username, profile_answers, last_recommendations, ...}
_chat_sessions: dict[str, dict[str, Any]] = {}

# Node names -> user-friendly labels for progress display
NODE_LABELS: dict[str, str] = {
    "inject_context": "Preparing context...",
    "check_profile": "Checking your profile...",
    "profile_init": "Setting up your profile...",
    "profile_collector": "Collecting your preferences...",
    "load_profile": "Loading your profile...",
    "recommender": "Finding plant recommendations...",
    "resolve_selection": "Resolving your selection...",
    "resolve_single_plant": "Identifying plant...",
    "compare_selected": "Comparing plants...",
    "explain_plant": "Getting plant details...",
    "pick_plant": "Adding to your garden...",
}


def _get_or_create_session(session_id: str | None) -> tuple[str, dict]:
    """Get existing session or create new one. Returns (session_id, state)."""
    sid = session_id or str(uuid.uuid4())
    if sid not in _chat_sessions:
        questions = get_profile_questions()
        _chat_sessions[sid] = {
            "messages": [],
            "username": None,
            "profile_answers": {},
            "pending_questions": list(questions) if questions else [],
            "last_recommendations": [],
        }
    return sid, _chat_sessions[sid]


def _build_state_from_session(session: dict, user_message: str) -> dict:
    """Build GraphState from session + new user message."""
    messages = list(session.get("messages", []))
    messages.append(HumanMessage(content=user_message))

    # First user message as username if not set
    username = session.get("username")
    if not username and messages:
        first_content = getattr(messages[0], "content", None) or (messages[0].get("content") if isinstance(messages[0], dict) else "")
        if first_content and len(str(first_content).strip()) <= 50:
            username = str(first_content).strip()

    return {
        "messages": messages,
        "username": username or session.get("username"),
        "profile_answers": session.get("profile_answers", {}),
        "pending_questions": session.get("pending_questions", []),
        "last_recommendations": session.get("last_recommendations", []),
    }


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    response: str
    recommendations: list[dict] = []
    greeting: str | None = None


def _to_chat_response(session_id: str, result: dict) -> ChatResponse:
    """Convert graph result to ChatResponse."""
    messages = result.get("messages", [])
    last_ai = _get_last_ai_content(messages)
    recs = result.get("recommendations", [])
    greeting = result.get("greeting")

    # Normalize recommendations
    out_recs = []
    for r in recs:
        name = r.get("name") or r.get("latin", "Plant")
        out_recs.append({
            "name": name,
            "image_url": r.get("image_url") or r.get("img_url", ""),
            "explanation": r.get("explanation", ""),
        })

    return ChatResponse(
        session_id=session_id,
        response=last_ai or "",
        recommendations=out_recs,
        greeting=greeting,
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Sync chat endpoint. Processes message and returns full response."""
    session_id, session = _get_or_create_session(request.session_id)
    state = _build_state_from_session(session, request.message)

    result = graph_app.invoke(state)

    # Update session from result
    session["messages"] = result.get("messages", session["messages"])
    session["username"] = result.get("username", session["username"])
    session["profile_answers"] = result.get("profile_answers", session["profile_answers"])
    session["pending_questions"] = result.get("pending_questions", session["pending_questions"])
    session["last_recommendations"] = result.get("last_recommendations", session["last_recommendations"])

    return _to_chat_response(session_id, result)


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest):
    """Streaming chat endpoint. Sends progress events via SSE, then final response."""
    session_id, session = _get_or_create_session(request.session_id)
    state = _build_state_from_session(session, request.message)

    async def generate():
        result = None
        try:
            # Stream both updates (for progress) and values (for final state)
            async for mode, chunk in graph_app.astream(
                state, stream_mode=["updates", "values"]
            ):
                if mode == "updates":
                    # chunk: {node_name: update} or (namespace, {node_name: update})
                    if isinstance(chunk, tuple):
                        _, node_updates = chunk
                    else:
                        node_updates = chunk
                    for node_name, _ in (node_updates or {}).items():
                        label = NODE_LABELS.get(
                            node_name, node_name.replace("_", " ").title()
                        )
                        yield f"data: {json.dumps({'type': 'progress', 'label': label})}\n\n"
                elif mode == "values":
                    result = chunk

            if result is None:
                result = graph_app.invoke(state)

            # Update session from final result
            session["messages"] = result.get("messages", session["messages"])
            session["username"] = result.get("username", session["username"])
            session["profile_answers"] = result.get(
                "profile_answers", session["profile_answers"]
            )
            session["pending_questions"] = result.get(
                "pending_questions", session["pending_questions"]
            )
            session["last_recommendations"] = result.get(
                "last_recommendations", session["last_recommendations"]
            )

            resp = _to_chat_response(session_id, result)
            yield f"data: {json.dumps({'type': 'done', 'data': resp.model_dump()})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/user-info/questions")
def get_questions():
    """Return profile questions for the frontend."""
    return {"questions": get_profile_questions() or []}


@app.get("/health")
def health():
    return {"status": "ok"}
