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
from agent.recommender import _fetch_profile_from_db
from garden.death import apply_prevention_actions, remove_plant_from_garden, submit_death_report
from garden.garden import get_plant_by_id, get_user_garden_plants

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

    # First user message as username if not set (short message only)
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
    profile: dict = {}
    username: str | None = None


def _to_chat_response(session_id: str, result: dict) -> ChatResponse:
    """Convert graph result to ChatResponse."""
    messages = result.get("messages", [])
    response = _get_last_ai_content(messages) or ""
    profile = result.get("profile_answers", {}) or {}
    username = result.get("username")
    return ChatResponse(
        session_id=session_id,
        response=response,
        recommendations=result.get("recommendations", []),
        greeting=result.get("greeting"),
        profile=profile,
        username=username,
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Sync chat endpoint."""
    sid, session = _get_or_create_session(req.session_id)
    state = _build_state_from_session(session, req.message)
    result = graph_app.invoke(state)
    session["messages"] = result.get("messages", session["messages"])
    session["username"] = result.get("username", session["username"])
    session["profile_answers"] = result.get("profile_answers", session["profile_answers"])
    session["pending_questions"] = result.get("pending_questions", session.get("pending_questions", []))
    session["last_recommendations"] = result.get("last_recommendations", [])
    return _to_chat_response(sid, result)


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest):
    """Stream chat with progress events and final done payload."""

    def generate():
        sid, session = _get_or_create_session(req.session_id)
        state = _build_state_from_session(session, req.message)

        try:
            result = None
            for chunk in graph_app.stream(state, stream_mode="values"):
                result = chunk
                yield f"data: {json.dumps({'type': 'progress', 'label': 'Working...'})}\n\n"

            if result is None:
                result = graph_app.invoke(state)

            session["messages"] = result.get("messages", session["messages"])
            session["username"] = result.get("username", session["username"])
            session["profile_answers"] = result.get(
                "profile_answers", session["profile_answers"]
            )
            session["pending_questions"] = result.get(
                "pending_questions", session.get("pending_questions", [])
            )
            session["last_recommendations"] = result.get("last_recommendations", [])

            resp = _to_chat_response(sid, result)
            yield f"data: {json.dumps({'type': 'done', 'data': resp.model_dump()})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/api/plant/{plant_id}")
def get_plant(plant_id: str):
    """Fetch plant details by ID."""
    plant = get_plant_by_id(plant_id)
    if not plant:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Plant not found")
    return plant


class DeathReportRequest(BaseModel):
    username: str
    plant_id: str
    cause: list[str]
    details: str = ""
    reminder_system: bool = False
    where_placed: str = ""
    last_watered_date: str | None = None
    travel_away: bool = False


@app.post("/api/death-report")
def death_report(req: DeathReportRequest):
    """Submit death report via JSON body."""
    return submit_death_report(
        username=req.username,
        plant_id=req.plant_id,
        cause=req.cause,
        details=req.details,
        reminder_system=req.reminder_system,
        where_placed=req.where_placed,
        last_watered_date=req.last_watered_date,
        travel_away=req.travel_away,
    )


class ApplyPreventionsRequest(BaseModel):
    username: str
    selected_preventions: list[str]
    plant_id: str | None = None


@app.post("/api/death-report/apply-preventions")
def apply_preventions(req: ApplyPreventionsRequest):
    """Apply selected prevention actions to the user's profile and remove the plant from garden."""
    result = apply_prevention_actions(req.username, req.selected_preventions)
    if req.plant_id and req.username:
        removed = remove_plant_from_garden(req.username.strip(), req.plant_id.strip())
        result["plant_removed"] = removed
    return result


@app.delete("/api/garden")
def delete_garden_plant(username: str = "", plant_id: str = ""):
    """Remove a plant from user's garden."""
    if not username or not plant_id:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="username and plant_id required")
    removed = remove_plant_from_garden(username.strip(), plant_id.strip())
    if not removed:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Plant not found in garden")
    return {"ok": True}


@app.get("/api/garden")
def get_garden(username: str = ""):
    """Fetch user's garden plants by username."""
    if not username or not username.strip():
        return {"plants": []}
    plants = get_user_garden_plants(username.strip())
    return {"plants": plants}


@app.get("/api/profile")
def get_profile(username: str = ""):
    """Fetch user's profile by username."""
    if not username or not username.strip():
        return {"profile": {}}
    profile = _fetch_profile_from_db(username.strip())
    return {"profile": profile or {}}


@app.get("/api/user-info/questions")
def get_questions():
    """Return profile questions for the frontend."""
    return {"questions": get_profile_questions() or []}


@app.get("/health")
def health():
    return {"status": "ok"}
