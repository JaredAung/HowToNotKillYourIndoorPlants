import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv(Path(__file__).resolve().parent / ".env")

from agent.manager import get_profile_questions, app as graph_app, _get_last_ai_content

app = FastAPI(
    title="How To Not Kill Your Indoor Plants",
    description="API for indoor plant care tracking",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Welcome to How To Not Kill Your Indoor Plants API"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


def _check_profile_exists(username: str) -> bool:
    uri = os.getenv("MONGO_URI", "").strip().strip('"')
    database = os.getenv("MONGO_DATABASE", "HowNotToKillYourPlants")
    collection_name = os.getenv("MONGO_USER_PROFILES_COLLECTION", "UserProfiles")
    if not uri:
        return False
    coll = MongoClient(uri)[database][collection_name]
    return coll.find_one({"username": username}) is not None


@app.get("/api/user-info/questions")
async def get_user_info_questions(username: str = ""):
    """If profile doesn't exist for username, return questions to ask one by one. Otherwise return profile_exists=True."""
    if not username:
        raise HTTPException(400, "username required")
    if _check_profile_exists(username):
        return {"profile_exists": True, "questions": []}
    questions = get_profile_questions().split("\n")
    return {"profile_exists": False, "questions": questions}


# In-memory session store for chat state (use Redis/DB for production)
_chat_sessions: dict[str, dict] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Send a message and get the agent's response. State is persisted by session_id."""
    if not request.message.strip():
        raise HTTPException(400, "message cannot be empty")

    session_id = request.session_id or str(uuid.uuid4())
    state = _chat_sessions.get(session_id, {"messages": []})

    state["messages"] = state.get("messages", []) + [{"role": "user", "content": request.message.strip()}]
    result = graph_app.invoke(state)
    _chat_sessions[session_id] = result

    last_ai = _get_last_ai_content(result.get("messages", []))
    return ChatResponse(response=last_ai or "(no response)", session_id=session_id)
