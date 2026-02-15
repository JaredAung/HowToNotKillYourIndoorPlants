"""
FastAPI backend for How To Not Kill Your Indoor Plants.
Chat API (sync and streaming) with LangGraph agent.
Garden lookup/chat, plant chat with voice (ElevenLabs), death reports.
"""
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from pymongo import MongoClient

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv(Path(__file__).resolve().parent / ".env")

from agent.manager import app as graph_app, get_profile_questions, _get_last_ai_content
from agent.recommender import _fetch_profile_from_db
from garden.death import apply_prevention_actions, remove_plant_from_garden, submit_death_report
from garden.garden import get_plant_by_id, get_user_garden_plants
from garden.plant_chat import build_plant_persona, generate_plant_reply, synthesize_plant_speech

app = FastAPI(
    title="How To Not Kill Your Indoor Plants",
    description="API for indoor plant care tracking",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000"],
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


# --- Session & chat ---
_chat_sessions: dict[str, dict[str, Any]] = {}


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
    """If username given: return profile_exists + questions when profile missing. If no username: return questions only."""
    if not username:
        return {"questions": (get_profile_questions() or "").split("\n")}
    if _check_profile_exists(username):
        return {"profile_exists": True, "questions": []}
    questions = get_profile_questions().split("\n")
    return {"profile_exists": False, "questions": questions}


def _get_or_create_session(session_id: str | None) -> tuple[str, dict]:
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
    messages = list(session.get("messages", []))
    messages.append(HumanMessage(content=user_message))
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
    username: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    recommendations: list[dict] = []
    greeting: str | None = None
    profile: dict = {}
    username: str | None = None


def _to_chat_response(session_id: str, result: dict) -> ChatResponse:
    messages = result.get("messages", [])
    response = _get_last_ai_content(messages) or ""
    profile = result.get("profile_answers", {}) or {}
    username = result.get("username")
    return ChatResponse(
        response=response,
        session_id=session_id,
        recommendations=result.get("recommendations", []),
        greeting=result.get("greeting"),
        profile=profile,
        username=username,
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Sync chat endpoint."""
    sid, session = _get_or_create_session(req.session_id)
    if req.username:
        session["username"] = req.username
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
        if req.username:
            session["username"] = req.username
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
            session["profile_answers"] = result.get("profile_answers", session["profile_answers"])
            session["pending_questions"] = result.get("pending_questions", session.get("pending_questions", []))
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


# --- Garden lookup & chat (Solocup) ---
def _get_garden_database() -> tuple[Any | None, str | None]:
    mongo_uri = os.getenv("MONGO_URI", "").strip().strip('"')
    db_name = os.getenv("MONGO_DATABASE", os.getenv("MONGO_DB_NAME", "HowNotToKillYourPlants"))
    if not mongo_uri:
        return None, "Garden is unavailable right now. Please try again later."
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
        client.admin.command("ping")
        return client[db_name], None
    except Exception:
        return None, "Garden is unavailable right now. Please try again later."


def _get_user_garden_doc(db: Any, username: str) -> dict | None:
    coll_name = os.getenv("MONGO_USER_GARDEN_COLLECTION", "User_Garden_Collection")
    coll = db[coll_name]
    doc = coll.find_one({"username": username})
    if doc:
        return doc
    escaped = re.escape(username)
    return coll.find_one({"username": {"$regex": f"^{escaped}$", "$options": "i"}})


def _serialize_plant_doc(plant: dict | None) -> dict | None:
    if not isinstance(plant, dict):
        return None
    common = plant.get("common")
    if isinstance(common, list):
        common_names = [str(x).strip() for x in common if str(x).strip()]
    elif isinstance(common, str) and common.strip():
        common_names = [common.strip()]
    else:
        common_names = []
    description = plant.get("description") if isinstance(plant.get("description"), dict) else {}
    care_guidelines = plant.get("care_guidelines") if isinstance(plant.get("care_guidelines"), dict) else {}
    return {
        "id": plant.get("id"),
        "latin": plant.get("latin"),
        "common": common_names,
        "climate": plant.get("climate"),
        "ideallight": plant.get("ideallight"),
        "toleratedlight": plant.get("toleratedlight"),
        "watering": plant.get("watering"),
        "care_level": plant.get("care_level"),
        "care_guidelines": care_guidelines,
        "description": description,
        "image_url": plant.get("image_url"),
    }


def _resolve_garden_plant(db: Any, user_doc: dict) -> tuple[dict | None, int | None]:
    plants_coll_name = os.getenv("MONGO_PLANTS_COLLECTION", os.getenv("MONGO_COLLECTION", "PlantsRawCollection"))
    plants_coll = db[plants_coll_name]
    plant_id: int | None = None
    raw_id = user_doc.get("plant_id")
    if isinstance(raw_id, int):
        plant_id = raw_id
    elif isinstance(raw_id, str) and raw_id.isdigit():
        plant_id = int(raw_id)

    if plant_id is not None:
        found = plants_coll.find_one({"id": plant_id})
        if found:
            return _serialize_plant_doc(found), plant_id

    embedded = user_doc.get("plant")
    if isinstance(embedded, dict):
        embedded_id = embedded.get("id")
        if isinstance(embedded_id, int):
            found = plants_coll.find_one({"id": embedded_id})
            if found:
                return _serialize_plant_doc(found), embedded_id
        return _serialize_plant_doc(embedded), embedded_id if isinstance(embedded_id, int) else plant_id

    return None, plant_id


def _plant_name(plant: dict) -> str:
    common = plant.get("common") or []
    if isinstance(common, list) and common:
        return str(common[0])
    latin = plant.get("latin")
    return str(latin) if latin else "your plant"


def _contains_any(text: str, words: list[str]) -> bool:
    low = text.lower()
    return any(word in low for word in words)


def _temperament_from_climate(climate: str) -> str:
    c = (climate or "").lower()
    if "humid" in c:
        return "sensitive"
    if "arid" in c:
        return "stoic"
    if "tropical" in c:
        return "expressive"
    if "subtropical" in c:
        return "balanced"
    return "calm"


def _strictness_from_care_level(care_level: str) -> str:
    c = (care_level or "").lower()
    if "high" in c or "hard" in c:
        return "high standards"
    if "easy" in c or "low" in c:
        return "forgiving"
    return "realistic"


def _watering_style(watering_text: str) -> str:
    w = (watering_text or "").lower()
    if _contains_any(w, ["frequent", "moist", "daily"]):
        return "anxious"
    if _contains_any(w, ["dry", "drought", "infrequent", "minimal"]):
        return "independent"
    if _contains_any(w, ["moderate", "weekly"]):
        return "balanced"
    return "chill"


def _recommended_actions(plant: dict) -> list[str]:
    actions: list[str] = []
    if plant.get("watering"):
        actions.append("Check soil moisture")
    if plant.get("ideallight"):
        actions.append("Move to brighter indirect light")
    if plant.get("toleratedlight"):
        actions.append("Avoid sudden light changes")
    care_guidelines = plant.get("care_guidelines") if isinstance(plant.get("care_guidelines"), dict) else {}
    if care_guidelines.get("humidity"):
        actions.append("Monitor humidity")
    if care_guidelines.get("temperature"):
        actions.append("Keep temperature stable")
    return actions[:3] or ["Observe leaf health today"]


def _build_plant_reply(message: str, plant: dict, history: list[dict[str, str]] | None) -> tuple[str, str, list[str]]:
    name = _plant_name(plant)
    climate = str(plant.get("climate") or "")
    care_level = str(plant.get("care_level") or "")
    watering = str(plant.get("watering") or "")
    description = plant.get("description") if isinstance(plant.get("description"), dict) else {}
    symbolism = str(description.get("symbolism") or "").strip()
    interesting = str(description.get("interesting_fact") or "").strip()

    temperament = _temperament_from_climate(climate)
    strictness = _strictness_from_care_level(care_level)
    watering_vibe = _watering_style(watering)
    actions = _recommended_actions(plant)

    if _contains_any(message, ["how do i care", "how to care", "care for you", "watering", "light", "sun"]):
        ideal = str(plant.get("ideallight") or "bright indirect light")
        tolerated = str(plant.get("toleratedlight") or "some flexible light conditions")
        water = str(plant.get("watering") or "steady, moderate watering")
        guidelines = plant.get("care_guidelines") if isinstance(plant.get("care_guidelines"), dict) else {}
        if guidelines:
            first_key = next(iter(guidelines.keys()))
            first_val = guidelines.get(first_key)
            extra = f" Also, {first_key}: {first_val}."
        else:
            extra = ""
        reply = (
            f"I'm {name}, and I thrive with {ideal} and {water}. "
            f"I can tolerate {tolerated}, but consistency helps me feel great.{extra}"
        )
        return reply, "supportive", actions

    if _contains_any(message, ["weather", "movie", "news", "sports", "politics", "stock", "crypto", "joke"]):
        suffix = f" {interesting}" if interesting else ""
        reply = (
            f"I'm {name}, so I mostly think about light and watering.{suffix} "
            "Tell me how my spot feels and I'll help us grow together."
        )
        return reply, "playful", actions

    tone_map = {
        "expressive": "cheerful",
        "stoic": "proud",
        "calm": "sleepy",
        "sensitive": "concerned",
        "balanced": "calm",
    }
    tone_tag = tone_map.get(temperament, "calm")
    memory_line = " I remember our earlier chat." if history else ""
    symbolism_line = f" {symbolism}" if symbolism else ""
    reply = (
        f"Hi, I'm {name} - your {temperament} plant friend with {strictness} needs and a {watering_vibe} watering style.{memory_line} "
        f"Stay close to my care rhythm and we'll do great together.{symbolism_line}"
    )
    return reply, tone_tag, actions


class GardenLookupRequest(BaseModel):
    username: str


class GardenLookupResponse(BaseModel):
    found: bool
    username: str | None = None
    message: str | None = None
    plant_id: int | None = None
    plant: dict | None = None
    resolved_plant: dict | None = None


class GardenChatHistoryItem(BaseModel):
    role: Literal["user", "plant"]
    content: str


class GardenChatRequest(BaseModel):
    username: str
    message: str
    history: list[GardenChatHistoryItem] | None = None


class GardenChatResponse(BaseModel):
    reply: str
    tone_tag: str
    recommended_actions: list[str]
    plant_id: int


@app.post("/api/garden/lookup", response_model=GardenLookupResponse)
async def garden_lookup(request: GardenLookupRequest):
    username = request.username.strip()
    if not username:
        raise HTTPException(400, "username required")

    db, err = _get_garden_database()
    if err or db is None:
        raise HTTPException(503, err or "Garden is unavailable right now. Please try again later.")

    user_doc = _get_user_garden_doc(db, username)
    if not user_doc:
        return GardenLookupResponse(
            found=False,
            message="You must first pick a plant through the original start screen.",
        )

    resolved_plant, plant_id = _resolve_garden_plant(db, user_doc)
    embedded = _serialize_plant_doc(user_doc.get("plant") if isinstance(user_doc.get("plant"), dict) else None)
    return GardenLookupResponse(
        found=True,
        username=str(user_doc.get("username") or username),
        plant_id=plant_id,
        plant=embedded,
        resolved_plant=resolved_plant,
    )


@app.post("/api/garden/chat", response_model=GardenChatResponse)
async def garden_chat(request: GardenChatRequest):
    username = request.username.strip()
    message = request.message.strip()
    if not username:
        raise HTTPException(400, "username required")
    if not message:
        raise HTTPException(400, "message cannot be empty")

    db, err = _get_garden_database()
    if err or db is None:
        raise HTTPException(503, err or "Garden is unavailable right now. Please try again later.")

    user_doc = _get_user_garden_doc(db, username)
    if not user_doc:
        raise HTTPException(404, "You must first pick a plant through the original start screen.")

    resolved_plant, plant_id = _resolve_garden_plant(db, user_doc)
    if not resolved_plant:
        raise HTTPException(
            400,
            "We found your Garden entry, but couldn't load your plant. Please reselect a plant from the start screen.",
        )

    history = [
        {"role": item.role, "content": item.content.strip()}
        for item in (request.history or [])
        if item.content and item.content.strip()
    ]
    reply, tone_tag, actions = _build_plant_reply(message, resolved_plant, history)
    resolved_id = plant_id
    if resolved_id is None and isinstance(resolved_plant.get("id"), int):
        resolved_id = int(resolved_plant.get("id"))
    return GardenChatResponse(
        reply=reply,
        tone_tag=tone_tag,
        recommended_actions=actions,
        plant_id=resolved_id or 0,
    )


# --- Plant chat with LLM + TTS ---
class PlantChatMessage(BaseModel):
    role: str
    content: str


class PlantChatRequest(BaseModel):
    plant: dict
    userMessage: str
    history: list[PlantChatMessage] = []


class PlantChatResponse(BaseModel):
    reply: str
    tone_tag: str
    recommended_actions: list[str]
    audio_base64: str | None = None


class PlantTTSRequest(BaseModel):
    text: str


class PlantTTSResponse(BaseModel):
    audio_base64: str | None = None


@app.post("/api/garden/plant-chat", response_model=PlantChatResponse)
def garden_plant_chat(request: PlantChatRequest):
    """Plant chat endpoint used by Garden chat UI (Ollama + optional ElevenLabs TTS)."""
    message = (request.userMessage or "").strip()
    if not message:
        return PlantChatResponse(
            reply="I'm feeling a little quiet today 🌿",
            tone_tag="quiet",
            recommended_actions=[],
        )

    persona = build_plant_persona(request.plant or {})
    history = [{"role": m.role, "content": m.content} for m in (request.history or [])]
    result = generate_plant_reply(
        plant=request.plant or {},
        persona=persona,
        user_message=message,
        history=history,
    )
    return PlantChatResponse(
        reply=result.get("reply", "I'm feeling a little quiet today 🌿"),
        tone_tag=result.get("tone_tag", "quiet"),
        recommended_actions=result.get("recommended_actions", []),
        audio_base64=result.get("audio_base64"),
    )


@app.post("/api/garden/plant-tts", response_model=PlantTTSResponse)
def garden_plant_tts(request: PlantTTSRequest):
    text = (request.text or "").strip()
    if not text:
        raise HTTPException(400, "text is required")
    return PlantTTSResponse(audio_base64=synthesize_plant_speech(text))


# --- Plant, garden, death, profile ---
@app.get("/api/plant/{plant_id}")
def get_plant(plant_id: str):
    """Fetch plant details by ID."""
    plant = get_plant_by_id(plant_id)
    if not plant:
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


@app.get("/api/garden")
def get_garden(username: str = ""):
    """Fetch user's garden plants by username."""
    if not username or not username.strip():
        return {"plants": []}
    plants = get_user_garden_plants(username.strip())
    return {"plants": plants}


@app.delete("/api/garden")
def delete_garden_plant(username: str = "", plant_id: str = ""):
    """Remove a plant from user's garden."""
    if not username or not plant_id:
        raise HTTPException(status_code=400, detail="username and plant_id required")
    removed = remove_plant_from_garden(username.strip(), plant_id.strip())
    if not removed:
        raise HTTPException(status_code=404, detail="Plant not found in garden")
    return {"ok": True}


@app.get("/api/profile")
def get_profile(username: str = ""):
    """Fetch user's profile by username."""
    if not username or not username.strip():
        return {"profile": {}}
    profile = _fetch_profile_from_db(username.strip())
    return {"profile": profile or {}}


