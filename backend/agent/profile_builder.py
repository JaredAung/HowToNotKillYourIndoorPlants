"""
Profile builder agent: schema, extraction, and profile collection logic.
Collects user plant preferences via freeform text + follow-up questions.
"""
import json
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage

from agent.llm import get_ollama_llm

# Valid use values from house_plants_enriched_schema.json (placement/style)
USE_VALUES = [
    "Colors / Forms", "Flower", "Ground cover", "Hanging", "Potted plant",
    "Primary", "Secondary", "Table top", "Tertiary",
]

# Valid climate values from house_plants_enriched_schema.json
CLIMATE_VALUES = [
    "Arid Tropical", "Subtropical", "Subtropical arid", "Tropical",
    "Tropical humid", "Tropical to subtropical",
]

# User profile schema - fields to collect for plant recommendations
USER_PROFILE_SCHEMA = {
    "experience_level": "plant care experience (beginner, intermediate, expert)",
    "time_to_commit": "time available for plant care per week",
    "symbolism": "why they want plants (decor, air quality, hobby)",
    "use": "placement/style: one of " + ", ".join(USE_VALUES),
    "climate": "climate zone: one of " + ", ".join(CLIMATE_VALUES),
    "average_room_temp": "typical room temperature (celsius or fahrenheit)",
    "average_sunlight_type": "type of sunlight (direct, indirect, low, bright)",
    "average_sunlight_time": "hours of sunlight per day",
    "watering_preferences": "watering frequency preference (frequent, moderate, minimal)",
    "humidity_level": "humidity level (dry, moderate, humid)",
    "max_plant_size_preference": "max preferred plant size (small, medium, large)",
    "physical_desc": "optional: user's preference for how the plant should look (e.g. bright leaves, tall, trailing, colorful, variegated). Only extract if mentioned.",
}

# Map each question to its schema key (must match PROFILE_QUESTIONS 1:1)
QUESTION_TO_SCHEMA_KEY = [
    "experience_level", "time_to_commit", "symbolism",
    "use", "climate", "average_room_temp", "average_sunlight_type", "average_sunlight_time",
    "watering_preferences", "humidity_level", "max_plant_size_preference",
]

PROFILE_QUESTIONS = [
    "What's your plant care experience level? (beginner, intermediate, expert)",
    "How much time can you commit to plant care per week?",
    "What motivates you to get plants? (e.g. decor, air quality, hobby)",
    "What placement or style do you want? (Potted plant, Table top, Hanging, Flower, Ground cover, Colors/Forms, Primary, Secondary, Tertiary)",
    "What's your local climate? (Tropical, Tropical humid, Subtropical, Arid Tropical, etc.)",
    "What's the average room temperature? (celsius or fahrenheit)",
    "What type of sunlight does your space get? (direct, indirect, low, bright)",
    "How many hours of sunlight per day does your space get?",
    "What's your watering preference? (frequent, moderate, minimal)",
    "What's the humidity level? (dry, moderate, humid)",
    "What's your max preferred plant size? (small, medium, large)",
]

FREEFORM_SENTINEL = "__freeform__"

# Map extracted keys (LLM may use alternate names) to QUESTION_TO_SCHEMA_KEY
EXTRACTED_KEY_TO_SCHEMA = {}


def get_profile_questions() -> str:
    """Returns profile questions as newline-separated string."""
    return "\n".join(PROFILE_QUESTIONS)


def _extract_json_from_response(raw: str) -> Optional[dict]:
    """Extract JSON object from LLM response (handles markdown, extra text)."""
    raw = (raw or "").strip()
    if "```" in raw:
        parts = raw.split("```")
        for p in parts:
            p = p.strip()
            if p.lower().startswith("json"):
                p = p[4:].strip()
            if p.startswith("{"):
                raw = p
                break
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    for i, c in enumerate(raw[start:], start):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start : i + 1])
                except json.JSONDecodeError:
                    pass
    return None


def _extract_fields_from_text(text: str) -> dict:
    """Use LLM to extract schema fields from user's freeform text. Returns dict of key -> value."""
    llm = get_ollama_llm(temperature=0.2)
    keys = list(USER_PROFILE_SCHEMA.keys())
    prompt = f"""Extract plant-care profile fields from this text. Return a JSON object with these keys (use null if not mentioned).

CRITICAL: Do NOT infer or guess. Only include a field if the user explicitly mentions it. 
E.g. location does NOT imply room temp, sunlight, or experience level. "Desk plant" is use only—NOT physical_desc. "Dry climate" is climate only—NOT watering. Use null for anything not directly stated.

Keys: {keys}

Examples from text:
- mention of temperature -> average_room_temp: "40°C"
- mention of sunlight type -> average_sunlight_type: "direct"
- mention of sunlight hours -> average_sunlight_time: "4"
- mention of plant size -> max_plant_size_preference: "small"
- mention of plant description -> physical_desc: "bright leaves" or "tall" or "trailing"
- mention of experience level -> experience_level: "beginner"
- mention of use -> use: "Table top" or "Potted plant"
- mention of climate -> climate: "Subtropical" or "Tropical humid"

User text: "{text}"

Return ONLY valid JSON, no other text."""
    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        raw_resp = getattr(resp, "content", None) or ""
        data = _extract_json_from_response(raw_resp)
        if data is None:
            print("[Extract] Failed to parse JSON. Raw response:", raw_resp[:500])
            return {}
        result = {}
        for k, v in data.items():
            if k not in USER_PROFILE_SCHEMA or v is None or not str(v).strip():
                continue
            result[k] = str(v).strip()
        return result
    except Exception as e:
        print("[Extract] Error:", e)
        return {}


def _normalize_answer(raw_answer: str, question: str, schema_key: str) -> str:
    """Use LLM to normalize user's raw answer to a standardized format for the schema field."""
    schema_hint = USER_PROFILE_SCHEMA.get(schema_key, "")
    llm = get_ollama_llm(temperature=0.2)
    prompt = f"""Normalize this user answer for a plant care profile. Return ONLY the normalized value, nothing else.

Question: {question}
Schema field: {schema_key}
Schema hint: {schema_hint}

User's raw answer: "{raw_answer}"

Rules:
- experience_level: "beginner", "intermediate", or "expert"
- watering_preferences: "frequent", "moderate", or "minimal"
- time to commit: numeric value (2,3)
- average_sunlight_type: "direct", "indirect", "low", or "bright"
- humidity_level: "dry", "moderate", or "humid"
- max_plant_size_preference: "small", "medium", or "large"
- average_room_temp: keep number and unit (e.g. "72°F" or "22°C")
- use: map to exactly one of: Colors / Forms, Flower, Ground cover, Hanging, Potted plant, Primary, Secondary, Table top, Tertiary. Examples: indoor decor/bedroom/office -> Potted plant or Table top; desk/side table -> Table top; hanging basket -> Hanging; focal point/statement -> Primary; accent/filler -> Secondary or Tertiary; colorful/variegated -> Colors / Forms
- climate: map to exactly one of: Arid Tropical, Subtropical, Subtropical arid, Tropical, Tropical humid, Tropical to subtropical. Examples: SF/NYC/LA -> Subtropical; Miami/Singapore -> Tropical humid; Phoenix/desert -> Arid Tropical; humid tropics -> Tropical humid
- For other fields: keep essence, be concise
- If empty, return "N/A" """

    try:
        resp = llm.invoke([HumanMessage(content=prompt)])
        normalized = (getattr(resp, "content", None) or "").strip().strip('"')
        return normalized if normalized else raw_answer
    except Exception:
        return raw_answer


def _extract_username_from_messages(messages: list) -> Optional[str]:
    """Extract username from check_user_profile_exists tool call in messages."""
    for m in reversed(messages):
        tool_calls = getattr(m, "tool_calls", None) or []
        for tc in tool_calls:
            name = getattr(tc, "name", None) or (tc.get("name") if isinstance(tc, dict) else None)
            if name == "check_user_profile_exists":
                args = getattr(tc, "args", None) or (tc.get("args") if isinstance(tc, dict) else {}) or {}
                if isinstance(args, dict) and "username" in args:
                    return args["username"]
    for m in messages:
        role = getattr(m, "type", None) or getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None)
        if role in ("human", "user"):
            content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else None)
            if content and isinstance(content, str) and len(content.strip()) < 50:
                return content.strip()
    return None


def _get_last_user_content(messages: list) -> Optional[str]:
    """Extract content of last user/human message."""
    for m in reversed(messages):
        content = getattr(m, "content", None)
        if content is None and isinstance(m, dict):
            content = m.get("content")
        if content:
            role = getattr(m, "type", None) or getattr(m, "role", None)
            if role is None and isinstance(m, dict):
                role = m.get("role", m.get("type"))
            if role in ("human", "user"):
                return str(content)
    return None


def _init_profile_null() -> dict:
    """Initialize all schema fields to null."""
    return {k: None for k in QUESTION_TO_SCHEMA_KEY}


def _get_null_fields(answers: dict) -> list[str]:
    """Return schema keys that are still null (need to be asked)."""
    return [k for k in QUESTION_TO_SCHEMA_KEY if answers.get(k) is None or str(answers.get(k)).strip() == ""]


def _complete_profile(answers: dict, username: str) -> dict:
    """Save profile to Mongo, reiterate captured info, return completion state. Keeps profile_answers so reasoning agent can use them."""
    filled = {k: v for k, v in answers.items() if v is not None and str(v).strip()}
    profile_summary = "\n".join(f"  • {k}: {v}" for k, v in filled.items())
    completion_msg = (
        "Here's the information I've captured:\n\n"
        f"**{username}'s Profile**\n{profile_summary}\n\n"
        "(Not saved to database yet.) Let me recommend some plants for you..."
    )
    return {
        "messages": [AIMessage(content=completion_msg)],
        "pending_questions": [],
        "profile_answers": filled,  # Keep for reasoning agent to use on next turn
        "username": username,
    }


def profile_init(state: dict) -> dict:
    """When profile not found: set handoff state so builder can greet. No user-facing message here."""
    username = _extract_username_from_messages(state.get("messages", [])) or ""
    return {
        "active_agent": "builder",
        "handoff_reason": "profile_not_found",
        "pending_questions": [FREEFORM_SENTINEL],
        "profile_answers": _init_profile_null(),
        "username": username,
    }


def profile_collector(state: dict) -> dict:
    """Treat last user message: if freeform, extract fields; else answer current question."""
    messages = state.get("messages", [])
    pending = list(state.get("pending_questions") or [])
    answers = dict(state.get("profile_answers") or {})
    username = state.get("username") or ""
    reason = state.get("handoff_reason")

    if reason == "profile_not_found" and pending and pending[0] == FREEFORM_SENTINEL:
        msg = (
            f"Hi{f' {username}' if username else ''}! I couldn't find a saved profile for you.\n\n"
            "Tell me your plant preferences in a short paragraph (experience, climate, light, watering, humidity, "
            "temperature, size, how you want the plant to look—e.g. bright leaves, tall, trailing). I'll extract what I can and only ask follow-ups for anything missing."
        )
        return {
            "messages": [AIMessage(content=msg)],
            "handoff_reason": None,
        }

    user_content = _get_last_user_content(messages)
    if user_content is None:
        return {"messages": [AIMessage(content="Please share your answer.")]}

    if not pending:
        return {"messages": [AIMessage(content="All done! What would you like to know?")]}

    if pending[0] == FREEFORM_SENTINEL:
        extracted = _extract_fields_from_text(user_content)
        print("\n[Extracted JSON]", json.dumps(extracted, indent=2))
        for k, v in extracted.items():
            if v is None or not str(v).strip():
                continue
            schema_key = EXTRACTED_KEY_TO_SCHEMA.get(k, k)
            if schema_key == "physical_desc":
                answers[schema_key] = str(v).strip()
                continue
            if schema_key not in QUESTION_TO_SCHEMA_KEY:
                continue
            q = PROFILE_QUESTIONS[QUESTION_TO_SCHEMA_KEY.index(schema_key)]
            answers[schema_key] = _normalize_answer(str(v), q, schema_key)
        pending.pop(0)
        pending = _get_null_fields(answers)
        if not pending:
            return _complete_profile(answers, username)
        first_key = pending[0]
        first_q = PROFILE_QUESTIONS[QUESTION_TO_SCHEMA_KEY.index(first_key)]
        return {
            "messages": [AIMessage(content=first_q)],
            "pending_questions": pending,
            "profile_answers": answers,
            "username": username,
        }

    current_schema_key = pending[0]
    current_q = PROFILE_QUESTIONS[QUESTION_TO_SCHEMA_KEY.index(current_schema_key)] if current_schema_key in QUESTION_TO_SCHEMA_KEY else current_schema_key
    normalized = _normalize_answer(user_content, current_q, current_schema_key)
    answers[current_schema_key] = normalized
    pending.pop(0)

    if not pending:
        return _complete_profile(answers, username)

    next_key = pending[0]
    next_q = PROFILE_QUESTIONS[QUESTION_TO_SCHEMA_KEY.index(next_key)] if next_key in QUESTION_TO_SCHEMA_KEY else next_key
    return {
        "messages": [AIMessage(content=next_q)],
        "pending_questions": pending,
        "profile_answers": answers,
        "username": username,
    }
