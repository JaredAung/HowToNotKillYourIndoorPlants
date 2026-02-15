"""
Profile collection for plant recommendations.
Collects user preferences before running the recommender.
"""
from typing import Any

from langchain_core.messages import AIMessage

FREEFORM_SENTINEL = "__FREEFORM__"

# Question keys used by the two-tower model (profile_answers_to_user_raw)
PROFILE_QUESTIONS = [
    "experience_level",
    "climate",
    "light_availability",
    "room_size",
    "has_pets",
    "time_to_commit",
    "average_room_temp",
    "use",
    "watering_preferences",
]


def get_profile_questions() -> list[str]:
    """Return ordered list of profile question keys."""
    return list(PROFILE_QUESTIONS)


def _get_last_user_message(state: dict) -> str:
    """Extract last user message content."""
    for m in reversed(state.get("messages", [])):
        role = getattr(m, "type", None) or getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else None)
        if role in ("human", "user"):
            return getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else "") or ""
    return ""


def profile_init(state: dict) -> dict:
    """Initialize profile collection. Sets pending_questions if not already set."""
    pending = state.get("pending_questions") or []
    if not pending:
        questions = get_profile_questions()
        return {"pending_questions": list(questions)}
    return {}


# Valid answers for experience_level (so we don't treat "new" or "beginner" as a name)
EXPERIENCE_KEYWORDS = frozenset({
    "beginner", "intermediate", "advanced", "new", "novice", "expert",
    "beginning", "starting", "low", "medium", "high",
})

QUESTION_LABELS = {
    "experience_level": "What's your plant care experience? (beginner/intermediate/advanced)",
    "climate": "What's your climate? (tropical/temperate/dry/etc.)",
    "light_availability": "How much light do you have? (low/medium/bright)",
    "room_size": "What's your room size? (small/medium/large)",
    "has_pets": "Do you have pets? (yes/no)",
    "time_to_commit": "How much time can you commit? (low/medium/high)",
    "average_room_temp": "Average room temperature? (e.g. 22°C or 72°F)",
    "use": "What do you want plants for? (e.g. air purification, decoration)",
    "watering_preferences": "Watering preference? (e.g. low maintenance, regular)",
}


def profile_collector(state: dict) -> dict:
    """Collect one profile answer from the last user message."""
    pending = state.get("pending_questions") or []
    profile_answers = dict(state.get("profile_answers") or {})
    user_msg = _get_last_user_message(state).strip()

    if not pending:
        return {}

    current_key = pending[0]

    # If first message looks like a name (short, no question answered yet), ask first question.
    # But don't treat valid experience answers like "new" or "beginner" as a name.
    if not profile_answers and user_msg and len(user_msg) <= 50 and "\n" not in user_msg:
        msg_lower = user_msg.lower().strip()
        is_experience_answer = (
            current_key == "experience_level"
            and (msg_lower in EXPERIENCE_KEYWORDS or msg_lower.startswith(("beginner", "intermediate", "advanced", "new")))
        )
        if not is_experience_answer:
            next_question = QUESTION_LABELS.get(current_key, f"What is your {current_key.replace('_', ' ')}?")
            return {"messages": [AIMessage(content=next_question)]}

    value = user_msg if user_msg else None

    if value:
        profile_answers[current_key] = value
        new_pending = pending[1:]
    else:
        new_pending = pending

    next_question = None
    if new_pending:
        key = new_pending[0]
        next_question = QUESTION_LABELS.get(key, f"What is your {key.replace('_', ' ')}?")

    updates: dict[str, Any] = {
        "profile_answers": profile_answers,
        "pending_questions": new_pending,
    }

    if next_question:
        updates["messages"] = [AIMessage(content=next_question)]
    elif not new_pending:
        updates["messages"] = [AIMessage(content="Thanks! Let me find some plants for you.")]

    return updates
