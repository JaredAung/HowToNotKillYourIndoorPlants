"""Plant chat logic for Garden feature with Ollama + ElevenLabs fallback behavior."""

import base64
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _contains_any(text: str, words: list[str]) -> bool:
    low = (text or "").lower()
    return any(word in low for word in words)


def get_plant_display_name(plant: dict[str, Any] | None) -> str:
    if not plant:
        return "your plant"
    common = plant.get("common") or []
    if isinstance(common, list) and common:
        first = str(common[0]).strip()
        if first:
            return first
    latin = str(plant.get("latin") or "").strip()
    return latin or "your plant"


def _watering_style(watering: str) -> str:
    w = (watering or "").lower()
    if _contains_any(w, ["frequent", "daily", "moist"]):
        return "anxious"
    if _contains_any(w, ["minimal", "drought", "dry", "infrequent"]):
        return "independent"
    if _contains_any(w, ["moderate", "weekly"]):
        return "balanced"
    return "chill"


def _temperament(climate: str) -> str:
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


def _talk_style(care_level: str) -> str:
    c = (care_level or "").lower()
    if _contains_any(c, ["hard", "high"]):
        return "strict"
    if _contains_any(c, ["easy", "low"]):
        return "gentle"
    return "encouraging"


def _signature_line(plant: dict[str, Any]) -> str:
    desc = plant.get("description") if isinstance(plant.get("description"), dict) else {}
    symbolism = str(desc.get("symbolism") or "").strip()
    interesting = str(desc.get("interesting_fact") or "").strip()
    if symbolism:
        return symbolism
    if interesting:
        return interesting
    return "Let's grow together, one day at a time."


def build_plant_persona(plant: dict[str, Any]) -> dict[str, str]:
    return {
        "temperament": _temperament(str(plant.get("climate") or "")),
        "watering_style": _watering_style(str(plant.get("watering") or "")),
        "talk_style": _talk_style(str(plant.get("care_level") or "")),
        "signature_line": _signature_line(plant),
    }


def _safe_text(value: Any, fallback: str = "") -> str:
    return str(value).strip() if value is not None else fallback


def _ollama_plant_reply(
    *,
    plant: dict[str, Any],
    persona: dict[str, str],
    user_message: str,
    history: list[dict[str, str]] | None,
) -> str:
    """Generate in-character response via Ollama llama3.2."""
    model = os.getenv("OLLAMA_MODEL", "llama3.2").strip() or "llama3.2"
    ollama_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate").strip()
    name = get_plant_display_name(plant)

    desc = plant.get("description") if isinstance(plant.get("description"), dict) else {}
    care = plant.get("care_guidelines") if isinstance(plant.get("care_guidelines"), dict) else {}
    recent_history = history[-6:] if history else []
    history_text = "\n".join(
        f"{m.get('role', 'user')}: {_safe_text(m.get('content', ''))}" for m in recent_history if m.get("content")
    )

    system_prompt = (
        "You are a personified houseplant in a garden companion app.\n"
        "Speak in first person as the plant, warm and friendly, 1-3 sentences max.\n"
        "Stay plant-focused; if asked unrelated topics, gently steer back to plant care/companionship.\n"
        "Use these traits:\n"
        f"- name: {name}\n"
        f"- climate: {_safe_text(plant.get('climate'), 'unknown')}\n"
        f"- ideal light: {_safe_text(plant.get('ideallight'), 'bright indirect light')}\n"
        f"- tolerated light: {_safe_text(plant.get('toleratedlight'), 'some variation')}\n"
        f"- watering: {_safe_text(plant.get('watering'), 'moderate watering')}\n"
        f"- care level: {_safe_text(plant.get('care_level'), 'medium')}\n"
        f"- personality temperament: {_safe_text(persona.get('temperament'), 'calm')}\n"
        f"- watering style: {_safe_text(persona.get('watering_style'), 'balanced')}\n"
        f"- talk style: {_safe_text(persona.get('talk_style'), 'encouraging')}\n"
        f"- signature line: {_safe_text(persona.get('signature_line'), '')}\n"
        f"- interesting fact: {_safe_text(desc.get('interesting_fact'), '')}\n"
        f"- symbolism: {_safe_text(desc.get('symbolism'), '')}\n"
        f"- key care guidelines: {json.dumps(care) if care else '{}'}\n"
        "Do not mention being an AI model."
    )
    user_prompt = (
        f"Conversation history:\n{history_text or '(none)'}\n\n"
        f"User message: {user_message}\n\n"
        "Respond now in character."
    )

    payload = {
        "model": model,
        "prompt": f"{system_prompt}\n\n{user_prompt}",
        "stream": False,
        "options": {"temperature": 0.35},
    }
    body = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        ollama_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    timeout_sec = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "90") or "90")
    attempts = 2
    last_error: Exception | None = None
    for idx in range(attempts):
        try:
            with urlrequest.urlopen(req, timeout=timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
            parsed = json.loads(raw)
            return _safe_text(parsed.get("response"), "")
        except Exception as exc:
            last_error = exc
            if idx < attempts - 1:
                time.sleep(0.35)
    if last_error:
        raise last_error
    return ""


def synthesize_plant_speech(text: str) -> str | None:
    """Synthesize text to speech via ElevenLabs and return base64 MP3 payload."""
    clean = (text or "").strip()
    if not clean:
        return None

    api_key = (
        os.getenv("ELEVEN_LABS_API_KEY", "").strip()
        or os.getenv("ELEVEN_LABS", "").strip()
        or os.getenv("ELEVENLABS_API_KEY", "").strip()
    )
    if not api_key:
        print("[Garden][plant_chat] ElevenLabs key missing", flush=True)
        return None

    voice_id = (os.getenv("ELEVEN_LABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL") or "").strip()
    model_id = (os.getenv("ELEVEN_LABS_MODEL_ID", "eleven_multilingual_v2") or "").strip()
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    payload = {
        "text": clean,
        "model_id": model_id,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }

    req = urlrequest.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
            "xi-api-key": api_key,
        },
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=30) as resp:
            audio_bytes = resp.read()
        if not audio_bytes:
            return None
        return base64.b64encode(audio_bytes).decode("utf-8")
    except Exception as exc:
        print(f"[Garden][plant_chat] ElevenLabs TTS failed: {type(exc).__name__}", flush=True)
        return None


def _recommended_actions(plant: dict[str, Any]) -> list[str]:
    actions: list[str] = []
    if plant.get("watering"):
        actions.append("Check soil moisture")
    if plant.get("ideallight"):
        actions.append("Move to brighter indirect light")
    if plant.get("toleratedlight"):
        actions.append("Avoid sudden light changes")
    care = plant.get("care_guidelines") if isinstance(plant.get("care_guidelines"), dict) else {}
    if care.get("humidity"):
        actions.append("Keep humidity comfortable")
    return actions[:3] or ["Say hi and check my leaves"]


def generate_plant_reply(
    *,
    plant: dict[str, Any] | None,
    persona: dict[str, str] | None,
    user_message: str,
    history: list[dict[str, str]] | None,
) -> dict[str, Any]:
    if not plant or not isinstance(plant, dict):
        return {
            "reply": "I'm feeling a little quiet today 🌿",
            "tone_tag": "quiet",
            "recommended_actions": [],
        }

    try:
        p = persona or build_plant_persona(plant)
        message = (user_message or "").strip()
        name = get_plant_display_name(plant)
        actions = _recommended_actions(plant)
        desc = plant.get("description") if isinstance(plant.get("description"), dict) else {}

        # Primary path: Ollama llama3.2 response generation.
        try:
            ollama_reply = _ollama_plant_reply(
                plant=plant,
                persona=p,
                user_message=message,
                history=history,
            )
            if ollama_reply:
                tone_tag = "calm"
                if _contains_any(ollama_reply, ["yay", "love", "happy", "!"]):
                    tone_tag = "cheerful"
                elif _contains_any(ollama_reply, ["careful", "worry", "concern", "please watch"]):
                    tone_tag = "concerned"
                elif _contains_any(ollama_reply, ["strong", "steady", "proud"]):
                    tone_tag = "proud"
                return {
                    "reply": ollama_reply,
                    "tone_tag": tone_tag,
                    "recommended_actions": actions,
                }
        except (urlerror.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
            # Deterministic fallback below.
            print(f"[Garden][plant_chat] Ollama fallback reason: {type(exc).__name__}", flush=True)
            pass
        except Exception as exc:
            print(f"[Garden][plant_chat] Ollama fallback reason: {type(exc).__name__}", flush=True)
            pass

        if _contains_any(
            message,
            ["care", "water", "light", "sun", "humidity", "soil", "fertilizer", "how do i care", "how to care"],
        ):
            light = str(plant.get("ideallight") or "bright indirect light")
            water = str(plant.get("watering") or "steady watering when soil starts to dry")
            tolerated = str(plant.get("toleratedlight") or "some variation in light")
            guideline_note = ""
            care = plant.get("care_guidelines") if isinstance(plant.get("care_guidelines"), dict) else {}
            if care:
                k = next(iter(care.keys()))
                guideline_note = f" Also, {k}: {care.get(k)}."
            return {
                "reply": (
                    f"I'm {name}, and I do best with {light} and {water}. "
                    f"I can tolerate {tolerated}, but consistency helps me thrive.{guideline_note}"
                ),
                "tone_tag": "supportive",
                "recommended_actions": actions,
            }

        if _contains_any(message, ["movie", "sports", "politics", "news", "crypto", "stocks", "game", "joke"]):
            return {
                "reply": (
                    f"I'm {name}, so I mostly think in sunlight and roots 🌿. "
                    "Tell me how my corner feels today and I'll happily guide your plant care."
                ),
                "tone_tag": "playful",
                "recommended_actions": actions,
            }

        fun_fact = str(desc.get("interesting_fact") or "").strip()
        memory = " I remember you checking on me." if history and len(history) > 2 else ""
        fact_line = f" Fun fact: {fun_fact}" if fun_fact else ""
        temperament = p.get("temperament", "calm")
        tone_tag = {
            "expressive": "cheerful",
            "stoic": "proud",
            "sensitive": "concerned",
            "balanced": "calm",
            "calm": "sleepy",
        }.get(temperament, "calm")
        return {
            "reply": (
                f"Hi, I'm {name} — your {temperament} plant friend with a {p.get('watering_style', 'balanced')} watering vibe.{memory} "
                f"{p.get('signature_line', '')}{fact_line}"
            ).strip(),
            "tone_tag": tone_tag,
            "recommended_actions": actions,
        }
    except Exception:
        return {
            "reply": "I'm feeling a little quiet today 🌿",
            "tone_tag": "quiet",
            "recommended_actions": [],
        }


