"""Shared LLM configuration for agents. Uses Gemini when API key is set, otherwise Ollama."""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

_project_root = Path(__file__).resolve().parent.parent.parent
_backend_dir = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")
load_dotenv(_backend_dir / ".env")


def get_llm(temperature: float = 0.7) -> BaseChatModel:
    """Return Gemini if GOOGLE_API_KEY/GEMINI_API_KEY is set, otherwise Ollama."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if api_key:
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            google_api_key=api_key.strip().strip('"'),
            temperature=temperature,
        )
    return get_ollama_llm(temperature)


def get_ollama_llm(temperature: float = 0.7) -> ChatOllama:
    """Return a configured Ollama LLM. Fallback when Gemini API key is not set."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    return ChatOllama(model=model, base_url=base_url, temperature=temperature)
