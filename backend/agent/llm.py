"""Shared LLM configuration for agents."""
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

_project_root = Path(__file__).resolve().parent.parent.parent
_backend_dir = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")
load_dotenv(_backend_dir / ".env")


def get_ollama_llm(temperature: float = 0.7) -> ChatOllama:
    """Return a configured Ollama LLM. Use this for all agents to avoid redefining setup."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    return ChatOllama(model=model, base_url=base_url, temperature=temperature)
