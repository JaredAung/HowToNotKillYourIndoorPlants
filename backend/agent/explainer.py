"""
Plant care explainer using the agent manager (Ollama).
"""
import sys
from pathlib import Path

# Ensure backend is on path when running explainer directly
_backend = Path(__file__).resolve().parent.parent
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))

from dotenv import load_dotenv

load_dotenv(_backend.parent / ".env")
load_dotenv(_backend / ".env")

from agent.manager import get_llm, test_llm

if __name__ == "__main__":
    print("Testing LangChain-Ollama connection...")
    try:
        result = test_llm()
        print(f"✓ LLM response: {result}")
    except Exception as e:
        print(f"✗ Error: {e}")
