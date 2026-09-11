"""
LLM client factory.

Uses a local Ollama model (e.g. llama3.1, qwen2.5-coder, codellama, ...)
via langchain-ollama. Configure with environment variables:

    OLLAMA_BASE_URL  (default: http://localhost:11434)
    OLLAMA_MODEL     (default: llama3.1)
"""
import os
from functools import lru_cache

from langchain_ollama import ChatOllama


@lru_cache(maxsize=1)
def get_llm() -> ChatOllama:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:3b")
    temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0"))
    return ChatOllama(model=model, base_url=base_url, temperature=temperature)
