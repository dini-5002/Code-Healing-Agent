"""Separate Ollama clients for the Tester (call-selection) and Fixer roles.

DEFAULT_MODEL is the single source of truth for the fallback model name.
Every place that reports or instantiates a model reads from here, so the
API/UI can never claim a different model than the one actually running
(this used to drift: main.py said "qwen2.5-coder:7b" while this file
instantiated "qwen2.5-coder:3b").
"""
import os
from functools import lru_cache
from langchain_ollama import ChatOllama

DEFAULT_MODEL = "qwen2.5-coder:3b"


def _kwargs(role: str):
    return {
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "temperature": float(os.getenv(
            "TESTER_TEMPERATURE" if role == "tester" else "FIXER_TEMPERATURE",
            "0.2" if role == "tester" else "0",
        )),
    }


def tester_model_name() -> str:
    return os.getenv("TESTER_MODEL", DEFAULT_MODEL)


def fixer_model_name() -> str:
    return os.getenv("FIXER_MODEL", DEFAULT_MODEL)


@lru_cache(maxsize=1)
def get_tester_llm() -> ChatOllama:
    return ChatOllama(model=tester_model_name(), **_kwargs("tester"))


@lru_cache(maxsize=1)
def get_fixer_llm() -> ChatOllama:
    return ChatOllama(model=fixer_model_name(), **_kwargs("fixer"))