import os
import uuid
from functools import lru_cache

import chromadb

COLLECTION_NAME = "bug-reports"


@lru_cache(maxsize=1)
def get_client():
    path = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
    os.makedirs(path, exist_ok=True)
    return chromadb.PersistentClient(path=path)


def get_collection():
    return get_client().get_or_create_collection(name=COLLECTION_NAME)


def add_memory(document: str) -> None:
    get_collection().add(ids=[str(uuid.uuid4())], documents=[document])


def list_memories(limit: int = 100) -> list[dict]:
    d = get_collection().get(limit=limit)
    ids = d.get("ids", [])
    docs = d.get("documents", [])
    return [{"id": i, "document": docs[n] if n < len(docs) else ""} for n, i in enumerate(ids)]


def clear_memories() -> None:
    c = get_collection()
    ids = c.get().get("ids", [])
    if ids:
        c.delete(ids=ids)


# ---------------------------------------------------------------------------
# Best-effort helpers used by the graph. Memory is a convenience for giving
# the Fixer LLM extra context -- it must never be able to fail or block a run
# (e.g. Chroma not installed/running), so every call here swallows errors.
# ---------------------------------------------------------------------------
def save_bug_pattern(text: str) -> None:
    try:
        add_memory(text)
    except Exception:
        pass


def recent_patterns(query: str, limit: int = 3) -> list[str]:
    try:
        collection = get_collection()
        count = collection.count()
        if not count:
            return []
        results = collection.query(query_texts=[query], n_results=min(limit, count))
        return results.get("documents", [[]])[0]
    except Exception:
        return []