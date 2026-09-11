"""
Vector-memory (ChromaDB) helpers for storing / searching / updating
bug-report patterns, ported from the original notebook.
"""
import os
from functools import lru_cache

import chromadb

COLLECTION_NAME = "bug-reports"


@lru_cache(maxsize=1)
def get_client() -> chromadb.ClientAPI:
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.PersistentClient(path=persist_dir)


def get_collection():
    client = get_client()
    return client.get_or_create_collection(name=COLLECTION_NAME)


def list_memories(limit: int = 100):
    """Return all stored bug-pattern memories (for the frontend 'Memory' tab)."""
    collection = get_collection()
    data = collection.get(limit=limit)
    memories = []
    for i, mem_id in enumerate(data.get("ids", [])):
        memories.append(
            {
                "id": mem_id,
                "document": data["documents"][i] if data.get("documents") else "",
            }
        )
    return memories


def clear_memories():
    collection = get_collection()
    existing = collection.get()
    ids = existing.get("ids", [])
    if ids:
        collection.delete(ids=ids)
