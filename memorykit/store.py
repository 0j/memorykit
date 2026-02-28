"""
store.py: Stores memories and their vector embeddings.
Uses pure Python + JSON (no external DB needed, works on any Python version).
"""

import json
import os
import math
from typing import List
from datetime import datetime


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorStore:
    """
    Lightweight local vector store using plain JSON files.
    No external dependencies — works on any Python version.
    """

    def __init__(self, db_path: str = "./memorykit_db"):
        self.db_path = db_path
        self.memories_file = os.path.join(db_path, "memories.json")
        self.embeddings_file = os.path.join(db_path, "embeddings.json")
        os.makedirs(db_path, exist_ok=True)
        self._memories = self._load(self.memories_file, {})
        self._embeddings = self._load(self.embeddings_file, {})
        print(f"✓ Vector store ready at: {self.db_path}")

    def _load(self, path: str, default):
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return default

    def _save(self):
        with open(self.memories_file, "w") as f:
            json.dump(self._memories, f, indent=2)
        with open(self.embeddings_file, "w") as f:
            json.dump(self._embeddings, f)

    def save(self, memory: dict, embedding: List[float]) -> None:
        self._memories[memory["id"]] = memory
        self._embeddings[memory["id"]] = embedding
        self._save()

    def query(self, embedding: List[float], agent_id: str, top_k: int = 5) -> List[dict]:
        results = []
        for mem_id, mem in self._memories.items():
            if mem.get("agent_id") != agent_id:
                continue
            if mem_id not in self._embeddings:
                continue
            score = _cosine_similarity(embedding, self._embeddings[mem_id])
            results.append({**mem, "relevance_score": score})
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:top_k]

    def get_older_than(self, agent_id: str, cutoff_date: str) -> List[dict]:
        return [
            m for m in self._memories.values()
            if m.get("agent_id") == agent_id
            and not m.get("summarized", False)
            and m.get("created_at", "") < cutoff_date
        ]

    def mark_summarized(self, agent_id: str, cutoff_date: str) -> None:
        for mem in self.get_older_than(agent_id, cutoff_date):
            self._memories[mem["id"]]["summarized"] = True
        self._save()

    def delete(self, memory_id: str) -> bool:
        if memory_id in self._memories:
            del self._memories[memory_id]
            self._embeddings.pop(memory_id, None)
            self._save()
            return True
        return False

    def wipe_agent(self, agent_id: str) -> None:
        to_delete = [k for k, v in self._memories.items() if v.get("agent_id") == agent_id]
        for k in to_delete:
            del self._memories[k]
            self._embeddings.pop(k, None)
        self._save()
