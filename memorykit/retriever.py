"""
retriever.py: Retrieves and re-ranks memories.
Combines relevance (vector similarity) + recency + importance.
This is MemoryKit's secret sauce.
"""

from typing import List
from datetime import datetime


class Retriever:
    """
    Smarter than raw similarity search.
    Re-ranks results by combining: relevance + recency + importance.
    """

    def __init__(self, store, embedder, recency_weight: float = 0.3, importance_weight: float = 0.2):
        self.store = store
        self.embedder = embedder
        self.recency_weight = recency_weight
        self.importance_weight = importance_weight
        self.relevance_weight = 1 - recency_weight - importance_weight

    def search(self, query: str, agent_id: str, top_k: int = 5, min_relevance: float = 0.2) -> List[dict]:
        """
        Find memories relevant to a query.
        Re-ranks by blending relevance, recency, and importance.
        """
        # Embed the query
        query_embedding = self.embedder.embed(query)

        # Get raw vector search results (fetch more than needed for re-ranking)
        raw_results = self.store.query(
            embedding=query_embedding,
            agent_id=agent_id,
            top_k=top_k * 2  # fetch extra to re-rank
        )

        # Filter by minimum relevance
        filtered = [m for m in raw_results if m.get("relevance_score", 0) >= min_relevance]

        # Score and re-rank
        scored = [self._score(m) for m in filtered]
        scored.sort(key=lambda x: x["final_score"], reverse=True)

        return scored[:top_k]

    def _score(self, memory: dict) -> dict:
        """
        Compute final score blending relevance + recency + importance.
        """
        relevance = memory.get("relevance_score", 0.5)
        importance = memory.get("importance", 0.7)
        recency = self._recency_score(memory.get("created_at", ""))

        final_score = (
            relevance * self.relevance_weight +
            recency * self.recency_weight +
            importance * self.importance_weight
        )

        memory["recency_score"] = recency
        memory["final_score"] = round(final_score, 4)
        return memory

    def _recency_score(self, created_at: str) -> float:
        """
        Exponential decay: recent memories score higher.
        Score = 1.0 (today) → 0.5 (30 days ago) → ~0.1 (6 months ago)
        """
        if not created_at:
            return 0.5

        try:
            created = datetime.fromisoformat(created_at)
            age_days = (datetime.utcnow() - created).days
            import math
            return math.exp(-0.023 * age_days)  # half-life of ~30 days
        except Exception:
            return 0.5
