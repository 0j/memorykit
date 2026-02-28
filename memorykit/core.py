"""
MemoryKit - Persistent memory layer for AI agents
core.py: Main interface
"""

import uuid
import json
from datetime import datetime, timedelta
from typing import List, Optional
from .embedder import Embedder
from .store import VectorStore
from .retriever import Retriever
from .summarizer import Summarizer


class Memory:
    """
    Give your AI agent a brain that persists across sessions.

    Usage:
        mem = Memory(agent_id="my_agent")
        mem.store("User's name is Sarah, she's building a startup")
        results = mem.retrieve("What do I know about this user?")
    """

    def __init__(self, agent_id: str, db_path: str = "./memorykit_db"):
        self.agent_id = agent_id
        self.embedder = Embedder()
        self.store = VectorStore(db_path=db_path)
        self.retriever = Retriever(self.store, self.embedder)
        self.summarizer = Summarizer()

    def remember(self, content: str, tags: List[str] = [], importance: float = 0.7) -> dict:
        """
        Store a memory. Call this after any meaningful interaction.

        Args:
            content: What to remember (natural language)
            tags: Optional labels e.g. ["preference", "goal"]
            importance: 0.0 to 1.0 — how important is this memory?

        Returns:
            The stored memory object
        """
        embedding = self.embedder.embed(content)

        memory = {
            "id": f"mem_{uuid.uuid4().hex[:8]}",
            "agent_id": self.agent_id,
            "content": content,
            "tags": tags,
            "importance": importance,
            "created_at": datetime.utcnow().isoformat(),
            "decay_score": 1.0,  # starts fresh, decays over time
            "summarized": False
        }

        self.store.save(memory, embedding)
        print(f"✓ Memory stored: \"{content[:60]}...\"" if len(content) > 60 else f"✓ Memory stored: \"{content}\"")
        return memory

    def recall(self, query: str, top_k: int = 5, min_relevance: float = 0.2) -> List[dict]:
        """
        Retrieve memories relevant to a query.
        Call this before generating any AI response.

        Args:
            query: What you're looking for (natural language)
            top_k: Max number of memories to return
            min_relevance: Minimum similarity threshold (0.0 to 1.0)

        Returns:
            List of relevant memory objects, ranked by relevance + recency
        """
        results = self.retriever.search(
            query=query,
            agent_id=self.agent_id,
            top_k=top_k,
            min_relevance=min_relevance
        )

        if not results:
            print(f"No memories found for: \"{query}\"")
        else:
            print(f"✓ Found {len(results)} relevant memories")

        return results

    def compress(self, older_than_days: int = 30) -> Optional[str]:
        """
        Compress old memories into a summary to save space and cost.
        Run this periodically (weekly/monthly).

        Args:
            older_than_days: Compress memories older than this

        Returns:
            Summary string or None if nothing to compress
        """
        cutoff = (datetime.utcnow() - timedelta(days=older_than_days)).isoformat()
        old_memories = self.store.get_older_than(self.agent_id, cutoff)

        if not old_memories:
            print("Nothing to compress yet.")
            return None

        summary = self.summarizer.compress(old_memories)
        self.store.mark_summarized(self.agent_id, cutoff)
        # Store the summary itself as a high-importance memory
        self.remember(f"[SUMMARY] {summary}", tags=["summary"], importance=1.0)
        print(f"✓ Compressed {len(old_memories)} memories into summary")
        return summary

    def forget(self, memory_id: str) -> bool:
        """Delete a specific memory by ID."""
        return self.store.delete(memory_id)

    def wipe(self) -> None:
        """Clear ALL memories for this agent. Use carefully."""
        self.store.wipe_agent(self.agent_id)
        print(f"✓ All memories wiped for agent: {self.agent_id}")

    def context_block(self, query: str, max_tokens: int = 500) -> str:
        """
        Get a ready-to-inject context block for your LLM prompt.
        The most useful method for building AI agents.

        Usage:
            context = mem.context_block("current conversation topic")
            prompt = f"{context}\\n\\nUser: {user_message}"

        Returns:
            Formatted string to inject into your system prompt
        """
        memories = self.recall(query, top_k=5)
        if not memories:
            return ""

        lines = ["## What I remember about this user:\n"]
        char_count = 0

        for m in memories:
            line = f"- {m['content']}\n"
            if char_count + len(line) > max_tokens * 4:  # rough token estimate
                break
            lines.append(line)
            char_count += len(line)

        return "".join(lines)
