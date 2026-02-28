"""
summarizer.py: Compresses clusters of old memories into concise summaries.
Uses LLM to generate intelligent summaries, not just truncation.
"""

import os
from typing import List, Optional


SUMMARY_PROMPT = """You are compressing a list of memories from an AI agent into a concise summary.
Preserve the most important facts, preferences, and context.
Be specific, not vague. Write in third person (e.g. "User prefers...").
Keep the summary under 200 words.

Memories to compress:
{memories}

Summary:"""


class Summarizer:
    """
    Compresses old memories into concise summaries.
    This keeps the memory store lean and cost-efficient over time.
    """

    def __init__(self):
        self._client = None

    def _load_client(self):
        if self._client is None:
            if os.getenv("OPENAI_API_KEY"):
                from openai import OpenAI
                self._client = ("openai", OpenAI())
            elif os.getenv("ANTHROPIC_API_KEY"):
                import anthropic
                self._client = ("anthropic", anthropic.Anthropic())
            else:
                # Fallback: simple rule-based compression (no LLM needed)
                self._client = ("local", None)

    def compress(self, memories: List[dict]) -> str:
        """
        Compress a list of memories into a single summary string.
        """
        self._load_client()

        memory_text = "\n".join([f"- {m['content']}" for m in memories])

        provider, client = self._client

        if provider == "openai":
            return self._summarize_openai(client, memory_text)
        elif provider == "anthropic":
            return self._summarize_anthropic(client, memory_text)
        else:
            return self._summarize_local(memories)

    def _summarize_openai(self, client, memory_text: str) -> str:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": SUMMARY_PROMPT.format(memories=memory_text)}],
            max_tokens=300
        )
        return response.choices[0].message.content.strip()

    def _summarize_anthropic(self, client, memory_text: str) -> str:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            messages=[{"role": "user", "content": SUMMARY_PROMPT.format(memories=memory_text)}]
        )
        return response.content[0].text.strip()

    def _summarize_local(self, memories: List[dict]) -> str:
        """
        Fallback: simple compression without LLM.
        Just joins the most important memories.
        """
        sorted_mems = sorted(memories, key=lambda m: m.get("importance", 0.5), reverse=True)
        top = sorted_mems[:5]
        return "Key facts: " + " | ".join([m["content"][:100] for m in top])
