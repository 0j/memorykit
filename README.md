# MemoryKit 🧠

**Give your AI agent a memory that persists across sessions.**

Most AI agents forget everything the moment a session ends. MemoryKit fixes that — a lightweight, open-source memory layer that stores, retrieves, and intelligently compresses context for any AI agent.

```bash
pip install memorykit
```

```python
from memorykit import Memory

mem = Memory(agent_id="my_agent")

# Store what matters
mem.remember("User's name is Alex, building an AI startup")
mem.remember("Prefers concise technical answers", importance=0.9)

# Recall before responding
context = mem.context_block("What should I know about this user?")
# → Inject into your LLM system prompt

# Your agent now remembers across sessions
```

---

## Why MemoryKit?

| Without MemoryKit | With MemoryKit |
|---|---|
| Agent forgets everything each session | Agent builds knowledge over time |
| Users repeat themselves constantly | Seamless, personalized experience |
| No context across conversations | Rich, persistent context |
| Expensive re-explanation every time | Efficient, compressed memory |

---

## Features

- **🔍 Semantic retrieval** — finds relevant memories by meaning, not just keywords
- **⏱ Recency-aware ranking** — recent memories naturally score higher
- **📦 Auto-compression** — old memories get summarized to stay lean
- **🏠 Local-first** — works entirely on your machine, no cloud required
- **🔌 LLM-agnostic** — works with OpenAI, Anthropic, or any model
- **⚡ Dead simple API** — `remember()`, `recall()`, `compress()`. That's it.

---

## Installation

**Minimal (local, free, no API keys):**
```bash
pip install memorykit
```

**With OpenAI embeddings:**
```bash
pip install memorykit[openai]
export OPENAI_API_KEY=your_key
```

**With Anthropic:**
```bash
pip install memorykit[anthropic]
export ANTHROPIC_API_KEY=your_key
```

---

## Quick Start

### 1. Basic memory storage and retrieval

```python
from memorykit import Memory

mem = Memory(agent_id="user_123")

# Store memories
mem.remember("User is a software engineer")
mem.remember("User dislikes verbose explanations", importance=0.9)
mem.remember("User is working on a robotics project", tags=["project"])

# Retrieve relevant memories
results = mem.recall("How should I communicate with this user?")
for r in results:
    print(f"[{r['final_score']:.2f}] {r['content']}")
```

### 2. Inject memory into your LLM prompt

```python
from memorykit import Memory
from openai import OpenAI

mem = Memory(agent_id="user_123")
client = OpenAI()

def chat(user_message: str) -> str:
    # Get relevant memory context
    context = mem.context_block(user_message)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"You are a helpful assistant.\n{context}"},
            {"role": "user", "content": user_message}
        ]
    )

    reply = response.choices[0].message.content

    # Store the interaction
    mem.remember(f"User asked about: {user_message[:100]}", importance=0.5)

    return reply
```

### 3. Compress old memories (run weekly)

```python
# Compress memories older than 30 days into a summary
summary = mem.compress(older_than_days=30)
print(summary)
```

---

## Architecture

```
Your App / AI Agent
        │
        ▼
  ┌─────────────┐
  │  MemoryKit  │
  │   SDK       │
  └──────┬──────┘
         │
   ┌─────┴──────────────────┐
   │                        │
   ▼                        ▼
Vector Store           Summary Store
(ChromaDB local)      (Compressed LLM summaries)
```

**Scoring formula:** `final_score = (relevance × 0.5) + (recency × 0.3) + (importance × 0.2)`

---

## Roadmap

- [x] Core memory store/retrieve/compress
- [x] Local vector storage (ChromaDB)
- [x] OpenAI + Anthropic embedding support
- [x] Recency + importance re-ranking
- [ ] Multi-agent shared memory
- [ ] Cloud sync (Pinecone, Weaviate)
- [ ] Memory visualization dashboard
- [ ] REST API / hosted service

---

## Contributing

PRs welcome. This is an early-stage open source project — stars and issues help a lot.

---

## License

MIT — use it, build on it, ship it.

---

*MemoryKit is the first step toward AI agents that truly know you.*
