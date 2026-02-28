"""
examples/chatbot.py

A simple AI chatbot that actually remembers you across sessions.
Run this to see MemoryKit in action.

Requirements:
    pip install memorykit openai
    export OPENAI_API_KEY=your_key_here
"""

import os
from memorykit import Memory

# Initialize memory for this user
mem = Memory(agent_id="demo_user")

# Pre-load some example memories (in real usage, these build up over time)
print("\n=== Loading example memories ===")
mem.remember("User's name is Alex", tags=["identity"], importance=1.0)
mem.remember("Alex is building an AI startup called MemoryKit", tags=["project"], importance=0.9)
mem.remember("Alex prefers concise, technical answers", tags=["preference"], importance=0.8)
mem.remember("Alex has a background in software engineering and finance", tags=["background"], importance=0.7)
mem.remember("Alex is based in Stockholm", tags=["location"], importance=0.6)

print("\n=== Starting chatbot (type 'quit' to exit) ===\n")

# Simple chat loop
try:
    from openai import OpenAI
    client = OpenAI()

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            break

        # THE KEY STEP: inject relevant memories into the prompt
        context = mem.context_block(user_input)

        system_prompt = f"""You are a helpful AI assistant with memory of past conversations.
{context}
Use this context naturally — don't say "I remember that..." just incorporate it."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
        )

        reply = response.choices[0].message.content
        print(f"\nAssistant: {reply}\n")

        # Store the interaction as a new memory
        mem.remember(
            f"User asked: '{user_input}' | Assistant replied about: {reply[:100]}",
            tags=["conversation"],
            importance=0.5
        )

except ImportError:
    # Demo without OpenAI — just shows memory retrieval working
    print("(OpenAI not installed — showing memory retrieval demo instead)\n")

    test_queries = [
        "What is this user working on?",
        "Where does this person live?",
        "How should I communicate with this user?"
    ]

    for query in test_queries:
        print(f"Query: '{query}'")
        memories = mem.recall(query, top_k=2)
        for m in memories:
            print(f"  → [{m['final_score']:.2f}] {m['content']}")
        print()
