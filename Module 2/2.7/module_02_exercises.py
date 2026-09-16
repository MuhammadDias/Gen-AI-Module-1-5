# 2.7 Module 02 Exercises
from typing import Iterable, Generator, Any

def filter_responses(responses: list[dict]) -> list[dict]:
    filtered = [r for r in responses if r.get("latency_ms", 0) < 500]
    return sorted(filtered, key=lambda x: x.get("tokens", 0))

def conversation_stats(messages: list[dict]) -> dict:
    total = len(messages)
    user_turns = sum(1 for m in messages if m["role"] == "user")
    assistant_turns = sum(1 for m in messages if m["role"] == "assistant")
    total_words = sum(len(m["content"].split()) for m in messages)
    return {
        "total_messages": total,
        "user_turns": user_turns,
        "assistant_turns": assistant_turns,
        "avg_words_per_message": round(total_words / total, 2) if total else 0.0
    }

def batch_items(items: Iterable[Any], batch_size: int) -> Generator[list[Any], None, None]:
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def fast_and_cheap(fast_models: list[str], cheap_models: list[str]) -> set[str]:
    return set(fast_models) & set(cheap_models)

if __name__ == "__main__":
    print("Batches:", list(batch_items(range(10), 3)))
    print("Fast & Cheap:", fast_and_cheap(["gpt-4o-mini", "claude-haiku", "gpt-4o"], ["gpt-4o-mini", "claude-haiku"]))
