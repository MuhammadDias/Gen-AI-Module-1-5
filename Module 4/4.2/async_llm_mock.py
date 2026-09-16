# 4.2 Async Mocking LLM API Calls
import asyncio

async def call_model_mock(model: str, prompt: str) -> dict:
    await asyncio.sleep(0.05)
    return {
        "model": model,
        "response": f"[{model}] Answer to: {prompt[:30]}",
        "tokens": 42,
    }

async def compare_models(prompt: str, models: list[str]) -> list[dict]:
    tasks = [call_model_mock(m, prompt) for m in models]
    return await asyncio.gather(*tasks)

if __name__ == "__main__":
    responses = asyncio.run(compare_models(
        "What is RAG?",
        ["claude-sonnet-4-5", "gpt-4o", "gemini-1.5-pro"]
    ))
    for r in responses:
        print(f"{r['model']}: {r['response']}")
