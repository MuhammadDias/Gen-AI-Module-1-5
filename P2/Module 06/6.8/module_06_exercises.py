"""Module 06 exercises using the xKiro OpenAI-compatible API."""

import asyncio
import os
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI, RateLimitError

load_dotenv()

DEFAULT_MODEL = "qwen/qwen3.7-flash:free"
BASE_URL = "https://api.xkiro.com/v1"


def create_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["XKIRO_API_KEY"],
        base_url=BASE_URL,
    )


def create_async_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=os.environ["XKIRO_API_KEY"],
        base_url=BASE_URL,
    )


# Exercise 1: retry a chat completion after rate limiting.
def retry_on_rate_limit(
    client: Any,
    messages: list[dict[str, str]],
    max_retries: int = 5,
    model: str = DEFAULT_MODEL,
    base_delay: float = 1.0,
) -> Any:
    """Retry rate-limited chat calls with exponential backoff.

    ``max_retries`` is the number of retries after the initial attempt.
    The last RateLimitError is re-raised when all attempts are exhausted.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be zero or greater")
    if base_delay < 0:
        raise ValueError("base_delay must be zero or greater")

    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
            )
        except RateLimitError:
            if attempt == max_retries:
                raise
            delay = base_delay * (2**attempt)
            time.sleep(delay)

    raise RuntimeError("Retry loop exited unexpectedly")


# Exercise 2: track total input and output tokens across calls.
class BudgetExceeded(Exception):
    """Raised when recorded usage goes beyond the configured token budget."""


class TokenBudgetManager:
    def __init__(self, token_limit: int):
        if token_limit < 0:
            raise ValueError("token_limit must be zero or greater")
        self.token_limit = token_limit
        self.input_tokens = 0
        self.output_tokens = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def record_usage(self, input_tokens: int, output_tokens: int) -> int:
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("Token counts must be zero or greater")

        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        if self.total_tokens > self.token_limit:
            raise BudgetExceeded(
                f"Token budget exceeded: {self.total_tokens} > {self.token_limit}"
            )
        return self.total_tokens

    def record_response(self, response: Any) -> int:
        """Record token usage from an OpenAI-compatible completion response."""
        usage = response.usage
        if usage is None:
            raise ValueError("The response did not include token usage")
        return self.record_usage(
            input_tokens=usage.prompt_tokens or 0,
            output_tokens=usage.completion_tokens or 0,
        )


# Exercise 3: call several models concurrently and return a DataFrame.
async def compare_models(
    prompt: str,
    models: list[str],
) -> pd.DataFrame:
    """Send one prompt to each model concurrently via xKiro."""
    if not models:
        return pd.DataFrame(
            columns=[
                "model",
                "response_text",
                "input_tokens",
                "output_tokens",
                "latency_ms",
            ]
        )

    async with create_async_client() as client:
        async def call_model(model: str) -> dict[str, Any]:
            started = time.perf_counter()
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            latency_ms = (time.perf_counter() - started) * 1000
            usage = response.usage
            return {
                "model": model,
                "response_text": response.choices[0].message.content or "",
                "input_tokens": usage.prompt_tokens if usage else 0,
                "output_tokens": usage.completion_tokens if usage else 0,
                "latency_ms": round(latency_ms, 2),
            }

        rows = await asyncio.gather(*(call_model(model) for model in models))

    return pd.DataFrame(
        rows,
        columns=[
            "model",
            "response_text",
            "input_tokens",
            "output_tokens",
            "latency_ms",
        ],
    )


# Exercise 4: write streamed response chunks to disk as they arrive.
def stream_to_file(
    prompt: str,
    output_path: str | Path,
    model: str = DEFAULT_MODEL,
) -> None:
    """Stream a chat completion to a UTF-8 file, flushing each received chunk."""
    client = create_client()
    with Path(output_path).open("w", encoding="utf-8") as output_file:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            text = chunk.choices[0].delta.content
            if text:
                output_file.write(text)
                output_file.flush()


def _test_retry_with_mock() -> None:
    class MockRateLimitError(Exception):
        pass

    class MockCompletions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs: Any) -> dict[str, str]:
            self.calls += 1
            if self.calls < 3:
                raise MockRateLimitError("simulated rate limit")
            return {"content": "success"}

    class MockClient:
        def __init__(self):
            self.chat = type("Chat", (), {})()
            self.chat.completions = MockCompletions()

    client = MockClient()
    with patch(__name__ + ".RateLimitError", MockRateLimitError), patch(
        __name__ + ".time.sleep"
    ) as sleep:
        result = retry_on_rate_limit(
            client,
            [{"role": "user", "content": "test"}],
            max_retries=3,
            base_delay=0.25,
        )

    assert result == {"content": "success"}
    assert client.chat.completions.calls == 3
    assert [call.args[0] for call in sleep.call_args_list] == [0.25, 0.5]


def _test_token_budget() -> None:
    manager = TokenBudgetManager(token_limit=10)
    manager.record_usage(input_tokens=4, output_tokens=3)
    assert manager.total_tokens == 7
    try:
        manager.record_usage(input_tokens=2, output_tokens=2)
    except BudgetExceeded:
        pass
    else:
        raise AssertionError("Expected BudgetExceeded")


if __name__ == "__main__":
    _test_retry_with_mock()
    _test_token_budget()
    print("Mock retry and token-budget tests passed.")
    print("compare_models() and stream_to_file() are ready for xKiro API calls.")
