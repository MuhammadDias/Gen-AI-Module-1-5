# 3.2 Inheritance and Abstract Base Classes (ABC)
from abc import ABC, abstractmethod

class BaseLLMClient(ABC):
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    @abstractmethod
    def complete(self, messages: list[dict], **kwargs) -> str:
        pass

    def count_words(self, text: str) -> int:
        return len(text.split())

class MockAnthropicClient(BaseLLMClient):
    def complete(self, messages: list[dict], **kwargs) -> str:
        last_user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        return f"[Mock Anthropic] Echo: {last_user}"

class MockOpenAIClient(BaseLLMClient):
    def complete(self, messages: list[dict], **kwargs) -> str:
        return f"[Mock OpenAI] Received {len(messages)} messages."

if __name__ == "__main__":
    clients = [MockAnthropicClient("k1", "claude-sonnet-4-5"), MockOpenAIClient("k2", "gpt-4o")]
    for c in clients:
        print(c.complete([{"role": "user", "content": "Hello"}]))
