from abc import ABC, abstractmethod
from dataclasses import dataclass
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


@dataclass
class ChatMessage:
	role: str
	content: str


@dataclass
class ChatResponse:
	text: str
	input_tokens: int
	output_tokens: int
	model: str


class BaseLLMClient(ABC):
	@abstractmethod
	def chat(
		self,
		messages: list[ChatMessage],
		system: str = "",
		max_tokens: int = 1024,
		temperature: float = 0.7,
	) -> ChatResponse:
		...


class XKIROClient(BaseLLMClient):
	def __init__(self, model: str = "qwen/qwen3.7-flash:free"):
		self.model = model
		self._client = OpenAI(
			api_key=os.environ["XKIRO_API_KEY"],
			base_url="https://api.xkiro.com/v1",
		)

	def chat(
		self,
		messages: list[ChatMessage],
		system: str = "",
		max_tokens: int = 1024,
		temperature: float = 0.7,
	) -> ChatResponse:
		api_messages = []
		if system:
			api_messages.append({"role": "system", "content": system})
		api_messages.extend(
			{"role": message.role, "content": message.content}
			for message in messages
		)
		response = self._client.chat.completions.create(
			model=self.model,
			max_tokens=max_tokens,
			temperature=temperature,
			messages=api_messages,
		)
		return ChatResponse(
			text=response.choices[0].message.content or "",
			input_tokens=response.usage.prompt_tokens,
			output_tokens=response.usage.completion_tokens,
			model=self.model,
		)


#sage:
client: BaseLLMClient = XKIROClient()
msgs = [ChatMessage(role="user", content="What is a vector database?")]
result = client.chat(msgs, system="Be concise.")
print(result.text)
