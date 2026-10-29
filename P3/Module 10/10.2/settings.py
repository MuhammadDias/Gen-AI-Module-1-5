import os

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()


def configure_settings() -> None:
    api_key = os.environ["OPEN_ROUTER_API_KEY"]
    Settings.llm = OpenAI(
        model=os.getenv("LLM_MODEL", "qwen/qwen3.7-flash:free"),
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
        temperature=0.1,
        max_tokens=1024,
    )
    Settings.embed_model = OpenAIEmbedding(
        model="openai/text-embedding-3-small",
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
    )
    Settings.chunk_size = 512
    Settings.chunk_overlap = 64


if __name__ == "__main__":
    configure_settings()
    print(f"Configured chunk size: {Settings.chunk_size}")