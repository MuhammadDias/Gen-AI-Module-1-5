import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()
PERSIST_DIR = Path(__file__).with_name("llama_index_store")


def configure_models() -> None:
    api_key = os.environ["OPEN_ROUTER_API_KEY"]
    Settings.llm = OpenAI(
        model=os.getenv("LLM_MODEL", "qwen/qwen3.7-flash:free"),
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
    )
    Settings.embed_model = OpenAIEmbedding(
        model="openai/text-embedding-3-small",
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
    )


def main() -> None:
    configure_models()
    if PERSIST_DIR.exists():
        storage = StorageContext.from_defaults(persist_dir=str(PERSIST_DIR))
        index = load_index_from_storage(storage)
    else:
        index = VectorStoreIndex.from_documents(
            [
                Document(text="LlamaIndex persists indexes to disk."),
                Document(text="Loading a persisted index avoids re-embedding documents."),
            ]
        )
        index.storage_context.persist(persist_dir=str(PERSIST_DIR))
    answer = index.as_query_engine().query("How does LlamaIndex handle persistence?")
    print("Answer after load/build:", answer)


if __name__ == "__main__":
    main()