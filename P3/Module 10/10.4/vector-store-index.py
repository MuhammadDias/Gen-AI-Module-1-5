import os

from dotenv import load_dotenv
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()

DOCUMENTS = [
    "RAG combines retrieval with generation to answer questions from external knowledge.",
    "Fine-tuning trains a model on new data to improve performance on a specific task.",
    "Prompt engineering improves LLM outputs without changing model weights.",
    "Vector databases store high-dimensional embeddings and support similarity search.",
]


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
    index = VectorStoreIndex.from_documents(
        [Document(text=text) for text in DOCUMENTS]
    )
    response = index.as_query_engine(similarity_top_k=3).query(
        "What is the difference between RAG and fine-tuning?"
    )
    print("Answer:", response)
    for source_node in response.source_nodes:
        print(f"[{source_node.score:.4f}] {source_node.text[:100]}")


if __name__ == "__main__":
    main()