import os

from dotenv import load_dotenv
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

load_dotenv()

DOCUMENTS = [
    "HNSW is a graph algorithm used by modern vector search systems.",
    "IVF partitions vectors into clusters for approximate search.",
    "Product quantization compresses vectors to reduce memory usage.",
    "Exact nearest-neighbour search is O(n) in the number of vectors.",
]


def main() -> None:
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
    index = VectorStoreIndex.from_documents([Document(text=text) for text in DOCUMENTS])
    retriever = VectorIndexRetriever(index=index, similarity_top_k=2)
    for mode in (ResponseMode.COMPACT, ResponseMode.TREE_SUMMARIZE):
        synthesizer = get_response_synthesizer(response_mode=mode)
        engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=synthesizer,
        )
        response = engine.query("What algorithms are used in vector search?")
        print(f"[{mode}] {str(response)[:200]}\n")


if __name__ == "__main__":
    main()