import os

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
INDEX_NAME = "ai-knowledge-base"


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]


def main() -> None:
    api_client = OpenAI(
        api_key=os.environ["OPEN_ROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    pinecone = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    if INDEX_NAME not in pinecone.list_indexes().names():
        pinecone.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    index = pinecone.Index(INDEX_NAME)
    texts = [
        "RAG connects LLMs to external knowledge bases.",
        "Fine-tuning updates model weights using task-specific data.",
        "ChromaDB is an open-source local vector database.",
    ]
    vectors = embed_texts(api_client, texts)
    index.upsert(
        vectors=[
            {"id": f"d{number}", "values": vector, "metadata": {"text": text, "source": "guide"}}
            for number, (text, vector) in enumerate(zip(texts, vectors), start=1)
        ]
    )
    query_vector = embed_texts(api_client, ["How can an LLM use external knowledge?"])[0]
    results = index.query(vector=query_vector, top_k=3, include_metadata=True)
    for match in results.matches:
        print(f"[{match.score:.4f}] {match.metadata['text']}")

    filtered = index.query(
        vector=query_vector,
        top_k=3,
        filter={"source": {"$eq": "guide"}},
        include_metadata=True,
    )
    print("Filtered matches:", len(filtered.matches))


if __name__ == "__main__":
    main()