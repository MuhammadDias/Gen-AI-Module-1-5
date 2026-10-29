import os

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model="openai/text-embedding-3-small",
        input=texts,
    )
    ordered = sorted(response.data, key=lambda item: item.index)
    return [item.embedding for item in ordered]


def main() -> None:
    client = OpenAI(
        api_key=os.environ["OPEN_ROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    database = chromadb.PersistentClient(path="chroma_db")
    collection = database.get_or_create_collection(
        name="ai_knowledge_base",
        metadata={"hnsw:space": "cosine"},
    )
    documents = [
        "RAG connects LLMs to external knowledge bases at inference time.",
        "Fine-tuning adapts model weights using task-specific training data.",
        "Prompt engineering guides LLM outputs without changing model weights.",
        "Vector databases index embeddings for fast similarity search.",
    ]
    ids = [f"doc_{index}" for index in range(len(documents))]
    vectors = embed_texts(client, documents)
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=vectors,
        metadatas=[{"category": "technique", "index": index} for index in range(len(documents))],
    )
    query = "How can I add knowledge to an LLM without retraining?"
    results = collection.query(
        query_embeddings=embed_texts(client, [query]),
        n_results=3,
        include=["documents", "distances", "metadatas"],
    )
    for document, distance, metadata in zip(
        results["documents"][0], results["distances"][0], results["metadatas"][0]
    ):
        print(f"distance={distance:.4f} | {document} | category={metadata['category']}")

    filtered = collection.query(
        query_embeddings=embed_texts(client, [query]),
        n_results=3,
        where={"category": {"$eq": "technique"}},
        include=["documents", "distances"],
    )
    print("Filtered result count:", len(filtered["documents"][0]))

    updated_text = "RAG retrieves external documents and adds them to the LLM context."
    collection.update(
        ids=["doc_0"],
        documents=[updated_text],
        embeddings=embed_texts(client, [updated_text]),
    )
    collection.delete(ids=["doc_3"])
    print("Collection size after update and delete:", collection.count())


if __name__ == "__main__":
    main()