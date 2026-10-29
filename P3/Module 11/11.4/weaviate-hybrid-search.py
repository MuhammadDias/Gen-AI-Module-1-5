import os

import weaviate
import weaviate.classes as wvc
from dotenv import load_dotenv
from openai import OpenAI
from weaviate.classes.query import Filter

load_dotenv()


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]


def main() -> None:
    api_client = OpenAI(
        api_key=os.environ["OPEN_ROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    client = weaviate.connect_to_local()
    try:
        if not client.collections.exists("AIDocument"):
            client.collections.create(
                name="AIDocument",
                properties=[
                    wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="source", data_type=wvc.config.DataType.TEXT),
                    wvc.config.Property(name="category", data_type=wvc.config.DataType.TEXT),
                ],
                vectorizer_config=wvc.config.Configure.Vectorizer.none(),
            )
        collection = client.collections.get("AIDocument")
        texts = [
            "RAG retrieves relevant documents to ground language model answers.",
            "Vector databases store embeddings for semantic search.",
            "BM25 ranks documents using keyword frequency and rarity.",
        ]
        vectors = embed_texts(api_client, texts)
        with collection.batch.dynamic() as batch:
            for text, vector in zip(texts, vectors):
                batch.add_object(
                    properties={"text": text, "source": "guide", "category": "ai"},
                    vector=vector,
                )

        query_vector = embed_texts(api_client, ["external knowledge for LLMs"])[0]
        results = collection.query.hybrid(
            query="external knowledge LLM",
            vector=query_vector,
            alpha=0.5,
            limit=3,
            return_properties=["text", "source"],
        )
        for result in results.objects:
            print(result.properties)

        filtered = collection.query.near_vector(
            near_vector=query_vector,
            limit=3,
            filters=Filter.by_property("category").equal("ai"),
            return_properties=["text", "source"],
        )
        print("Filtered objects:", len(filtered.objects))
    finally:
        client.close()


if __name__ == "__main__":
    main()