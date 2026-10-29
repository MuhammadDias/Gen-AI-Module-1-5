"""Vector databases add persistence, filtering, scale, and tenant isolation to vector search."""

REQUIREMENTS = {
    "local development": "ChromaDB",
    "managed production": "Pinecone",
    "hybrid keyword and vector search": "Weaviate",
    "existing PostgreSQL deployment": "pgvector",
}


if __name__ == "__main__":
    for need, choice in REQUIREMENTS.items():
        print(f"{need}: {choice}")