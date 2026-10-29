RECOMMENDATIONS = {
    "local development": "ChromaDB",
    "managed production": "Pinecone",
    "hybrid keyword and vector search": "Weaviate",
    "existing PostgreSQL": "pgvector",
    "self-hosted scale": "Qdrant or Weaviate",
}


def recommend(requirement: str) -> str:
    try:
        return RECOMMENDATIONS[requirement]
    except KeyError as error:
        raise ValueError(f"Unknown requirement: {requirement}") from error


if __name__ == "__main__":
    for requirement, database in RECOMMENDATIONS.items():
        print(f"{requirement}: {database}")