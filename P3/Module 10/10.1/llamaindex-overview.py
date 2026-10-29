"""LlamaIndex provides abstractions for ingesting, indexing, and querying data."""

CORE_CONCEPTS = {
    "Document": "Input content and metadata",
    "Node": "A parsed text chunk with metadata",
    "Index": "A structure built from nodes for a query pattern",
    "Retriever": "Finds relevant nodes without generating an answer",
    "Query engine": "Retrieves nodes and synthesizes an answer",
}


if __name__ == "__main__":
    for name, purpose in CORE_CONCEPTS.items():
        print(f"{name}: {purpose}")