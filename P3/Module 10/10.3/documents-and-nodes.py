from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

documents = [
    Document(
        text=(
            "LlamaIndex is a data framework for LLM applications. It provides tools for "
            "ingesting, structuring, and querying data. Its core abstraction is the Index."
        ),
        metadata={"source": "llamaindex_overview", "category": "framework"},
    ),
    Document(
        text=(
            "Vector store indexes keep embeddings in a vector database. They support "
            "semantic search and integrate with several storage backends."
        ),
        metadata={"source": "vector_store_guide", "category": "retrieval"},
    ),
]


if __name__ == "__main__":
    parser = SentenceSplitter(chunk_size=128, chunk_overlap=20)
    nodes = parser.get_nodes_from_documents(documents)
    print(f"Documents: {len(documents)}, nodes: {len(nodes)}")
    for node in nodes:
        print(f"Node {node.node_id[:8]} | {len(node.text)} chars | source={node.metadata.get('source')}")