import os

from dotenv import load_dotenv
from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()

DOCUMENTS = [
    Document(text="ChromaDB is an open-source embedding database.", metadata={"tag": "vector-db"}),
    Document(text="Pinecone is a managed vector database for production.", metadata={"tag": "vector-db"}),
    Document(text="FAISS is a library for efficient dense-vector similarity search.", metadata={"tag": "vector-search"}),
    Document(text="pgvector adds vector similarity search to PostgreSQL.", metadata={"tag": "vector-db"}),
]


if __name__ == "__main__":
    Settings.embed_model = OpenAIEmbedding(
        model="openai/text-embedding-3-small",
        api_key=os.environ["OPEN_ROUTER_API_KEY"],
        api_base="https://openrouter.ai/api/v1",
    )
    index = VectorStoreIndex.from_documents(DOCUMENTS)
    retriever = VectorIndexRetriever(index=index, similarity_top_k=3)
    for node in retriever.retrieve("open source vector databases"):
        print(f"[{node.score:.4f}] {node.text}")
        print(f"tag: {node.metadata.get('tag')}")