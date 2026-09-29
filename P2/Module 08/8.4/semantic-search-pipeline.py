from dataclasses import dataclass, field
from typing import Optional
import os

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openrouter_client = OpenAI(
    api_key=os.environ["OPEN_ROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)


@dataclass
class Document:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class SearchResult:
    document: Document
    score: float
    rank: int


def embed_batch(
    texts: list[str], model: str = "openai/text-embedding-3-small"
) -> np.ndarray:
    """Embed texts in one API call and return a float32 array."""
    response = openrouter_client.embeddings.create(input=texts, model=model)
    vectors = sorted(response.data, key=lambda item: item.index)
    return np.array([vector.embedding for vector in vectors], dtype=np.float32)


class VectorStore:
    """In-memory vector store for semantic search."""

    def __init__(self, embed_model: str = "openai/text-embedding-3-small"):
        self.embed_model = embed_model
        self._documents: list[Document] = []
        self._matrix: Optional[np.ndarray] = None

    def add_documents(self, documents: list[Document]) -> None:
        """Embed and index a list of documents."""
        texts = [document.text for document in documents]
        vectors = embed_batch(texts, model=self.embed_model)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = (vectors / norms).astype(np.float32)

        for document, vector in zip(documents, normalized):
            document.embedding = vector
            self._documents.append(document)

        self._matrix = np.array(
            [document.embedding for document in self._documents],
            dtype=np.float32,
        )
        print(f"Index now contains {len(self._documents)} documents.")

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        """Return the k most similar documents for a query string."""
        if self._matrix is None or not self._documents:
            raise RuntimeError("No documents indexed yet.")

        query_vector = embed_batch([query], model=self.embed_model)[0]
        query_norm = np.linalg.norm(query_vector)
        if query_norm == 0:
            return []
        query_vector = (query_vector / query_norm).astype(np.float32)

        scores = self._matrix @ query_vector
        top_k = min(k, len(self._documents))
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            SearchResult(
                document=self._documents[int(index)],
                score=float(scores[index]),
                rank=rank + 1,
            )
            for rank, index in enumerate(top_indices)
        ]

    @property
    def size(self) -> int:
        return len(self._documents)


CORPUS = [
    Document(
        "d01",
        "Retrieval-Augmented Generation (RAG) combines information retrieval "
        "with language model generation to answer questions using external knowledge.",
    ),
    Document(
        "d02",
        "Vector databases store high-dimensional embeddings and enable fast "
        "approximate nearest-neighbour search using algorithms like HNSW and IVF.",
    ),
    Document(
        "d03",
        "Fine-tuning adapts a pre-trained language model to a specific task by "
        "continuing training on a curated dataset with task-specific examples.",
    ),
    Document(
        "d04",
        "Prompt engineering involves designing and optimising input prompts to "
        "guide language models toward producing the desired output.",
    ),
    Document(
        "d05",
        "LangChain is a Python framework that provides abstractions for building "
        "applications with large language models, including chains, agents, and memory.",
    ),
    Document(
        "d06",
        "Cosine similarity measures the angle between two vectors and is the "
        "standard metric for comparing text embeddings in semantic search.",
    ),
    Document(
        "d07",
        "RLHF (Reinforcement Learning from Human Feedback) aligns language models "
        "with human preferences by training a reward model on human rankings.",
    ),
    Document(
        "d08",
        "Chunking strategies for RAG include fixed-size chunks, sentence-aware "
        "splits, and recursive character splitting with configurable overlap.",
    ),
    Document(
        "d09",
        "The transformer architecture uses self-attention mechanisms to model "
        "relationships between all tokens in a sequence simultaneously.",
    ),
    Document(
        "d10",
        "Agents use language models as a reasoning engine, enabling them to plan "
        "multi-step tasks, call tools, and take actions based on observations.",
    ),
]

if __name__ == "__main__":
    store = VectorStore()
    store.add_documents(CORPUS)

    QUERIES = [
        "How does RAG work?",
        "What algorithms do vector databases use?",
        "How do I split documents for embedding?",
    ]
    for query in QUERIES:
        print(f"\nQuery: {query!r}")
        results = store.search(query, k=3)
        for result in results:
            print(
                f"  [{result.rank}] score={result.score:.4f} | "
                f"{result.document.text[:80]}..."
            )
