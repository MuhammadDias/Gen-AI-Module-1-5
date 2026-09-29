from dataclasses import dataclass, field
from typing import Any, Callable, Optional
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
class FilteredDocument:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


def embed_texts(texts: list[str]) -> np.ndarray:
    response = openrouter_client.embeddings.create(
        input=texts, model="openai/text-embedding-3-small"
    )
    vectors = np.array(
        [item.embedding for item in sorted(response.data, key=lambda item: item.index)],
        dtype=np.float32,
    )
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


class FilteredVectorStore:
    def __init__(self):
        self._docs: list[FilteredDocument] = []

    def add(self, docs: list[FilteredDocument]) -> None:
        embeddings = embed_texts([document.text for document in docs])
        for document, embedding in zip(docs, embeddings):
            document.embedding = embedding
            self._docs.append(document)

    def search(
        self,
        query: str,
        k: int = 5,
        filter_fn: Optional[Callable[[FilteredDocument], bool]] = None,
    ) -> list[tuple[FilteredDocument, float]]:
        """Apply an optional metadata filter before ranking documents."""
        candidates = (
            self._docs
            if filter_fn is None
            else [document for document in self._docs if filter_fn(document)]
        )
        if not candidates or k <= 0:
            return []

        query_vector = embed_texts([query])[0]
        matrix = np.array(
            [document.embedding for document in candidates], dtype=np.float32
        )
        scores = matrix @ query_vector
        top_k = min(k, len(candidates))
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(candidates[index], float(scores[index])) for index in top_indices]


docs = [
    FilteredDocument("a1", "GPT-4o supports vision and function calling.", {"category": "openai", "year": 2024}),
    FilteredDocument("a2", "Claude 3.5 Sonnet excels at coding tasks.", {"category": "anthropic", "year": 2024}),
    FilteredDocument("a3", "GPT-4o-mini is a smaller, cheaper model.", {"category": "openai", "year": 2024}),
    FilteredDocument("a4", "Claude Opus 4 is Anthropic's most capable model.", {"category": "anthropic", "year": 2025}),
    FilteredDocument("a5", "GPT-4 Turbo has a 128K context window.", {"category": "openai", "year": 2023}),
]

store = FilteredVectorStore()
store.add(docs)

print("=== All docs ===")
results = store.search("which model is good at coding?", k=3)
for document, score in results:
    print(f"  [{score:.4f}] {document.id}: {document.text}")

print("\n=== Anthropic only ===")
results = store.search(
    "which model is good at coding?",
    k=3,
    filter_fn=lambda document: document.metadata["category"] == "anthropic",
)
for document, score in results:
    print(f"  [{score:.4f}] {document.id}: {document.text}")
