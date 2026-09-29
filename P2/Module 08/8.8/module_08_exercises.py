from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_EMBED_MODEL = "openai/text-embedding-3-small"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
EmbeddingFunction = Callable[[list[str], str], np.ndarray]


def create_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["OPEN_ROUTER_API_KEY"],
        base_url=OPENROUTER_BASE_URL,
    )


def embed_texts(
    texts: list[str],
    model: str = DEFAULT_EMBED_MODEL,
) -> np.ndarray:
    """Request a batch of embeddings from OpenRouter."""
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    response = create_client().embeddings.create(input=texts, model=model)
    vectors = sorted(response.data, key=lambda item: item.index)
    return np.asarray([item.embedding for item in vectors], dtype=np.float32)


def _normalise_rows(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("Expected a two-dimensional embedding matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.where(norms == 0, 1.0, norms)


def _normalise_vector(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(values))
    return values / norm if norm else values


# Exercise 1: find document pairs whose embedding cosine exceeds a threshold.
@dataclass(frozen=True)
class DuplicatePair:
    first_index: int
    second_index: int
    first_text: str
    second_text: str
    similarity: float


class DuplicateDetector:
    def __init__(
        self,
        documents: Sequence[str],
        threshold: float = 0.95,
        model: str = DEFAULT_EMBED_MODEL,
        embed_fn: EmbeddingFunction = embed_texts,
    ):
        if not -1.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between -1 and 1")
        self.documents = list(documents)
        self.threshold = threshold
        self.model = model
        vectors = embed_fn(self.documents, model)
        self.embeddings = _normalise_rows(vectors)
        if len(self.embeddings) != len(self.documents):
            raise ValueError("The embedder must return one vector per document")

    def find_duplicates(self) -> list[DuplicatePair]:
        similarities = self.embeddings @ self.embeddings.T
        pairs = []
        for first_index in range(len(self.documents)):
            for second_index in range(first_index + 1, len(self.documents)):
                score = float(similarities[first_index, second_index])
                if score > self.threshold:
                    pairs.append(
                        DuplicatePair(
                            first_index=first_index,
                            second_index=second_index,
                            first_text=self.documents[first_index],
                            second_text=self.documents[second_index],
                            similarity=score,
                        )
                    )
        return sorted(pairs, key=lambda pair: pair.similarity, reverse=True)


CORPUS_50 = [
    f"Document {index:02d}: notes about topic {index % 7} and language model evaluation."
    for index in range(49)
]
CORPUS_50.append(CORPUS_50[0])


# Exercise 2: blend normalized semantic similarity with a BM25-style score.
@dataclass(frozen=True)
class HybridResult:
    document: str
    score: float
    semantic_score: float
    keyword_score: float


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridSearch:
    def __init__(
        self,
        documents: Sequence[str],
        alpha: float = 0.5,
        model: str = DEFAULT_EMBED_MODEL,
        embed_fn: EmbeddingFunction = embed_texts,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if k1 <= 0 or not 0.0 <= b <= 1.0:
            raise ValueError("BM25 parameters require k1 > 0 and 0 <= b <= 1")
        self.documents = list(documents)
        self.alpha = alpha
        self.model = model
        self.embed_fn = embed_fn
        self.k1 = k1
        self.b = b
        self._tokenized = [_tokens(document) for document in self.documents]
        vectors = embed_fn(self.documents, model)
        self._embeddings = _normalise_rows(vectors)
        if len(self._embeddings) != len(self.documents):
            raise ValueError("The embedder must return one vector per document")

    def _bm25_scores(self, query: str) -> np.ndarray:
        query_terms = _tokens(query)
        document_count = len(self.documents)
        if not document_count:
            return np.empty(0, dtype=np.float32)
        lengths = [len(tokens) for tokens in self._tokenized]
        average_length = sum(lengths) / document_count if document_count else 0.0
        document_frequency = {
            term: sum(term in set(tokens) for tokens in self._tokenized)
            for term in set(query_terms)
        }

        scores = []
        for tokens, document_length in zip(self._tokenized, lengths):
            term_counts = {term: tokens.count(term) for term in set(query_terms)}
            score = 0.0
            for term in query_terms:
                term_frequency = term_counts[term]
                if not term_frequency:
                    continue
                frequency = document_frequency[term]
                inverse_document_frequency = math.log(
                    1.0
                    + (document_count - frequency + 0.5)
                    / (frequency + 0.5)
                )
                length_normalizer = (
                    term_frequency
                    + self.k1
                    * (1.0 - self.b + self.b * document_length / average_length)
                    if average_length
                    else term_frequency + self.k1
                )
                score += (
                    inverse_document_frequency
                    * term_frequency
                    * (self.k1 + 1.0)
                    / length_normalizer
                )
            scores.append(score)
        return np.asarray(scores, dtype=np.float32)

    def search(self, query: str, k: int = 5) -> list[HybridResult]:
        if k <= 0 or not self.documents:
            return []
        query_vectors = _normalise_rows(self.embed_fn([query], self.model))
        semantic_scores = self._embeddings @ query_vectors[0]
        raw_keyword_scores = self._bm25_scores(query)
        maximum = float(raw_keyword_scores.max()) if raw_keyword_scores.size else 0.0
        keyword_scores = raw_keyword_scores / maximum if maximum else raw_keyword_scores
        combined_scores = self.alpha * semantic_scores + (1.0 - self.alpha) * keyword_scores
        indices = np.argsort(combined_scores)[::-1][: min(k, len(self.documents))]
        return [
            HybridResult(
                document=self.documents[int(index)],
                score=float(combined_scores[index]),
                semantic_score=float(semantic_scores[index]),
                keyword_score=float(keyword_scores[index]),
            )
            for index in indices
        ]


# Exercise 3: mutable VectorStore with the same core API as section 8.4.
@dataclass
class VectorDocument:
    id: str
    text: str
    embedding: np.ndarray | None = None


@dataclass(frozen=True)
class VectorSearchResult:
    document: VectorDocument
    score: float
    rank: int


class VectorStore:
    """Standalone 8.4-style vector store with delete and update support."""

    def __init__(
        self,
        model: str = DEFAULT_EMBED_MODEL,
        embed_fn: EmbeddingFunction = embed_texts,
    ):
        self.model = model
        self.embed_fn = embed_fn
        self._documents: list[VectorDocument] = []
        self._matrix = np.empty((0, 0), dtype=np.float32)

    def _rebuild_matrix(self) -> None:
        if not self._documents:
            self._matrix = np.empty((0, 0), dtype=np.float32)
            return
        if any(document.embedding is None for document in self._documents):
            raise RuntimeError("Cannot rebuild matrix while a document lacks an embedding")
        self._matrix = np.vstack(
            [document.embedding for document in self._documents]
        ).astype(np.float32)

    def add_documents(self, documents: Sequence[VectorDocument]) -> None:
        existing_ids = {document.id for document in self._documents}
        new_ids = [document.id for document in documents]
        if len(new_ids) != len(set(new_ids)) or existing_ids.intersection(new_ids):
            raise ValueError("Document IDs must be unique in the vector store")
        if not documents:
            return
        vectors = _normalise_rows(
            self.embed_fn([document.text for document in documents], self.model)
        )
        if len(vectors) != len(documents):
            raise ValueError("The embedder must return one vector per document")
        for document, vector in zip(documents, vectors):
            document.embedding = vector
            self._documents.append(document)
        self._rebuild_matrix()

    def delete(self, doc_id: str) -> bool:
        original_count = len(self._documents)
        self._documents = [
            document for document in self._documents if document.id != doc_id
        ]
        deleted = len(self._documents) != original_count
        if deleted:
            self._rebuild_matrix()
        return deleted

    def update(self, doc_id: str, new_text: str) -> None:
        for document in self._documents:
            if document.id == doc_id:
                vector = self._normalise_one(self.embed_fn([new_text], self.model)[0])
                document.text = new_text
                document.embedding = vector
                self._rebuild_matrix()
                return
        raise KeyError(f"No document with id {doc_id!r}")

    @staticmethod
    def _normalise_one(vector: np.ndarray) -> np.ndarray:
        return _normalise_vector(vector).astype(np.float32)

    def search(self, query: str, k: int = 5) -> list[VectorSearchResult]:
        if k <= 0 or not self._documents:
            return []
        query_vector = self._normalise_one(self.embed_fn([query], self.model)[0])
        if not np.any(query_vector):
            return []
        scores = self._matrix @ query_vector
        indices = np.argsort(scores)[::-1][: min(k, len(self._documents))]
        return [
            VectorSearchResult(
                document=self._documents[int(index)],
                score=float(scores[index]),
                rank=rank + 1,
            )
            for rank, index in enumerate(indices)
        ]

    @property
    def size(self) -> int:
        return len(self._documents)


# Exercise 4: persistent embedding cache keyed by SHA-256(text + model).
def _cache_key(text: str, model: str) -> str:
    payload = f"{model}\0{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def embed_with_cache(
    texts: str | Sequence[str],
    model: str = DEFAULT_EMBED_MODEL,
    cache_path: str | Path = "embeddings.sqlite3",
    embed_fn: EmbeddingFunction = embed_texts,
) -> np.ndarray:
    """Return vectors from SQLite cache, embedding only unique cache misses."""
    input_was_text = isinstance(texts, str)
    text_list = [texts] if input_was_text else list(texts)
    if not text_list:
        return np.empty((0, 0), dtype=np.float32)

    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    keys = [_cache_key(text, model) for text in text_list]
    cached: dict[str, np.ndarray] = {}
    with sqlite3.connect(cache_file) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS embeddings (
                cache_key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                vector_json TEXT NOT NULL
            )"""
        )
        for key in set(keys):
            row = connection.execute(
                "SELECT vector_json FROM embeddings WHERE cache_key = ?",
                (key,),
            ).fetchone()
            if row is not None:
                cached[key] = np.asarray(json.loads(row[0]), dtype=np.float32)

        missing_by_key: dict[str, str] = {}
        for key, text in zip(keys, text_list):
            if key not in cached:
                missing_by_key.setdefault(key, text)

        if missing_by_key:
            missing_keys = list(missing_by_key)
            missing_texts = [missing_by_key[key] for key in missing_keys]
            vectors = np.asarray(embed_fn(missing_texts, model), dtype=np.float32)
            if len(vectors) != len(missing_texts):
                raise ValueError("The embedder must return one vector per text")
            for key, text, vector in zip(missing_keys, missing_texts, vectors):
                vector_json = json.dumps(vector.tolist(), separators=(",", ":"))
                text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                connection.execute(
                    "INSERT OR REPLACE INTO embeddings "
                    "(cache_key, model, text_hash, vector_json) VALUES (?, ?, ?, ?)",
                    (key, model, text_hash, vector_json),
                )
                cached[key] = vector

    result = np.vstack([cached[key] for key in keys]).astype(np.float32)
    return result[0] if input_was_text else result


def _hash_embed(texts: list[str], model: str) -> np.ndarray:
    """Small deterministic stand-in for embedding API calls in local tests."""
    vectors = []
    for text in texts:
        digest = hashlib.sha256(f"{model}\0{text}".encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big")
        vectors.append(np.random.default_rng(seed).standard_normal(32))
    return np.asarray(vectors, dtype=np.float32)


def run_local_tests() -> None:
    pairs = DuplicateDetector(
        CORPUS_50,
        threshold=0.99,
        embed_fn=_hash_embed,
    ).find_duplicates()
    assert len(CORPUS_50) == 50
    assert any(pair.first_index == 0 and pair.second_index == 49 for pair in pairs)

    hybrid = HybridSearch(
        ["red apple fruit", "blue car vehicle", "green apple tree"],
        alpha=0.0,
        embed_fn=_hash_embed,
    )
    assert hybrid.search("apple", k=1)[0].document in {
        "red apple fruit",
        "green apple tree",
    }

    store = VectorStore(embed_fn=_hash_embed)
    store.add_documents(
        [VectorDocument("a", "alpha document"), VectorDocument("b", "beta document")]
    )
    assert store._matrix.shape == (2, 32)
    store.update("a", "updated alpha document")
    assert store._matrix.shape == (2, 32)
    assert store._documents[0].text == "updated alpha document"
    assert store.delete("b") is True
    assert store._matrix.shape == (1, 32)
    assert store.delete("missing") is False

    calls: list[list[str]] = []

    def counting_embedder(batch: list[str], model: str) -> np.ndarray:
        calls.append(list(batch))
        return _hash_embed(batch, model)

    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        db_path = Path(directory) / "cache.sqlite3"
        first = embed_with_cache(
            ["same", "same", "other"],
            cache_path=db_path,
            embed_fn=counting_embedder,
        )
        second = embed_with_cache(
            ["same", "new"],
            cache_path=db_path,
            embed_fn=counting_embedder,
        )
        assert first.shape == (3, 32)
        assert second.shape == (2, 32)
        assert calls == [["same", "other"], ["new"]]


if __name__ == "__main__":
    run_local_tests()
    print("Module 08 exercise tests passed (mock embeddings; no API requests).")
