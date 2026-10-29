from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import time
from dataclasses import dataclass
from typing import Any


def _mock_embedding(text: str, dimensions: int = 32) -> list[float]:
    digest = hashlib.sha512(text.encode("utf-8")).digest()
    raw = [((digest[index % len(digest)] / 255.0) * 2.0) - 1.0 for index in range(dimensions)]
    norm = math.sqrt(sum(value * value for value in raw))
    return [value / norm for value in raw] if norm else raw


def _validate_tenant_id(tenant_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", tenant_id):
        raise ValueError("tenant_id may contain only letters, digits, '_' and '-' (max 40)")
    return f"tenant_{tenant_id.lower()}"


# Exercise 1: insert mock vectors and report average/p95 latency and query throughput.
@dataclass(frozen=True)
class BenchmarkReport:
    document_count: int
    query_count: int
    average_latency_ms: float
    p95_latency_ms: float
    queries_per_second: float


class VectorDBBenchmark:
    def __init__(self, chroma_client: Any, collection_name: str = "benchmark") -> None:
        self.client = chroma_client
        self.collection_name = collection_name

    def run(self, document_count: int = 1000, query_count: int = 100) -> BenchmarkReport:
        if document_count <= 0 or query_count <= 0:
            raise ValueError("document_count and query_count must be positive")
        existing_names = {
            item if isinstance(item, str) else item.name
            for item in self.client.list_collections()
        }
        if self.collection_name in existing_names:
            self.client.delete_collection(self.collection_name)
        collection = self.client.create_collection(
            self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        documents = [f"Mock document {index} about topic {index % 20}." for index in range(document_count)]
        collection.add(
            ids=[f"doc-{index}" for index in range(document_count)],
            documents=documents,
            embeddings=[_mock_embedding(text) for text in documents],
        )
        latencies = []
        started_all = time.perf_counter()
        for index in range(query_count):
            query = f"Find documents about topic {index % 20}"
            started = time.perf_counter()
            collection.query(
                query_embeddings=[_mock_embedding(query)],
                n_results=min(5, document_count),
            )
            latencies.append((time.perf_counter() - started) * 1000)
        elapsed = time.perf_counter() - started_all
        ordered = sorted(latencies)
        p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
        return BenchmarkReport(
            document_count=document_count,
            query_count=query_count,
            average_latency_ms=statistics.fmean(latencies),
            p95_latency_ms=ordered[p95_index],
            queries_per_second=query_count / elapsed if elapsed else 0.0,
        )


# Exercise 2: search named Chroma collections and merge using cosine similarity.
@dataclass(frozen=True)
class MergedResult:
    collection: str
    id: str
    text: str
    score: float
    metadata: dict[str, Any]


class MultiCollectionSearch:
    def __init__(self, chroma_client: Any, collection_names: list[str]) -> None:
        self.client = chroma_client
        self.collection_names = list(dict.fromkeys(collection_names))

    def search(self, query_embedding: list[float], k: int = 5) -> list[MergedResult]:
        if k <= 0 or not self.collection_names:
            return []
        merged = []
        for name in self.collection_names:
            collection = self.client.get_collection(name)
            result = collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "distances", "metadatas"],
            )
            for doc_id, text, distance, metadata in zip(
                result["ids"][0],
                result["documents"][0],
                result["distances"][0],
                result["metadatas"][0],
            ):
                merged.append(
                    MergedResult(
                        collection=name,
                        id=doc_id,
                        text=text or "",
                        score=1.0 - float(distance),
                        metadata=metadata or {},
                    )
                )
        merged.sort(key=lambda result: (-result.score, result.collection, result.id))
        return merged[:k]


# Exercise 3: route operations to a separate collection for each tenant.
class TenantVectorStore:
    def __init__(self, chroma_client: Any) -> None:
        self.client = chroma_client

    def _collection(self, tenant_id: str) -> Any:
        return self.client.get_or_create_collection(
            name=_validate_tenant_id(tenant_id),
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, tenant_id: str, ids: list[str], texts: list[str]) -> None:
        if len(ids) != len(texts):
            raise ValueError("ids and texts must have the same length")
        if len(ids) != len(set(ids)):
            raise ValueError("document IDs must be unique within a batch")
        if not ids:
            return
        self._collection(tenant_id).upsert(
            ids=ids,
            documents=texts,
            embeddings=[_mock_embedding(text) for text in texts],
            metadatas=[{"tenant_id": tenant_id} for _ in texts],
        )

    def search(self, tenant_id: str, query: str, k: int = 5) -> list[dict[str, Any]]:
        if k <= 0:
            return []
        result = self._collection(tenant_id).query(
            query_embeddings=[_mock_embedding(query)],
            n_results=k,
            include=["documents", "distances", "metadatas"],
        )
        return [
            {"id": doc_id, "text": text, "distance": distance, "metadata": metadata}
            for doc_id, text, distance, metadata in zip(
                result["ids"][0],
                result["documents"][0],
                result["distances"][0],
                result["metadatas"][0],
            )
        ]


# Exercise 4: export/import Chroma records including vector and metadata.
def export_collection(collection: Any) -> list[dict[str, Any]]:
    records = collection.get(include=["documents", "embeddings", "metadatas"])
    return [
        {
            "id": doc_id,
            "text": text,
            "embedding": embedding.tolist() if hasattr(embedding, "tolist") else list(embedding),
            "metadata": metadata or {},
        }
        for doc_id, text, embedding, metadata in zip(
            records["ids"],
            records["documents"],
            records["embeddings"],
            records["metadatas"],
        )
    ]


def save_collection_json(collection: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as output_file:
        json.dump(export_collection(collection), output_file, ensure_ascii=False, indent=2)


def import_collection_json(chroma_client: Any, collection_name: str, path: str) -> Any:
    with open(path, encoding="utf-8") as input_file:
        records = json.load(input_file)
    if not isinstance(records, list):
        raise ValueError("Collection export must be a JSON list")
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    for record in records:
        if not all(key in record for key in ("id", "text", "embedding", "metadata")):
            raise ValueError("Every record must include id, text, embedding, and metadata")
    if records:
        collection.upsert(
            ids=[record["id"] for record in records],
            documents=[record["text"] for record in records],
            embeddings=[record["embedding"] for record in records],
            metadatas=[record["metadata"] for record in records],
        )
    return collection


def run_local_tests() -> None:
    assert len(_mock_embedding("same")) == 32
    assert _mock_embedding("same") == _mock_embedding("same")
    assert _validate_tenant_id("Team-1") == "tenant_team-1"
    try:
        _validate_tenant_id("../other")
    except ValueError:
        pass
    else:
        raise AssertionError("Tenant collection names must be validated")

    class FakeCollection:
        def __init__(self, name: str) -> None:
            self.name = name
            self.records: dict[str, dict[str, Any]] = {}

        def upsert(self, ids: list[str], documents: list[str], embeddings: list[list[float]], metadatas: list[dict[str, Any]] | None = None) -> None:
            metadata_values = metadatas or [{} for _ in ids]
            for doc_id, text, embedding, metadata in zip(ids, documents, embeddings, metadata_values):
                self.records[doc_id] = {
                    "text": text,
                    "embedding": list(embedding),
                    "metadata": dict(metadata),
                }

        add = upsert

        def query(self, query_embeddings: list[list[float]], n_results: int, include: list[str] | None = None) -> dict[str, list[list[Any]]]:
            query_vector = query_embeddings[0]
            scored = []
            for doc_id, record in self.records.items():
                similarity = sum(left * right for left, right in zip(query_vector, record["embedding"]))
                scored.append((1 - similarity, doc_id, record))
            scored.sort(key=lambda item: item[0])
            selected = scored[:n_results]
            return {
                "ids": [[item[1] for item in selected]],
                "documents": [[item[2]["text"] for item in selected]],
                "distances": [[item[0] for item in selected]],
                "metadatas": [[item[2]["metadata"] for item in selected]],
            }

        def get(self, include: list[str] | None = None) -> dict[str, list[Any]]:
            return {
                "ids": list(self.records),
                "documents": [record["text"] for record in self.records.values()],
                "embeddings": [record["embedding"] for record in self.records.values()],
                "metadatas": [record["metadata"] for record in self.records.values()],
            }

    class FakeClient:
        def __init__(self) -> None:
            self.collections: dict[str, FakeCollection] = {}

        def list_collections(self) -> list[FakeCollection]:
            return list(self.collections.values())

        def delete_collection(self, name: str) -> None:
            self.collections.pop(name, None)

        def create_collection(self, name: str, metadata: dict[str, Any] | None = None) -> FakeCollection:
            collection = FakeCollection(name)
            self.collections[name] = collection
            return collection

        def get_collection(self, name: str) -> FakeCollection:
            return self.collections[name]

        def get_or_create_collection(self, name: str, metadata: dict[str, Any] | None = None) -> FakeCollection:
            return self.collections.setdefault(name, FakeCollection(name))

    fake_client = FakeClient()
    report = VectorDBBenchmark(fake_client, "bench-test").run(document_count=8, query_count=3)
    assert report.document_count == 8 and report.query_count == 3
    assert report.average_latency_ms >= 0 and report.p95_latency_ms >= 0
    assert report.queries_per_second > 0

    first_collection = fake_client.get_or_create_collection("first")
    second_collection = fake_client.get_or_create_collection("second")
    first_collection.upsert(["a"], ["alpha document"], [_mock_embedding("alpha document")], [{}])
    second_collection.upsert(["b"], ["beta document"], [_mock_embedding("beta document")], [{}])
    merged = MultiCollectionSearch(fake_client, ["first", "second"]).search(_mock_embedding("alpha document"), k=2)
    assert len(merged) == 2 and merged[0].text == "alpha document"

    tenant_store = TenantVectorStore(fake_client)
    tenant_store.add("team-a", ["private"], ["private alpha notes"])
    tenant_store.add("team-b", ["other"], ["different beta notes"])
    assert tenant_store.search("team-a", "private alpha", k=3)[0]["id"] == "private"
    assert tenant_store.search("team-b", "private alpha", k=3)[0]["id"] == "other"

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        export_path = str(Path(directory) / "collection.json")
        save_collection_json(first_collection, export_path)
        restored_client = FakeClient()
        restored = import_collection_json(restored_client, "restored", export_path)
        assert export_collection(restored) == export_collection(first_collection)


if __name__ == "__main__":
    run_local_tests()
    print("Module 11 exercise tests passed (mock embedding helpers; Chroma not required).")