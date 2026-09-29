from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys

import numpy as np


@dataclass
class RetrievalEvalCase:
    query: str
    relevant_doc_ids: list[str]


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Return the fraction of the top-k results that are relevant."""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Return the fraction of all relevant documents found in the top-k."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Return the reciprocal rank of the first relevant result."""
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(store, eval_cases: list[RetrievalEvalCase], k: int = 5) -> dict:
    """Run all evaluation cases and return aggregate metrics."""
    precision_scores = []
    recall_scores = []
    mrr_scores = []
    for case in eval_cases:
        results = store.search(case.query, k=k)
        retrieved_ids = [result.document.id for result in results]
        precision_scores.append(
            precision_at_k(retrieved_ids, case.relevant_doc_ids, k)
        )
        recall_scores.append(recall_at_k(retrieved_ids, case.relevant_doc_ids, k))
        mrr_scores.append(
            mean_reciprocal_rank(retrieved_ids, case.relevant_doc_ids)
        )
    return {
        f"precision@{k}": round(float(np.mean(precision_scores)), 4),
        f"recall@{k}": round(float(np.mean(recall_scores)), 4),
        "MRR": round(float(np.mean(mrr_scores)), 4),
    }


eval_cases = [
    RetrievalEvalCase("How does RAG work?", ["d01", "d08"]),
    RetrievalEvalCase("What are vector databases?", ["d02", "d06"]),
    RetrievalEvalCase("How do agents use language models?", ["d10"]),
    RetrievalEvalCase("What is fine-tuning?", ["d03"]),
    RetrievalEvalCase("How do transformers model token relationships?", ["d09"]),
]


def load_vector_pipeline():
    pipeline_path = (
        Path(__file__).resolve().parents[1]
        / "8.4"
        / "semantic-search-pipeline.py"
    )
    spec = importlib.util.spec_from_file_location("semantic_search_pipeline", pipeline_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load vector pipeline from {pipeline_path}")
    pipeline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pipeline
    spec.loader.exec_module(pipeline)
    return pipeline


if __name__ == "__main__":
    pipeline = load_vector_pipeline()
    store = pipeline.VectorStore()
    store.add_documents(pipeline.CORPUS)
    metrics = evaluate_retrieval(store, eval_cases, k=3)
    print(metrics)
