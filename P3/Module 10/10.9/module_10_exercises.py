from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# Exercise 1: create a folder-backed LlamaIndex Q&A function with source filenames.
def build_folder_qa(folder: str | Path, llm: Any = None, embed_model: Any = None) -> Callable[[str], dict[str, Any]]:
    from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex

    directory = Path(folder)
    if not directory.is_dir():
        raise FileNotFoundError(f"Document folder not found: {directory}")
    documents = SimpleDirectoryReader(
        input_dir=str(directory),
        required_exts=[".txt"],
        filename_as_id=True,
    ).load_data()
    if not documents:
        raise ValueError(f"No .txt documents found in {directory}")
    if llm is not None:
        Settings.llm = llm
    if embed_model is not None:
        Settings.embed_model = embed_model
    query_engine = VectorStoreIndex.from_documents(documents).as_query_engine()

    def ask(question: str) -> dict[str, Any]:
        response = query_engine.query(question)
        sources = sorted(
            {
                Path(str(node.node.metadata.get("file_name", node.node.metadata.get("file_path", "unknown")))).name
                for node in response.source_nodes
            }
        )
        return {"answer": str(response), "sources": sources}

    return ask


# Exercise 2: compare vector retrieval with document-wide summary synthesis.
def compare_index_answers(
    paragraphs: Sequence[str],
    questions: Sequence[str],
    query_fn: Callable[[str, str, Sequence[str]], str],
) -> list[dict[str, str]]:
    if not paragraphs:
        raise ValueError("At least one paragraph is required")
    results = []
    for index_type in ("VectorStoreIndex", "SummaryIndex"):
        for question in questions:
            results.append(
                {
                    "index_type": index_type,
                    "question": question,
                    "answer": query_fn(index_type, question, paragraphs),
                }
            )
    return results


def compare_llamaindex_indexes(
    paragraphs: Sequence[str],
    questions: Sequence[str],
    llm: Any = None,
    embed_model: Any = None,
) -> list[dict[str, str]]:
    from llama_index.core import Document, Settings, SummaryIndex, VectorStoreIndex

    if llm is not None:
        Settings.llm = llm
    if embed_model is not None:
        Settings.embed_model = embed_model
    documents = [Document(text=paragraph) for paragraph in paragraphs]
    indexes = {
        "VectorStoreIndex": VectorStoreIndex.from_documents(documents),
        "SummaryIndex": SummaryIndex.from_documents(documents),
    }
    engines = {name: index.as_query_engine() for name, index in indexes.items()}
    return compare_index_answers(
        paragraphs,
        questions,
        lambda index_type, question, _paragraphs: str(engines[index_type].query(question)),
    )


# Exercise 3: enforce both metadata constraints before ranking/retrieving.
def filter_nodes_by_metadata(nodes: Iterable[Any], topic: str, year: int) -> list[Any]:
    return [
        node
        for node in nodes
        if node.metadata.get("topic") == topic and node.metadata.get("year") == year
    ]


def build_metadata_filtered_retriever(index: Any, topic: str, year: int, top_k: int = 5) -> Any:
    from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    filters = MetadataFilters(
        filters=[
            ExactMatchFilter(key="topic", value=topic),
            ExactMatchFilter(key="year", value=year),
        ]
    )
    return index.as_retriever(filters=filters, similarity_top_k=top_k)


@dataclass(frozen=True)
class RerankedNode:
    node: Any
    score: float


# Exercise 4: score every retrieved node and return the highest-scoring three.
def rerank_nodes(
    question: str,
    nodes: Sequence[Any],
    score_fn: Callable[[str, str], float],
    top_k: int = 3,
) -> list[RerankedNode]:
    if top_k <= 0:
        return []
    scored = []
    for position, node in enumerate(nodes):
        text = getattr(node, "text", str(node))
        score = float(score_fn(question, text))
        if not 0 <= score <= 10:
            raise ValueError("Relevance scores must be between 0 and 10")
        scored.append((position, RerankedNode(node=node, score=score)))
    scored.sort(key=lambda item: (-item[1].score, item[0]))
    return [result for _, result in scored[:top_k]]


def build_llm_relevance_scorer(llm: Any) -> Callable[[str, str], float]:
    def score(question: str, text: str) -> float:
        response = llm.complete(
            "Rate how relevant the passage is to the question from 0 to 10. "
            "Return only a number.\n"
            f"Question: {question}\nPassage: {text}"
        )
        raw = getattr(response, "text", str(response)).strip()
        try:
            value = float(raw)
        except ValueError as error:
            raise ValueError(f"Invalid relevance score: {raw!r}") from error
        if not 0 <= value <= 10:
            raise ValueError("Relevance scores must be between 0 and 10")
        return value

    return score


def run_local_tests() -> None:
    paragraphs = ["p1", "p2"]
    compared = compare_index_answers(
        paragraphs,
        ["specific?", "overall?"],
        lambda index_type, question, _paragraphs: f"{index_type}:{question}",
    )
    assert len(compared) == 4
    assert compared[0]["answer"] == "VectorStoreIndex:specific?"
    assert compared[2]["index_type"] == "SummaryIndex"

    nodes = [
        type("Node", (), {"metadata": {"topic": "embeddings", "year": 2024}})(),
        type("Node", (), {"metadata": {"topic": "RAG", "year": 2024}})(),
        type("Node", (), {"metadata": {"topic": "embeddings", "year": 2023}})(),
    ]
    assert filter_nodes_by_metadata(nodes, "embeddings", 2024) == [nodes[0]]

    ranked = rerank_nodes("question", ["low", "high", "middle", "last"], lambda _q, text: {"low": 1, "high": 9, "middle": 6, "last": 3}[text])
    assert [item.node for item in ranked] == ["high", "middle", "last"]
    assert rerank_nodes("question", [], lambda _q, _text: 0) == []


if __name__ == "__main__":
    run_local_tests()
    print("Module 10 exercise tests passed (mocked; no API requests).")