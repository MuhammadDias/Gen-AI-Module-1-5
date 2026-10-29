from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from typing import Any


def _parse_json_list(text: str, expected_length: int | None = None) -> list[str]:
    cleaned = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE)
    if match:
        cleaned = match.group(1)
    value = json.loads(cleaned)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("The model response must be a JSON list of strings")
    if expected_length is not None and len(value) != expected_length:
        raise ValueError(f"Expected exactly {expected_length} items, got {len(value)}")
    return value


# Exercise 1: extract facts, then generate one quiz question per fact.
def build_fact_quiz_chain(llm: Any) -> Any:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda

    extract_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Extract exactly 3 distinct key facts. Return a JSON list of strings only."),
            ("human", "Document:\n{document}"),
        ]
    )
    quiz_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Write one clear quiz question for each fact. Return a JSON list of strings only."),
            ("human", "Facts:\n{facts}"),
        ]
    )
    return (
        extract_prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(lambda raw: {"facts": json.dumps(_parse_json_list(raw, 3))})
        | quiz_prompt
        | llm
        | StrOutputParser()
        | RunnableLambda(lambda raw: _parse_json_list(raw, 3))
    )


# Exercise 2: retrieve three paraphrases and deduplicate by document content hash.
class MultiQueryRetriever:
    def __init__(
        self,
        paraphrase: Callable[[str], Sequence[str]],
        retrieve: Callable[[str], Sequence[Any]],
    ) -> None:
        self.paraphrase = paraphrase
        self.retrieve = retrieve

    def invoke(self, query: str) -> list[Any]:
        queries = list(self.paraphrase(query))
        if len(queries) != 3 or any(not item.strip() for item in queries):
            raise ValueError("The query generator must return 3 non-empty paraphrases")

        unique: dict[str, Any] = {}
        for paraphrase in queries:
            for document in self.retrieve(paraphrase):
                content = getattr(document, "page_content", document)
                if not isinstance(content, str):
                    raise TypeError("Retrieved documents must contain text")
                content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                unique.setdefault(content_hash, document)
        return list(unique.values())


def _split_with_overlap(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0")
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return [chunk for chunk in chunks if chunk]


# Exercise 3: map chunk summaries, then reduce summaries until one remains.
def summarize_long_document(
    document: str,
    summarize: Callable[[str], str],
    chunk_size: int = 1200,
    overlap: int = 100,
) -> str:
    chunks = _split_with_overlap(document, chunk_size, overlap)
    if not chunks:
        return ""
    summaries = [summarize(chunk) for chunk in chunks]
    while len(summaries) > 1:
        combined = "\n\n".join(summaries)
        reduced_chunks = _split_with_overlap(combined, chunk_size, overlap)
        next_summaries = [summarize(chunk) for chunk in reduced_chunks]
        if len(next_summaries) >= len(summaries):
            raise RuntimeError("Summarizer did not reduce the intermediate summaries")
        summaries = next_summaries
    return summaries[0]


def build_map_reduce_chain(llm: Any, chunk_size: int = 2000) -> Any:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)
    prompt = ChatPromptTemplate.from_messages(
        [("system", "Summarize the supplied text accurately and concisely."), ("human", "{text}")]
    )
    summarize = prompt | llm | StrOutputParser()

    def map_reduce(values: dict[str, str]) -> str:
        chunks = splitter.split_text(values["document"])
        summaries = [summarize.invoke({"text": chunk}) for chunk in chunks]
        while len(summaries) > 1:
            reduced = [
                summarize.invoke({"text": chunk})
                for chunk in splitter.split_text("\n\n".join(summaries))
            ]
            if len(reduced) >= len(summaries):
                raise RuntimeError("Summarizer did not reduce the intermediate summaries")
            summaries = reduced
        return summaries[0] if summaries else ""

    return RunnableLambda(map_reduce)


# Exercise 4: grade a generated answer against the exact context used for RAG.
def score_groundedness(
    context: str,
    answer: str,
    grader: Callable[[str, str], int],
) -> int:
    score = grader(context, answer)
    if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 10:
        raise ValueError("Groundedness score must be an integer from 0 to 10")
    return score


def build_rag_quality_chain(llm: Any) -> Any:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer using only the retrieved context. Say when it is insufficient.\n{context}"),
            ("human", "{question}"),
        ]
    )
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Score how well the answer is supported by the context from 0 to 10. Return only an integer."),
            ("human", "Context:\n{context}\n\nAnswer:\n{answer}"),
        ]
    )

    def parse_score(raw: str) -> int:
        match = re.fullmatch(r"\s*(10|[0-9])\s*", raw)
        if not match:
            raise ValueError("The grader must return an integer from 0 to 10")
        return int(match.group(1))

    return (
        RunnablePassthrough.assign(answer=answer_prompt | llm | StrOutputParser())
        .assign(groundedness=grade_prompt | llm | StrOutputParser() | RunnableLambda(parse_score))
        | RunnableLambda(lambda result: {"answer": result["answer"], "groundedness": result["groundedness"]})
    )


def run_local_tests() -> None:
    assert _parse_json_list('["fact 1", "fact 2", "fact 3"]', 3) == [
        "fact 1",
        "fact 2",
        "fact 3",
    ]

    class Document:
        def __init__(self, page_content: str) -> None:
            self.page_content = page_content

    retrieval_calls: list[str] = []

    def retrieve(query: str) -> list[Document]:
        retrieval_calls.append(query)
        return [Document("shared"), Document(query)]

    retriever = MultiQueryRetriever(lambda _: ["variant a", "variant b", "variant c"], retrieve)
    retrieved = retriever.invoke("original")
    assert retrieval_calls == ["variant a", "variant b", "variant c"]
    assert len(retrieved) == 4
    assert retrieved[0].page_content == "shared"

    summarized = summarize_long_document(
        "alpha beta gamma delta epsilon zeta",
        lambda chunk: chunk.split()[0],
        chunk_size=12,
        overlap=2,
    )
    assert summarized == "alpha"
    assert summarize_long_document("", lambda chunk: chunk) == ""

    assert score_groundedness("context", "answer", lambda _context, _answer: 8) == 8
    try:
        score_groundedness("context", "answer", lambda _context, _answer: 11)
    except ValueError:
        pass
    else:
        raise AssertionError("Out-of-range groundedness scores must be rejected")


if __name__ == "__main__":
    run_local_tests()
    print("Module 09 exercise tests passed (mocked; no API requests).")