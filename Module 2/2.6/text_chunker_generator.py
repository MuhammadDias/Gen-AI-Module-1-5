# 2.6 Generators and Lazy Evaluation
from typing import Generator

def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> Generator[str, None, None]:
    """Yield overlapping text chunks for embedding/RAG pipelines."""
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        yield text[start:end]
        start += chunk_size - overlap

if __name__ == "__main__":
    document = "Python is a versatile language for GenAI. " * 30
    chunk_count = 0
    for chunk in chunk_text(document, chunk_size=100, overlap=20):
        chunk_count += 1
    print(f"Processed {chunk_count} chunks lazily.")
