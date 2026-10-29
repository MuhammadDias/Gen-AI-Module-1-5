from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

raw_documents = [
    Document(
        page_content=(
            "LangChain provides a standard interface for interacting with language models. "
            "LCEL makes pipelines declarative and easy to debug."
        ),
        metadata={"source": "langchain_docs.txt", "page": 1},
    ),
    Document(
        page_content=(
            "RAG combines a retriever with a generator. Retrieved documents are added to "
            "the generator context so the model can use external knowledge."
        ),
        metadata={"source": "rag_guide.txt", "page": 1},
    ),
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=120,
    chunk_overlap=20,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
)


if __name__ == "__main__":
    chunks = splitter.split_documents(raw_documents)
    print(f"Original documents: {len(raw_documents)}, chunks: {len(chunks)}")
    for index, chunk in enumerate(chunks):
        print(
            f"Chunk {index}: {len(chunk.page_content)} chars | "
            f"source: {chunk.metadata['source']}"
        )