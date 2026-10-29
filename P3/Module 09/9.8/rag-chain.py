import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

load_dotenv()

DOCUMENTS = [
    "RAG stands for Retrieval-Augmented Generation. It connects LLMs to external knowledge bases.",
    "Retrieval uses embeddings and vector search to find relevant documents for a query.",
    "Generation uses retrieved context and the question to produce a grounded answer.",
    "Vector databases store embeddings for fast similarity search.",
    "Fine-tuning changes model weights; RAG retrieves knowledge at inference time.",
]


def format_documents(documents: list[Document]) -> str:
    return "\n\n".join(document.page_content for document in documents)


def main() -> None:
    api_key = os.environ["OPEN_ROUTER_API_KEY"]
    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    documents = [Document(page_content=text) for text in DOCUMENTS]
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=200, chunk_overlap=30
    ).split_documents(documents)
    vectorstore = Chroma.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                'Answer using ONLY this context. If it is insufficient, say "I don\'t have that information."\n{context}',
            ),
            ("human", "{question}"),
        ]
    )
    model = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "qwen/qwen3.7-flash:free"),
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )
    chain = (
        {"context": retriever | format_documents, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    for question in (
        "What does RAG stand for?",
        "How is RAG different from fine-tuning?",
        "What is the capital of France?",
    ):
        print(f"Q: {question}\nA: {chain.invoke(question)}\n")


if __name__ == "__main__":
    main()