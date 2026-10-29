import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_openai import ChatOpenAI

load_dotenv()


def create_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "qwen/qwen3.7-flash:free"),
        api_key=os.environ["OPEN_ROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        max_tokens=512,
    )


def main() -> None:
    model = create_model()
    parser = StrOutputParser()

    summary_prompt = ChatPromptTemplate.from_messages(
        [("system", "Summarise this in one sentence."), ("human", "{text}")]
    )
    keyword_prompt = ChatPromptTemplate.from_messages(
        [("system", "Extract 3 keywords. Return them comma-separated."), ("human", "{text}")]
    )
    parallel = RunnableParallel(
        summary=summary_prompt | model | parser,
        keywords=keyword_prompt | model | parser,
    )
    text = "RAG improves LLM answers by retrieving relevant documents at inference time."
    print(parallel.invoke({"text": text}))

    code_chain = (
        ChatPromptTemplate.from_messages(
            [("system", "Give a complete Python code example."), ("human", "{input}")]
        )
        | model
        | parser
    )
    prose_chain = (
        ChatPromptTemplate.from_messages(
            [("system", "Explain the concept clearly."), ("human", "{input}")]
        )
        | model
        | parser
    )

    def route(values: dict[str, str]) -> str:
        question = values["input"].lower()
        wants_code = any(word in question for word in ("write", "code", "implement", "example"))
        return (code_chain if wants_code else prose_chain).invoke(values)

    router = RunnableLambda(route)
    print(router.invoke({"input": "Write a function to chunk text with overlap."}))
    print(router.invoke({"input": "What is attention in transformers?"}))


if __name__ == "__main__":
    main()