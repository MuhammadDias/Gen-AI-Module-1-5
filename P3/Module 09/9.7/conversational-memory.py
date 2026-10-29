import os

from dotenv import load_dotenv
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

load_dotenv()


def main() -> None:
    model = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "qwen/qwen3.7-flash:free"),
        api_key=os.environ["OPEN_ROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful AI tutor. Be concise."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )
    chain = prompt | model | StrOutputParser()
    store: dict[str, InMemoryChatMessageHistory] = {}

    def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    config = {"configurable": {"session_id": "user-123"}}
    print("Turn 1:", chain_with_history.invoke({"input": "What is RAG?"}, config=config))
    print(
        "Turn 2:",
        chain_with_history.invoke(
            {"input": "What are the main components I just asked about?"}, config=config
        ),
    )
    print("Stored messages:", len(get_session_history("user-123").messages))


if __name__ == "__main__":
    main()