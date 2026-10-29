import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()


def create_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "qwen/qwen3.8-27b:free"),
        temperature=0.7,
        max_tokens=512,
        api_key=os.environ["OPEN_ROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )


if __name__ == "__main__":
    model = create_model()
    messages = [
        SystemMessage(content="You are a concise technical assistant."),
        HumanMessage(content="What is LangChain Expression Language?"),
    ]
    print(model.invoke(messages).content)

    print("\nStreaming:")
    for chunk in model.stream(messages):
        print(chunk.content, end="", flush=True)
    print()

    responses = model.batch(
        [[HumanMessage(content="What is RAG?")], [HumanMessage(content="What is a vector database?")]]
    )
    for response in responses:
        print(response.content[:120])