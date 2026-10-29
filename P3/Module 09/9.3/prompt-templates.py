from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a {role}. Answer in {language}."),
        ("human", "{question}"),
    ]
)

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Today's context: {context}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)


if __name__ == "__main__":
    messages = prompt.format_messages(
        role="Python tutor",
        language="English",
        question="What is a generator?",
    )
    print(messages)

    formatted = chat_prompt.format_messages(
        context="User is learning about RAG",
        history=[HumanMessage(content="What is retrieval?")],
        input="How is it different from fine-tuning?",
    )
    print(f"Total messages: {len(formatted)}")