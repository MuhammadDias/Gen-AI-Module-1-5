from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class WorkflowState(TypedDict):
    question: str
    steps: list[str]
    answer: str


def retrieve(state: WorkflowState) -> dict[str, list[str]]:
    return {"steps": [*state["steps"], "retrieved context"]}


def respond(state: WorkflowState) -> dict[str, str]:
    return {"answer": f"A response to: {state['question']}"}


def build_graph():
    workflow = StateGraph(WorkflowState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("respond", respond)
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "respond")
    workflow.add_edge("respond", END)
    return workflow.compile()


if __name__ == "__main__":
    graph = build_graph()
    result = graph.invoke(
        {"question": "What is a stateful graph?", "steps": [], "answer": ""}
    )
    print(result)