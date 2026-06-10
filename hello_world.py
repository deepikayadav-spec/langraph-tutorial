from langgraph.graph import StateGraph, END
from typing import TypedDict


class State(TypedDict):
    message: str


def hello_node(state: State) -> State:
    return {"message": "Hello, World! from LangGraph"}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("hello", hello_node)
    graph.set_entry_point("hello")
    graph.add_edge("hello", END)
    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    result = app.invoke({"message": ""})
    print(result["message"])
