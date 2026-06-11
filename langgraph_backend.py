import re
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from typing import TypedDict, Annotated, List


class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


class MockChatModel:
    """Keyword-based mock that simulates basic conversational AI."""

    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        if not messages:
            return AIMessage(content="Hello! How can I assist you today?")

        last_msg = messages[-1].content.lower() if messages else ""
        all_text = " ".join(
            m.content.lower() for m in messages if hasattr(m, "content")
        )

        # Extract name from conversation history
        name_match = re.search(r"my name is (\w+)", all_text)
        user_name = name_match.group(1).capitalize() if name_match else None

        if any(w in last_msg for w in ["hi", "hello", "hey", "namaste"]):
            return AIMessage(content="Hello! How can I assist you today?")

        if "my name is" in last_msg:
            name = re.search(r"my name is (\w+)", last_msg)
            n = name.group(1).capitalize() if name else "there"
            return AIMessage(content=f"Nice to meet you, {n}! How can I help you?")

        if "what is my name" in last_msg or "what's my name" in last_msg:
            if user_name:
                return AIMessage(content=f"Your name is {user_name}!")
            return AIMessage(
                content="You haven't told me your name yet. What is it?"
            )

        if "capital" in last_msg and "india" in last_msg:
            return AIMessage(content="The capital of India is New Delhi.")

        if "capital" in last_msg and "kerala" in last_msg:
            return AIMessage(content="The capital of Kerala is Thiruvananthapuram.")

        if "capital" in last_msg and "france" in last_msg:
            return AIMessage(content="The capital of France is Paris.")

        if "pasta" in last_msg or ("recipe" in last_msg and "pasta" in all_text):
            return AIMessage(
                content=(
                    "Simple pasta recipe:\n"
                    "1. Boil pasta in salted water.\n"
                    "2. Sauté garlic in olive oil.\n"
                    "3. Add tomatoes and simmer 10 min.\n"
                    "4. Toss pasta in sauce.\n"
                    "5. Serve with parmesan."
                )
            )

        if "joke" in last_msg:
            return AIMessage(
                content="Why do programmers prefer dark mode? Because light attracts bugs!"
            )

        if "langgraph" in last_msg:
            return AIMessage(
                content=(
                    "LangGraph is a library for building stateful, multi-actor "
                    "applications with LLMs. It models workflows as graphs where "
                    "nodes are functions and edges define execution flow."
                )
            )

        if any(w in last_msg for w in ["bye", "goodbye", "thanks", "thank you"]):
            return AIMessage(content="Goodbye! Feel free to chat again anytime.")

        return AIMessage(
            content=(
                "That's an interesting question! I'm a demo chatbot built with "
                "LangGraph. Ask me about capitals, pasta recipes, LangGraph, or "
                "tell me your name!"
            )
        )


llm = MockChatModel()


def chat_node(state: ChatState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


checkpointer = MemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

CONFIG = {"configurable": {"thread_id": "streamlit_user_001"}}
