"""
Video 20: Human-in-the-Loop (HITL) — basic example.

Concept:
  HITL = placing a human checkpoint inside the AI workflow.
  The graph pauses at the interrupt point, saves its state to the checkpointer,
  and waits for the human to provide a decision. On resume it picks up from exactly
  the same point.

This example: the human must approve BEFORE the LLM processes the question.

LangGraph API:
  interrupt(value)         — pauses the graph, sends `value` to the caller
  Command(resume=value)    — resumes the paused graph with the given value

The interrupt value is accessible at result["__interrupt__"][0].value after the
first invoke call. Resuming requires calling invoke() again with Command(resume=...)
and the SAME config (same thread_id so the checkpointer can restore the state).

Checkpointer is REQUIRED for HITL — it is how the graph state is persisted
between the first (pause) and second (resume) invocations.
"""

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from typing import Annotated, List, TypedDict


# ─── State ────────────────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# ─── Mock LLM ─────────────────────────────────────────────────────────────────

def mock_llm_response(question: str) -> str:
    """Simulate a simple LLM response."""
    q = question.lower()
    if "gradient descent" in q:
        return (
            "Gradient descent is an optimisation algorithm that iteratively moves "
            "model parameters in the direction that minimises a loss function. "
            "It updates weights by subtracting the gradient multiplied by a learning rate."
        )
    if "transformer" in q:
        return (
            "The Transformer architecture uses self-attention to process all tokens "
            "in parallel rather than sequentially. It was introduced in 'Attention is "
            "All You Need' (2017) and forms the backbone of models like GPT and BERT."
        )
    return f"Here is my answer to your question: '{question}'."


# ─── HITL node ────────────────────────────────────────────────────────────────

def chat_node(state: State) -> dict:
    """
    Human approval checkpoint: pause before calling the LLM.
    The interrupt value is the dict sent back to the caller.
    When resumed, `decision` holds whatever the human passed in Command(resume=...).
    """
    last_question = state["messages"][-1].content

    # Pause here — the graph saves state and returns to the caller
    decision = interrupt({
        "question": last_question,
        "instruction": "Do you approve sending this question to the LLM? (yes/no)",
    })

    # Resume point — decision now contains the human's response
    approved = str(decision.get("approved", "no")).lower()

    if approved != "yes":
        return {
            "messages": [
                AIMessage(content="Your request was not sent to the LLM. Approval declined.")
            ]
        }

    # Human approved — call the LLM
    answer = mock_llm_response(last_question)
    return {"messages": [AIMessage(content=answer)]}


# ─── Graph assembly ───────────────────────────────────────────────────────────

checkpointer = MemorySaver()

graph = StateGraph(State)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "hitl-basic-001"}}


# ─── Interactive CLI ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== HITL Basic Example ===")
    print("The LLM will only respond AFTER you approve.")
    print()

    user_input = input("You: ").strip()
    if not user_input:
        print("No input provided. Exiting.")
        exit()

    # First invoke — hits interrupt, returns with "__interrupt__" key
    result = chatbot.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config,
    )

    # Check for interrupt
    interrupt_data = result.get("__interrupt__")
    if interrupt_data:
        interrupt_value = interrupt_data[0].value
        print()
        print("[HITL] Approval required before sending to LLM.")
        print(f"  Question   : {interrupt_value['question']}")
        print(f"  Instruction: {interrupt_value['instruction']}")
        print()

        user_decision = input("Your decision (yes/no): ").strip().lower()

        # Second invoke — resumes from the interrupt with the human's decision
        result = chatbot.invoke(
            Command(resume={"approved": user_decision}),
            config=config,
        )

    # Print the final AI message
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage):
            print()
            print(f"Bot: {msg.content}")
            break
