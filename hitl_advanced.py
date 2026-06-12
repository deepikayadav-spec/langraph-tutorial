"""
Video 20: Human-in-the-Loop (HITL) — advanced example (stock purchase chatbot).

Architecture:
  Two tools with different trust levels:
    - get_stock_price  : no HITL — safe read operation, run automatically
    - purchase_stocks  : HITL inside the tool — financial action needs human sign-off

HITL is placed INSIDE the tool function (not in the node). This is cleaner because:
  - The tool itself knows it is dangerous
  - The graph / node code stays generic
  - Different tools can have different approval requirements

Flow for purchase:
  1. User: "Buy 10 shares of TSLA"
  2. chat_node → AIMessage with tool_call for purchase_stocks
  3. ToolNode runs purchase_stocks → interrupt("Approve buying 10 shares of TSLA?")
  4. Graph pauses, state saved to MemorySaver
  5. CLI shows approval prompt
  6. User types "yes" or "no"
  7. chatbot.invoke(Command(resume="yes"), config) resumes
  8. purchase_stocks picks up after the interrupt — decision = "yes"
  9. Executes or cancels the purchase
  10. Returns result string → chat_node summarises

The re-invoke uses the SAME config (same thread_id) so the checkpointer restores
the exact graph state, including the partial ToolNode execution.
"""

import re
import uuid
from typing import Annotated, List, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt


# ─── State ────────────────────────────────────────────────────────────────────

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# ─── Tools ────────────────────────────────────────────────────────────────────

@tool
def get_stock_price(company: str) -> str:
    """
    Get the current stock price for a company.
    Accepts company name (e.g. 'Tesla', 'Apple') or ticker (e.g. 'TSLA', 'AAPL').
    No approval required — this is a safe read operation.
    """
    company_lower = company.lower().strip()
    mock = {
        "apple": ("AAPL", 213.49),
        "aapl": ("AAPL", 213.49),
        "tesla": ("TSLA", 248.23),
        "tsla": ("TSLA", 248.23),
        "google": ("GOOGL", 176.50),
        "googl": ("GOOGL", 176.50),
        "microsoft": ("MSFT", 425.27),
        "msft": ("MSFT", 425.27),
        "amazon": ("AMZN", 198.12),
        "amzn": ("AMZN", 198.12),
        "meta": ("META", 552.14),
        "nvidia": ("NVDA", 131.38),
        "nvda": ("NVDA", 131.38),
        "netflix": ("NFLX", 1285.00),
        "nflx": ("NFLX", 1285.00),
    }
    info = mock.get(company_lower)
    if info:
        ticker, price = info
        return f"{ticker} is currently trading at ${price:.2f} USD."
    return f"No data for '{company}'. Try: Apple, Tesla, Google, Microsoft, Amazon, Meta, Nvidia, Netflix."


@tool
def purchase_stocks(company: str, quantity: int) -> str:
    """
    Purchase stocks for a given company.
    REQUIRES human approval before executing — this is a financial action.
    company: company name (e.g. 'Tesla', 'Apple')
    quantity: number of shares to buy
    """
    # HITL: interrupt pauses the tool, sends the message to the caller,
    # and waits for Command(resume=...) before continuing here.
    decision = interrupt(
        f"Approve buying {quantity} share(s) of {company}? "
        f"Type 'yes' to confirm or 'no' to cancel."
    )

    # Resume point — decision holds what the human passed in Command(resume=...)
    if isinstance(decision, str) and decision.strip().lower() == "yes":
        order_id = uuid.uuid4().hex[:8].upper()
        return (
            f"Purchase confirmed. Order #{order_id}: "
            f"{quantity} share(s) of {company} placed successfully."
        )

    return f"Purchase cancelled. Order for {quantity} share(s) of {company} was rejected."


tools = [get_stock_price, purchase_stocks]


# ─── Mock LLM ─────────────────────────────────────────────────────────────────

class MockChatModelWithTools:

    def __init__(self, bound_tools: list = None):
        self._tools = {t.name: t for t in (bound_tools or [])}

    def bind_tools(self, tool_list) -> "MockChatModelWithTools":
        return MockChatModelWithTools(bound_tools=list(tool_list))

    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        if not messages:
            return AIMessage(content="Hello! How can I help?")

        last = messages[-1]
        if isinstance(last, ToolMessage):
            return AIMessage(content=str(last.content))

        text = (last.content or "").lower()

        # Buy / purchase → purchase_stocks (HITL)
        if any(w in text for w in ["buy", "purchase", "order", "acquire"]):
            nums = re.findall(r"\d+", text)
            qty = int(nums[0]) if nums else 1
            companies = {
                "apple": "Apple", "tesla": "Tesla", "google": "Google",
                "microsoft": "Microsoft", "amazon": "Amazon", "meta": "Meta",
                "nvidia": "Nvidia", "netflix": "Netflix",
            }
            company = next((v for k, v in companies.items() if k in text), "Apple")
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "purchase_stocks",
                    "args": {"company": company, "quantity": qty},
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "tool_call",
                }],
            )

        # Stock price check → get_stock_price (no HITL)
        if any(w in text for w in ["price", "stock", "share", "trading"]):
            companies = {
                "apple": "Apple", "tesla": "Tesla", "google": "Google",
                "microsoft": "Microsoft", "amazon": "Amazon", "meta": "Meta",
                "nvidia": "Nvidia", "netflix": "Netflix",
            }
            company = next((v for k, v in companies.items() if k in text), "Apple")
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "get_stock_price",
                    "args": {"company": company},
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "tool_call",
                }],
            )

        # Greeting / fallback
        if any(w in text for w in ["hi", "hello", "hey"]):
            return AIMessage(
                content=(
                    "Hello! I'm your stock assistant with Human-in-the-Loop controls.\n\n"
                    "Try:\n"
                    "- 'What is Tesla's stock price?' (automatic)\n"
                    "- 'Buy 5 shares of Apple' (requires your approval)"
                )
            )
        return AIMessage(
            content=(
                "I can check stock prices or help you buy shares.\n"
                "Try: 'What is Nvidia's price?' or 'Buy 10 shares of Tesla'."
            )
        )


llm = MockChatModelWithTools().bind_tools(tools)


# ─── Graph ────────────────────────────────────────────────────────────────────

def chat_node(state: ChatState) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


checkpointer = MemorySaver()

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


# ─── Interactive CLI ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== HITL Stock Chatbot ===")
    print("Commands: check price / buy shares / exit")
    print()

    session_id = str(uuid.uuid4())

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "bye"):
            print("Goodbye!")
            break

        config = {"configurable": {"thread_id": session_id}}

        result = chatbot.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=config,
        )

        # Check if the graph paused at a HITL interrupt
        interrupt_data = result.get("__interrupt__")
        if interrupt_data:
            interrupt_msg = interrupt_data[0].value
            print()
            print(f"[APPROVAL REQUIRED] {interrupt_msg}")
            decision = input("Your decision (yes/no): ").strip()

            # Resume from the interrupt with the human's decision
            result = chatbot.invoke(
                Command(resume=decision),
                config=config,
            )

        # Print the final AI message
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                print(f"\nBot: {msg.content}\n")
                break
