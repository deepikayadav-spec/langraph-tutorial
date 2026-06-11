"""
Video 17: Tools in LangGraph.

Graph structure:
  START → chat_node → [tools_condition] → tools → chat_node  (loop)
                                        ↘ END

Three tools:
  1. web_search     — DuckDuckGoSearchRun (real internet search)
  2. calculator     — pure Python arithmetic
  3. get_stock_price — Alpha Vantage API (mock data when key missing)

ToolNode    : prebuilt node that executes whichever tool the LLM chose.
tools_condition : prebuilt edge router — "tools" if tool_calls present, END otherwise.
llm.bind_tools  : attaches tool schemas so the LLM knows what's available.

The loop (tools → chat_node) is critical: without it the raw JSON tool response
would land in the user's chat instead of being summarised by the LLM.

Without real OpenAI key: MockChatModelWithTools detects intent via keywords
and generates proper AIMessage(tool_calls=[...]) so ToolNode runs the real tools.
"""

import os
import re
import sqlite3
import uuid

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, List, TypedDict


# ─── Tools ────────────────────────────────────────────────────────────────────

@tool
def web_search(query: str) -> str:
    """Search the internet using DuckDuckGo for current news and real-time information."""
    from langchain_community.tools import DuckDuckGoSearchRun
    return DuckDuckGoSearchRun().run(query)


@tool
def calculator(first_number: float, second_number: float, operation: str) -> str:
    """
    Perform basic arithmetic on two numbers.
    operation must be one of: 'add', 'subtract', 'multiply', 'divide'
    """
    op = operation.lower().strip()
    if op == "add":
        return str(first_number + second_number)
    if op == "subtract":
        return str(first_number - second_number)
    if op == "multiply":
        return str(first_number * second_number)
    if op == "divide":
        if second_number == 0:
            return "Error: division by zero"
        return str(first_number / second_number)
    return f"Unknown operation '{operation}'. Use: add, subtract, multiply, divide"


@tool
def get_stock_price(ticker: str) -> str:
    """
    Get the current stock price for a ticker symbol (e.g. AAPL, TSLA, GOOGL).
    Requires ALPHAVANTAGE_API_KEY env var; returns mock data when key is absent.
    """
    ticker = ticker.upper().strip()
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")

    if api_key:
        import requests
        url = (
            "https://www.alphavantage.co/query"
            f"?function=GLOBAL_QUOTE&symbol={ticker}&apikey={api_key}"
        )
        try:
            data = requests.get(url, timeout=10).json()
            quote = data.get("Global Quote", {})
            price = quote.get("05. price")
            if price:
                return f"{ticker} is trading at ${float(price):.2f} USD."
            return f"No data returned for {ticker}. Check the ticker symbol."
        except Exception as e:
            return f"API error: {e}"

    # Fallback mock prices when no API key is set
    mock = {
        "AAPL": 213.49, "TSLA": 248.23, "GOOGL": 176.50,
        "MSFT": 425.27, "AMZN": 198.12, "META": 552.14,
        "NVDA": 131.38, "NFLX": 1285.00,
    }
    price = mock.get(ticker)
    if price:
        return (
            f"[Mock] {ticker} is trading at ${price:.2f} USD. "
            "Set ALPHAVANTAGE_API_KEY in .env for live prices."
        )
    return (
        f"No mock data for '{ticker}'. "
        "Supported: AAPL, TSLA, GOOGL, MSFT, AMZN, META, NVDA, NFLX. "
        "Set ALPHAVANTAGE_API_KEY for live prices on any ticker."
    )


tools = [web_search, calculator, get_stock_price]


# ─── State ────────────────────────────────────────────────────────────────────

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# ─── Mock LLM with tool-calling support ──────────────────────────────────────
# In production: replace with ChatOpenAI(model="gpt-4o").bind_tools(tools)
# The real LLM reads tool schemas and decides which tool to call automatically.
# This mock uses keyword detection to simulate that decision.

class MockChatModelWithTools:
    """
    Simulates a tool-calling LLM for demo purposes.

    bind_tools(tools) → returns self with tool schemas stored.
    invoke(messages)  → returns AIMessage with tool_calls when keyword matches,
                         or a natural language summary after a tool result.
    """

    def __init__(self, bound_tools: list = None):
        self._tools = {t.name: t for t in (bound_tools or [])}

    def bind_tools(self, tool_list) -> "MockChatModelWithTools":
        return MockChatModelWithTools(bound_tools=list(tool_list))

    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        if not messages:
            return AIMessage(content="Hello! How can I help you?")

        last = messages[-1]

        # Tool result received → summarise in natural language
        if isinstance(last, ToolMessage):
            return self._handle_tool_result(last)

        # Otherwise → decide whether to call a tool or reply directly
        text = (last.content or "").lower()
        tool_call = self._route_to_tool(text)
        if tool_call:
            return AIMessage(content="", tool_calls=[tool_call])

        return self._chat_response(text)

    # ── private helpers ───────────────────────────────────────────────────────

    def _route_to_tool(self, text: str) -> dict | None:
        """Return a tool_call dict or None."""

        # Calculator — detect two numbers + arithmetic intent
        if "calculator" in self._tools:
            nums = re.findall(r"\d+(?:\.\d+)?", text)
            if len(nums) >= 2 and any(
                w in text for w in ["product", "multiply", "multiplied", "times",
                                     "calculate", "sum", "plus", "minus",
                                     "subtract", "divide", "divided", "add", "×", "x "]
            ):
                op = "multiply"
                if any(w in text for w in ["add", "sum", "plus", "+"]): op = "add"
                elif any(w in text for w in ["subtract", "minus", "difference"]): op = "subtract"
                elif any(w in text for w in ["divide", "quotient"]): op = "divide"
                return {
                    "name": "calculator",
                    "args": {"first_number": float(nums[0]), "second_number": float(nums[1]),
                             "operation": op},
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "tool_call",
                }

        # Stock price — detect company names or ticker symbols
        if "get_stock_price" in self._tools and any(
            w in text for w in ["stock", "share price", "shares", "trading at", "price of"]
        ):
            companies = {
                "apple": "AAPL", "aapl": "AAPL",
                "tesla": "TSLA", "tsla": "TSLA",
                "google": "GOOGL", "googl": "GOOGL", "alphabet": "GOOGL",
                "microsoft": "MSFT", "msft": "MSFT",
                "amazon": "AMZN", "amzn": "AMZN",
                "meta": "META", "facebook": "META",
                "nvidia": "NVDA", "nvda": "NVDA",
                "netflix": "NFLX", "nflx": "NFLX",
            }
            ticker = next((v for k, v in companies.items() if k in text), "AAPL")
            return {
                "name": "get_stock_price",
                "args": {"ticker": ticker},
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "tool_call",
            }

        # Web search — detect news / current events intent
        if "web_search" in self._tools and any(
            w in text for w in ["news", "latest", "who won", "what happened",
                                 "search", "today", "current", "find out", "recent",
                                 "last week", "this week", "2025", "2026"]
        ):
            return {
                "name": "web_search",
                "args": {"query": text[:160]},
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "tool_call",
            }

        return None

    def _handle_tool_result(self, msg: ToolMessage) -> AIMessage:
        name = getattr(msg, "name", "") or ""
        content = msg.content

        if "calculator" in name:
            return AIMessage(content=f"The result is **{content}**.")
        if "stock" in name:
            return AIMessage(content=f"Here is the latest stock information: {content}")
        # web_search
        snippet = content[:400] + ("..." if len(content) > 400 else "")
        return AIMessage(content=f"Here is what I found:\n\n{snippet}")

    def _chat_response(self, text: str) -> AIMessage:
        if any(w in text for w in ["hi", "hello", "hey", "namaste"]):
            return AIMessage(
                content="Hello! I'm a LangGraph chatbot with tools. I can:\n"
                        "- **Calculate** — 'What is 245 × 378?'\n"
                        "- **Stock prices** — 'What is Tesla's stock price?'\n"
                        "- **Web search** — 'What is the latest news in AI?'"
            )
        if "what can you do" in text or "help" in text:
            return AIMessage(
                content=(
                    "I can help with:\n"
                    "- **Math**: 'What is 25 multiplied by 48?'\n"
                    "- **Stock prices**: 'What is Apple's stock price?'\n"
                    "- **Web search**: 'Latest news on India's space mission'"
                )
            )
        return AIMessage(
            content=(
                "I'm here to help! Try asking me to calculate something, "
                "look up a stock price, or search the web for current news."
            )
        )


llm = MockChatModelWithTools()
llm_with_tools = llm.bind_tools(tools)


# ─── Graph nodes ──────────────────────────────────────────────────────────────

def chat_node(state: ChatState) -> dict:
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


tool_node = ToolNode(tools)


# ─── Graph assembly ───────────────────────────────────────────────────────────

conn = sqlite3.connect("chatbot_tools.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)          # must be named "tools" — tools_condition returns this string

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)   # → "tools" or END
graph.add_edge("tools", "chat_node")                        # loop: LLM sees tool result and replies

chatbot = graph.compile(checkpointer=checkpointer)


# ─── Utility ──────────────────────────────────────────────────────────────────

def retrieve_all_threads() -> list:
    """Return list of unique thread IDs stored in the SQLite DB."""
    seen: set = set()
    for ct in checkpointer.list(None):
        seen.add(ct.config["configurable"]["thread_id"])
    return list(seen)


# ─── Main (quick smoke test) ──────────────────────────────────────────────────

if __name__ == "__main__":
    config = {
        "configurable": {"thread_id": "test-tools-001"},
        "metadata": {"thread_id": "test-tools-001"},
        "run_name": "chat-turn",
    }

    def chat(user_text: str):
        print(f"\nUser: {user_text}")
        result = chatbot.invoke(
            {"messages": [HumanMessage(content=user_text)]},
            config=config,
        )
        reply = result["messages"][-1].content
        print(f"Bot : {reply}")

    chat("Hi!")
    chat("What is 245 multiplied by 378?")
    chat("What is Tesla's stock price?")
    chat("What is the latest news in AI?")
