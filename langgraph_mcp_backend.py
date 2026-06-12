"""
Video 18: MCP (Model Context Protocol) client in LangGraph.

Why MCP instead of raw tools?
  Raw tools are brittle — if a 3rd-party API changes, you fix the tool code in every
  chatbot. MCP separates concerns: the server owns the tool logic; the client just
  holds a short config block. Server changes never touch the client.

Architecture:
  - Two MCP servers (local, stdio transport): math + expense tracker
  - Two regular LangChain tools kept alongside: web_search, get_stock_price
  - MultiServerMCPClient fetches tool schemas from both servers at startup
  - All tools (MCP + regular) merged and bound to the LLM

Key async requirement:
  langchain-mcp-adapters is async-only. Therefore:
    • chat_node becomes  async def
    • llm.invoke → await llm.ainvoke
    • SqliteSaver → AsyncSqliteSaver (requires aiosqlite)
    • graph.compile() returns an AsyncCompiledGraph
    • callers use await graph.ainvoke() / async for … astream()

  ToolNode is already async internally — no changes needed there.

Streamlit note:
  Streamlit is synchronous at its core. Using it with async code requires asyncio.run()
  calls and nest_asyncio. This is "hacky" as the instructor says — in production use
  FastAPI + React/Next.js instead.

In production: replace MockChatModelWithTools with ChatOpenAI(model="gpt-4o-mini")
"""

import asyncio
import os
import re
import sys
import uuid

import aiosqlite
import nest_asyncio
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated, List, TypedDict

nest_asyncio.apply()  # allows asyncio.run() inside Streamlit's event loop


# ─── Regular (non-MCP) tools ─────────────────────────────────────────────────

@tool
def web_search(query: str) -> str:
    """Search the internet using DuckDuckGo for current news and information."""
    from langchain_community.tools import DuckDuckGoSearchRun
    return DuckDuckGoSearchRun().run(query)


@tool
def get_stock_price(ticker: str) -> str:
    """Get the current stock price for a ticker symbol (e.g. AAPL, TSLA)."""
    ticker = ticker.upper().strip()
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if api_key:
        import requests
        url = (
            "https://www.alphavantage.co/query"
            f"?function=GLOBAL_QUOTE&symbol={ticker}&apikey={api_key}"
        )
        try:
            quote = requests.get(url, timeout=10).json().get("Global Quote", {})
            price = quote.get("05. price")
            if price:
                return f"{ticker} is trading at ${float(price):.2f} USD."
        except Exception as e:
            return f"API error: {e}"
    mock = {
        "AAPL": 213.49, "TSLA": 248.23, "GOOGL": 176.50,
        "MSFT": 425.27, "AMZN": 198.12, "META": 552.14,
        "NVDA": 131.38, "NFLX": 1285.00,
    }
    price = mock.get(ticker)
    if price:
        return f"[Mock] {ticker} is trading at ${price:.2f} USD."
    return f"No data for '{ticker}'. Supported: AAPL TSLA GOOGL MSFT AMZN META NVDA NFLX"


REGULAR_TOOLS = [web_search, get_stock_price]


# ─── MCP client config ────────────────────────────────────────────────────────
# Local servers: transport="stdio" — client spawns the server as a subprocess.
# Remote servers would use transport="streamable_http" with a URL instead.

_HERE = os.path.dirname(os.path.abspath(__file__))

MCP_CONFIG = {
    "math": {
        "transport": "stdio",
        "command": sys.executable,
        "args": [os.path.join(_HERE, "mcp_math_server.py")],
    },
    "expense_tracker": {
        "transport": "stdio",
        "command": sys.executable,
        "args": [os.path.join(_HERE, "mcp_expense_server.py")],
    },
}


# ─── State ────────────────────────────────────────────────────────────────────

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# ─── Mock LLM with tool-calling support (async) ───────────────────────────────
# Replace with ChatOpenAI(model="gpt-4o-mini") when OPENAI_API_KEY is available.
# ainvoke() is called by LangGraph's async node execution.

class MockChatModelWithTools:

    def __init__(self, tools: list = None):
        self._tools = {t.name: t for t in (tools or [])}

    def bind_tools(self, tools) -> "MockChatModelWithTools":
        return MockChatModelWithTools(tools=list(tools))

    async def ainvoke(self, messages: List[BaseMessage]) -> AIMessage:
        return self.invoke(messages)

    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        if not messages:
            return AIMessage(content="Hello! How can I help you?")

        last = messages[-1]

        if isinstance(last, ToolMessage):
            return self._handle_tool_result(last)

        text = (last.content or "").lower()
        call = self._route_to_tool(text)
        if call:
            return AIMessage(content="", tool_calls=[call])

        return self._chat_response(text)

    def _route_to_tool(self, text: str) -> dict | None:
        nums = re.findall(r"\d+(?:\.\d+)?", text)

        # ── MCP math tools ────────────────────────────────────────────────────
        math_ops = {
            "add":      ["plus", "add", "sum", "addition"],
            "subtract": ["minus", "subtract", "difference"],
            "multiply": ["multiply", "multiplied", "product", "times", "×"],
            "divide":   ["divide", "divided", "quotient"],
            "power":    ["power", "exponent", "raised to", "squared", "cubed"],
            "modulus":  ["modulus", "remainder", "mod"],
        }
        if len(nums) >= 2:
            for op_name, keywords in math_ops.items():
                if op_name in self._tools and any(kw in text for kw in keywords):
                    # power uses (base, exponent); all others use (a, b)
                    args = (
                        {"base": float(nums[0]), "exponent": float(nums[1])}
                        if op_name == "power"
                        else {"a": float(nums[0]), "b": float(nums[1])}
                    )
                    return {
                        "name": op_name,
                        "args": args,
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "type": "tool_call",
                    }

        # ── MCP expense tools ─────────────────────────────────────────────────
        if "add_expense" in self._tools and any(
            w in text for w in ["add expense", "spent", "expense for", "bought", "purchased"]
        ):
            amount = float(nums[0]) if nums else 100.0
            return {
                "name": "add_expense",
                "args": {
                    "amount": amount,
                    "description": text[:80],
                    "date": "2026-06-11",
                    "category": "General",
                },
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "tool_call",
            }

        if "list_expenses" in self._tools and any(
            w in text for w in ["list expense", "show expense", "my expense", "all expense"]
        ):
            return {
                "name": "list_expenses",
                "args": {},
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "tool_call",
            }

        if "summarize_expenses" in self._tools and any(
            w in text for w in ["summarize expense", "total expense", "expense summary", "how much spent"]
        ):
            return {
                "name": "summarize_expenses",
                "args": {},
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "tool_call",
            }

        # ── Regular tools ─────────────────────────────────────────────────────
        if "get_stock_price" in self._tools and any(
            w in text for w in ["stock", "share price", "shares of"]
        ):
            companies = {
                "apple": "AAPL", "tesla": "TSLA", "google": "GOOGL",
                "microsoft": "MSFT", "amazon": "AMZN", "meta": "META",
                "nvidia": "NVDA", "netflix": "NFLX",
            }
            ticker = next((v for k, v in companies.items() if k in text), "AAPL")
            return {
                "name": "get_stock_price",
                "args": {"ticker": ticker},
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "tool_call",
            }

        if "web_search" in self._tools and any(
            w in text for w in ["news", "latest", "search", "find out", "today", "recent", "2026"]
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

        # MCP tools return content as a list of text-block dicts: [{'type':'text','text':'...'}]
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            ).strip()

        if name in ("add", "subtract", "multiply", "divide", "power", "modulus"):
            return AIMessage(content=f"The result is **{content}**.")
        if "expense" in name:
            return AIMessage(content=str(content))
        if "stock" in name:
            return AIMessage(content=f"Stock info: {content}")
        return AIMessage(content=f"Here is what I found:\n\n{str(content)[:400]}")

    def _chat_response(self, text: str) -> AIMessage:
        if any(w in text for w in ["hi", "hello", "hey", "namaste"]):
            return AIMessage(
                content=(
                    "Hello! I'm a LangGraph + MCP chatbot. I can:\n"
                    "- **Math** (via MCP): 'What is 12 × 15?', '7 power 3'\n"
                    "- **Expense tracking** (via MCP): 'Add expense 500 for a book'\n"
                    "- **Stock prices**: 'Tesla stock price?'\n"
                    "- **Web search**: 'Latest AI news'"
                )
            )
        if "what can you do" in text or "help" in text:
            return AIMessage(
                content=(
                    "I can:\n"
                    "- Math operations: add, subtract, multiply, divide, power, modulus\n"
                    "- Expense tracking: add, list, and summarize expenses\n"
                    "- Stock prices for AAPL, TSLA, GOOGL, MSFT, AMZN, META, NVDA, NFLX\n"
                    "- Web search for current news"
                )
            )
        return AIMessage(
            content=(
                "Try: 'What is 25 multiplied by 48?', 'Tesla stock price?', "
                "'Latest news on AI', or 'Add expense 300 for lunch'."
            )
        )


# ─── Async graph builder ──────────────────────────────────────────────────────

async def build_graph(db_path: str = "chatbot_mcp.db"):
    """
    Fetches tools from MCP servers, merges with regular tools,
    builds and compiles the LangGraph chatbot.
    Returns (chatbot, checkpointer_connection) tuple.
    """
    # Connect to MCP servers and fetch their tools
    # langchain-mcp-adapters 0.1.0: no context manager — just call get_tools() directly
    mcp_client = MultiServerMCPClient(MCP_CONFIG)
    mcp_tools = await mcp_client.get_tools()    # starts servers and fetches schemas

    all_tools = REGULAR_TOOLS + mcp_tools

    llm = MockChatModelWithTools()
    llm_with_tools = llm.bind_tools(all_tools)

    # Nodes — chat_node must be async because llm.ainvoke is async
    async def chat_node(state: ChatState) -> dict:
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    tool_node = ToolNode(all_tools)

    # Graph
    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    # AsyncSqliteSaver — the async counterpart of SqliteSaver (requires aiosqlite)
    conn = await aiosqlite.connect(db_path)
    checkpointer = AsyncSqliteSaver(conn)

    chatbot = graph.compile(checkpointer=checkpointer)
    return chatbot, conn


# ─── Module-level initialization ─────────────────────────────────────────────
# Run once when this module is imported (by Streamlit or scripts).
# MCP server subprocesses and aiosqlite connection stay alive for process lifetime.

_chatbot, _db_conn = asyncio.run(build_graph())
chatbot = _chatbot


# ─── Thread retrieval ─────────────────────────────────────────────────────────

async def _retrieve_threads_async() -> list:
    seen: set = set()
    async for ct in _chatbot.checkpointer.alist(None):
        seen.add(ct.config["configurable"]["thread_id"])
    return list(seen)


def retrieve_all_threads() -> list:
    return asyncio.run(_retrieve_threads_async())


# ─── Main (smoke test) ────────────────────────────────────────────────────────

if __name__ == "__main__":
    async def main():
        config = {
            "configurable": {"thread_id": "test-mcp-001"},
            "metadata": {"thread_id": "test-mcp-001"},
            "run_name": "chat-turn",
        }

        async def chat(text: str):
            print(f"\nUser: {text}")
            result = await chatbot.ainvoke(
                {"messages": [HumanMessage(content=text)]},
                config=config,
            )
            print(f"Bot : {result['messages'][-1].content}")

        await chat("Hi!")
        await chat("What is 17 multiplied by 23?")
        await chat("What is 100 power 2?")
        await chat("Tesla stock price?")

    asyncio.run(main())
