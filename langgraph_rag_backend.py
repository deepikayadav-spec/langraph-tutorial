"""
Video 19: RAG (Retrieval-Augmented Generation) in LangGraph.

Why RAG?
  LLMs have a training cutoff — they don't know your private documents.
  RAG fixes this by retrieving relevant chunks from your documents at query time
  and giving them to the LLM as context, reducing hallucination and enabling
  answers from up-to-date or private content.

RAG pipeline (ingest_pdf):
  PDF → PyPDFLoader → RecursiveCharacterTextSplitter → HuggingFaceEmbeddings
      → FAISS vector store → Retriever

The retriever is wrapped as a @tool so the LLM decides WHEN to use it — just like
web_search or calculator. The graph structure is identical to the tool chatbot.

Embeddings note:
  The instructor uses OpenAIEmbeddings (text-embedding-3-small). This version uses
  HuggingFaceEmbeddings (all-MiniLM-L6-v2) — free, no API key required.
  The model (~90 MB) is downloaded on first use via sentence-transformers.

In production: replace MockChatModelWithTools with ChatOpenAI(model="gpt-4o-mini")
and use OpenAIEmbeddings for higher-quality embeddings.
"""

import os
import re
import sqlite3
import uuid
from typing import Annotated, List, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# ─── Global retriever (set by ingest_pdf) ────────────────────────────────────
# A module-level variable lets the tool read whichever retriever was last built.

_retriever = None


# ─── PDF ingestion pipeline ───────────────────────────────────────────────────

def ingest_pdf(pdf_path: str):
    """
    Load a PDF, split it into chunks, embed with HuggingFace, and store in FAISS.
    Sets the global _retriever used by the rag_search tool.
    Returns the retriever for callers who want a reference.
    """
    global _retriever

    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    # Load: PyPDFLoader reads each page as a Document
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split: overlapping windows so context isn't cut mid-sentence
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # Embed: all-MiniLM-L6-v2 is a compact, free sentence-transformer model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Store: FAISS is an in-memory vector store — fast for prototyping
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Retriever: top-4 most similar chunks by cosine similarity
    _retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )
    return _retriever


# ─── Tools ────────────────────────────────────────────────────────────────────

@tool
def rag_search(query: str) -> str:
    """
    Search the uploaded PDF document for information relevant to the query.
    Use this whenever the user asks about the content of an uploaded document.
    Returns the most relevant text chunks plus page numbers.
    """
    if _retriever is None:
        return "No document has been uploaded yet. Please upload a PDF file first."

    results = _retriever.invoke(query)
    if not results:
        return "No relevant content found in the document for that query."

    parts = []
    for i, doc in enumerate(results, 1):
        page = doc.metadata.get("page", "?")
        source = os.path.basename(doc.metadata.get("source", "document"))
        parts.append(f"[Chunk {i} | {source} · Page {page}]\n{doc.page_content.strip()}")

    return "\n\n---\n\n".join(parts)


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


tools = [rag_search, web_search, get_stock_price]


# ─── State ────────────────────────────────────────────────────────────────────

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# ─── Mock LLM with tool-calling support ──────────────────────────────────────
# Replace with ChatOpenAI(model="gpt-4o-mini") when OPENAI_API_KEY is available.

class MockChatModelWithTools:

    def __init__(self, bound_tools: list = None):
        self._tools = {t.name: t for t in (bound_tools or [])}

    def bind_tools(self, tool_list) -> "MockChatModelWithTools":
        return MockChatModelWithTools(bound_tools=list(tool_list))

    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        if not messages:
            return AIMessage(content="Hello! How can I help you?")

        last = messages[-1]
        if isinstance(last, ToolMessage):
            return self._handle_tool_result(last)

        text = (last.content or "").lower()
        tool_call = self._route_to_tool(text)
        if tool_call:
            return AIMessage(content="", tool_calls=[tool_call])
        return self._chat_response(text)

    def _route_to_tool(self, text: str) -> dict | None:
        # RAG: route to rag_search when a PDF is loaded and the question is substantive
        if (
            "rag_search" in self._tools
            and _retriever is not None
            and not any(w in text for w in ["hi", "hello", "hey", "help", "what can you"])
            and not self._is_stock_query(text)
            and not self._is_search_query(text)
        ):
            return {
                "name": "rag_search",
                "args": {"query": text[:300]},
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "tool_call",
            }

        # Stock price
        if self._is_stock_query(text) and "get_stock_price" in self._tools:
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

        # Web search
        if self._is_search_query(text) and "web_search" in self._tools:
            return {
                "name": "web_search",
                "args": {"query": text[:160]},
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "tool_call",
            }

        return None

    def _is_stock_query(self, text: str) -> bool:
        return any(w in text for w in ["stock", "share price", "shares of"])

    def _is_search_query(self, text: str) -> bool:
        return any(w in text for w in ["news", "latest", "search", "find out", "today", "recent", "2026"])

    def _handle_tool_result(self, msg: ToolMessage) -> AIMessage:
        name = getattr(msg, "name", "") or ""
        content = msg.content

        # Normalise content — can be a list of content blocks (from MCP/LangChain)
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            ).strip()

        if "rag" in name:
            if "No document" in content:
                return AIMessage(content=content)
            return AIMessage(
                content=f"Based on the uploaded document:\n\n{content}"
            )
        if "stock" in name:
            return AIMessage(content=f"Stock information: {content}")
        snippet = content[:400] + ("..." if len(content) > 400 else "")
        return AIMessage(content=f"Here is what I found:\n\n{snippet}")

    def _chat_response(self, text: str) -> AIMessage:
        if any(w in text for w in ["hi", "hello", "hey", "namaste"]):
            return AIMessage(
                content=(
                    "Hello! I'm a RAG-enabled LangGraph chatbot.\n\n"
                    "**Upload a PDF** in the sidebar to ask questions about its content.\n\n"
                    "I can also:\n"
                    "- Look up stock prices: 'Tesla stock price?'\n"
                    "- Search the web: 'Latest AI news'"
                )
            )
        if "what can you do" in text or "help" in text:
            return AIMessage(
                content=(
                    "I can:\n"
                    "- **Answer questions from your PDF** — upload one in the sidebar first\n"
                    "- **Stock prices**: 'What is Apple's stock price?'\n"
                    "- **Web search**: 'What is the latest news in AI?'"
                )
            )
        if _retriever is None:
            return AIMessage(
                content="Please upload a PDF in the sidebar to get started. I'll then be able to answer questions from its content."
            )
        return AIMessage(
            content="I'll search the uploaded document for that. Try asking a specific question about the PDF's content."
        )


llm = MockChatModelWithTools()
llm_with_tools = llm.bind_tools(tools)


# ─── Graph nodes ──────────────────────────────────────────────────────────────

def chat_node(state: ChatState) -> dict:
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


tool_node = ToolNode(tools)


# ─── Graph assembly ───────────────────────────────────────────────────────────

conn = sqlite3.connect("chatbot_rag.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


# ─── Utility ──────────────────────────────────────────────────────────────────

def retrieve_all_threads() -> list:
    seen: set = set()
    for ct in checkpointer.list(None):
        seen.add(ct.config["configurable"]["thread_id"])
    return list(seen)


# ─── Main (smoke test) ────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = {
        "configurable": {"thread_id": "test-rag-001"},
        "metadata": {"thread_id": "test-rag-001"},
        "run_name": "chat-turn",
    }

    def chat(user_text: str):
        print(f"\nUser: {user_text}")
        result = chatbot.invoke(
            {"messages": [HumanMessage(content=user_text)]},
            config=config,
        )
        print(f"Bot : {result['messages'][-1].content}")

    chat("Hi!")
    chat("What is Tesla's stock price?")
    chat("What is the latest news in AI?")
    print("\nSmoke test passed. Upload a PDF in Streamlit to test RAG search.")
