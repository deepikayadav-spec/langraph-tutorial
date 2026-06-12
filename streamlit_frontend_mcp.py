"""
Video 18: Streamlit frontend for the MCP-powered chatbot.

Key differences from streamlit_tool_frontend.py:
  1. Imports from langgraph_mcp_backend (async graph + AsyncSqliteSaver)
  2. Every graph call must go through asyncio.run() because the graph is async
  3. chatbot.astream() instead of chatbot.stream()
  4. chatbot.aget_state() instead of chatbot.get_state()
  5. nest_asyncio is applied in the backend so asyncio.run() works inside Streamlit

Production note: this asyncio.run()-inside-Streamlit approach is "hacky" as the
instructor notes. Production systems should use FastAPI + React/Next.js instead.
"""

import asyncio
import uuid
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langgraph_mcp_backend import chatbot, retrieve_all_threads


# ─── Utility functions ────────────────────────────────────────────────────────

def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []
    add_thread(thread_id)


def load_conversation(thread_id):
    async def _get():
        config = {"configurable": {"thread_id": thread_id}}
        state = await chatbot.aget_state(config)
        return state.values.get("messages", [])
    return asyncio.run(_get())


# ─── Session setup ────────────────────────────────────────────────────────────

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

add_thread(st.session_state["thread_id"])


# ─── Sidebar UI ───────────────────────────────────────────────────────────────

st.sidebar.title("LangGraph MCP Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")

for tid in reversed(st.session_state["chat_threads"]):
    if st.sidebar.button(str(tid), key=tid):
        st.session_state["thread_id"] = tid
        raw_messages = load_conversation(tid)
        temp_messages = []
        for msg in raw_messages:
            if isinstance(msg, (HumanMessage, AIMessage)) and msg.content:
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                temp_messages.append({"role": role, "content": msg.content})
        st.session_state["message_history"] = temp_messages


# ─── Main chat UI ─────────────────────────────────────────────────────────────

st.title("LangGraph Chatbot (MCP Tools)")
st.caption("Math via MCP · Expense tracking via MCP · Stock prices · Web search")

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask me anything...")

if user_input:
    config = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat-turn",
    }

    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state["message_history"].append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        # Collect async stream — astream with mode="messages" yields (chunk, metadata) tuples
        async def _collect_stream():
            chunks = []
            async for chunk, metadata in chatbot.astream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages",
            ):
                chunks.append((chunk, metadata))
            return chunks

        stream_output = asyncio.run(_collect_stream())

        # Show st.status badge for each tool that was invoked
        for chunk, _ in stream_output:
            if isinstance(chunk, AIMessage) and getattr(chunk, "tool_calls", None):
                for tc in chunk.tool_calls:
                    tool_name = tc.get("name", "tool")
                    with st.status(f"Using {tool_name}...", state="complete"):
                        st.write(f"`{tool_name}` finished.")

        # Stream only AIMessage text — ToolMessage JSON stays hidden
        def ai_content_stream():
            for chunk, _ in stream_output:
                if isinstance(chunk, AIMessage) and chunk.content:
                    yield chunk.content

        ai_message = st.write_stream(ai_content_stream())

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )
