"""
Video 17: Streamlit frontend for the tool-enabled chatbot.

Key differences from streamlit_database_frontend.py:
  1. Imports from langgraph_tool_backend (ToolNode + tools_condition graph)
  2. Stream filtered to AIMessage only — ToolMessage JSON never shown to user
  3. st.status() container shows which tool ran while the bot was thinking
"""

import uuid
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langgraph_tool_backend import chatbot, retrieve_all_threads


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
    config = {"configurable": {"thread_id": thread_id}}
    state = chatbot.get_state(config)
    return state.values.get("messages", [])


# ─── Session setup ────────────────────────────────────────────────────────────

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

add_thread(st.session_state["thread_id"])


# ─── Sidebar UI ───────────────────────────────────────────────────────────────

st.sidebar.title("LangGraph Chatbot (Tools)")

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

st.title("LangGraph Chatbot with Tools")
st.caption("Can search the web, calculate, and look up stock prices.")

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
        # Collect the full stream first so we can show tool status before the reply
        stream_output = list(
            chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages",
            )
        )

        # Show st.status badge for every tool that was called
        for chunk, _ in stream_output:
            if isinstance(chunk, AIMessage) and getattr(chunk, "tool_calls", None):
                for tc in chunk.tool_calls:
                    tool_name = tc.get("name", "tool")
                    with st.status(f"Using {tool_name}...", state="complete"):
                        st.write(f"`{tool_name}` finished.")

        # Stream only AIMessage content — ToolMessage JSON is filtered out
        def ai_content_stream():
            for chunk, _ in stream_output:
                if isinstance(chunk, AIMessage) and chunk.content:
                    yield chunk.content

        ai_message = st.write_stream(ai_content_stream())

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )
