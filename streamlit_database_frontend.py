import uuid
import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph_database_backend import chatbot, retrieve_all_threads


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
    # Load existing threads from DB on first load — persist across restarts
    st.session_state["chat_threads"] = retrieve_all_threads()

# Always register the current thread (new or resumed)
add_thread(st.session_state["thread_id"])


# ─── Sidebar UI ───────────────────────────────────────────────────────────────

st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("My Conversations")

# Newest thread first
for tid in reversed(st.session_state["chat_threads"]):
    if st.sidebar.button(str(tid), key=tid):
        st.session_state["thread_id"] = tid
        raw_messages = load_conversation(tid)
        temp_messages = []
        for msg in raw_messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_messages.append({"role": role, "content": msg.content})
        st.session_state["message_history"] = temp_messages


# ─── Main chat UI ─────────────────────────────────────────────────────────────

st.title("LangGraph Chatbot")

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type here...")

if user_input:
    config = {"configurable": {"thread_id": st.session_state["thread_id"]}}

    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state["message_history"].append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            message_chunk.content
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages",
            )
            if message_chunk.content
        )
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )
