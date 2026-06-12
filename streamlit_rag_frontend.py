"""
Video 19: Streamlit frontend for the RAG chatbot.

Key additions vs streamlit_tool_frontend.py:
  1. PDF file uploader in the sidebar — triggers ingest_pdf()
  2. Imports ingest_pdf from langgraph_rag_backend to build the retriever
  3. st.success / st.spinner feedback while embeddings are being built
  4. Same thread management and tool-status display as the tool frontend

Upload flow:
  1. User drags a PDF into the sidebar uploader
  2. File is saved to a temp path
  3. ingest_pdf() loads → splits → embeds → stores in FAISS
  4. The global _retriever in the backend is now set
  5. User asks a question → rag_search tool fires → answer from document
"""

import os
import tempfile
import uuid
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langgraph_rag_backend import chatbot, ingest_pdf, retrieve_all_threads


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

if "pdf_loaded" not in st.session_state:
    st.session_state["pdf_loaded"] = False

if "pdf_name" not in st.session_state:
    st.session_state["pdf_name"] = None

add_thread(st.session_state["thread_id"])


# ─── Sidebar UI ───────────────────────────────────────────────────────────────

st.sidebar.title("LangGraph RAG Chatbot")

# ── PDF uploader ──────────────────────────────────────────────────────────────
st.sidebar.header("Upload Document")

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF to chat with it",
    type=["pdf"],
    help="The PDF will be split into chunks and embedded for semantic search.",
)

if uploaded_file is not None and uploaded_file.name != st.session_state["pdf_name"]:
    with st.sidebar:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            # Save uploaded file to a temp path so PyPDFLoader can read it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                ingest_pdf(tmp_path)
                st.session_state["pdf_loaded"] = True
                st.session_state["pdf_name"] = uploaded_file.name
                st.success(f"Ready: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Failed to process PDF: {e}")
            finally:
                os.unlink(tmp_path)

elif st.session_state["pdf_loaded"]:
    st.sidebar.success(f"Loaded: {st.session_state['pdf_name']}")
else:
    st.sidebar.info("No document loaded yet.")

# ── Thread management ─────────────────────────────────────────────────────────
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

st.title("LangGraph RAG Chatbot")
if st.session_state["pdf_loaded"]:
    st.caption(f"Chatting with: **{st.session_state['pdf_name']}**")
else:
    st.caption("Upload a PDF in the sidebar to get started.")

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask a question about your document...")

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
        # Collect stream — same pattern as tool frontend
        stream_output = list(
            chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages",
            )
        )

        # Show which tool ran
        for chunk, _ in stream_output:
            if isinstance(chunk, AIMessage) and getattr(chunk, "tool_calls", None):
                for tc in chunk.tool_calls:
                    tool_name = tc.get("name", "tool")
                    label = "Searching document..." if tool_name == "rag_search" else f"Using {tool_name}..."
                    with st.status(label, state="complete"):
                        st.write(f"`{tool_name}` finished.")

        def ai_content_stream():
            for chunk, _ in stream_output:
                if isinstance(chunk, AIMessage) and chunk.content:
                    yield chunk.content

        ai_message = st.write_stream(ai_content_stream())

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )
