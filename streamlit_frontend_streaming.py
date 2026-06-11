import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph_backend import chatbot, CONFIG

st.title("LangGraph Chatbot (Streaming)")

# session_state persists conversation history across reruns
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# Replay full conversation history on each rerun
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type here...")

if user_input:
    # Display and store user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state["message_history"].append({"role": "user", "content": user_input})

    # Stream AI response with typewriter effect
    # chatbot.stream(stream_mode="messages") yields (message_chunk, metadata) tuples;
    # st.write_stream() consumes the generator token-by-token and returns the full string.
    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            message_chunk.content
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            )
            if message_chunk.content
        )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )
