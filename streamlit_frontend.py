import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph_backend import chatbot, CONFIG

st.title("LangGraph Chatbot")

# session_state persists across reruns (each Enter press);
# a plain Python list would reset every time
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

# Display full conversation history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input box fixed at the bottom
user_input = st.chat_input("Type here...")

if user_input:
    # Show and store user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )

    # Invoke LangGraph chatbot (MemorySaver tracks full conversation)
    response = chatbot.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=CONFIG,
    )
    ai_message = response["messages"][-1].content

    # Show and store assistant reply
    with st.chat_message("assistant"):
        st.markdown(ai_message)
    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )
