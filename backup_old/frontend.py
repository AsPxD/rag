import streamlit as st
import requests
from dotenv import load_dotenv
import os

load_dotenv()

# Backend URL (update to your deployed URL after deployment)
BACKEND_URL = "http://localhost:8000/ask"  # Change to deployed URL, e.g., https://your-backend.onrender.com/ask

st.title("Medical RAG Chatbot")
st.warning("Disclaimer: This is not professional medical advice. Consult a doctor.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask a medical question:")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(BACKEND_URL, json={"query": user_input})
                response.raise_for_status()
                data = response.json()
                st.markdown(data["answer"])
                st.markdown(f"**Sources:** {data['sources']}")
                answer = data["answer"]
            except Exception as e:
                st.error(f"Error: {e}")
                answer = "Sorry, an error occurred."

    st.session_state.messages.append({"role": "assistant", "content": answer})
