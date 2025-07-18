import streamlit as st
from chatbot.utils import get_openai_api_key
from chatbot.rag_engine import load_json_to_documents, create_vectorstore
from chatbot.chat_interface import ask_chatgpt

@st.cache_resource
def setup_chatbot():
    documents = load_json_to_documents("data/alrajhi_accounts_cleaned.json")
    return create_vectorstore(documents)

st.set_page_config(page_title="Al Rajhi Chatbot", layout="wide")

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/4/49/Al_Rajhi_Bank_Logo.svg", width=200)
    st.markdown("### 💬 Al Rajhi Chatbot")
    st.markdown("Ask me about account types, savings plans, or eligibility.")
    st.markdown("---")
    st.caption("Built with ❤️ using ChatGPT API and Streamlit.")

st.title("🤖 Al Rajhi Bank Smart Assistant")

vectorstore = setup_chatbot()

query = st.text_input("💬 Type your question:", placeholder="e.g. What is the Million Account?")

if query:
    with st.spinner("Thinking..."):
        try:
            response = ask_chatgpt(query, vectorstore)
            st.markdown("### 🧠 Answer:")
            st.success(response)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
