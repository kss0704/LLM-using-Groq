import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Retrieve Groq API key from Streamlit secrets or environment
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY not found. Please set it in .env or Streamlit secrets.")
    st.stop()

# Initialize the Groq model
try:
    model = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)
except Exception as e:
    st.error(f"Failed to initialize Groq model: {str(e)}")
    st.stop()

# Set page config
st.set_page_config(page_title="LLM-Chat Assistant using Groq", page_icon="💬", layout="wide")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# App title and description
st.title("LLM-Chat Assistant using Groq")
st.markdown("""
    This is a conversational AI assistant powered by Groq's Gemma2-9b-It model.
    Ask any question, and the assistant will respond!
""")

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful and knowledgeable assistant. Answer the user's question accurately and concisely. If the user asks for code, provide a clear and accurate code snippet with a brief explanation."),
        ("user", "Question: {question}")
    ]
)

# Chain setup
output_parser = StrOutputParser()
chain = prompt | model | output_parser

# Chat input
with st.form(key="chat_form", clear_on_submit=True):
    input_text = st.text_input("What question do you have in mind?", placeholder="Type here...")
    submit_button = st.form_submit_button(label="Send")

# Process input and generate response
if submit_button and input_text:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": input_text})

    # Generate response
    try:
        response = chain.invoke({"question": input_text})
    except Exception as e:
        response = f"Error generating response: {str(e)}"

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f"**You**: {message['content']}")
    else:
        st.markdown(f"**Assistant**: {message['content']}")

# Sidebar with model info and clear chat option
with st.sidebar:
    st.header("Model Info")
    st.write("Model: Gemma2-9b-It")
    st.write("Provider: Groq")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.experimental_rerun()