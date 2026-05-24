import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(
    page_title="Groq Q&A Chatbot",
    page_icon="🤖",
    layout="centered"
)

# Get Groq API key from Streamlit secrets first, then .env
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY is missing. Add it in .env locally or Streamlit Secrets after deployment.")
    st.stop()

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Optional LangSmith tracking
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot With Groq"

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer the user's question clearly and simply."
        ),
        (
            "user",
            "Question: {question}"
        )
    ]
)

def generate_response(question, model_name, temperature, max_tokens):
    llm = ChatGroq(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    answer = chain.invoke({"question": question})
    return answer

# UI
st.title("Enhanced Q&A Chatbot With Groq")

model_name = st.sidebar.selectbox(
    "Select Groq Model",
    [
        "llama-3.1-8b-instant",
        "llama-3.3-70b-versatile",
        "mixtral-8x7b-32768",
        "gemma2-9b-it"
    ]
)

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.7
)

max_tokens = st.sidebar.slider(
    "Max Tokens",
    min_value=50,
    max_value=1000,
    value=300
)

st.write("Go ahead and ask any question.")

user_input = st.text_input("You:")

if user_input:
    with st.spinner("Generating response..."):
        try:
            response = generate_response(
                user_input,
                model_name,
                temperature,
                max_tokens
            )
            st.write(response)
        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.info("Please provide the user input.")