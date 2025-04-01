import os 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
if not os.getenv("GEMINI_KEY"):
    raise ValueError("GEMINI_KEY is not set. Check your .env file.")

os.environ['GEMINI_KEY'] = os.getenv("GEMINI_KEY")
os.environ["LANGCHAIN_TRACING_V2"]= "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_KEY')

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please repond to the quiries "),
        ("user", "Question: {question}")
    ]
)


st.title('Demo with open ai')
input_text=st.text_input("search the topic u want ")


llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', google_api_key = os.getenv('GEMINI_KEY'))
output_parser = StrOutputParser()
chain =prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))