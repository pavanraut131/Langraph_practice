from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st 
import os 
from dotenv import load_dotenv

load_dotenv()

os.environ['GEMINI_KEY'] = os.getenv("GEMINI_KEY")
os.environ["LANGCHAIN_TRACING_V2"]= "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_KEY')

prompt = ChatPromptTemplate.from_messages(
    [
        ('system', 'you are a helpful assistant. please resposne to the queries  '),
        ('user', "Question:{question}")
    ]
)
st.title('demo with ollama')
input_text = st.text_input("search the input you want ")

# llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', google_api_key = os.getenv('GEMINI_KEY'))
llm = Ollama(model='gemma')
output_parser = StrOutputParser()
chain =prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))