from fastapi import FastAPI
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import  google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="fastapi",
    description='api that integrates with fastapi and langchain',
    version='1.0'
)
GOOGLE_API_KEY = os.getenv('GEMINI_KEY')
print(GOOGLE_API_KEY)
LANGCHAIN_KEY = os.getenv('LANGCHAIN_KEY')
print(LANGCHAIN_KEY)
os.environ["LANGCHAIN_TRACING_V2"]= "true"

llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash', google_api_key=GOOGLE_API_KEY)

prompt = PromptTemplate(
    input_variables=['question'],
    template='you are helpful ai. answer the following question: {question}'
)
chain = prompt | llm

class ChatRequest(BaseModel):
    question:str


@app.get('/')
def home():
    return {"message": "welcome to the home page"}

@app.post('/chat')
def chat_gemini(request:ChatRequest):
    response=chain.invoke({"question":request.question})
    return {"response":response}