from fastapi import FastAPI 
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langserve import add_routes
import uvicorn
import os 
from langchain_community.llms import Ollama
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
gemini_key = os.getenv("GEMINI_KEY")

if not gemini_key:
    raise ValueError("⚠️ ERROR: GEMINI_KEY is missing! Check your .env file.")

# Initialize FastAPI
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API server"
)

# Define Models
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=gemini_key)
llm = Ollama(model="gemma")

# Define Prompts
prompt1 = ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
prompt2 = ChatPromptTemplate.from_template("Write me a poem about {topic} with 100 words")

# Create Chains
essay_chain = prompt1 | model
poem_chain = prompt2 | llm

# Add Routes
add_routes(app, model, path="/chat")
add_routes(app, essay_chain, path="/essay")
add_routes(app, poem_chain, path="/poem")

# Run FastAPI
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
