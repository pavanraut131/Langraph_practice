from langchain.agents import initialize_agent, Tool
from langchain_google_genai import  ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
load_dotenv()

os.environ['GEMINI_KEY'] = os.getenv('GEMINI_KEY')

def getwheather(loaction:str):
    return f"Current weather in {loaction} is sunny"


tool = Tool(
    name="get_weather",
    func=getwheather,
    description="Get the current weather "
)

llm =  ChatGoogleGenerativeAI(model='gemini-2.0-flash', google_api_key=os.getenv("GEMINI_KEY"))

agent = initialize_agent(tools=[tool], llm=llm, agent="openai-functions", verbose=True)

response=agent.invoke({"input":"what is the weather in new york"})
print(response)  