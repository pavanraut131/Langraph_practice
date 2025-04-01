import streamlit as st
import os 
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from dotenv import  load_dotenv
from langchain_community.vectorstores import FAISS
import time
from langchain_core.documents import Document

load_dotenv()

grok_api_key = os.getenv('GROQ_KEY')
os.environ["USER_AGENT"] = "MyLangChainApp/1.0"
 
if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model='gemma')
    st.session_state.loader = WebBaseLoader('https://docs.smith.langchain.com/')
    st.session_state.docs =st.session_state.loader.load()
    print(type(st.session_state.docs))
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text = "\n\n".join([doc.page_content for doc in st.session_state.docs])
    st.session_state.final_doc = st.session_state.text_splitter.split_text(text)
    docs = [Document(page_content=text) for text in st.session_state.final_doc]
    st.session_state.vectordb = FAISS.from_documents(docs, st.session_state.embeddings)



st.title('chatgrok demo')
llm = ChatGroq(groq_api_key=grok_api_key, model='gemma2-9b-it')

prompt= ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided conetxt only.
    Please provie the most accurate response based on the question 
    <context>
    {context}
    </context>
    Question:{input}

"""
)

document_chain = create_stuff_documents_chain(llm, prompt)
retriver = st.session_state.vectordb.as_retriever()

retrival_chain = create_retrieval_chain(retriver,document_chain)

prompt = st.text_input("Input your prompt here")

if prompt:
    start = time.process_time()
    response = retrival_chain.invoke({"input":prompt})
    print("response time", time.process_time()-start)
    st.write(response['answer'])

    # with st.expander("Document similarity search"):
    #     for i, doc in enumerate(response['context']):
    #         st.write(doc.page_content)
    #         st.write('---------------------')
