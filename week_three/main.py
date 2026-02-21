import os
import yaml
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

# LangChain & AI Imports
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# --- 1. CONFIGURATION & INITIALIZATION ---
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"
PDF_PATH = BASE_DIR / "pdfs" / "iesc104.pdf"
DB_DIR = BASE_DIR / "chroma_db"

with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)

embeddings = HuggingFaceEmbeddings(model_name=config["EMBEDDING_MODEL"])
llm = ChatGroq(
    api_key=config["GROQ_API_KEY"],
    model="llama-3.3-70b-versatile",
    temperature=0
)

# --- 2. VECTOR DATABASE SETUP ---
# Create or Load the Vector Store
if not DB_DIR.exists():
    print("Indexing documents... this may take a moment.")
    loader = PyPDFLoader(str(PDF_PATH)).load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(loader)
    vector_db = Chroma.from_documents(
        documents=texts, 
        embedding=embeddings, 
        persist_directory=str(DB_DIR)
    )
else:
    print("Loading existing vector database...")
    vector_db = Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)

retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# --- 3. RAG CHAIN SETUP ---
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a factual QA assistant. Answer ONLY from the given context.
    Context: {context}
    Question: {question}
    Answer:"""
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)

# --- 4. FASTAPI APPLICATION ---
app = FastAPI()

class Query(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_file = BASE_DIR / "index.html"
    if index_file.exists():
        with open(index_file, "r") as f:
            return f.read()
    return "<h1>index.html not found</h1>"

@app.post("/ask")
async def ask_rag(query: Query):
    try:
        response = qa_chain.invoke(query.text)
        return {"answer": response["result"]}
    except Exception as e:
        return {"answer": f"Backend Error: {str(e)}"}

if __name__ == "__main__":
    print("\n🚀 Server running at http://127.0.0.1:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)