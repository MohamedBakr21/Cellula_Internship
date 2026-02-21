import os
import yaml
from pathlib import Path
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

class RagSystem:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.embeddings = HuggingFaceEmbeddings(model_name=self.config["EMBEDDING_MODEL"])
        self.llm = ChatGroq(
            api_key=self.config["GROQ_API_KEY"],
            model="llama-3.3-70b-versatile",
            temperature=0
        )
        self.vector_db = self._load_vector_db()
        self.qa_chain = self._build_rag_chain()

    def _load_config(self, path):
        with open(path, "r") as file:
            return yaml.safe_load(file)

    def _load_vector_db(self):
        # In production, specify a 'persist_directory' so you don't re-index every time
        return Chroma(
            persist_directory="./chroma_db", 
            embedding_function=self.embeddings
        )

    def _build_rag_chain(self):
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="You are a factual QA assistant. Answer ONLY from context.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        )
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vector_db.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt}
        )

    def ask(self, query: str):
        try:
            return self.qa_chain.invoke(query)
        except Exception as e:
            return {"error": str(e), "result": "Failed to process request."}