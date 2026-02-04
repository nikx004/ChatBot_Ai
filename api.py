# api.py

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

from ingest import ingest_documents

# -----------------------------
# Constants
# -----------------------------
VECTOR_DIR = "vectorstore"

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(
    title="Internal Document Chatbot",
    version="0.1.0"
)

# -----------------------------
# Request model
# -----------------------------
class Question(BaseModel):
    question: str

# -----------------------------
# Startup: build vectorstore if missing
# -----------------------------
@app.on_event("startup")
def startup_event():
    if not os.path.exists(VECTOR_DIR):
        print("⚠️ Vectorstore not found. Running ingestion...")
        ingest_documents()
    else:
        print("✅ Vectorstore already exists")

# -----------------------------
# Load embeddings + vectorstore
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    persist_directory=VECTOR_DIR,
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k": 4})

# -----------------------------
# LLM (local CPU-safe model)
# -----------------------------
llm = HuggingFacePipeline.from_model_id(
    model_id="google/flan-t5-base",
    task="text2text-generation",
    pipeline_kwargs={"max_new_tokens": 512}
)

# -----------------------------
# Prompt
# -----------------------------
PROMPT = PromptTemplate(
    template="""
You are an internal company assistant.
Answer ONLY using the provided context.
If the answer is not found, say:
"Information not found in the provided documents."

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["context", "question"]
)

# -----------------------------
# QA Chain
# -----------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=False
)

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Internal Document Chatbot API running"}

@app.post("/ask")
def ask_question(payload: Question):
    try:
        result = qa_chain.run(payload.question)
        return {"answer": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

