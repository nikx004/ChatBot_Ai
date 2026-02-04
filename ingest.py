# ingest.py
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

VECTOR_DIR = "vectorstore"
DATA_DIR = "newdata"

def ingest_documents():
    print("🔹 Starting ingestion...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    docs = []
    for file in os.listdir(DATA_DIR):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_DIR, file))
            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)

    Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DIR
    )

    print("✅ Vectorstore created successfully")

if __name__ == "__main__":
    ingest_documents()

