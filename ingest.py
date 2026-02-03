import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_DIR = "newdata"   # folder where your PDFs are
VECTOR_DIR = "vectorstore"

documents = []

for file in os.listdir(DATA_DIR):
    path = os.path.join(DATA_DIR, file)

    if file.endswith(".pdf"):
        documents.extend(PyPDFLoader(path).load())

    elif file.endswith(".docx"):
        documents.extend(Docx2txtLoader(path).load())

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma.from_documents(
    chunks,
    embedding=embeddings,
    persist_directory=VECTOR_DIR
)

print("✅ Documents indexed successfully")

