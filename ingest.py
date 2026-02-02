from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

DATA_PATH = "newdata"
DB_PATH = "vectorstore"

documents = []

for file in os.listdir(DATA_PATH):
    path = os.path.join(DATA_PATH, file)
    if file.endswith(".pdf"):
        documents.extend(PyPDFLoader(path).load())
    elif file.endswith(".docx"):
        documents.extend(Docx2txtLoader(path).load())

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory=DB_PATH
)

db.persist()
print("Documents indexed successfully")

