from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

app = FastAPI(title="Internal Document Chatbot")

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load vector database
db = Chroma(
    persist_directory="vectorstore",
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k": 10})

# Load LLM
llm = Ollama(
    model="mistral",
    temperature=0
)

class Question(BaseModel):
    question: str

@app.post("/ask")
def ask(q: Question):
    docs = retriever.invoke(q.question)

    if not docs:
        return {"answer": "Information not found in provided documents."}

    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
Answer ONLY using the context below.
If the answer is not present, say so clearly.

Context:
{context}

Question:
{q.question}
"""

    answer = llm.invoke(prompt)
    return {"answer": answer}

