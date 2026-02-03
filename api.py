import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_community.vectorstores import Chroma

app = FastAPI(title="Internal Document Chatbot")

embeddings = None
db = None
retriever = None
llm = None


class Question(BaseModel):
    question: str


@app.on_event("startup")
def startup_event():
    global embeddings, db, retriever, llm

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.environ["GEMINI_API_KEY"],
    )

    db = Chroma(
        persist_directory="vectorstore",
        embedding_function=embeddings,
    )

    retriever = db.as_retriever(search_kwargs={"k": 10})

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        google_api_key=os.environ["GEMINI_API_KEY"],
    )


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

    response = llm.invoke(prompt)
    return {"answer": response.content}

