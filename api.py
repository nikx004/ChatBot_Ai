import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

app = FastAPI(title="Document AI Chatbot")

# ---- Lazy initialization (CRITICAL for Render) ----
embeddings = None
db = None
retriever = None
llm = None

class Question(BaseModel):
    question: str

@app.on_event("startup")
def startup_event():
    global embeddings, db, retriever, llm

    embeddings = OpenAIEmbeddings(
        openai_api_key=os.environ["OPENAI_API_KEY"]
    )

    db = Chroma(
        persist_directory="vectorstore",
        embedding_function=embeddings
    )

    retriever = db.as_retriever(search_kwargs={"k": 10})

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        openai_api_key=os.environ["OPENAI_API_KEY"]
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

    answer = llm.invoke(prompt)
    return {"answer": answer}

