import os

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
import os

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
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=os.environ["OPENAI_API_KEY"]
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("api:app", host="0.0.0.0", port=port)

