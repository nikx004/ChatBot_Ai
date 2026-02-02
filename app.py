import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

st.title("📄 Internal Document Chatbot")

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Vector DB
db = Chroma(
    persist_directory="vectorstore",
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k": 10})

# LLM
llm = Ollama(
    model="mistral",
    temperature=0,
    system="Answer ONLY using the provided documents. If the answer is not found, say so clearly."
)

query = st.text_input("Ask a question based on the documents:")

if query:
    docs = retriever.invoke(query)

    if not docs:
        st.write("❌ Information not found in provided documents.")
    else:
        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = f"""
You must answer ONLY from the context below.
If the answer is not present, say "Information not found in provided documents."

Context:
{context}

Question:
{query}
"""

        response = llm.invoke(prompt)
        st.write("### Answer")
        st.write(response)

