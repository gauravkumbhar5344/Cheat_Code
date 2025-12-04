import os
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, END
from typing import TypedDict, List


# -----------------------
# LangGraph State
# -----------------------
class RAGState(TypedDict):
    question: str
    context: List[str]
    answer: str


# -----------------------
# Load PDF & Build Vector Store (Run once)
# -----------------------

loader = PyPDFLoader("example.pdf")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

splits = text_splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model="all-minilm:latest")
vector_store = FAISS.from_documents(splits, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})


# -----------------------
# PROMPT
# -----------------------
prompt = ChatPromptTemplate.from_template(
    """
Answer strictly based on the context below. 
If the answer is not found in the context, say "Answer not found in the document."

Context:
{context}

Question:
{question}

Answer:
"""
)


# -----------------------
# LLM
# -----------------------
llm = ChatGroq(
    api_key="",
    model="llama-3.3-70b-versatile"
)


# -----------------------
# Graph Nodes
# -----------------------

def retrieve_docs(state: RAGState):
    docs = retriever.invoke(state["question"])
    context_texts = [d.page_content for d in docs]
    return {"context": context_texts}


def generate_answer(state: RAGState):
    final_prompt = prompt.format(
        context="\n\n".join(state["context"]),
        question=state["question"]
    )
    result = llm.invoke(final_prompt)
    return {"answer": result.content}


# -----------------------
# Build Graph
# -----------------------

graph = StateGraph(RAGState)

graph.add_node("retrieve", retrieve_docs)
graph.add_node("generate", generate_answer)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

app = graph.compile()


# -----------------------
# Streamlit UI
# -----------------------

st.title("📄 LangGraph-based RAG Chatbot")

user_query = st.text_input("Ask a question about the PDF")

if user_query:
    with st.spinner("Thinking..."):
        output = app.invoke({"question": user_query})
        st.write(output["answer"])
