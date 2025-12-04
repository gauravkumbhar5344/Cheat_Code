import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, END
from typing import TypedDict, List


# -----------------------
# LangGraph STATE with MEMORY
# -----------------------
class RAGState(TypedDict):
    question: str
    context: List[str]
    answer: str
    history: List[dict]   # NEW MEMORY


# -----------------------
# Load PDF and create vector DB
# -----------------------
loader = PyPDFLoader("example.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
)
splits = splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model="all-minilm:latest")
vector_store = FAISS.from_documents(splits, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})


# -----------------------
# Prompt Template
# -----------------------
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use the context below to answer.
If answer is not in context, say: "Not found in the document."

Context:
{context}

Chat History:
{history}

Question:
{question}

Answer:
""")


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
    context_list = [d.page_content for d in docs]
    return {"context": context_list}


def generate_answer(state: RAGState):
    final_prompt = prompt.format(
        context="\n\n".join(state["context"]),
        question=state["question"],
        history="\n".join(
            [f"User: {x['user']}\nAssistant: {x['assistant']}" for x in state["history"]]
        )
    )

    result = llm.invoke(final_prompt)
    answer = result.content

    # update memory
    updated_history = state["history"] + [
        {"user": state["question"], "assistant": answer}
    ]

    return {
        "answer": answer,
        "history": updated_history
    }


# -----------------------
# Graph Definition
# -----------------------
graph = StateGraph(RAGState)

graph.add_node("retrieve", retrieve_docs)
graph.add_node("generate", generate_answer)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

app = graph.compile()


# -----------------------
# Streamlit Chat UI
# -----------------------

st.title("🧠 LangGraph RAG Chatbot with Memory")

# Initialize chat history in Streamlit session
if "history" not in st.session_state:
    st.session_state.history = []

# Show previous messages
for chat in st.session_state.history:
    st.chat_message("user").write(chat["user"])
    st.chat_message("assistant").write(chat["assistant"])

# Chat input
user_query = st.chat_input("Ask a question about the PDF...")

if user_query:
    # Run graph
    with st.spinner("Thinking..."):
        output = app.invoke({
            "question": user_query,
            "history": st.session_state.history
        })

    # extract answer
    answer = output["answer"]

    # update streamlit memory
    st.session_state.history = output["history"]

    # show latest exchange
    st.chat_message("user").write(user_query)
    st.chat_message("assistant").write(answer)
