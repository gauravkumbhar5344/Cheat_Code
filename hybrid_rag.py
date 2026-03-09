import os
import pickle

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
#from langchain_chroma.vectorstores import Chroma
#from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
#from langchain.retrievers import EnsembleRetriever

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate
#from langchain.chains import create_retrieval_chain
#from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# -----------------------------
# 1. Load LLM
# -----------------------------
llm = ChatGoogleGenerativeAI(
    google_api_key="AIzaSyDm_h9-X4GRJdxfo-xpp8yvJlzU5mGN9rU",
    model="gemini-2.5-flash",
    temperature=0
)


# -----------------------------
# 2. Embedding Model
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -----------------------------
# 3. Load PDF
# -----------------------------
loader = PyPDFLoader("document.pdf")
documents = loader.load()


# -----------------------------
# 4. Chunk Documents
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)

docs = text_splitter.split_documents(documents)


# -----------------------------
# 5. Dense Vector Store
# -----------------------------
if os.path.exists("vectorstore"):
    vectorstore = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("vectorstore")

dense_retriever = vectorstore.as_retriever(search_kwargs={"k":4})


# -----------------------------
# 6. BM25 Retriever
# -----------------------------
if os.path.exists("bm25.pkl"):
    with open("bm25.pkl", "rb") as f:
        bm25_retriever = pickle.load(f)
else:
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 4

    with open("bm25.pkl", "wb") as f:
        pickle.dump(bm25_retriever, f)


# -----------------------------
# 7. Hybrid Retriever
# -----------------------------
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.5, 0.5]
)


# -----------------------------
# 8. Prompt Template
# -----------------------------
prompt = ChatPromptTemplate.from_template("""
Answer the question using the provided context.

Context:
{context}

Question:
{input}

Answer:
""")


"""question = hybrid_retriever.invoke("Model Architecture")
print(question)"""


# -----------------------------
# 9. Document Chain
# -----------------------------
document_chain = create_stuff_documents_chain(
    llm,
    prompt
)


# -----------------------------
# 10. Retrieval Chain
# -----------------------------
rag_chain = create_retrieval_chain(
    hybrid_retriever,
    document_chain
)

# -----------------------------
# 11. Ask Questions
# -----------------------------
while True:

    query = input("\nAsk Question (type exit to quit): ")

    if query.lower() == "exit":
        break

    response = rag_chain.invoke({
        "input": query
    })

    print("\nAnswer:\n", response["answer"])