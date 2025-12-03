"""
rag_chroma_groq.py

Requirements (install via pip):
pip install langchain chromadb langchain-groq python-docx pandas PyPDF2 unstructured[local] \
            sentence_transformers huggingface-hub

Notes:
- Replace HUGGINGFACE_API_TOKEN and GROQ_API_KEY with your keys.
- ChatGroq LLM is used via langchain_groq.ChatGroq (assumes that package works as a LangChain LLM wrapper).
- Adjust chunk_size / overlap for your documents.
"""

import os
from pathlib import Path
import pandas as pd

# LangChain imports
#from langchain.docstore.document import Document
#from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings.huggingface import HuggingFaceEmbeddings
#from langchain.vectorstores import Chroma
#from langchain.chains import create_retrieval_chain
#from langchain.chains.combine_documents import create_stuff_documents_chain

# File loaders
#from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.document_loaders import PyPDFLoader , Docx2txtLoader , TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
#from langchain_community.chains import RetrievalQA
# Groq LLM wrapper (user already referenced ChatGroq)
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings



# ----------- User settings (edit) -----------
HUGGINGFACEHUB_API_TOKEN = ""   # your Hugging Face token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

GROQ_API_KEY = ""           # your Groq API key
GROQ_MODEL = "llama-3.1-8b-instant"  # as requested
DATA_DIR = "dataset"           # folder that contains pdf/docx/csv/xlsx files
CHROMA_PERSIST_DIR = "./chroma_db" # where chroma will persist vectors
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # small, fast; change if you want a HF model
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
# --------------------------------------------

def load_docs_from_folder(folder: str):
    """Walk folder and load files into a list of LangChain Documents"""
    docs = []
    folder = Path(folder)
    for root, _, files in os.walk(folder):
        for fname in files:
            path = Path(root) / fname
            lower = fname.lower()
            try:
                if lower.endswith(".pdf"):
                    loader = PyPDFLoader(str(path))
                    pages = loader.load()
                    docs.extend(pages)
                elif lower.endswith(".docx") or lower.endswith(".doc"):
                    # Docx loader (docx only; if .doc, consider conversion)
                    loader = Docx2txtLoader(str(path))
                    d = loader.load()
                    docs.extend(d)
                elif lower.endswith(".txt"):
                    loader = TextLoader(str(path), encoding="utf8")
                    docs.extend(loader.load())
                elif lower.endswith(".csv"):
                    # Use pandas to read CSV, then join rows into text
                    df = pd.read_csv(str(path), dtype=str, keep_default_na=False)
                    # Create a text blob per row (or one document per file)
                    row_texts = []
                    for i, row in df.iterrows():
                        row_text = "\n".join([f"{col}: {row[col]}" for col in df.columns])
                        row_texts.append(Document(page_content=row_text, metadata={"source": str(path), "row": i}))
                    docs.extend(row_texts)
                elif lower.endswith(".xlsx") or lower.endswith(".xls"):
                    df = pd.read_excel(str(path), dtype=str, keep_default_na=False)
                    row_texts = []
                    for i, row in df.iterrows():
                        row_text = "\n".join([f"{col}: {row[col]}" for col in df.columns])
                        row_texts.append(Document(page_content=row_text, metadata={"source": str(path), "row": i}))
                    docs.extend(row_texts)
                else:
                    # skip unknown types
                    print(f"Skipping unsupported file: {path.name}")
            except Exception as e:
                print(f"Error loading {path}: {e}")
    print(f"Loaded {len(docs)} raw documents/records from {folder}")
    return docs


def chunk_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    new_docs = []
    for doc in docs:
        chunks = splitter.split_documents([doc])
        new_docs.extend(chunks)
    print(f"Split into {len(new_docs)} chunks (chunk_size={chunk_size}, overlap={chunk_overlap})")
    return new_docs


def create_or_load_chroma(docs, persist_dir=CHROMA_PERSIST_DIR):
    # Initialize HuggingFace embeddings wrapper
    #embeddings = HuggingFaceEmbeddings(model_name=embedding_model, huggingfacehub_api_token=hf_token)
    embeddings = OllamaEmbeddings(model="all-minilm:latest")
    # If persist directory exists and has a collection, Chroma will load it. We initialize the vectorstore with persistence.
    #vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
    
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="chroma_db"   # auto-persist enabled
    )
    
    # If empty, add docs
    if vectordb._collection.count() == 0:
        print("Chroma DB is empty. Adding documents and persisting...")
        vectordb.add_documents(docs)
        #vectordb.persist()
        print("Documents indexed and saved to Chroma persist directory.")
    else:
        print("Loaded existing Chroma DB from persist directory.")
    return vectordb


"""def build_qa_chain(vectordb, groq_api_key=GROQ_API_KEY, groq_model=GROQ_MODEL):
    # Setup Groq LLM wrapper (temperature, max tokens can be tuned)
    llm = ChatGroq(api_key=groq_api_key, model=groq_model, temperature=0.0)

    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    return qa
"""


def main():
    # 1) Load all documents from folder
    docs = load_docs_from_folder(DATA_DIR)

    if len(docs) == 0:
        print("No documents found. Exiting.")
        return

    # 2) Chunk/split documents
    chunks = chunk_documents(docs)

    # 3) Create or load Chroma (embedding + index)
    vectordb = create_or_load_chroma(chunks)

    # 4) Build QA chain
    #qa_chain = build_qa_chain(vectordb)

    # 5) Interactive QA loop
    print("\nRAG ready. Enter questions (type 'exit' to quit).")


if __name__ == "__main__":
    main()
