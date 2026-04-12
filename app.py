import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

st.title("🏦 Enterprise Document GPT (Local RAG)")

# 1. File Uploader
uploaded_file = st.file_uploader("Upload any PDF document", type="pdf")

if uploaded_file:
    # Save the file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    # 2. Process & Embed (The generic part)
    loader = PyPDFLoader("temp.pdf")
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(data)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(docs, embeddings)

    # 3. Chat Interface
    query = st.text_input("Ask a question about this document:")
    if query:
        results = vector_db.similarity_search(query, k=3)
        st.subheader("Top Matches from Document:")
        for res in results:
            st.write(f"--- \n {res.page_content}")