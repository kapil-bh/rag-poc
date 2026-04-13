import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA


st.set_page_config(page_title="Enterprise RAG", layout="wide")
st.title("🏦 Enterprise Document GPT (Local RAG)")

# --- ARCHITECTURAL FIX: CACHING ---
# This function only runs when a NEW file is uploaded. 
# Otherwise, it returns the existing database from memory.
@st.cache_resource
def process_pdf(file_bytes):
    with open("temp.pdf", "wb") as f:
        f.write(file_bytes)
    
    loader = PyPDFLoader("temp.pdf")
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(data)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # We use a fresh in-memory Chroma instance each time the file changes
    return Chroma.from_documents(docs, embeddings)

# 1. File Uploader
uploaded_file = st.file_uploader("Upload any PDF document", type="pdf")

if uploaded_file:
    # Trigger the cached processing
    vector_db = process_pdf(uploaded_file.getvalue())

    # 2. Chat Interface (Using a Form to prevent 'double-fire' on enter)
    with st.form("my_query_form", clear_on_submit=False):
        query = st.text_input("Ask a question about this document:")
        submitted = st.form_submit_button("Submit")
    
        if submitted and query:
            llm = ChatOllama(model="llama3.2:1b")
            qa_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=vector_db.as_retriever())

            # Get the actual human-readable answer
            response = qa_chain.invoke(query)
            st.write(response["result"])

           # results = vector_db.similarity_search(query, k=3)
            
           # st.subheader("Top Matches from Document:")
            # Displaying in a clean way
           # for i, res in enumerate(results):
            #    with st.expander(f"Source Chunk {i+1}"):
             #       st.write(res.page_content)