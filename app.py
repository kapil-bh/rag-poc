import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. CONFIGURATION & UI SETUP ---
st.set_page_config(page_title="Enterprise RAG Architect", layout="wide")
st.title("🏦 Enterprise Document GPT (Hybrid RAG)")
st.markdown("---")

# Initialize Session State to persist the vector database across reruns
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None

# --- 2. CACHED PROCESSING LOGIC ---
@st.cache_resource
def process_pdf(file_bytes):
    """Saves PDF, chunks text, and creates vector embeddings."""
    with open("temp.pdf", "wb") as f:
        f.write(file_bytes)
    
    loader = PyPDFLoader("temp.pdf")
    data = loader.load()
    
    # Chunking: 1000 chars with 150 char overlap to maintain context
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(data)
    
    # Embedding model (CPU-friendly for local and cloud)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    return Chroma.from_documents(docs, embeddings)

# --- 3. SIDEBAR: UPLOAD & ENGINE STATUS ---
with st.sidebar:
    st.header("Admin Panel")
    uploaded_file = st.file_uploader("Upload Regulatory PDF", type="pdf")
    
    if uploaded_file:
        with st.spinner("Analyzing document..."):
            st.session_state.vector_db = process_pdf(uploaded_file.getvalue())
        st.success("Document Indexing Complete!")

    # --- HYBRID ENGINE LOGIC ---
    # Checks Streamlit Secrets (Web) or Local Environment Variables
    groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    
    if groq_api_key:
        llm = ChatGroq(
            temperature=0, 
            model_name="llama-3.3-70b-versatile", 
            groq_api_key=groq_api_key
        )
        st.sidebar.success("Connected to Groq Cloud ⚡")
    else:
        llm = ChatOllama(model="llama3.2:1b")
        st.sidebar.info("Running on Local Ollama engine")

# --- 4. MAIN CHAT INTERFACE ---
if st.session_state.vector_db is not None:
    # Architecting the System Prompt
    template = """You are a Senior Solutions Architect and Compliance Expert. 
    Use the following pieces of retrieved context to answer the question accurately. 
    If the context doesn't contain the answer, state that you don't know based on the provided document.

    Context: {context}

    Question: {question}
    
    Helpful Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)

    with st.form("query_form"):
        query = st.text_input("Ask a question about the uploaded document:")
        submitted = st.form_submit_button("Submit Query")

        if submitted and query:
            with st.spinner("Retrieving facts..."):
                # Use the DB from session state
                retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 3})
                
                # LCEL Chain Logic
                chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                
                # Execute and display result
                response = chain.invoke(query)
                st.markdown("### 🤖 AI Architect Response")
                st.write(response)
                
                # Source Transparency Section
                with st.expander("View Source Document Chunks"):
                    sources = retriever.invoke(query)
                    for i, doc in enumerate(sources):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.info(doc.page_content)
else:
    st.warning("Please upload a PDF in the sidebar to begin.")