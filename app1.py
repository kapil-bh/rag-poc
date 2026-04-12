import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 1. CONFIGURATION & UI SETUP ---
st.set_page_config(page_title="Enterprise RAG Architect", layout="wide")
st.title("🏦 Enterprise Document GPT (Local RAG)")
st.markdown("---")

# Initialize Session State for the Vector Database
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None

# --- 2. CACHED PROCESSING LOGIC ---
@st.cache_resource
def process_pdf(file_bytes):
    """Processes the PDF once and stores it in a local vector database."""
    with open("temp.pdf", "wb") as f:
        f.write(file_bytes)
    
    # Load and Split
    loader = PyPDFLoader("temp.pdf")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(data)
    
    # Local Embeddings (CPU friendly)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create In-Memory Vector Store
    return Chroma.from_documents(docs, embeddings)

# --- 3. SIDEBAR: UPLOAD & STATUS ---
with st.sidebar:
    st.header("Admin Panel")
    uploaded_file = st.file_uploader("Upload Policy PDF" , type="pdf")
    
    if uploaded_file:
        with st.spinner("Analyzing document..."):
            st.session_state.vector_db = process_pdf(uploaded_file.getvalue())
        st.success("Document Indexing Complete!")
    
    st.info("Stack: Streamlit + LangChain + ChromaDB + Ollama")

# --- 4. MAIN CHAT INTERFACE ---
if st.session_state.vector_db is not None:
    # Define the Brain (Make sure you ran: ollama pull llama3.2:1b)
    llm = ChatOllama(model="llama3.2:1b")

    # Architecting the Prompt
    template = """You are a Senior Banking Compliance Architect. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Keep the answer concise and professional.

    Context: {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # The Query Form
    with st.form("query_form"):
        query = st.text_input("Enter your regulatory question:")
        submitted = st.form_submit_button("Ask AI")

        if submitted and query:
            with st.spinner("Consulting the knowledge base..."):
                # Define the LCEL Chain
                retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 3})
                
                chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                    | StrOutputParser()
                )
                
                # Execute
                answer = chain.invoke(query)
                
                # Output
                st.markdown("### 🤖 Senior Architect Response")
                st.write(answer)
                
                # Show Sources for Transparency
                with st.expander("View Source Document Chunks"):
                    sources = retriever.invoke(query)
                    for i, doc in enumerate(sources):
                        st.markdown(f"**Source {i+1}:**")
                        st.info(doc.page_content)
else:
    st.warning("Please upload a PDF in the sidebar to begin.")