# 🏦 Enterprise Document GPT (Local-First RAG)

A high-performance, privacy-centric Retrieval-Augmented Generation (RAG) pipeline. This project demonstrates how to architect a secure AI consultant capable of interrogating complex, regulated documents—such as **APRA Prudential Standards**—without the data ever leaving your local infrastructure.



## 🏗️ The Architecture
This implementation follows a "Local-First" philosophy to solve the **Data Sovereignty** and **Privacy** hurdles often found in enterprise AI adoption.

1.  **Ingestion:** PDF documents are parsed and shredded into semantic chunks using `RecursiveCharacterTextSplitter`.
2.  **Vectorization:** Text chunks are transformed into 384-dimensional mathematical vectors using `all-MiniLM-L6-v2`.
3.  **Storage:** Vectors are stored in a high-speed, in-memory `ChromaDB` instance.
4.  **Orchestration:** Built with **LangChain Expression Language (LCEL)** to ensure a transparent and modular data pipeline.
5.  **Inference:** Powered by `Ollama` (Llama 3.2), ensuring all language processing stays within the local environment.



## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10 or higher
- [Ollama](https://ollama.com/) installed and running

### 2. Environment Setup
Clone the repository and install the required Python dependencies:
```bash
pip install -r requirements.txt
3. Initialize the Inference Engine
In a separate terminal, start the Ollama service and pull the model:

Bash
ollama serve
ollama pull llama3.2:1b

4. Launch the Application
Bash
streamlit run app.py
🛠️ Technical Stack
UI: Streamlit

Orchestration: LangChain (LCEL)

Vector Database: ChromaDB

Embeddings: HuggingFace

Local LLM: Ollama

📝 Compliance Use Case: APRA APS 210
This tool is specifically tuned for the rigors of financial regulation.

Target: APRA APS 210 (Liquidity Coverage Ratio).

Query Example: "What are the qualitative requirements for the LCR stress test?"

Result: The system retrieves specific clauses from the Prudential Standard and provides an executive summary with full source transparency.