<div align="center">

# RAG: Chat with PDFs

**An intelligent conversational interface for document interaction using Retrieval-Augmented Generation.**

<a href="https://github.com/shashankAcharya3/RAG-Chat-with-pdfs/commits/main">
<img src="https://img.shields.io/badge/Maintained-Yes-151515.svg?style=flat-square" alt="Maintained">
</a>
<a href="https://github.com/shashankAcharya3/RAG-Chat-with-pdfs/blob/main/LICENSE">
<img src="https://img.shields.io/badge/License-MIT-151515.svg?style=flat-square" alt="License">
</a>
<img src="https://img.shields.io/badge/Language-Python-151515.svg?style=flat-square" alt="Python">

</div>

---

### Overview

This repository features a **Retrieval-Augmented Generation (RAG)** system designed to allow users to have context-aware conversations with standard PDF documents. By combining document parsing, vector embeddings, and Large Language Models (LLMs), the application grounds its responses strictly in the provided text, minimizing hallucinations and enabling rapid information extraction.

### Core Features

- **Document Ingestion:** Automated parsing and chunking of complex PDF documents.
- **Semantic Search:** Generation of high-dimensional text embeddings to retrieve the most relevant document chunks based on user queries.
- **Context-Aware Generation:** Feeds retrieved context to the LLM to synthesize precise, accurate answers.
- **Interactive Chat Interface:** A streamlined UI for seamless document uploads and natural language querying.

### System Architecture & Tech Stack

- **Language**: Python
- **LLM Framework**: LangChain / LlamaIndex 
- **Vector Database**: FAISS / ChromaDB / Pinecone 
- **Embeddings**: OpenAI Embeddings / HuggingFace BGE 
- **Frontend/UI**: Streamlit / Gradio 

### Getting Started

To run the RAG application locally, ensure you have Python 3.8+ installed. You can deploy the entire stack at once by running the consolidated bash commands below.

```bash
# 1. Clone the repository and navigate inside
git clone [https://github.com/shashankAcharya3/RAG-Chat-with-pdfs.git](https://github.com/shashankAcharya3/RAG-Chat-with-pdfs.git)
cd RAG-Chat-with-pdfs

# 2. Initialize and activate the virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# 3. Install all required dependencies
pip install -r requirements.txt

# 4. Set up your environment variables
echo 'OPENAI_API_KEY="your_api_key_here"' > .env

# 5. Launch the application
streamlit run app.py
