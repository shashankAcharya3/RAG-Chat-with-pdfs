import os
import streamlit as st
import pdfplumber

from langchain.schema.document import Document
from langchain.vectorstores import Annoy
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import google.generativeai as genai

# API KEY and model configuration
api_key = "AIzaSyA1x_Qbe63gbyJHdSPkNxvnbDKFDSK8FBE"
genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name="models/gemini-1.5-pro")

# Load PDFs
def load_pdf(data_path="data"):
    documents = []
    for pdf_file in os.listdir(data_path):
        if pdf_file.endswith(".pdf"):
            with pdfplumber.open(os.path.join(data_path, pdf_file)) as pdf:
                text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
                documents.append(Document(page_content=text))
    return documents

# Split into chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

# App UI
st.title("📄 Gemini PDF QA")
st.markdown("Ask questions based on your PDF documents.")

# Load and process documents once
@st.cache_resource
def load_and_process():
    docs = load_pdf()
    chunks = split_documents(docs)
    texts = [chunk.page_content for chunk in chunks]

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    annoy_index = Annoy.from_texts(texts, embeddings)
    return texts, embeddings, annoy_index

texts, embeddings, annoy_store = load_and_process()

# User input
query = st.text_input("Enter your question:")
if query:
    with st.spinner("Finding relevant context..."):
        query_embedding = embeddings.embed_query(query)
        similar_docs = annoy_store.similarity_search_by_vector(query_embedding, k=5)

        if similar_docs:
            pdf_context = " ".join(doc.page_content for doc in similar_docs)
            st.subheader("🔍 Most Relevant Text from PDF:")
            st.text_area("Context", pdf_context, height=250)
        else:
            pdf_context = "No relevant context found."
            st.warning("No relevant context found.")

    if pdf_context != "No relevant context found.":
        with st.spinner("Generating response from Gemini..."):
            full_prompt = f"""
            You are an AI assistant. Answer the user's question based on the extracted PDF context.

            **Question:** {query}

            **PDF Context:** {pdf_context}

            Provide a detailed and informative response. Explain thoroughly, provide examples, and ensure clarity. If needed, break your response into sections for better readability.
            """
            generation_config = genai.types.GenerationConfig(
                temperature=0.7,
                top_p=0.9,
                max_output_tokens=1024,
            )

            response = model.generate_content(full_prompt, generation_config=generation_config)

            if response and hasattr(response, "candidates") and response.candidates:
                full_text = " ".join(part.text for part in response.candidates[0].content.parts)
                st.subheader("💬 Gemini's Response:")
                st.markdown(full_text)
            else:
                st.error("No valid response received from Gemini AI.")
