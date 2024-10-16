import streamlit as st
import os
import cohere
import faiss
import numpy as np
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import Cohere
from dotenv import load_dotenv
import pandas as pd

# Load environment variables (for Cohere API key)
load_dotenv()

# Initialize Cohere client
co = cohere.Client(os.getenv("COHERE_API_KEY"))

# Streamlit UI
st.set_page_config(page_title="Budget 2024 ChatBot", layout="wide")
st.title("Interactive Question-Answering Bot for Document-Based Insights📕")
st.markdown("Upload a PDF document to extract insights based on its content..")

# Sidebar for file upload and query input
st.sidebar.header("Upload and Query")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

# Initialize history storage
if 'history' not in st.session_state:
    st.session_state.history = []

# File upload and processing
if uploaded_file is not None:
    with open("temp_pdf.pdf", "wb") as f:
        f.write(uploaded_file.read())
    
    # Function to read and process PDF
    def read_pdf(file_path):
        file_loader = PyPDFLoader(file_path)
        documents = file_loader.load()
        return documents
    
    # Load the saved PDF file
    doc = read_pdf("temp_pdf.pdf")
    
    # Chunk the document using a Recursive Text Splitter
    def chunk_data(docs, chunk_size=800, chunk_overlap=50):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        docs_chunked = text_splitter.split_documents(docs)
        return docs_chunked

    # Chunk the PDF document into smaller parts
    documents = chunk_data(docs=doc)
    
    # Cohere Embedding Function
    class CohereEmbeddings(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            response = co.embed(texts=texts)
            return response.embeddings
        
        def embed_query(self, text: str) -> list[float]:
            response = co.embed(texts=[text])
            return response.embeddings[0]
    
    # Store embeddings in FAISS
    def store_faiss_embeddings(embeddings, documents):
        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings))
        return index

    # Generate Cohere embeddings for the document chunks
    texts = [doc.page_content for doc in documents]
    cohere_embedder = CohereEmbeddings()
    embeddings = cohere_embedder.embed_documents(texts)

    # Store embeddings in FAISS
    faiss_index = store_faiss_embeddings(embeddings, documents)

    # Search FAISS for relevant documents based on query
    def search_faiss_index(query):
        query_embedding = cohere_embedder.embed_query(query)
        query_vector = np.array([query_embedding])
        D, I = faiss_index.search(query_vector, k=5)
        return I

    # Query input from the user
    query = st.sidebar.text_input("Enter your Query Regarding Budget 2024:", placeholder="e.g., What is the budget for X?")

    if query:
        result_indices = search_faiss_index(query)
        
        # Run Question Answering Chain on matched documents
        llm = Cohere(cohere_api_key=os.getenv("COHERE_API_KEY"))
        qa_chain = load_qa_chain(llm, chain_type="stuff")

        # Get the matching documents based on FAISS search results
        matching_documents = [documents[i] for i in result_indices[0]]

        # Run QA chain
        answer = qa_chain.run(input_documents=matching_documents, question=query)
        
        # Display the answer
        st.markdown(f"### Answer:")
        st.success(answer)
        
        # Save the query and answer to history
        st.session_state.history.append({"query": query, "answer": answer})

# Display query history
st.sidebar.markdown("### Query History")
if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.sidebar.dataframe(history_df, use_container_width=True)
else:
    st.sidebar.write("No queries yet.")

# Add footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and Cohere.")
