# Interactive Question-Answering Bot for Document-Based Insights

This project is a **Streamlit application** that enables users to upload PDF documents, extract their text, and ask questions about the content. It utilizes **Cohere's language model** to generate embeddings for document chunks and stores them in a **FAISS** index for efficient retrieval. Users can query the document, and relevant sections are returned as answers.

![Streamlit App Preview]()
![Screenshot 2024-10-10 111857](https://github.com/user-attachments/assets/4e9471e8-9f16-4078-9e3b-06259919aadc)

## Features

- Upload and extract text from PDF documents.
- Split large documents into smaller chunks for optimized processing.
- Generate document embeddings using the **Cohere API**.
- Store embeddings in a **FAISS** index for fast similarity searches.
- Ask questions about the uploaded document and retrieve answers based on context.

## How FAISS Works in This Project

**FAISS (Facebook AI Similarity Search)** is utilized to store and search embeddings efficiently. When a PDF is uploaded, it is chunked into smaller sections, embeddings are generated using Cohere, and these embeddings are then stored in the FAISS index.

FAISS allows for:

- **Fast similarity search**: Quickly find relevant sections of the document based on embeddings.
- **Scalability**: Efficient for large datasets, making it ideal for handling large PDF documents.

In this project, we use the **L2 distance** for measuring similarity, but FAISS supports various distance metrics.

### Choosing FAISS for Embeddings Storage

FAISS is ideal for vector storage and retrieval in scenarios where:

- You need **fast retrieval** from a large collection of document chunks.
- Embeddings need to be **compared based on similarity**.
- You want **scalability** with minimal performance loss as your dataset grows.

This project leverages FAISS with **flat indexing** for simplicity. However, further optimizations can be explored using **IVF** or **HNSW** indexes for larger datasets.

### Note on Pinecone

Pinecone is a paid service for vector storage and retrieval. Instead, this project uses **FAISS**, which is open-source and suitable for our needs without incurring costs.

## Tech Stack

- **Streamlit**: For building the web interface and handling user input.
- **Cohere**: Used to generate embeddings for both document chunks and queries.
- **FAISS**: For efficient similarity searches among document chunks.
- **LangChain**: Manages the workflow of embedding creation, query processing, and QA.
- **PyPDFLoader**: For reading and processing PDF files.
- **Docker**: Containerized deployment for easy setup.

## How to Run the Project Locally

### Prerequisites

Make sure you have the following installed:

- **Python 3.9 or higher**
- **Cohere API Key** (Sign up at [Cohere](https://cohere.ai))
- **Docker** (if using containerized deployment)

### Installation Steps

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/streamlit-cohere-pdf-bot.git
   cd streamlit-cohere-pdf-bot

