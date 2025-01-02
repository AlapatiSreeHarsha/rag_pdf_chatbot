# Conversational RAG with PDF Uploads and Chat History

This project is a Streamlit-based Conversational Retrieval-Augmented Generation (RAG) application. The app allows users to upload PDF files, extract their content, and interact with the content through a conversational interface. It uses Hugging Face for document embeddings and Groq API for large language model (LLM) processing.

---

## Technologies Used

1. **Streamlit**: A Python library to create interactive web applications for data science and machine learning.
2. **LangChain**: A framework to build applications that use language models with external data sources like documents, APIs, and databases.
3. **FAISS**: A library for efficient similarity search and clustering of dense vectors (used to search through document embeddings).
4. **Hugging Face**: Provides pre-trained models for text processing, such as embeddings for documents.
5. **Groq API**: A powerful API for accessing large language models and performing NLP tasks (e.g., question answering).

---

## Step-by-Step Guide

### 1. **Understanding the Code**

The `chatbot31.py` file is the core of this application. Here’s what it does:

- **PDF Upload and Processing**:
    - Users can upload multiple PDF files.
    - The app extracts text from the PDFs using `PyPDFLoader`.
    - The text is then split into smaller chunks using `RecursiveCharacterTextSplitter` to manage document size for embedding.

- **Embedding the Text**:
    - The `HuggingFaceEmbeddings` class is used to generate embeddings for the extracted text. This allows the app to perform similarity-based search for relevant content based on user queries.

- **History-Aware Retrieval**:
    - The app maintains a session-based chat history. Each user’s input and prior conversations are considered to refine responses using a history-aware retrieval mechanism.

- **Question Answering**:
    - Using the embeddings and retrieval chains, the app answers user questions based on the context from the PDF documents.

- **Session Management**:
    - The app stores chat history for each session using `st.session_state`, allowing the conversation to persist over multiple interactions.

---

### 2. **Installation Instructions**

Follow the steps below to install the required libraries and set up the environment.

#### Step 1: Clone the Repository

bash
git clone https://github.com/your-repo/conversational-rag.git
cd conversational-rag


pip install streamlit langchain langchain-huggingface faiss-cpu python-dotenv


##Touble shooting
pip install <package-name>



streamlit run chatbot31.py
