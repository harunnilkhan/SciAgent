# SciAgent - Scientific Document Analysis Tool

A powerful RAG-based application for analyzing scientific papers and documents using local LLM models. SciAgent enables users to upload PDF documents, ask questions, generate summaries, and perform semantic search - all running locally without requiring cloud API keys.

##  Features

- **PDF Document Processing**: Upload and process scientific papers and documents
- **Question-Answering**: Ask questions about your documents and get grounded answers
- **Document Summarization**: Generate comprehensive summaries of uploaded documents
- **Semantic Search**: Perform detailed searches using keywords and concepts
- **Local LLM Integration**: Works with Ollama models - no cloud APIs required
- **Vector Database**: Efficient document retrieval using ChromaDB/FAISS
- **User-Friendly Interface**: Clean Streamlit-based web interface

##  Installation & Setup

### 1. Create Virtual Environment & Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root and add the following configuration:

```env
# LLM model to use (must be downloaded with ollama pull)
LLM_MODEL=llama2

# Directory for storing index files
VECTOR_STORE_PATH=./data/vector_store
```

**Note**: This project requires **no cloud API keys** - everything runs locally!

### 3. Prerequisites

Make sure you have Ollama installed and the required model downloaded:

```bash
# Install Ollama (visit https://ollama.ai for installation instructions)
# Download the model
ollama pull llama2
```

### 4. Launch the Application

```bash
streamlit run app.py
```

##  Usage Guide

### PDF Upload
1. Navigate to the **PDF Upload** tab
2. Select your scientific paper or document
3. Click **"Upload & Process"** to process the document

### Question-Answering
1. Go to the **Q&A** tab
2. Type your question about the document in the text box
3. Click **"Submit"**
4. SciAgent will perform chunk retrieval and generate a grounded answer using Llama 2

### Document Summarization
1. Navigate to the **Summary** tab
2. Get an instant comprehensive summary of your document

### Detailed Search
1. Use the **Detailed Search** tab
2. Enter specific keywords or concepts
3. View semantically relevant text chunks from your document

##  Project Structure

```
sciagent/
│
├── .env                      # Environment configuration
├── requirements.txt          # Python dependencies
├── README.md                # This file
│
├── app.py                   # Streamlit web interface
│
├── sci_agent/
│   ├── config.py            # Environment & model settings
│   ├── document_processor.py # PDF → text chunks processing
│   ├── embeddings.py        # Chunk → embedding conversion
│   ├── retriever.py         # Dense + token re-ranking retrieval
│   ├── chain.py             # LangChain RAG chains
│   └── agent.py             # LangGraph agent workflow
│
└── data/
    ├── raw/                 # Uploaded PDF files
    ├── processed/           # Processed chunks & intermediate data
    └── vector_store/        # ChromaDB / FAISS storage directory
```

##  Technical Stack

- **Frontend**: Streamlit
- **LLM**: Ollama (Local)
- **Embeddings**: Local embedding models
- **Vector Database**: ChromaDB / FAISS
- **Framework**: LangChain + LangGraph
- **Document Processing**: PDF parsing and chunking
- **Language**: Python

##  Key Components

- **Document Processor**: Handles PDF parsing and text chunking
- **Embeddings Engine**: Converts text chunks to vector embeddings
- **Retriever**: Implements dense retrieval with token re-ranking
- **RAG Chain**: Orchestrates retrieval and generation pipeline
- **Agent Workflow**: Manages complex query processing flows

##  Getting Started

1. Clone the repository
2. Follow the installation steps above
3. Ensure Ollama is running with your chosen model
4. Launch the application with `streamlit run app.py`
5. Upload a PDF and start asking questions!

##  Requirements

- Python 3.8+
- Ollama installed and running
- Sufficient disk space for vector storage
- RAM requirements depend on chosen LLM model

##  Contributing

Feel free to contribute to this project by:
- Reporting bugs
- Suggesting new features
- Submitting pull requests
- Improving documentation


##  Support

If you encounter any issues:
1. Check that Ollama is running and the model is downloaded
2. Verify environment variables are correctly set
3. Ensure all dependencies are installed
4. Check the logs for detailed error messages

---

**SciAgent** - Making scientific document analysis accessible and powerful through local AI technology.
