import os
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# Root directory of the project
ROOT_DIR = Path(__file__).parent.parent.resolve()

# --- LLM Provider Configuration ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_IF_YOU_USE_IT")
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-3.5-turbo")

# HuggingFace Settings
HF_API_KEY = os.getenv("HF_API_KEY")
HF_LLM_MODEL = os.getenv("HF_LLM_MODEL", "distilgpt2")

# Ollama Settings
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama2")

# --- Embedding Model Configuration ---
EMBEDDING_MODEL_TYPE = os.getenv("EMBEDDING_MODEL_TYPE", "sentence_transformers").lower()
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# --- Storage Path Configuration ---
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

VECTOR_STORE_DIR_NAME = "vector_store"
VECTOR_STORE_PATH = DATA_DIR / VECTOR_STORE_DIR_NAME
VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)

PDF_STORAGE_DIR_NAME = "raw_pdfs"
PDF_STORAGE_PATH = DATA_DIR / PDF_STORAGE_DIR_NAME
PDF_STORAGE_PATH.mkdir(parents=True, exist_ok=True)

PROCESSED_DATA_DIR_NAME = "processed_texts"
PROCESSED_DATA_PATH = DATA_DIR / PROCESSED_DATA_DIR_NAME
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

# --- LLM & Processing Parameters ---
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.1))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# --- Vector Database Configuration ---
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "sciagent_ollama_docs")

VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "chroma").lower()

# --- Agent Configuration (If using LangGraph agent.py) ---
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", 10))

# --- Retrieval Configuration ---
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", 5))

# --- Application UI Configuration ---
APP_TITLE = "SciAgent - Scientific Paper Assistant"
APP_DESCRIPTION = (
    "SciAgent is an AI assistant that allows you to upload scientific papers, "
    "ask questions about them, and create summaries. "
    "This version uses local Ollama models by default."
)

# Log level
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
