import os
from typing import List, Optional
import logging

from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings
from langchain_community.vectorstores import Chroma, FAISS

from sci_agent.config import (
    EMBEDDING_MODEL_TYPE,
    EMBEDDING_MODEL_NAME,
    VECTOR_STORE_PATH,
    COLLECTION_NAME as DEFAULT_COLLECTION_NAME,
    VECTOR_DB_TYPE,
    OLLAMA_HOST,
    OLLAMA_MODEL as OLLAMA_EMBED_MODEL,
    LOG_LEVEL
)

# Logging setup
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)


def get_embeddings_model() -> Embeddings:
    """
    Initializes and returns the embedding model based on configuration.
    """
    logger.info(
        f"Loading embedding model: Type={EMBEDDING_MODEL_TYPE}, "
        f"Model Name/ID={EMBEDDING_MODEL_NAME if EMBEDDING_MODEL_TYPE == 'sentence_transformers' else OLLAMA_EMBED_MODEL}"
    )
    if EMBEDDING_MODEL_TYPE == "sentence_transformers":
        try:
            return HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={'device': 'cpu'},  # Consider 'cuda' if GPU available
                encode_kwargs={'normalize_embeddings': True}  # Often recommended
            )
        except ImportError:
            msg = "Failed to load HuggingFaceEmbeddings. Please run 'pip install sentence-transformers langchain-community'."
            logger.error(msg)
            raise ImportError(msg)
        except Exception as e:
            logger.error(f"Error loading SentenceTransformers embedding model ({EMBEDDING_MODEL_NAME}): {e}")
            raise

    elif EMBEDDING_MODEL_TYPE == "ollama":
        try:
            return OllamaEmbeddings(
                model=OLLAMA_EMBED_MODEL,
                base_url=OLLAMA_HOST
            )
        except ImportError:
            msg = "Failed to load OllamaEmbeddings. Please run 'pip install langchain-community'."
            logger.error(msg)
            raise ImportError(msg)
        except Exception as e:
            logger.error(f"Error loading Ollama embedding model ({OLLAMA_EMBED_MODEL}): {e}")
            raise
    else:
        msg = f"Unsupported embedding model type: {EMBEDDING_MODEL_TYPE}"
        logger.error(msg)
        raise ValueError(msg)


def get_vector_store_path(collection_name: str) -> str:
    """Helper to get the full path for a given collection."""
    return str(VECTOR_STORE_PATH / collection_name)


def create_vector_store(documents: List[Document], collection_name: Optional[str] = None):
    """
    Creates a vector store from document chunks.
    """
    current_collection_name = collection_name if collection_name else DEFAULT_COLLECTION_NAME
    if not documents:
        logger.warning(f"Empty document list for '{current_collection_name}', no vector store will be created.")
        return None

    logger.info(f"Creating vector store for '{current_collection_name}' ({VECTOR_DB_TYPE})...")
    embedding_model = get_embeddings_model()
    db_path = get_vector_store_path(current_collection_name)
    VECTOR_STORE_PATH.mkdir(parents=True, exist_ok=True)  # Ensure base path exists

    if VECTOR_DB_TYPE == "chroma":
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory=db_path
        )
        logger.info(f"Chroma vector store created and saved: {db_path}")
    elif VECTOR_DB_TYPE == "faiss":
        vector_store = FAISS.from_documents(
            documents=documents,
            embedding=embedding_model
        )
        vector_store.save_local(db_path)
        logger.info(f"FAISS vector store created and saved: {db_path}")
    else:
        raise ValueError(f"Unsupported vector database type: {VECTOR_DB_TYPE}")
    return vector_store


def load_vector_store(collection_name: Optional[str] = None):
    """
    Loads an existing vector store.
    """
    current_collection_name = collection_name if collection_name else DEFAULT_COLLECTION_NAME
    db_path = get_vector_store_path(current_collection_name)

    if not os.path.exists(db_path):
        logger.warning(f"Vector store path not found: {db_path}. Cannot load.")
        return None

    logger.info(f"Loading vector store for '{current_collection_name}' ({VECTOR_DB_TYPE})...")
    embedding_model = get_embeddings_model()

    try:
        if VECTOR_DB_TYPE == "chroma":
            vector_store = Chroma(
                persist_directory=db_path,
                embedding_function=embedding_model
            )
            logger.info(f"Chroma vector store loaded: {db_path}")
        elif VECTOR_DB_TYPE == "faiss":
            try:
                vector_store = FAISS.load_local(
                    db_path,
                    embeddings=embedding_model,
                    allow_dangerous_deserialization=True  # Add this for safety with pickled FAISS indexes
                )
            except TypeError:
                logger.warning("FAISS.load_local does not support 'allow_dangerous_deserialization' argument. Trying without it.")
                vector_store = FAISS.load_local(db_path, embeddings=embedding_model)
            logger.info(f"FAISS vector store loaded: {db_path}")
        else:
            raise ValueError(f"Unsupported vector database type: {VECTOR_DB_TYPE}")
        return vector_store
    except Exception as e:
        logger.error(f"Error loading vector store ({db_path}): {e}")
        return None


def add_documents_to_vector_store(
    documents: List[Document],
    collection_name: Optional[str] = None
):
    """
    Adds new documents to an existing vector store. Creates one if it doesn't exist.
    """
    current_collection_name = collection_name if collection_name else DEFAULT_COLLECTION_NAME
    if not documents:
        logger.warning(f"No documents to add for '{current_collection_name}'.")
        return

    vector_store = load_vector_store(current_collection_name)

    if vector_store:
        logger.info(f"Adding {len(documents)} documents to existing '{current_collection_name}' vector store...")
        vector_store.add_documents(documents)
        if VECTOR_DB_TYPE == "faiss":  # FAISS needs explicit save after adding
            db_path = get_vector_store_path(current_collection_name)
            vector_store.save_local(db_path)
            logger.info(f"FAISS vector store updated and saved: {db_path}")
        elif VECTOR_DB_TYPE == "chroma":
            logger.info("Documents added to Chroma vector store.")
    else:
        logger.info(f"Vector store for '{current_collection_name}' not found, creating a new one...")
        create_vector_store(documents, current_collection_name)


def query_vector_store(
    query: str,
    collection_name: Optional[str] = None,
    n_results: int = 5
) -> List[Document]:
    """
    Retrieves relevant documents from the vector store for a given query.
    """
    current_collection_name = collection_name if collection_name else DEFAULT_COLLECTION_NAME
    logger.debug(f"Querying '{current_collection_name}' collection with '{query[:50]}...', requesting {n_results} results.")

    vector_store = load_vector_store(current_collection_name)
    if not vector_store:
        logger.warning(f"Vector store '{current_collection_name}' could not be loaded. Returning empty list.")
        return []
    try:
        relevant_docs = vector_store.similarity_search(query, k=n_results)
        logger.info(f"Found {len(relevant_docs)} relevant documents for query '{query[:50]}...'.")
        return relevant_docs
    except Exception as e:
        logger.error(f"Error performing query on vector store '{current_collection_name}': {e}")
        return []
