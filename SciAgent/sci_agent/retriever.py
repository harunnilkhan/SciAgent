from typing import List, Optional
import re
import logging

from langchain_core.documents import Document

from sci_agent.embeddings import query_vector_store
from sci_agent.config import TOP_K_RESULTS, COLLECTION_NAME as DEFAULT_COLLECTION_NAME, LOG_LEVEL

# Logging setup
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)


def optimize_query(query: str) -> str:
    """
    Normalizes the incoming query for general text searches by:
      - Converting to lowercase
      - Removing punctuation
      - Consolidating multiple spaces and trimming
    There are no longer any domain-specific keyword additions or checks.
    """
    # 1) Convert to lowercase
    q = query.lower()

    # 2) Remove punctuation
    q = re.sub(r'[^\w\s]', '', q)

    # 3) Consolidate spaces and trim
    q = re.sub(r'\s+', ' ', q).strip()

    logger.debug(f"Normalized query: '{query}' -> '{q}'")
    return q


def clean_text(text: str) -> str:
    """
    Cleans and formats text.
    """
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def retrieve_relevant_documents(query: str, collection_name: Optional[str] = None) -> List[Document]:
    """
    Retrieves the most relevant document chunks for a given query.
    """
    current_collection_name = collection_name if collection_name else DEFAULT_COLLECTION_NAME
    optimized_query = optimize_query(query)
    logger.info(f"Retrieving relevant documents for '{optimized_query}' from '{current_collection_name}' collection.")

    relevant_docs = query_vector_store(
        query=optimized_query,
        collection_name=current_collection_name,
        n_results=TOP_K_RESULTS
    )
    return relevant_docs


def format_retrieved_documents(docs: List[Document]) -> str:
    """
    Converts retrieved documents into a readable format.
    """
    if not docs:
        return "No relevant document chunks found."

    formatted_docs = []
    for i, doc in enumerate(docs):
        metadata = doc.metadata or {}
        content = clean_text(doc.page_content)

        source = metadata.get("source", "Unknown source")
        page = metadata.get("page", "Unknown page")
        chunk_id = metadata.get("chunk_id", f"c{i}")

        header = (f"### Document Chunk {i+1} (ID: {chunk_id})\n"
                  f"**Source:** {source}, **Page:** "
                  f"{page if not isinstance(page, int) else page + 1}")
        formatted_docs.append(f"{header}\n\n{content}\n")

    return "\n---\n\n".join(formatted_docs)


def retrieve_and_format_documents(query: str, collection_name: Optional[str] = None) -> str:
    """
    Retrieves and formats relevant document chunks for a given query.
    """
    docs = retrieve_relevant_documents(query, collection_name)
    return format_retrieved_documents(docs)


class BasicRetriever:
    """
    A simple document retriever for easier use in LangChain chains.
    """
    def __init__(self, collection_name: Optional[str] = None):
        self.collection_name = collection_name if collection_name else DEFAULT_COLLECTION_NAME

    def get_relevant_documents(self, query: str) -> List[Document]:
        return retrieve_relevant_documents(query, self.collection_name)

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)


def get_basic_retriever(collection_name: Optional[str] = None):
    """
    Creates an instance of the basic document retriever.
    """
    return BasicRetriever(collection_name)
