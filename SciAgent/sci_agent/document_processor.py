import os
from pathlib import Path
from typing import List, Optional
import uuid
import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # Updated import
from langchain_core.documents import Document

from sci_agent.config import (
    PDF_STORAGE_PATH,
    PROCESSED_DATA_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    LOG_LEVEL
)

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)


def save_uploaded_pdf(pdf_file_buffer, original_filename: Optional[str] = None) -> str:
    """
    Saves the uploaded PDF file and returns the file path.

    Args:
        pdf_file_buffer: The buffer of the PDF file uploaded via Streamlit (e.g., BytesIO)
        original_filename: The original filename (to preserve extension)

    Returns:
        str: The path to the saved PDF file
    """
    if not original_filename:
        filename = f"{uuid.uuid4()}.pdf"
    else:
        # Sanitize filename and ensure it has a .pdf extension
        base, ext = os.path.splitext(original_filename)
        safe_base = "".join(c if c.isalnum() or c in (' ', '.', '-') else '_' for c in base)
        filename = f"{safe_base}_{uuid.uuid4().hex[:8]}.pdf"

    file_path = PDF_STORAGE_PATH / filename
    PDF_STORAGE_PATH.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    try:
        with open(file_path, "wb") as f:
            f.write(pdf_file_buffer.getvalue())
        logger.info(f"PDF file successfully saved: {file_path}")
        return str(file_path)
    except Exception as e:
        logger.error(f"Error saving PDF file ({file_path}): {e}")
        raise


def load_pdf(file_path: str) -> List[Document]:
    """
    Loads the PDF file and returns a list of LangChain Document objects.
    """
    logger.info(f"Loading PDF: {file_path}")
    if not Path(file_path).exists():
        logger.error(f"PDF file not found: {file_path}")
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    loader = PyPDFLoader(file_path)
    try:
        documents = loader.load()
    except Exception as e:
        logger.error(f"Error loading PDF with PyPDFLoader: {e} (File: {file_path})")
        raise

    file_name = Path(file_path).name
    for doc in documents:
        doc.metadata["source"] = file_name
        doc.metadata["file_path"] = file_path
        if "page" not in doc.metadata:
            logger.warning(f"Missing 'page' metadata for page {doc.metadata.get('page_number', 'unknown')} in '{file_name}'.")

    logger.info(f"{len(documents)} pages loaded: {file_name}")
    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Splits Document objects into smaller chunks.
    """
    if not documents:
        logger.warning("No documents found to split.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # Common separators
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)

    # Add chunk_id for better tracking
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{chunk.metadata.get('source', 'unknown')}_p{chunk.metadata.get('page', 'N')}_c{i}"

    logger.info(f"{len(chunks)} chunks created.")
    return chunks


def process_pdf(file_path: str) -> List[Document]:
    """
    Loads, processes, and splits the PDF into text chunks.
    Also saves the processed chunks to a text file (for debugging).
    """
    logger.info(f"Processing PDF: {file_path}")
    documents = load_pdf(file_path)
    chunks = split_documents(documents)

    if chunks:
        file_stem = Path(file_path).stem
        output_filename = f"{file_stem}_chunks.txt"
        output_path = PROCESSED_DATA_PATH / output_filename
        PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for chunk in chunks:
                    f.write(f"--- Chunk ID: {chunk.metadata.get('chunk_id', 'N/A')} ---\n")
                    f.write(f"Source: {chunk.metadata.get('source', 'N/A')}, Page: {chunk.metadata.get('page', 'N/A')}\n")
                    f.write(chunk.page_content)
                    f.write("\n\n--------------------\n\n")
            logger.info(f"Processed chunks saved: {output_path}")
        except Exception as e:
            logger.error(f"Error saving processed chunks ({output_path}): {e}")

    return chunks
