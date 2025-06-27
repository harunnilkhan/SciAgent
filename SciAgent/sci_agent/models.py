from typing import Optional, List, Dict, Any
import logging

from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama

from sci_agent.config import (
    LLM_PROVIDER,
    OPENAI_API_KEY, OPENAI_LLM_MODEL,
    OLLAMA_HOST, OLLAMA_MODEL,
    TEMPERATURE,
    LOG_LEVEL
)

# Logging setup
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)


# Basic text truncation (not token-based, for rough limiting)
def truncate_text(text: str, max_length: int = 4000) -> str:
    """
    Truncates text to a maximum character length.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def get_llm(temperature: Optional[float] = None) -> BaseLanguageModel:
    """
    Returns the appropriate LLM model based on the selected provider.
    """
    current_temp = temperature if temperature is not None else TEMPERATURE
    logger.info(
        f"Loading LLM: Provider={LLM_PROVIDER}, "
        f"Model={(OLLAMA_MODEL if LLM_PROVIDER == 'ollama' else OPENAI_LLM_MODEL if LLM_PROVIDER == 'openai' else 'HF_MODEL')}, "
        f"Temperature={current_temp}"
    )

    if LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY or OPENAI_API_KEY == "YOUR_OPENAI_API_KEY_IF_YOU_USE_IT":
            raise ValueError("OpenAI API key (OPENAI_API_KEY) is not set. Please check your .env file or config.py.")
        return ChatOpenAI(
            model=OPENAI_LLM_MODEL,
            temperature=current_temp,
            openai_api_key=OPENAI_API_KEY
        )
    elif LLM_PROVIDER == "ollama":
        try:
            return Ollama(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_HOST,
                temperature=current_temp,
            )
        except Exception as e:
            logger.error(f"Error loading/connecting to Ollama LLM ({OLLAMA_MODEL} @ {OLLAMA_HOST}): {e}")
            logger.error("Please ensure the Ollama server is running and the model is downloaded (e.g., 'ollama run llama2').")
            raise
    else:
        raise ValueError(f"Unsupported LLM provider: {LLM_PROVIDER}")


# Example of formatting document context (can be adapted)
def format_document_context_for_llm(docs: List[Dict[str, Any]], max_tokens_estimate: int = 1800) -> str:
    """
    Formats a list of documents into a single string for LLM input.
    """
    if not docs:
        return "No relevant documents found."

    context_parts = []
    current_char_count = 0
    max_chars_estimate = max_tokens_estimate * 3.5

    for i, doc_data in enumerate(docs):
        content = doc_data.get("page_content", doc_data.get("content", ""))
        metadata = doc_data.get("metadata", {})

        source = metadata.get("source", "Unknown source")
        page = metadata.get("page", "Unknown page")

        doc_text = f"DOCUMENT {i+1}:\nSource: {source}, Page: {page}\nContent: {content}\n\n"

        if current_char_count + len(doc_text) > max_chars_estimate and i > 0:
            logger.warning(f"Reached context token limit ({max_chars_estimate} characters), added {i}/{len(docs)} documents.")
            break

        context_parts.append(doc_text)
        current_char_count += len(doc_text)

    full_context = "".join(context_parts)
    if not full_context:
        return "Could not build document context."
    return full_context
