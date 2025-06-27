import logging
from operator import itemgetter
from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from sci_agent.models import get_llm
from sci_agent.retriever import get_basic_retriever  # Uses BasicRetriever class
from sci_agent.config import LOG_LEVEL, COLLECTION_NAME as DEFAULT_COLLECTION_NAME

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)


def format_docs_for_prompt(docs: List[Document]) -> str:
    """
    Converts documents into a format suitable for LLM prompts.
    """
    if not docs:
        return "No context information available."
    return "\n\n".join(
        f"[DOCUMENT SOURCE: {doc.metadata.get('source', 'Unknown')}, PAGE: {doc.metadata.get('page', 'Unknown')}, CHUNK ID: {doc.metadata.get('chunk_id', 'N/A')}]\nCONTENT:\n{doc.page_content}"
        for doc in docs
    )


def create_qa_chain(collection_name: str = None):
    """
    Creates the question-answering chain.
    """
    current_collection_name = collection_name if collection_name else DEFAULT_COLLECTION_NAME
    logger.info(f"Creating QA chain for collection '{current_collection_name}'.")

    try:
        retriever_instance = get_basic_retriever(current_collection_name)
    except Exception as e:
        logger.error(f"Error creating retriever: {e}. Using empty retriever.")

        # Fallback dummy retriever
        class EmptyRetriever:
            def get_relevant_documents(self, query: str) -> List[Document]: return []

            async def aget_relevant_documents(self, query: str) -> List[Document]: return []

        retriever_instance = EmptyRetriever()

    llm = get_llm()

    # QA Prompt Template
    qa_system_prompt = """You are a scientific research assistant. Your task is to answer the USER QUESTION using the provided CONTEXT information.
Your answers should be direct and based only on the information in the CONTEXT.
If the CONTEXT does not contain the answer, respond with "The provided documents did not contain this information."
Do not guess or use information outside the CONTEXT. Use expressions from the CONTEXT as much as possible.
For each piece of information used, cite the DOCUMENT SOURCE, PAGE, and CHUNK ID in the format [SOURCE: ..., PAGE: ..., CHUNK ID: ...].
For example: "Scoliosis is a lateral curvature of the spine [SOURCE: paper1.pdf, PAGE: 2, CHUNK ID: paper1.pdf_p2_c5]."
"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "CONTEXT:\n{context}\n\nUSER QUESTION: {question}\n\nYOUR ANSWER:")
        ]
    )

    rag_chain = (
        {
            "context": itemgetter("question") | RunnableLambda(
                retriever_instance.get_relevant_documents) | RunnableLambda(format_docs_for_prompt),
            "question": itemgetter("question"),  # Pass the question through
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def create_summary_chain(collection_name: str = None):
    """
    Creates the summarization chain.
    """
    current_collection_name = collection_name if collection_name else DEFAULT_COLLECTION_NAME
    logger.info(f"Creating summary chain for collection '{current_collection_name}'.")

    try:
        retriever_instance = get_basic_retriever(current_collection_name)
    except Exception as e:
        logger.error(f"Error creating retriever: {e}. Using empty retriever.")

        class EmptyRetriever:
            def get_relevant_documents(self, query: str) -> List[Document]: return []

            async def aget_relevant_documents(self, query: str) -> List[Document]: return []

        retriever_instance = EmptyRetriever()

    llm = get_llm()

    # Summary Prompt Template
    summary_system_prompt = """You are a scientific research assistant. Your task is to create a comprehensive summary of the document using the provided DOCUMENT CHUNKS.
The summary should include the main ideas, key findings, and conclusions. Highlight important points.
Write the summary in fluent paragraphs. Avoid using bullet points.
If there is insufficient information or the documents are incoherent, respond with "A meaningful summary could not be generated from the documents."
"""
    summary_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", summary_system_prompt),
            ("human", "DOCUMENT CHUNKS:\n{context}\n\nCOMPREHENSIVE SUMMARY:")
        ]
    )

    fixed_summary_query = "information about the paper's general content, objectives, methods, and results"

    summarization_chain = (
        RunnableLambda(
            lambda _: retriever_instance.get_relevant_documents(fixed_summary_query))  # Input to retriever
        | RunnableLambda(format_docs_for_prompt)  # Format docs
        | (lambda formatted_docs: {"context": formatted_docs})  # Prepare for prompt
        | summary_prompt
        | llm
        | StrOutputParser()
    )
    return summarization_chain
