import sys
import importlib
import streamlit as st
import logging
from pathlib import Path


splitter_found = False
for pkg_path, class_name in [
    ("langchain_text_splitters", "RecursiveCharacterTextSplitter"),
    ("langchain.text_splitter", "RecursiveCharacterTextSplitter"),  # Older paths
    ("langchain_core.text_splitter", "RecursiveCharacterTextSplitter"),
    ("langchain_community.text_splitter", "RecursiveCharacterTextSplitter"),
]:
    try:
        module = importlib.import_module(pkg_path)
        RecursiveCharacterTextSplitter = getattr(module, class_name)
        # Make it available for other modules if they expect it at a common path
        sys.modules["langchain_text_splitters"] = module  # Preferred alias
        splitter_found = True
        break
    except (ImportError, AttributeError):
        continue

if not splitter_found:
    st.error(
        "⚠️ `RecursiveCharacterTextSplitter` not found. "
        "Please install `langchain-text-splitters` or the appropriate `langchain` package."
    )
    st.stop()

# --- Project Imports ---
try:
    from sci_agent.config import (
        APP_TITLE, APP_DESCRIPTION,
        COLLECTION_NAME as DEFAULT_COLLECTION_NAME,
        LOG_LEVEL
    )
    from sci_agent.document_processor import save_uploaded_pdf, process_pdf
    from sci_agent.embeddings import add_documents_to_vector_store  # query_vector_store is used by retriever
    from sci_agent.retriever import retrieve_and_format_documents
    from sci_agent.chain import create_qa_chain, create_summary_chain
except ImportError as e:
    st.error(f"An error occurred while loading the required modules: {e}. Please check your installation.")
    st.exception(e)
    st.stop()

# --- Logging Setup ---
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🤖",
        layout="wide",
    )
    st.title(f"📚 {APP_TITLE}")
    st.markdown(APP_DESCRIPTION)
    st.markdown("---")

    # Initialize session state for collection name if not already present
    if 'collection_name' not in st.session_state:
        st.session_state.collection_name = DEFAULT_COLLECTION_NAME
    if 'file_uploader_key' not in st.session_state:
        st.session_state.file_uploader_key = 0  # To reset file uploader

    # --- Sidebar: PDF Upload & Database Management ---
    with st.sidebar:
        st.header("📄 Upload PDF & Database")

        new_collection_name = st.text_input(
            "Database Collection Name:",
            value=st.session_state.collection_name,
            help="The collection where documents will be stored and searched. If you change it, a new database will be used/created."
        )
        if new_collection_name != st.session_state.collection_name:
            st.session_state.collection_name = new_collection_name
            st.success(f"Active collection changed to '{new_collection_name}'.")

        uploaded_file = st.file_uploader(
            f"Upload PDF to '{st.session_state.collection_name}' collection:",
            type="pdf",
            key=f"file_uploader_{st.session_state.file_uploader_key}"
        )

        if uploaded_file:
            if st.button(f"📥 Process and Add '{Path(uploaded_file.name).name}'"):
                with st.spinner(
                        f"Processing '{Path(uploaded_file.name).name}' and adding to '{st.session_state.collection_name}' collection..."):
                    try:
                        saved_pdf_path = save_uploaded_pdf(uploaded_file, original_filename=uploaded_file.name)
                        logger.info(f"PDF '{uploaded_file.name}' temporarily saved: {saved_pdf_path}")

                        chunks = process_pdf(saved_pdf_path)
                        if not chunks:
                            st.warning("No text chunks could be extracted from the PDF.")
                        else:
                            logger.info(f"{len(chunks)} chunks created.")
                            add_documents_to_vector_store(chunks, st.session_state.collection_name)
                            st.success(
                                f"{len(chunks)} chunks successfully added to '{st.session_state.collection_name}' collection.")
                            st.session_state.file_uploader_key += 1
                            st.rerun()

                    except Exception as e:
                        st.error(f"An error occurred while processing the PDF: {e}")
                        logger.error(f"PDF processing error: {e}", exc_info=True)
                        st.exception(e)
        else:
            st.info("Please upload a PDF file to perform analysis or ask questions.")

        st.markdown("---")
        st.caption(f"Active Collection: **{st.session_state.collection_name}**")

    # --- Main Content Tabs ---
    st.header("Processing Options")
    qa_tab, sum_tab, search_tab = st.tabs(
        ["📝 Ask Question", "📋 Create Summary", "🔍 Detailed Search"]
    )

    current_collection = st.session_state.collection_name

    # Q&A Tab
    with qa_tab:
        st.subheader(f"Ask a Question from the '{current_collection}' Collection")
        question = st.text_input("Enter your question here:", key="qa_question",
                                 placeholder="E.g.: What are the main findings of this paper?")

        if st.button("💬 Answer", key="qa_button", use_container_width=True):
            if not question:
                st.warning("Please enter a question.")
            elif not current_collection:
                st.warning("Please select or create a collection name from the sidebar.")
            else:
                with st.spinner("Generating answer... (The LLM is working, this may take a moment)"):
                    try:
                        qa_chain_instance = create_qa_chain(current_collection)
                        input_payload = {"question": question}

                        answer = qa_chain_instance.invoke(input_payload)
                        st.markdown("### Answer:")
                        st.markdown(answer)

                    except Exception as e:
                        st.error(f"An error occurred during the Q&A process: {e}")
                        logger.error(f"QA error: {e}", exc_info=True)
                        st.exception(e)

    # Summarization Tab
    with sum_tab:
        st.subheader(f"Summary for Article(s) in the '{current_collection}' Collection")
        if st.button("📄 Create Summary", key="sum_button", use_container_width=True):
            if not current_collection:
                st.warning("Please select or create a collection name from the sidebar.")
            else:
                with st.spinner("Generating summary... (The LLM is working, this may take a moment)"):
                    try:
                        summary_chain_instance = create_summary_chain(current_collection)
                        summary = summary_chain_instance.invoke({})
                        st.markdown("### Summary:")
                        st.markdown(summary)
                    except Exception as e:
                        st.error(f"An error occurred while creating the summary: {e}")
                        logger.error(f"Summary error: {e}", exc_info=True)
                        st.exception(e)

    # Detailed Search Tab
    with search_tab:
        st.subheader(f"Detailed Search in the '{current_collection}' Collection (Vector Similarity)")
        search_query = st.text_input("Enter search terms:", key="search_query",
                                     placeholder="E.g.: idiopathic scoliosis treatment")
        if st.button("🔎 Search", key="search_button", use_container_width=True):
            if not search_query:
                st.warning("Please enter a search term.")
            elif not current_collection:
                st.warning("Please select or create a collection name from the sidebar.")
            else:
                with st.spinner("Searching documents..."):
                    try:
                        results = retrieve_and_format_documents(search_query, current_collection)
                        st.markdown("### Search Results (Top Similar Chunks):")
                        if results == "No relevant document chunks found.":
                            st.info(results)
                        else:
                            st.markdown(results)
                    except Exception as e:
                        st.error(f"An error occurred during the search process: {e}")
                        logger.error(f"Search error: {e}", exc_info=True)
                        st.exception(e)


if __name__ == "__main__":
    main()
