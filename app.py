import os
import shutil
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# --- Configuration ---
st.set_page_config(page_title="DocuBot AI 🤖", page_icon="🤖", layout="wide")

# Define paths
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

# --- Helper Functions ---
def clear_vector_store():
    """Clears the existing vector store and data directory."""
    if os.path.exists(DB_FAISS_PATH):
        shutil.rmtree(DB_FAISS_PATH)
    if os.path.exists(DATA_PATH):
        shutil.rmtree(DATA_PATH)
    os.makedirs(DATA_PATH, exist_ok=True)
    st.toast("🧹 Cleared old data and vector store!", icon="✨")

def save_uploaded_files(uploaded_files):
    """Saves uploaded files to the data directory."""
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    for file in uploaded_files:
        with open(os.path.join(DATA_PATH, file.name), "wb") as f:
            f.write(file.getbuffer())

def build_vector_store():
    """Builds the FAISS vector store from PDF documents."""
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    if not documents:
        st.warning("No PDF documents found. Please upload at least one PDF.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)
    return db

def get_qa_chain():
    """Creates and returns the RetrievalQA chain."""
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

    CUSTOM_PROMPT_TEMPLATE = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer from the context, just say that you don't know, don't try to make up an answer.
    Provide a concise and direct answer.

    Context: {context}
    Question: {question}

    Helpful Answer:"""

    prompt = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0.0,
            groq_api_key=st.secrets["GROQ_API_KEY"],
        ),
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

# --- Streamlit UI ---
with st.sidebar:
    st.title("📄 DocuBot Controls")
    st.markdown("Upload your PDF documents and click 'Process' to get started.")

    uploaded_files = st.file_uploader(
        "Upload your PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files."
    )

    if st.button("Process Documents", use_container_width=True):
        if uploaded_files:
            with st.spinner("Processing documents... This may take a moment."):
                clear_vector_store()
                save_uploaded_files(uploaded_files)
                build_vector_store()
                st.session_state.is_processed = True
                st.success("✅ Documents processed and ready!")
                st.toast("You can now ask questions about your documents.", icon="🎉")
        else:
            st.warning("Please upload at least one PDF file.")

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.toast("Chat history cleared!", icon="🧹")

# --- Main Chat ---
st.title("🤖 DocuBot AI: Chat with Your PDFs")
st.markdown("Ask any question about the content of your uploaded documents.")

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'is_processed' not in st.session_state:
    st.session_state.is_processed = os.path.exists(DB_FAISS_PATH)

if not st.session_state.messages:
    st.info("Upload your PDFs and click 'Process Documents' to begin the conversation.")

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# Handle user input
if prompt := st.chat_input("Ask a question about your documents..."):
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    if not st.session_state.is_processed:
        st.warning("Please upload and process your documents before asking questions.")
    else:
        try:
            with st.spinner("Thinking..."):
                qa_chain = get_qa_chain()
                response = qa_chain.invoke({'query': prompt})
                result = response["result"]
                source_documents = response["source_documents"]

                with st.chat_message('assistant'):
                    st.markdown(result)

                    with st.expander("📚 Show Sources"):
                        for i, doc in enumerate(source_documents, 1):
                            st.markdown(f"**Source {i}** - Page: {doc.metadata.get('page', 'N/A')}")
                            st.info(doc.page_content)

                st.session_state.messages.append({'role': 'assistant', 'content': result})

        except Exception as e:
            st.error(f"⚠️ An error occurred: {str(e)}")
            st.exception(e)
