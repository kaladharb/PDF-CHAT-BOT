import os
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Paths
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

# Ensure vectorstore folder exists
os.makedirs(DB_FAISS_PATH, exist_ok=True)

# ------------------------
# Helper functions
# ------------------------

def load_pdf_files(data_path):
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(documents)

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_vectorstore():
    """
    Loads FAISS if exists; otherwise builds it from PDFs.
    """
    embedding_model = get_embedding_model()
    index_path = os.path.join(DB_FAISS_PATH, "index.faiss")

    if os.path.exists(index_path):
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    else:
        # st.info("🚀 Building FAISS vectorstore from PDFs...")
        documents = load_pdf_files(DATA_PATH)
        chunks = create_chunks(documents)
        db = FAISS.from_documents(chunks, embedding_model)
        db.save_local(DB_FAISS_PATH)
        # st.success("✅ FAISS vectorstore built and saved")
    return db

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# ------------------------
# Streamlit App
# ------------------------

st.title("Ask AI Health Assistant!!")

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

prompt = st.chat_input("Pass your prompt here")

if prompt:
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    CUSTOM_PROMPT_TEMPLATE = """
    Use the pieces of information provided in the context to answer user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Don't provide anything out of the given context.

    Context: {context}
    Question: {question}

    Start the answer directly. No small talk please.
    """

    try:
        vectorstore = get_vectorstore()
        if vectorstore is None:
            st.error("❌ Failed to load the vector store")

        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatGroq(
                model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
                temperature=0.0,
                groq_api_key=st.secrets["GROQ_API_KEY"],
            ),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
        )

        response = qa_chain.invoke({'query': prompt})

        result = response["result"]
        source_documents = response["source_documents"]

        # Show answer
        st.chat_message('assistant').markdown(result)

        # Expandable sources
        with st.expander("Show Sources"):
            for i, doc in enumerate(source_documents, 1):
                st.markdown(f"**Source {i}** - Page {doc.metadata.get('page', 'N/A')}")
                st.markdown(doc.page_content)

        st.session_state.messages.append({'role': 'assistant', 'content': result})

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")