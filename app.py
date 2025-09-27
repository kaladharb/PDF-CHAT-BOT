import os
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- PATHS AND CONFIG ---
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

# Ensure vectorstore folder exists
os.makedirs(DB_FAISS_PATH, exist_ok=True)

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Health Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS FOR STYLING ---
def load_css():
    st.markdown("""
    <style>
        /* Main app background */
        .stApp {
            background-color: #1E1E1E;
            color: #E0E0E0;
        }

        /* Chat messages styling */
        .st-emotion-cache-1c7y2kd { /* This class might change with Streamlit updates */
            border-radius: 20px;
            padding: 1rem 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
            transition: 0.3s;
        }

        /* User message styling */
        div[data-testid="stChatMessage"]:has(div[data-testid="stMarkdownContainer"] p) {
            background-color: #2F2F2F; /* Darker shade for user */
        }

        /* Assistant message styling */
        div[data-testid="stChatMessage"]:not(:has(div[data-testid="stMarkdownContainer"] p)) {
            background-color: #3C3C3C; /* Lighter shade for assistant */
        }
        
        /* Input box styling */
        .stTextInput > div > div > input {
            background-color: #2F2F2F;
            color: #E0E0E0;
            border-radius: 15px;
            border: 1px solid #4A4A4A;
        }

        /* Expander styling for sources */
        .stExpander {
            background-color: #2F2F2F;
            border-radius: 10px;
            border: 1px solid #4A4A4A;
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #2F2F2F;
        }

        /* Button styling */
        .stButton > button {
            border-radius: 10px;
            border: 1px solid #4A4A4A;
            background-color: #3C3C3C;
            color: #E0E0E0;
        }
        .stButton > button:hover {
            background-color: #4A4A4A;
            color: #FFFFFF;
        }

    </style>
    """, unsafe_allow_html=True)

load_css()


# ------------------------
# Helper functions (No changes here)
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
        with st.spinner("Building vector store from PDFs... This may take a moment."):
            documents = load_pdf_files(DATA_PATH)
            chunks = create_chunks(documents)
            db = FAISS.from_documents(chunks, embedding_model)
            db.save_local(DB_FAISS_PATH)
    return db

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# ------------------------
# Streamlit App UI
# ------------------------

# --- SIDEBAR ---
with st.sidebar:
    st.title("👨‍⚕️ AI Health Assistant")
    st.markdown("This app uses a local knowledge base to answer your health-related questions. The information provided is for educational purposes only.")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.info("⚠️ **Disclaimer:** This is not a substitute for professional medical advice.")

# --- HEADER ---
st.title("Ask Your AI Health Assistant!")
st.markdown("Get instant answers from your health documents.")

# --- CHAT INTERFACE ---
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display initial welcome message from assistant
if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🩺"):
        st.markdown("Hello! How can I help you today? Feel free to ask me anything about the provided health documents.")
        st.markdown("For example: *What are the common symptoms of diabetes?*")

# Display past messages
for message in st.session_state.messages:
    avatar = "🧑‍💻" if message['role'] == 'user' else "🩺"
    with st.chat_message(message['role'], avatar=avatar):
        st.markdown(message['content'])

# --- MAIN LOGIC ---
prompt = st.chat_input("Ask a question about your documents...")

if prompt:
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user', avatar="🧑‍💻"):
        st.markdown(prompt)

    CUSTOM_PROMPT_TEMPLATE = """
    Use the pieces of information provided in the context to answer user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Don't provide anything out of the given context.

    Context: {context}
    Question: {question}

    Start the answer directly. No small talk please.
    """

    try:
        with st.spinner("👩‍⚕️ Thinking..."):
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("❌ Failed to load the vector store")
                st.stop()

            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatGroq(
                    model_name="llama3-8b-8192", # Using a more recent model
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

            # Display assistant response
            with st.chat_message('assistant', avatar="🩺"):
                st.markdown(result)
                
                # Expandable sources
                with st.expander("📄 Show Sources"):
                    for i, doc in enumerate(source_documents, 1):
                        st.info(f"**Source {i}** - Page {doc.metadata.get('page', 'N/A')}")
                        st.markdown(f"> {doc.page_content}")

            st.session_state.messages.append({'role': 'assistant', 'content': result})

    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")