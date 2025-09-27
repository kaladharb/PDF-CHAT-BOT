import os
import streamlit as st
from dotenv import load_dotenv

# LangChain imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Gemini imports
import google.genai as genai
from langchain.llms.base import LLM
from pydantic import Field, BaseModel

# ------------------------
# Load environment variables
# ------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ------------------------
# Paths
# ------------------------
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"
os.makedirs(DB_FAISS_PATH, exist_ok=True)

# ------------------------
# Helper Functions
# ------------------------
def load_pdf_files(data_path):
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_vectorstore():
    embedding_model = get_embedding_model()
    index_path = os.path.join(DB_FAISS_PATH, "index.faiss")

    if os.path.exists(index_path):
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    else:
        documents = load_pdf_files(DATA_PATH)
        chunks = create_chunks(documents)
        db = FAISS.from_documents(chunks, embedding_model)
        db.save_local(DB_FAISS_PATH)
    return db

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# ------------------------
# Gemini LLM Wrapper (fixed for Pydantic)
# ------------------------
class GeminiLLM(LLM, BaseModel):
    client: genai.Client = Field(exclude=True)  # Exclude from Pydantic validation
    model: str = "gemini-2.5-flash"

    @property
    def _llm_type(self) -> str:
        return "gemini"

    def _call(self, prompt: str, stop=None) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        return response.text

# Initialize Gemini client
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
gemini_llm = GeminiLLM(client=gemini_client)

# ------------------------
# Streamlit Layout
# ------------------------
st.set_page_config(page_title="Med Bot", page_icon="💊", layout="wide")
st.title("💊 AI Health Assistant")
st.markdown("Ask health-related questions and get answers from the preloaded PDF!")

# Sidebar for PDF Upload / Info
with st.sidebar:
    st.header("📄 PDF Settings")
    st.write("Current PDF folder: `data/`")
    uploaded_file = st.file_uploader("Upload a new PDF", type=["pdf"])
    if uploaded_file:
        file_path = os.path.join(DATA_PATH, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✅ {uploaded_file.name} uploaded!")

# Chat history
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

for msg in st.session_state['messages']:
    st.chat_message(msg['role']).markdown(msg['content'])

# User input
prompt = st.chat_input("Ask your health question here...")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state['messages'].append({"role": "user", "content": prompt})

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
            st.error("❌ Failed to load vector store")

        # Use Gemini LLM here
        qa_chain = RetrievalQA.from_chain_type(
            llm=gemini_llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
        )

        response = qa_chain.invoke({"query": prompt})
        result = response["result"]
        source_documents = response["source_documents"]

        # Show bot response
        st.chat_message("assistant").markdown(result)
        st.session_state['messages'].append({"role": "assistant", "content": result})

        # Expandable sources
        with st.expander("📚 Show Sources"):
            for i, doc in enumerate(source_documents, 1):
                st.markdown(f"**Source {i}** - Page {doc.metadata.get('page', 'N/A')}")
                st.markdown(doc.page_content)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
