Med Bot – AI Health Assistant
Overview

Med Bot is an AI-powered chatbot designed to provide answers to health-related questions. The bot comes preloaded with a health PDF, so when users ask questions like “What is diabetes?” or “How to prevent high blood pressure?”, it searches the document and provides accurate, easy-to-understand responses.

It uses RAG (Retrieval-Augmented Generation) with vector embeddings to find the most relevant information and generate human-like answers.

How It Works :

1. Preloaded Health PDF: The bot is initialized with a health PDF containing medical information.

2. Text Chunking & Embeddings: The PDF content is split into small chunks and converted into vector embeddings.

3. Vector Database (Groq API): Chunks are stored in a vector database for fast semantic search.

4. User Query Handling: When a user asks a question, it is converted into a vector and compared against the stored chunks.

5. RAG Answer Generation: The retrieved information is used to generate a concise, accurate, and context-aware response.

Tech Stack :

Language: Python

Frontend: Streamlit (for web interface)

Vector Database: Groq API

Embeddings & RAG: custom transformer models

PDF Processing:  LangChain text loaders

Deployment (optional): Streamlit Cloud 

<img width="1404" height="867" alt="Screenshot 2025-09-27 230130" src="https://github.com/user-attachments/assets/15cd8cd1-cb3b-4b9e-87ed-549b30d1aae9" />

<img width="1672" height="887" alt="Screenshot 2025-09-27 225904" src="https://github.com/user-attachments/assets/1d7c9edf-7ce1-4db9-abe4-753d87f459e9" />



