AI-Powered PDF Chatbot

this bot that allows users to chat with any PDF they upload. Instead of manually searching through pages, the bot uses Retrieval-Augmented Generation (RAG) to find and explain information directly from the document. When a user uploads a PDF, the app extracts its content, splits it into smaller text chunks, and converts each chunk into vector embeddings (numerical representations that capture meaning). These embeddings are stored in a vector database.


How It Works:

1.When a user uploads a PDF, the app:

2.Extracts the content from the PDF.

3.Splits it into smaller text chunks for efficient processing.

4.Converts each chunk into vector embeddings — numerical representations that capture meaning.

5.Stores the embeddings in a vector database.

6.Whenever a user asks a question, the chatbot:

7.Converts the question into an embedding to understand its meaning.

8.Compares it with the stored document chunks using semantic similarity search.Retrieves the most relevant text from the document.

9.Passes the context to a language model (Gemini or Groq) to generate a clear, context-aware answer.

10.The entire system is built with Streamlit, providing a clean and interactive chat interface. Users can upload any type of PDF — such as health guides, research papers, or reports — and get quick answers to their questions.


Tech Stack :

Language: Python

Frontend: Streamlit (for web interface)

Vector Database: Groq API

Embeddings & RAG: Custom transformer models

PDF Processing: LangChain text loaders

Deployment (optional): Streamlit Cloud

Screenshots
<img width="1672" height="887" alt="Screenshot 2025-09-27 225904" src="https://github.com/user-attachments/assets/1d7c9edf-7ce1-4db9-abe4-753d87f459e9" /> <img width="1070" height="787" alt="Screenshot 2025-09-28 105251" src="https://github.com/user-attachments/assets/ba7c4f27-aff3-4a02-8b51-b688606d7527" /> <img width="1828" height="840" alt="Screenshot 2025-09-28 105222" src="https://github.com/user-attachments/assets/b125e3bf-2855-41d2-920a-0cc277d90f44" />
