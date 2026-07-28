# 📚 Conversational RAG Chatbot with LangChain

A **Conversational Retrieval-Augmented Generation (RAG)** chatbot built using **LangChain**, **ChromaDB**, **Hugging Face Embeddings**, and **Google Gemini**. The chatbot can answer questions from PDF documents while maintaining conversation history to understand follow-up questions.

---

## 🚀 Features

* 📄 Load and process PDF documents.
* ✂️ Split documents into semantic chunks.
* 🧠 Generate vector embeddings using Hugging Face.
* 🗄️ Store embeddings in Chroma Vector Database.
* 🔍 Retrieve the most relevant document chunks using semantic search.
* 💬 Maintain chat history for follow-up questions.
* 🔄 Rewrite contextual questions into standalone questions before retrieval.
* 🤖 Generate accurate responses using Google Gemini.
* 🛑 Responds with *"I don't know based on the provided document."* when the answer is not present in the retrieved context.

---

## 🛠️ Tech Stack

* Python
* LangChain
* ChromaDB
* Hugging Face Embeddings
* Google Gemini API
* PyPDFLoader
* RecursiveCharacterTextSplitter

---

## 📂 Project Structure

```text
RAG/
│
├── data/
│   └── document.pdf
│
├── vector_store/
│   └── Chroma Database
│
├── main.py
├── requirements.txt
├── .env
└── README.md
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <repository-url>
cd RAG
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment.

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root.

```env
GOOGLE_API_KEY=YOUR_API_KEY
```

---

## ▶️ Running the Project

```bash
python main.py
```

The chatbot will prompt you for questions interactively.

Type:

```text
exit
```

to end the conversation.

---

## 🏗️ Architecture

```text
                PDF Documents
                      │
                      ▼
               PyPDFLoader
                      │
                      ▼
      RecursiveCharacterTextSplitter
                      │
                      ▼
       HuggingFace Embeddings
                      │
                      ▼
              Chroma Vector DB
                      │
                      ▼
                Retriever
                      │
                      ▼
        Chat History + User Query
                      │
                      ▼
      Question Rewriter (Gemini)
                      │
                      ▼
         Standalone Question
                      │
                      ▼
              Semantic Retrieval
                      │
                      ▼
        Prompt + Retrieved Context
                      │
                      ▼
             Gemini LLM Response
                      │
                      ▼
          Update Conversation History
```

---

## 🧠 How It Works

1. Load the PDF document.
2. Split the document into smaller chunks.
3. Convert each chunk into vector embeddings.
4. Store the embeddings in ChromaDB.
5. Receive a user's question.
6. Rewrite follow-up questions into standalone questions using conversation history.
7. Retrieve the most relevant chunks from Chroma.
8. Combine the retrieved context with the user's question.
9. Generate a response using Gemini.
10. Save both the user's question and the assistant's response in the chat history for future context.

---

## 💬 Example Conversation

```text
You: What is Artificial Intelligence?

AI:
Artificial Intelligence is the ability of a computer or robot to perform tasks in a manner similar to intelligent beings.

You:
Tell me more.

AI:
Artificial Intelligence aims to simulate human intelligence, enabling systems to solve problems, learn from data, and make decisions. The document further discusses Narrow AI and its applications.
```

---

## 📌 Current Capabilities

* Semantic document search
* Conversational RAG
* Persistent vector database
* Multi-turn conversations
* Context-aware question rewriting
* Interactive command-line chatbot

---

## 🔮 Future Improvements

* Support multiple PDF documents.
* Display source citations for each answer.
* Add reranking for improved retrieval quality.
* Implement hybrid search (BM25 + vector search).
* Stream LLM responses token-by-token.
* Build a web interface using Streamlit or FastAPI.
* Support document uploads at runtime.
* Add conversation summarization for long chats.

---

## 📚 Learning Outcomes

This project demonstrates the complete workflow of building a conversational RAG system, including:

* Document loading
* Text chunking
* Vector embeddings
* Vector databases
* Semantic search
* Retrieval-Augmented Generation (RAG)
* Conversation memory
* Question rewriting
* LangChain pipelines
* Large Language Model integration

---

## 📄 License

This project is intended for educational and learning purposes. Feel free to modify and extend it for your own experiments.
