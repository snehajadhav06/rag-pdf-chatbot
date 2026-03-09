# 🤖 RAG PDF Chatbot

A production-style Retrieval-Augmented Generation (RAG) chatbot that lets you chat with any PDF — built with LangChain, Google Gemini, FAISS, and Streamlit.

---

## 🏗️ Architecture

```
PDF Upload → Text Chunking → Embeddings (MiniLM) → FAISS Vector Store
                                                           ↓
User Question → Embed Question → Similarity Search → Top-4 Chunks
                                                           ↓
                                              Gemini LLM → Answer
```

---

## 🛠️ Tech Stack

| Component     | Tool                              |
|---------------|-----------------------------------|
| LLM           | Google Gemini 1.5 Flash (free)    |
| Embeddings    | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store  | FAISS (local, no cost)            |
| Orchestration | LangChain                         |
| UI            | Streamlit                         |

---

## 🚀 Setup & Run

### 1. Clone / Download the project
```bash
cd rag-chatbot
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Get your free Gemini API key
1. Go to https://aistudio.google.com/app/apikey
2. Sign in with Google
3. Click **Create API Key**
4. Copy the key

### 5. Add API key
```bash
cp .env.example .env
# Open .env and paste your key
```

### 6. Run the app
```bash
streamlit run app.py
```

### 7. Use it!
- Upload any PDF in the sidebar
- Wait for indexing (10–30 seconds)
- Ask questions in the chat!

---

## 📁 Project Structure

```
rag-chatbot/
├── app.py              # Streamlit UI
├── rag_pipeline.py     # Core RAG logic
├── requirements.txt    # Dependencies
├── .env.example        # API key template
└── README.md
```

---

## 💡 Example Use Cases

- Chat with your **resume** → practice interview Q&A
- Chat with a **research paper** → get summaries
- Chat with a **textbook chapter** → study aid
- Chat with a **legal document** → understand key clauses

---

## 🚢 Deploy to Streamlit Cloud (Free)

1. Push this project to GitHub
2. Go to https://share.streamlit.io
3. Connect your GitHub repo
4. Set `GOOGLE_API_KEY` in Streamlit Secrets
5. Deploy! ✅

---

## 🔧 Possible Improvements (for your resume!)

- Add support for multiple PDFs
- Add chat memory (multi-turn conversation)
- Switch to ChromaDB for persistent storage
- Add document summarization feature
- Support DOCX and TXT files

---

Built by Sneha Jadhav | RAG Project Portfolio
