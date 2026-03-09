# rag_pipeline.py
# Core RAG logic: Load PDF → Chunk → Embed → Store → Retrieve → Answer
# LLM: Groq (free, no quota issues) with Llama 3

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

# -------------------------------------------------------------------
# STEP 1: Load & chunk the PDF
# -------------------------------------------------------------------
def load_and_chunk(pdf_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
    """Load a PDF and split it into overlapping chunks."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(pages)
    return chunks


# -------------------------------------------------------------------
# STEP 2: Embed chunks & build FAISS vector store
# -------------------------------------------------------------------
def build_vector_store(chunks):
    """Convert text chunks into embeddings and store in FAISS."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


# -------------------------------------------------------------------
# STEP 3: Build the RAG chain
# -------------------------------------------------------------------
def build_rag_chain(vector_store):
    """Connect retriever + Groq LLM (Llama 3) into a QA chain."""

    prompt_template = """
You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not found in the context, say "I couldn't find that in the document."

Context:
{context}

Question: {question}

Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Groq LLM — free, fast, no daily quota issues
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain


# -------------------------------------------------------------------
# STEP 4: Ask a question
# -------------------------------------------------------------------
def ask_question(qa_chain, question: str):
    """Run a question through the RAG chain and return answer + sources."""
    result = qa_chain.invoke({"query": question})
    answer = result["result"]
    sources = result["source_documents"]
    return answer, sources