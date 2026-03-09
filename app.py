# app.py
# Streamlit UI for the RAG chatbot — powered by Groq (free!)

import os
import tempfile
import streamlit as st

st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 RAG PDF Chatbot")
st.markdown("Upload any PDF and ask questions — powered by **Groq (Llama 3.3)** + LangChain + FAISS")

# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Setup")

    api_key = st.text_input(
        "🔑 Groq API Key",
        type="password",
        help="Get a FREE key at https://console.groq.com"
    )

    st.markdown("---")
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("1. Upload a PDF\n2. We chunk + embed it\n3. Ask any question\n4. Llama 3 answers using your doc!")
    st.markdown("---")
    st.caption("🆓 Groq is 100% free — no quota issues!")

# -------------------------------------------------------------------
# Only import heavy libs after user provides API key
# -------------------------------------------------------------------
def build_pipeline(pdf_bytes, groq_api_key):
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_groq import ChatGroq
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate

    # Save PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    # Load & chunk
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(pages)
    os.unlink(tmp_path)

    # Embeddings + vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Prompt
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

    # Groq LLM
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=groq_api_key,
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

    return qa_chain, len(chunks)


# -------------------------------------------------------------------
# Main app logic
# -------------------------------------------------------------------
if uploaded_file and api_key:

    file_key = uploaded_file.name + str(uploaded_file.size)

    if "file_key" not in st.session_state or st.session_state.file_key != file_key:
        with st.spinner("📚 Reading and indexing your PDF... please wait"):
            try:
                pdf_bytes = uploaded_file.read()
                qa_chain, num_chunks = build_pipeline(pdf_bytes, api_key)
                st.session_state.qa_chain = qa_chain
                st.session_state.file_key = file_key
                st.session_state.chat_history = []
                st.session_state.num_chunks = num_chunks
                st.success(f"✅ PDF indexed! ({num_chunks} chunks created)")
            except Exception as e:
                st.error(f"❌ Error loading PDF: {str(e)}")
                st.stop()

    # Chat interface
    st.markdown("---")
    st.subheader(f"💬 Chat with: `{uploaded_file.name}`")

    for msg in st.session_state.get("chat_history", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_question = st.chat_input("Ask anything about your document...")

    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Thinking..."):
                try:
                    result = st.session_state.qa_chain.invoke({"query": user_question})
                    answer = result["result"]
                    sources = result["source_documents"]

                    st.markdown(answer)

                    with st.expander("📎 Source passages used"):
                        for i, doc in enumerate(sources):
                            st.markdown(f"**Chunk {i+1}** (Page {doc.metadata.get('page', '?') + 1})")
                            st.caption(doc.page_content[:300] + "...")
                            st.markdown("---")

                    st.session_state.chat_history.append({"role": "assistant", "content": answer})

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

elif not api_key:
    st.info("👈 Enter your **Groq API key** in the sidebar to get started.")
    st.markdown("""
    ### 🆓 Get a Free Groq API Key:
    1. Go to [console.groq.com](https://console.groq.com)
    2. Sign up with Google (no card needed)
    3. Click **API Keys** → **Create API Key**
    4. Paste it in the sidebar ✅
    """)
elif not uploaded_file:
    st.info("👈 Upload a **PDF file** in the sidebar to get started.")
    st.markdown("""
    ### 💡 Try asking:
    - *"What is this document about?"*
    - *"Summarize the key points"*
    - *"What does it say about [topic]?"*
    """)