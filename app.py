# app.py
# Streamlit UI for the RAG chatbot — powered by Groq (free!)

import streamlit as st
import tempfile
import os
from rag_pipeline import load_and_chunk, build_vector_store, build_rag_chain, ask_question

st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 RAG PDF Chatbot")
st.markdown("Upload any PDF and ask questions about it — powered by **Groq (Llama 3)** + LangChain + FAISS")

# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Setup")

    api_key = st.text_input(
        "🔑 Groq API Key",
        type="password",
        help="Get a FREE key at https://console.groq.com — no credit card needed!"
    )
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key

    st.markdown("---")
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("1. Upload a PDF\n2. We chunk + embed it\n3. Ask any question\n4. Llama 3 answers using your doc!")
    st.markdown("---")
    st.caption("🆓 Groq is 100% free — no quota issues!")

# -------------------------------------------------------------------
# Main — process PDF and chat
# -------------------------------------------------------------------
if uploaded_file and api_key:

    file_key = uploaded_file.name + str(uploaded_file.size)
    if "file_key" not in st.session_state or st.session_state.file_key != file_key:
        with st.spinner("📚 Reading and indexing your PDF... please wait"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            chunks = load_and_chunk(tmp_path)
            vector_store = build_vector_store(chunks)
            qa_chain = build_rag_chain(vector_store)

            st.session_state.qa_chain = qa_chain
            st.session_state.file_key = file_key
            st.session_state.chat_history = []
            st.session_state.num_chunks = len(chunks)

            os.unlink(tmp_path)

        st.success(f"✅ PDF indexed! ({st.session_state.num_chunks} chunks created)")

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
            with st.spinner("🔍 Searching document..."):
                answer, sources = ask_question(st.session_state.qa_chain, user_question)

            st.markdown(answer)

            with st.expander("📎 Source passages used"):
                for i, doc in enumerate(sources):
                    st.markdown(f"**Chunk {i+1}** (Page {doc.metadata.get('page', '?') + 1})")
                    st.caption(doc.page_content[:300] + "...")
                    st.markdown("---")

        st.session_state.chat_history.append({"role": "assistant", "content": answer})

elif not api_key:
    st.info("👈 Please enter your **Groq API key** in the sidebar to get started.")
    st.markdown("""
    ### 🆓 Get a Free Groq API Key (30 seconds, no card needed):
    1. Go to [console.groq.com](https://console.groq.com)
    2. Sign up with Google or email
    3. Click **API Keys** → **Create API Key**
    4. Paste it in the sidebar ✅
                
    Groq gives you **free access to Llama 3** with generous limits — perfect for portfolio projects!
    """)

elif not uploaded_file:
    st.info("👈 Please upload a **PDF file** in the sidebar.")
    st.markdown("""
    ### 💡 Example questions you can ask:
    - *"What is this document about?"*
    - *"Summarize the key points"*
    - *"What does it say about [topic]?"*
    - *"List all the main findings"*
    """)