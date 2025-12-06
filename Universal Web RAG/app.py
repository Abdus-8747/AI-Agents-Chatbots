import streamlit as st
import os
import time
from dotenv import load_dotenv

# LangChain imports
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

# ------------------------------
# Environment
# ------------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("❌ GROQ_API_KEY not found in .env")

st.set_page_config(
    page_title="Universal RAG Agent",
    page_icon="🌐",
    layout="wide",
)

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.title("⚙️ Settings")

model_choice = st.sidebar.selectbox(
    "Select Groq Model",
    ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    index=0,
)

st.sidebar.markdown("### Embedding Model")
st.sidebar.info("Using **all-MiniLM-L6-v2** (HuggingFace)")

st.sidebar.markdown("---")
st.sidebar.markdown("Made by **Abdus Samad** 🔥")

# ------------------------------
# URL Input
# ------------------------------
st.title("🌐 Universal Web RAG Agent")
st.write("Enter ANY webpage URL. The model will extract, index, and answer questions based on that page.")

url_input = st.text_input(
    "🌍 Enter a webpage URL",
    placeholder="https://en.wikipedia.org/wiki/Machine_learning",
)

load_button = st.button("Load & Process Page")

# ------------------------------
# Load + Build Vector Store
# ------------------------------
if load_button:
    if not url_input.strip():
        st.error("❌ Please enter a valid URL.")
    else:
        with st.spinner("🔄 Scraping and processing webpage..."):
            try:
                # Embeddings
                st.session_state.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

                # Loader
                st.session_state.loader = WebBaseLoader(
                    url_input,
                    requests_kwargs={"headers": {"User-Agent": "Mozilla/5.0"}},
                )
                docs = st.session_state.loader.load()

                # Split
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200
                )
                final_docs = splitter.split_documents(docs)

                # Store
                st.session_state.vector = FAISS.from_documents(
                    final_docs, st.session_state.embeddings
                )

                st.success("✅ Webpage indexed successfully! You can now ask questions.")

            except Exception as e:
                st.error(f"❌ Error loading page: {e}")

# ------------------------------
# If vector exists, allow Q&A
# ------------------------------
if "vector" in st.session_state:

    # LLM
    llm = ChatGroq(
        api_key=groq_api_key,
        model=model_choice,
        temperature=0.2,
    )

    # Prompt
    prompt_template = ChatPromptTemplate.from_template("""
    You are a highly intelligent assistant. Answer STRICTLY using the context from the webpage.

    Context:
    {context}

    Question:
    {input}

    Provide a concise, accurate answer.
    """)

    doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template)
    retriever = st.session_state.vector.as_retriever()
    chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=doc_chain)

    # UI input
    user_q = st.text_input("💬 Ask your question about the webpage:", placeholder="e.g., What is the main idea of the article?")

    if user_q:
        with st.spinner("💡 Thinking..."):
            start = time.time()
            response = chain.invoke({"input": user_q})
            end = time.time()

        # Response
        st.subheader("📘 Answer")
        st.info(response["answer"])

        st.success(f"⚡ Response Time: {round(end - start, 2)} seconds")

        # Context
        st.subheader("📄 Retrieved Context")
        for i, doc in enumerate(response["context"]):
            with st.expander(f"Document {i+1}"):
                st.write(doc.page_content)
