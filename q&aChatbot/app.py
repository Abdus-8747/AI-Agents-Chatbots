import streamlit as st
import os
import time
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

# 🔑 Load API keys from .env
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


# 📌 Create Vector Embeddings
import asyncio

def vector_embeddings():
    if "vectors" not in st.session_state:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())

        # 🚀 FREE unlimited embeddings
        st.session_state.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        st.session_state.loader = PyPDFDirectoryLoader("./pdfs")
        st.session_state.docs = st.session_state.loader.load()

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )

        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs[:30]
        )

        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings
        )


# 🎯 Streamlit UI
st.title("📚 Q&A Chatbot with Groq + HuggingFace Embeddings + Streamlit")

# ✅ Use correct param for Groq
llm = ChatGroq(api_key=groq_api_key, model="llama-3.3-70b-versatile")

prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}
Question: {input}
Helpful Answer:
""")

user_question = st.text_input("Enter your question here:")

if st.button("Get Answer"):
    vector_embeddings()
    st.success("✅ Vector store created successfully.")

    if user_question:
        start = time.process_time()

        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()

        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({"input": user_question})

        st.write("⏱️ Response time:", round(time.process_time() - start, 2), "seconds")
        st.write("### 🤖 Answer:")
        st.write(response.get("answer", "No answer found."))

        with st.expander("🔍 Doc Similarity Search"):
            for i, doc in enumerate(response.get("context", [])):
                st.write(doc.page_content)
                st.write("---------------------------------------")
