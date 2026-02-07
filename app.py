# import streamlit as st
# import tempfile
# import os
#
# from langchain_community.chains import PebbloRetrievalQA
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_ollama import OllamaEmbeddings, OllamaLLM
# from langchain_community.vectorstores import FAISS
#
# from langchain_classic.chains.retrieval_qa.base import RetrievalQA
#
# st.set_page_config(page_title="PDF Q&A Bot", layout="centered")
# st.title("PDF Question and Answer Bot")
#
# @st.cache_resource
# def load_llm():
#     return OllamaLLM(model="llama3")
#
# @st.cache_resource
# def load_embeddings():
#     return OllamaEmbeddings(model="nomic-embed-text")
#
# uploaded_pdf = st.file_uploader("Upload a PDF File", type="pdf")
#
# if uploaded_pdf:
#     with st.spinner("Processing PDF..."):
#
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#             tmp_file.write(uploaded_pdf.read())
#             pdf_path = tmp_file.name
#
#         st.write("📄 Loading PDF...")
#         loader = PyPDFLoader(pdf_path)
#         documents = loader.load()
#
#         st.write("✂️ Splitting text...")
#         splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200
#         )
#         chunks = splitter.split_documents(documents)
#
#         st.write(f"🧠 Creating embeddings for {len(chunks)} chunks...")
#         embeddings = load_embeddings()
#         vector_store = FAISS.from_documents(chunks, embeddings)
#
#         st.write("🤖 Loading LLM...")
#         llm = load_llm()
#
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             retriever=vector_store.as_retriever()
#         )
#
#         os.remove(pdf_path)
#
#         st.success("PDF Processed! You can now ask questions.")
#
#         # -----------ASK QUESTION-------------
#         question = st.text_input("Ask a question from the pdf")
#
#         if question:
#             with st.spinner("Thinking..."):
#                 answer = qa_chain.run(question)
#
#                 st.subheader("Answer")
#                 st.write(answer)

import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="PDF Q&A Bot", layout="centered")
st.title("📄 PDF Question and Answer Bot")

# ------------------ CACHED RESOURCES ------------------
@st.cache_resource
def load_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")

@st.cache_resource
def load_llm():
    return OllamaLLM(model="phi3",
    num_predict = 256,
    temperature = 0.1
    )

@st.cache_resource
def build_vector_store(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    embeddings = load_embeddings()
    return FAISS.from_documents(chunks, embeddings)


@st.cache_resource
def build_qa_chain(_vector_store):
    # --------------------PROMPT---------------

    PROMPT = PromptTemplate(
        template="""
    Use ONLY the context below to answer.
    If the answer is not present, say "Not found in the document".
    Keep the answer short and clear.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """,
        input_variables=["context", "question"]
    )

    llm = load_llm()
    retriever = _vector_store.as_retriever(search_kwargs={"k": 4})

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs = {"prompt": PROMPT}
    )





# ------------------ SESSION STATE ------------------
if "processed" not in st.session_state:
    st.session_state.processed = False

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# ------------------ FILE UPLOAD ------------------
uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_pdf:
    # Detect new PDF upload
    if uploaded_pdf.name != st.session_state.pdf_name:
        st.session_state.processed = False
        st.session_state.pdf_name = uploaded_pdf.name

    if not st.session_state.processed:
        with st.spinner("Processing PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_pdf.read())
                pdf_path = tmp_file.name

            vector_store = build_vector_store(pdf_path)
            qa_chain = build_qa_chain(vector_store)

            st.session_state.qa_chain = qa_chain
            st.session_state.processed = True

            os.remove(pdf_path)

        st.success("✅ PDF processed successfully!")

# ------------------ Q&A ------------------
if st.session_state.get("processed", False):
    question = st.text_input("Ask a question from the PDF")

    if question:
        with st.spinner("Thinking..."):
            result = st.session_state.qa_chain.invoke(
                {"query": question}
            )

        st.subheader("Answer")
        st.write(result["result"])
