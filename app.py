import os
import time
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq

# Streamlit UI Setup
st.set_page_config(page_title="Research Assistant", layout="centered")

st.markdown("## 📚 Research RAG Assistant")
st.markdown("Upload your PDF document, generate embeddings, and ask questions based on it.")

# Session state to track embedding status
if "faiss_db" not in st.session_state:
    st.session_state.faiss_db = None
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Upload file
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Generate Embeddings
if uploaded_file:
    if st.button("📌 Generate Embeddings"):
        with st.spinner("Processing PDF and generating embeddings..."):
            # Save uploaded file
            pdf_path = os.path.join("temp", uploaded_file.name)
            os.makedirs("temp", exist_ok=True)
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.read())

            # Load and process PDF
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            if not documents:
                st.warning("❌ Could not extract any content from the uploaded PDF.")
            else:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
                texts = text_splitter.split_documents(documents)

                if not texts:
                    st.warning("❌ The PDF was loaded but no text was split. Please check the content.")
                else:
                    embeddings = HuggingFaceEmbeddings(
                        model_name="nomic-ai/nomic-embed-text-v1",
                        model_kwargs={"trust_remote_code": True, "revision": "289f532e14dbbbd5a04753fa58739e9ba766f3c7"}
                    )

                    # Build FAISS index
                    st.session_state.faiss_db = FAISS.from_documents(texts, embeddings)
                    st.success("✅ Embeddings generated! You can now ask questions.")

# Once embeddings are ready, allow Q&A
if st.session_state.faiss_db:
    db_retriever = st.session_state.faiss_db.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    prompt_template = """
    <s>[INST]
You are a highly specialized research assistant and subject-matter expert. Your role is to help researchers by answering questions based **strictly on the content provided from uploaded research papers or documents**. 

Your responses should:
- Be concise, accurate, and contextually relevant.
- Stay grounded in the content and not fabricate information.
- Avoid personal opinions or speculative answers.
- Use formal, academic tone suitable for research discussions.
- If the question is unrelated to the context, politely state that the information is not available in the document.

CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
[/INST]
"""

    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question', 'chat_history'])

    llm = ChatGroq(
        api_key="",
        model="llama3-70b-8192",
        temperature=0.5,
        max_tokens=1024
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=st.session_state.memory,
        retriever=db_retriever,
        combine_docs_chain_kwargs={'prompt': prompt}
    )

    for message in st.session_state.messages:
        with st.chat_message(message.get("role")):
            st.write(message.get("content"))

    input_prompt = st.chat_input("Ask your research/legal question")

    if input_prompt:
        with st.chat_message("user"):
            st.write(input_prompt)
        st.session_state.messages.append({"role": "user", "content": input_prompt})

        with st.chat_message("assistant"):
            with st.status("Thinking 💡...", expanded=True):
                result = qa.invoke(input=input_prompt)
                full_response = "⚠️ _Note: This information is based on uploaded document content._\n\n\n"
                for chunk in result["answer"]:
                    full_response += chunk
                    time.sleep(0.02)
                st.write(full_response)

        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})

    st.button("🔄 Reset Chat", on_click=lambda: [st.session_state.messages.clear(), st.session_state.memory.clear()])
else:
    st.info("👆 Upload a PDF and click 'Generate Embeddings' to start.")

