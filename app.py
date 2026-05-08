import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from groq import Groq

# Page config
st.set_page_config(page_title="HR Chatbot", layout="wide")

st.title("💬 HR Chat Assistant")

# session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# upload PDFs
uploaded_files = st.file_uploader(
    "Upload HR PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

# process docs
if st.button("Process Documents"):
    if not uploaded_files:
        st.warning("Upload at least one file")
    else:
        all_docs = []

        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                path = tmp.name

            loader = PyPDFLoader(path)
            docs = loader.load()

            for d in docs:
                d.metadata["source"] = file.name

            all_docs.extend(docs)
            os.remove(path)

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(all_docs)

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)

        st.success("✅ Documents processed!")

# chat UI
st.markdown("---")

if st.session_state.vector_store:
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

    user_input = st.chat_input("Ask about HR policies...")

    if user_input:
        st.session_state.chat_history.append(("user", user_input))

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                retriever = st.session_state.vector_store.as_retriever()
                docs = retriever.invoke(user_input)

                context = "\n".join([d.page_content for d in docs])

                prompt = f"""
                You are an HR assistant.
                Answer ONLY from context. If not found, say "Not in document".

                Context:
                {context}

                Question:
                {user_input}
                """

                client = Groq(api_key=os.getenv("GROQ_API_KEY"))

                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}]
                )

                answer = response.choices[0].message.content

                st.markdown(answer)

                st.markdown("**Sources:**")
                for d in docs[:3]:
                    st.markdown(f"- {d.metadata.get('source', 'unknown')}")

                st.session_state.chat_history.append(("assistant", answer))