import streamlit as st
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Pinecone as PC
import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
import os
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Load embeddings (this can be cached if required)
@st.cache_resource
def download_hugging_face_embeddings():
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings


# Streamlit app
def main():
    st.title("Medical ChatBot")
    st.markdown("**Keep yourself healthy!**")

    # Initialize session state for conversation history
    if "history" not in st.session_state:
        st.session_state.history = []

    # User input
    user_input = st.chat_input("Ask a medical question...")

    if user_input:
        # Append user message to history
        st.session_state.history.append({"role": "user", "content": user_input})

        embeddings = download_hugging_face_embeddings()

        pineconekey = os.getenv("PINECONE_API_KEY")
        pc = Pinecone(api_key=pineconekey)
        index_name = "medicalchatbot"

        docsearch = PC.from_existing_index(index_name=index_name, embedding=embeddings)

        prompt_template = """
        Use the following pieces of information to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}
        Question: {question}

        Only return the helpful answer below and nothing else.
        Helpful answer:
        """

        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        llm = CTransformers(
            model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
            model_type="llama",
            config={'max_new_tokens': 512, 'temperature': 0.8}
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

        result = qa({"query": user_input})

        # Append bot response to history
        st.session_state.history.append({"role": "bot", "content": result["result"]})

    # Display conversation history
    for message in st.session_state.history:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        else:
            with st.chat_message("bot"):
                st.write(message["content"])

if __name__ == "__main__":
    main()
