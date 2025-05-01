# Install once before:
# pip install streamlit langchain langchain-community sentence-transformers chromadb pandas

# --------- IMPORTS ----------

import pandas as pd
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["CHROMA_TELEMETRY"] = "false"

# --------- VERY FIRST: PAGE SETTINGS (Important) ----------
st.set_page_config(page_title="Product FAQ Chatbot", page_icon="🛒")

# --------- STEP 1: Load FAQ Data and Setup Retriever ----------
@st.cache_resource
def load_vectorstore():
    faq_data = pd.read_csv('fact-base-tesco.csv')

    docs = []
    for idx, row in faq_data.iterrows():
        content = f"Q: {row['Question']}\nA: {row['Answer']}"
        docs.append(Document(page_content=content, metadata={"source": idx}))

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db_streamlit")

    return vectorstore

vectorstore = load_vectorstore()

# Using retriever for basic invoke
retriever = vectorstore.as_retriever()

# --------- STEP 2: UI and Chat Memory ----------
st.title("🛒 Product FAQ Chatbot (Tesco Example)")
st.write("Ask any question related to the product FAQs!")

# Chat memory session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message("user").markdown(message["content"])
    else:
        st.chat_message("assistant").markdown(message["content"])

# --------- STEP 3: Handle New Query ----------
if user_query := st.chat_input("Type your question here..."):

    # Add user input to memory
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Search for relevant document
    relevant_docs = vectorstore.similarity_search_with_score(user_query, k=1)

    if relevant_docs:
        doc, score = relevant_docs[0]

        # Set Similarity Threshold
        threshold = 0.60  # Tune if needed

        if score >= threshold:
            answer_text = doc.page_content
        else:
            answer_text = "❗ Sorry, I couldn't find a confident answer. Please try rephrasing your question."
    else:
        answer_text = "❗ Sorry, no matching FAQ found."

    # Add assistant response to memory
    st.session_state.messages.append({"role": "assistant", "content": answer_text})

    # Display assistant response
    st.chat_message("assistant").markdown(answer_text)

# --------- STEP 4: Sidebar Info ----------
st.sidebar.title("About")
st.sidebar.info(
    "Built with ❤️ using LangChain, Chroma, HuggingFace, and Streamlit!\n\n"
    "Chatbot remembers conversation and safely handles unclear questions!"
)
