# --------- DISABLE CHROMA TELEMETRY ----------
import os
os.environ["CHROMA_TELEMETRY"] = "false"

# --------- IMPORTS ----------
import pandas as pd
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from chromadb.config import Settings

# --------- STREAMLIT PAGE CONFIG ----------
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

    # ✅ Disable Chroma telemetry + persistence to fix Streamlit Cloud error
    client_settings = Settings(anonymized_telemetry=False, persist_directory=None)

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        client_settings=client_settings
    )

    return vectorstore

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever()

# --------- STEP 2: UI and Chat Memory ----------
st.title("🛒 Product FAQ Chatbot (Tesco Example)")
st.write("Ask any question related to the product FAQs!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

# --------- STEP 3: Handle New Query ----------
if user_query := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": user_query})

    relevant_docs = vectorstore.similarity_search_with_score(user_query, k=1)

    if relevant_docs:
        doc, score = relevant_docs[0]
        threshold = 0.60  # Can be tuned
        if score >= threshold:
            answer_text = doc.page_content
        else:
            answer_text = "❗ Sorry, I couldn't find a confident answer. Please try rephrasing your question."
    else:
        answer_text = "❗ Sorry, no matching FAQ found."

    st.session_state.messages.append({"role": "assistant", "content": answer_text})
    st.chat_message("assistant").markdown(answer_text)

# --------- STEP 4: Sidebar Info ----------
st.sidebar.title("About")
st.sidebar.info(
    "Built with ❤️ using LangChain, Chroma, HuggingFace, and Streamlit!\n\n"
    "Chatbot remembers conversation and safely handles unclear questions!"
)
