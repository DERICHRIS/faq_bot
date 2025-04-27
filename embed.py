# Install once before:
# pip install langchain langchain-community sentence-transformers chromadb pandas

import os
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# STEP 1: Load FAQ
faq_data = pd.read_csv('fact-base-tesco.csv')

# STEP 2: Create documents
docs = []
for idx, row in faq_data.iterrows():
    content = f"Q: {row['Question']}\nA: {row['Answer']}"
    docs.append(Document(page_content=content, metadata={"source": idx}))

# STEP 3: Setup local embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# STEP 4: Setup ChromaDB
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")

retriever = vectorstore.as_retriever()

# STEP 5: Start chatbot loop
print("🔵 Chatbot is ready. Ask your product questions!")

while True:
    query = input("\nYou: ")
    if query.lower() == "exit":
        break

    relevant_docs = retriever.invoke(query)
    
    if relevant_docs:
        answer_text = relevant_docs[0].page_content
        print(f"\nBot Answer:\n{answer_text}")
    else:
        print("\nBot: Sorry, I couldn't find an answer for that.")
