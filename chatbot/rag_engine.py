import os
import json
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from chatbot.utils import get_openai_api_key

def load_json_to_documents(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    documents = []
    for item in data:
        content = item.get("content", "")
        title = item.get("title", "")
        if content.strip():
            documents.append(Document(page_content=content, metadata={"title": title}))
    return documents

def create_vectorstore(documents, index_path="vectorstore"):
    if os.path.exists(index_path):
        print("📦 Loading existing vector index from disk...")
        embeddings = OpenAIEmbeddings(openai_api_key=get_openai_api_key())
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    print("⚙️ Building new vector index...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=get_openai_api_key())
    vectorstore = FAISS.from_documents(chunks, embeddings)

    print("💾 Saving vector index to disk...")
    vectorstore.save_local(index_path)
    return vectorstore

def search_similar_docs(query, vectorstore, k=3):
    results = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in results])
