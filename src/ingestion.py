"""
ingestion.py — Document Ingestion Pipeline

Loads documents from the knowledge base, chunks them, generates embeddings,
and stores them in ChromaDB vector store.

"""

import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# Step 1: Load Documents

loader = DirectoryLoader(
    "data/knowledge_base",
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)

# Step 2: Split into Chunks 

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separators=["\n\n", "\n", " ", ""]
)

#  Step 3: Create Embedding Model 

embeddings = OllamaEmbeddings(model="nomic-embed-text")

#  Step 4: Build or Load ChromaDB 

if os.path.exists("data/chroma_db"):
    vectorstore = Chroma(
        persist_directory="data/chroma_db",
        embedding_function=embeddings,
        collection_name="knowledge_base"
    )
    print("Loaded existing ChromaDB")
else:
    # only loads and chunks documents if we need to build the vectorstore
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")

    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="data/chroma_db",
        collection_name="knowledge_base"
    )
    print(f"Stored {vectorstore._collection.count()} vectors in ChromaDB")