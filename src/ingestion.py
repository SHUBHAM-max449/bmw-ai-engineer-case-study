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

#  Configuration 

KNOWLEDGE_BASE_DIR = "data/knowledge_base"
CHROMA_PERSIST_DIR = "data/chroma_db"
COLLECTION_NAME    = "knowledge_base"
EMBEDDING_MODEL    = "nomic-embed-text"
CHUNK_SIZE         = 600
CHUNK_OVERLAP      = 50

# Step 1: Load Documents

loader = DirectoryLoader(
    KNOWLEDGE_BASE_DIR,
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)

# Step 2: Split into Chunks 

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""]
)

#  Step 3: Create Embedding Model 

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

#  Step 4: Build or Load ChromaDB 

if os.path.exists(CHROMA_PERSIST_DIR):
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
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
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=COLLECTION_NAME
    )
    print(f"Stored {vectorstore._collection.count()} vectors in ChromaDB")