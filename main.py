import os
from contextlib import asynccontextmanager
import time
import logging
from typing import List

from fastapi import FastAPI
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pydantic import BaseModel
from langchain_core.stores import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_community.document_loaders import DirectoryLoader
from langchain_classic.storage import LocalFileStore, create_kv_docstore

PERSIST_DIR = os.environ.get("PERSIST_DIR", "ragchroma")
PERSIST_DIR_DOCSTORE = os.environ.get("PERSIST_DIR_DOCSTORE", "ragdocstore")
BASE_URL = os.environ.get("BASE_URL", "")

recursive_retriever = None
markdown_retriever = None
parent_retriever = None
summary_retriever = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global recursive_retriever
    global parent_retriever
    global summary_retriever
    global markdown_retriever
    
    embeddings = OllamaEmbeddings(model="qwen3-embedding:8b", base_url=BASE_URL)

    print("Loading recursive...")
    vectordb = Chroma(persist_directory=PERSIST_DIR+"/recursive", embedding_function=embeddings)
    recursive_retriever = vectordb.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.3, "k": 3}  # Return top 3 most relevant chunks
    )

    print("Loading markdown...")
    vectordb = Chroma(persist_directory=PERSIST_DIR+"/markdown", embedding_function=embeddings)
    markdown_retriever = vectordb.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.3, "k": 3}  # Return top 3 most relevant chunks
    )

    print("Loading hierarchical...")
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=0)
    vectorstore = Chroma(persist_directory=PERSIST_DIR+"/hierarchical", embedding_function=embeddings)
    local_file_store = LocalFileStore(PERSIST_DIR_DOCSTORE+"/hierarchical")
    docstore = create_kv_docstore(local_file_store)
    parent_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    print("Loading summary...")
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    local_file_store = LocalFileStore(PERSIST_DIR_DOCSTORE+"/summary")
    docstore = create_kv_docstore(local_file_store)
    vectorstore = Chroma(persist_directory=PERSIST_DIR+"/summary", embedding_function=embeddings)
    summary_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key="original_id", # Links summaries to parent chunks
        child_splitter=child_splitter
    )

    print("All vector stores loaded.")

    yield
    # Clean up resources if needed
    print("Shutting down.")

app = FastAPI(lifespan=lifespan)

class QueryRequest(BaseModel):
    q: str

class QueryResponse(BaseModel):
    page_content: str
    metadata: dict


@app.post("/recursive", response_model=List[QueryResponse])
async def query_retriever(request: QueryRequest):
    """
    Queries the vector store with a given question and returns relevant documents.
    """
    print(f"Received query: {request.q}")
    
    start_time = time.monotonic()

    docs = recursive_retriever.invoke(request.q)

    end_time = time.monotonic()
    duration = end_time - start_time
    print(f"--- Time spent on retriever.invoke: {duration:.2f} seconds ---")

    return docs

@app.post("/markdown", response_model=List[QueryResponse])
async def query_retriever(request: QueryRequest):
    """
    Queries the vector store with a given question and returns relevant documents.
    """
    print(f"Received query: {request.q}")
    
    start_time = time.monotonic()

    docs = markdown_retriever.invoke(request.q)

    end_time = time.monotonic()
    duration = end_time - start_time
    print(f"--- Time spent on retriever.invoke: {duration:.2f} seconds ---")

    return docs


@app.post("/hierarchical", response_model=List[QueryResponse])
async def query_retriever(request: QueryRequest):
    """
    Queries the vector store with a given question and returns relevant documents.
    """
    print(f"Received query: {request.q}")
    
    start_time = time.monotonic()

    docs = parent_retriever.invoke(request.q)

    end_time = time.monotonic()
    duration = end_time - start_time
    print(f"--- Time spent on retriever.invoke: {duration:.2f} seconds ---")

    return docs

@app.post("/summary", response_model=List[QueryResponse])
async def query_retriever(request: QueryRequest):
    """
    Queries the vector store with a given question and returns relevant documents.
    """
    print(f"Received query: {request.q}")
    
    start_time = time.monotonic()

    docs = summary_retriever.invoke(request.q)

    end_time = time.monotonic()
    duration = end_time - start_time
    print(f"--- Time spent on retriever.invoke: {duration:.2f} seconds ---")

    return docs

