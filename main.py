import os
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pydantic import BaseModel

PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "chroma_db")
BASE_URL = os.environ.get("BASE_URL", "")

retriever = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever
    print("Loading model and vector store...")
    embeddings = OllamaEmbeddings(model="qwen3-embedding:8b", base_url=BASE_URL)
    vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    retriever = vectordb.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.3, "k": 3}  # Return top 3 most relevant chunks
    )
    print("Model and vector store loaded.")
    yield
    # Clean up resources if needed
    print("Shutting down.")

app = FastAPI(lifespan=lifespan)

class QueryRequest(BaseModel):
    q: str

class QueryResponse(BaseModel):
    page_content: str
    metadata: dict

@app.post("/query", response_model=List[QueryResponse])
async def query_retriever(request: QueryRequest):
    """
    Queries the vector store with a given question and returns relevant documents.
    """
    print(f"Received query: {request.q}")
    docs = retriever.invoke(request.q)
    return docs
