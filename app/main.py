"""
Promtior RAG Chatbot - Main Application
Uses LangChain + LangServe + Groq
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from langserve import add_routes
from app.rag_chain import create_rag_chain
from app.ingest import ingest_documents
import os

app = FastAPI(
    title="Promtior RAG Chatbot",
    description="A RAG-based chatbot that answers questions about Promtior",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Absolute path to the static folder — works regardless of where uvicorn is called from
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


@app.on_event("startup")
async def startup_event():
    print("Ingesting Promtior website content...")
    ingest_documents()
    print("Documents ingested. Setting up RAG chain...")
    rag_chain = create_rag_chain()
    add_routes(app, rag_chain, path="/chat")
    print("Chatbot ready!")


@app.get("/")
def root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health")
def health():
    return {"status": "healthy"}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
