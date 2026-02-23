"""
Ingest Promtior website content.
Usa SKLearnVectorStore en memoria - sin dependencias pesadas.
"""

import os
import json
import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

DOCS_FILE = "/code/chroma_db/documents.json"

PROMTIOR_URLS = [
    "https://www.promtior.ai",
    "https://www.promtior.ai/service",
    "https://www.promtior.ai/use-cases",
]


def scrape_url(url: str):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "head", "meta", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        lines = [line for line in text.splitlines() if len(line.strip()) > 30]
        clean_text = "\n".join(lines)
        if len(clean_text) < 100:
            return None
        print(f"  OK: {url} ({len(clean_text)} chars)")
        return {"content": clean_text, "source": url}
    except Exception as e:
        print(f"  Failed: {url} — {e}")
        return None


def ingest_documents():
    os.makedirs(os.path.dirname(DOCS_FILE), exist_ok=True)

    print("Scraping Promtior website...")
    scraped = []
    for url in PROMTIOR_URLS:
        doc = scrape_url(url)
        if doc:
            scraped.append(doc)

    all_docs = scraped

    if not scraped:
        print("Scraping failed.")
    else:
        print(f"Scraped {len(scraped)} pages.")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = []
    for doc in all_docs:
        splits = splitter.split_text(doc["content"])
        for split in splits:
            chunks.append({"content": split, "source": doc["source"]})

    # Save as plain JSON - no vector DB needed
    with open(DOCS_FILE, "w") as f:
        json.dump(chunks, f)

    print(f"Saved {len(chunks)} chunks to {DOCS_FILE}")


def get_relevant_chunks(question: str, k: int = 4) -> str:
    
    with open(DOCS_FILE, "r") as f:
        chunks = json.load(f)

    question_words = set(question.lower().split())

    # Score each chunk by keyword overlap
    scored = []
    for chunk in chunks:
        content_lower = chunk["content"].lower()
        score = sum(1 for word in question_words if word in content_lower)
        scored.append((score, chunk["content"]))

    # Return top-k chunks sorted by score
    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [content for _, content in scored[:k]]
    return "\n\n".join(top_chunks)    
