"""
RAG Chain usando LangChain + Groq.
Con cache en memoria de 10 minutos.
"""

import os
import time
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser
from app.ingest import get_relevant_chunks

SYSTEM_PROMPT = """You are an assistant that answers questions about Promtior using the context below.

Rules:
- Answer in a single, complete paragraph.
- Do NOT ask follow-up questions.
- Do NOT say "Would you like to know more?" or similar.
- State the facts clearly and end the response.

Context:
{context}
"""

FOUNDING_KEYWORDS = {"founded", "founding", "foundation", "established", "when", "year", "started"}
FOUNDING_SUFFIX = "\n\nBut... Someone told me Promtior was founded in 2023 😊."
CACHE_TTL = 600  # 10 minutos en segundos

# Cache en memoria: { pregunta: (respuesta, timestamp) }
_cache: dict = {}


def is_founding_question(question: str) -> bool:
    words = set(question.lower().split())
    return bool(FOUNDING_KEYWORDS & words)


def get_cached(question: str):
    """Retorna la respuesta cacheada si existe y no expiró."""
    key = question.strip().lower()
    if key in _cache:
        response, timestamp = _cache[key]
        if time.time() - timestamp < CACHE_TTL:
            print(f"Cache hit para: '{question}' (expira en {int(CACHE_TTL - (time.time() - timestamp))}s)")
            return response
        else:
            print(f"Cache expirado para: '{question}'")
            del _cache[key]
    return None


def set_cache(question: str, response: str):
    """Guarda la respuesta en cache."""
    key = question.strip().lower()
    _cache[key] = (response, time.time())
    print(f"Cache guardado para: '{question}'")


def create_rag_chain():
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0.1,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    def full_chain(question: str) -> str:
        # 1. Revisar cache primero
        cached = get_cached(question)
        if cached:
            return cached

        # 2. Si no hay cache, llamar al LLM
        response = (
            RunnableLambda(lambda q: {"context": get_relevant_chunks(q), "question": q})
            | prompt
            | llm
            | StrOutputParser()
        ).invoke(question)

        # 3. Appendear suffix si es pregunta de founding
        if is_founding_question(question):
            response += FOUNDING_SUFFIX

        # 4. Guardar en cache
        set_cache(question, response)

        return response

    return RunnableLambda(full_chain)
