# Promtior RAG Chatbot

Chatbot RAG que responde preguntas sobre [Promtior](https://www.promtior.ai) usando:

- **LangChain** — orquestación, chains, loaders, splitters
- **LangServe** — expone el chain como REST API
- **Groq API** — LLM gratuito y rápido (llama3-8b)
- **HuggingFace Embeddings** — embeddings locales sin API
- **ChromaDB** — vector store
- **FastAPI** — web framework

---

## Estructura del proyecto

```
promtior-rag-chatbot/
├── app/
│   ├── __init__.py          ← archivo vacío, necesario para Python
│   ├── main.py              ← FastAPI + LangServe + UI
│   ├── ingest.py            ← scraping + ChromaDB
│   ├── rag_chain.py         ← chain RAG con Groq
│   └── static/
│       └── index.html       ← interfaz de chat
├── terraform/
│   └── main.tf              ← infraestructura AWS
├── scripts/
│   └── deploy.sh            ← deploy a AWS
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Correr en local

### 1. Obtener API key gratuita de Groq
- Entrá a https://console.groq.com
- Creá una cuenta (gratis, sin tarjeta)
- Andá a *API Keys* → *Create API Key*

### 2. Crear archivo `.env`
```
GROQ_API_KEY=gsk_tu_key_aqui
```

### 3. Levantar con Docker
```bash
docker compose up --build
```

Abrí http://localhost:8000 en el navegador.

---

## Deploy en AWS

```bash
export GROQ_API_KEY=gsk_tu_key_aqui
export AWS_REGION=us-east-1
chmod +x scripts/deploy.sh
./scripts/deploy.sh
```
