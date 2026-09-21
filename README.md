# RAG Chatbot with Role-Based Access Control

A full-stack Retrieval-Augmented Generation service that answers questions
over enterprise documents, with **RBAC enforced at the vector-retrieval
layer** — not just at the API gate. A user can only ever retrieve (and the
LLM can only ever see) chunks their role and department are entitled to.

This project implements the "RAG Chatbot with Role-Based Access Control"
line item from my resume end-to-end: FastAPI + Flask REST APIs, LangChain
orchestration, ChromaDB/FAISS-style vector search, JWT auth with RBAC,
n8n-driven ingestion automation, and SQL-backed audit logging.

## Architecture

```
                         ┌─────────────────────────┐
   n8n (folder watch) ──►│  Flask ingestion webhook │──► chunk → embed → ChromaDB
                         │   app_flask_ingest.py    │        (RBAC metadata tagged)
                         └─────────────────────────┘              │
                                                                    │
   User ──► JWT login ──► FastAPI query API ──► RBAC filter ──► ChromaDB similarity search
            (app/main.py)   (app/routers/)      (app/rag/          │
                                                  vectorstore.py)   ▼
                                                              Prompt Template
                                                                    │
                                                                    ▼
                                                          LLM (OpenAI / extractive)
                                                                    │
                                                                    ▼
                                                          Answer + audit log (SQL)
```

**Two services, on purpose:** the FastAPI app serves live, low-latency user
queries; the Flask app is a separate, independently-scalable microservice
that only handles ingestion, triggered asynchronously by n8n. Decoupling
bulk/slow ingestion from the synchronous query path is what the resume's
"~40% reduction in query response latency" refers to — ingestion no longer
competes with query traffic for the same process's resources.

## Why RBAC lives in the retrieval layer, not just the API layer

A naive implementation checks "is this user allowed to use the chatbot?"
at the API gate and then lets any authenticated user query the whole
vector store. That's a data leak waiting to happen — the LLM will happily
summarize a confidential finance doc for an intern if it's in the
retrieved context.

Instead, every chunk is tagged at ingestion time with `department` and
`access_level` (`public` / `internal` / `confidential`). Every query
builds a ChromaDB `where` filter from the *authenticated* user's role and
department **before** the similarity search runs (`app/rag/vectorstore.py:
build_rbac_filter`). Unauthorized chunks are excluded from the candidate
set entirely — they never reach the prompt, so there's no way for the LLM
to leak them, no matter how the question is phrased.

| Role       | Access levels visible        | Department-scoped? |
|------------|-------------------------------|---------------------|
| admin      | public, internal, confidential | No — sees all departments |
| manager    | public, internal, confidential | Yes — own department + public |
| employee   | public, internal               | Yes — own department + public |
| guest      | public                          | No — public only |

## Project layout

```
app/
  main.py              FastAPI app + router registration
  config.py            Settings (env-driven)
  database.py           SQLAlchemy engine/session
  models.py             User, DocumentRecord, QueryLog ORM models
  schemas.py             Pydantic request/response models
  security.py            Password hashing, JWT create/decode
  auth.py                 get_current_user / require_roles dependencies
  rag/
    embeddings.py         Offline embedding function (see note below)
    ingestion.py           Load -> chunk -> embed -> store
    vectorstore.py          ChromaDB wrapper + RBAC filter builder
    chain.py                 Prompt template + retrieval-augmented generation
  routers/
    auth_router.py            /auth/register, /auth/login
    documents_router.py        /documents/upload, /documents/ (admin/manager)
    query_router.py             /query/ (the RAG endpoint)
app_flask_ingest.py    Flask microservice: n8n-triggered ingestion webhook
scripts/seed_demo.py   Creates demo users + ingests sample_docs/
data/sample_docs/      4 sample docs spanning departments & access levels
n8n/                   Documented n8n ingestion workflow (JSON export)
tests/test_rbac.py     Pytest suite: RBAC enforcement + API auth
Dockerfile, docker-compose.yml
```

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Seed demo users + sample documents (creates rbac_rag.db + chroma_db/)
python scripts/seed_demo.py

# Run the query API
uvicorn app.main:app --reload --port 8000

# In a second terminal, run the ingestion webhook (optional, for the n8n flow)
flask --app app_flask_ingest run --port 5001
```

Or with Docker: `docker compose up --build`.

### Demo users (created by seed_demo.py)

| username        | password       | role     | department  |
|-----------------|----------------|----------|-------------|
| alice_admin     | Admin123!      | admin    | general     |
| bob_manager     | Manager123!    | manager  | engineering |
| carol_employee  | Employee123!   | employee | hr          |
| dave_guest      | Guest123!      | guest    | general     |

### Try it

```bash
# Login
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -d "username=carol_employee&password=Employee123!" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['access_token'])")

# Ask a question - carol (HR) will NOT see the confidential finance doc
curl -s -X POST http://localhost:8000/query/ \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"question":"What was the Q3 revenue and margin?"}' | python3 -m json.tool

# Log in as alice_admin instead and re-run - she WILL see it.
```

Interactive API docs: `http://localhost:8000/docs`

### Run tests

```bash
pytest -v
```

## On the embedding model (read this before an interview)

This repo ships with `LocalHashingEmbeddings` (`app/rag/embeddings.py`) — a
scikit-learn `HashingVectorizer`-based embedding function with **zero
external dependencies**: no API key, no model download, no GPU. This was a
deliberate choice for this portfolio version so the entire pipeline is
runnable offline, in CI, and by anyone cloning the repo with no setup
friction.

In the version described in my resume, this slot was filled by a real
embedding model (sentence-transformers / OpenAI `text-embedding-3-small`),
which materially changes retrieval quality — the hashing trick captures
lexical overlap, not semantic similarity, so e.g. "cost" vs "price" won't
match without shared tokens. Swapping providers is a one-line change:
`get_embedding_function()` in `app/rag/embeddings.py` is the single seam.

## On the LLM

`LLM_PROVIDER=extractive` (default) returns the best-matching chunk
directly with no external call, so the whole system is demoable and
testable without an API key. Set `LLM_PROVIDER=openai` + `OPENAI_API_KEY`
in `.env` to get real generative answers from the retrieved context
(`app/rag/chain.py: _call_llm`).

## Roadmap / what I'd add next

- Swap `LocalHashingEmbeddings` for a real embedding model + reranker
- Streaming responses (SSE) from the query endpoint
- Alembic migrations instead of `create_all` on startup
- Per-document ACL overrides (beyond department/access_level) for
  cross-department exceptions
- Rate limiting per user/role on `/query/`
