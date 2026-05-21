# 🔍 Neat-RAG

**A production-ready Agentic Retrieval-Augmented Generation (RAG)** that turns private document collections — including **PDF, DOCX, Markdown, HTML, TXT, and Images** — into a queryable knowledge base. It offers ultimate flexibility by supporting both **high-performance cloud LLMs** (OpenAI, Gemini, Anthropic, DeepSeek) and **fully private, offline local models** via Ollama. Features include hybrid search, agentic tool routing, automatic inline citations, and a self-service API-key onboarding flow.

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](docker-compose.yml)
[![Pydantic AI](https://img.shields.io/badge/Pydantic_AI-Agent-E92063)](https://ai.pydantic.dev/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-17-4169E1?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-0080FF?logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![LLM Providers](https://img.shields.io/badge/LLM_Providers-Ollama_|_OpenAI_|_Anthropic_|_Google_|_DeepSeek-orange)](#-tech-stack)

<video src="https://github.com/user-attachments/assets/69214298-2aef-4eca-b346-4ef35f4adb87" controls="controls" muted="muted" loop="loop" autoplay="autoplay" width="100%"></video>

> ⭐️ **💖 If you like this project, a star would mean the world to me!**

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [How It Works](#️-how-it-works)
- [Installation](#-installation)
- [Usage](#-usage)
- [Access Control & Permissions](#-access-control--permissions)
- [Project Structure](#-project-structure)
- [Follow-up development plans](#-follow-up-development-plans)
- [Contributing](#-contributing)
- [License & Contact](#-license--contact)

---

## ✨ Features

| Feature | Description                                                                                            |
|---|--------------------------------------------------------------------------------------------------------|
| 📄 Multi-Format Ingestion | Uploads and indexes PDF, DOCX, Markdown, HTML, TXT, and Images files.                                  |
| 🔌 Flexible LLM Support | Works with any major cloud provider (OpenAI, Gemini, Anthropic, DeepSeek) or fully offline via Ollama. |
| 🔀 Hybrid Search | Finds relevant content by combining semantic vector search with keyword-based full-text search.        |
| 🎯 Smart Retrieval | Improves answer quality through query rewriting, multi-query decomposition, and neural reranking.      |
| 🔗 Inline Citations | Backs every answer with numbered `[1][2]` source references linked to the exact document passage.      |
| 🧠 Conversation Memory | Keeps track of chat history within a session for natural, multi-turn dialogue.                         |
| ⚡ Streaming Chat | Streams answers token-by-token for a responsive, real-time chat experience.                            |
| 🔑 User & Access Management | Controls who can access the system through an admin-managed API key and invite system.                 |
| 📊 Quality Evaluation | Measures answer quality with built-in RAGAS metrics (Faithfulness, Relevancy, Precision).              |
| 🖥️ Web UI | Provides a Streamlit interface for chatting, managing documents, and monitoring ingestion jobs.        |

---

## 🎬 Architecture

### System Architecture

```
┌─────────────────┐     REST / SSE      ┌──────────────────────────────────────────┐
│   Streamlit UI  │ ─────────────────▶  │              FastAPI (port 8058)          │
│   (port 8501)   │                      │  ┌──────────┐  ┌──────────┐  ┌────────┐ │
└─────────────────┘                      │  │  /chat   │  │  /docs   │  │ /admin │ │
                                         │  └────┬─────┘  └────┬─────┘  └────┬───┘ │
                                         │       │              │              │      │
                                         │  ┌────▼──────────────▼──────────────▼───┐ │
                                         │  │          Agent Orchestrator           │ │
                                         │  │  (Pydantic AI + tool routing)         │ │
                                         │  └──────┬─────────────────────┬──────────┘ │
                                         │  ┌──────▼──────┐    ┌─────────▼──────────┐ │
                                         │  │  Retrieval   │    │   LLM Provider     │ │
                                         │  │  · Hybrid    │    │  OpenAI / Gemini   │ │
                                         │  │  · HyDE/MQ   │    │  Anthropic / Ollama│ │
                                         │  │  · Reranker  │    └────────────────────┘ │
                                         │  └──────┬───────┘                           │
                                         └─────────│─────────────────────────────────  ┘
                                                   │
                          ┌────────────────────────┼──────────────────────────────┐
                          │                         │                              │
                   ┌──────▼──────┐          ┌───────▼──────┐             ┌────────▼──────┐
                   │  PostgreSQL  │          │    Qdrant     │             │     Redis      │
                   │  (pgvector)  │          │ (vector store)│             │  (job queue)   │
                   └─────────────┘          └──────────────┘             └───────────────┘
```

### API Key Onboarding Flow

```
Admin                          New User
  │                               │
  │── POST /admin/invites ──▶ [invite token]
  │                               │
  │              ◀── POST /auth/redeem ──── {token}
  │                               │
  │              ──── {api_key} ──▶
  │                               │
  │                        Uses X-API-Key header
```


---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.12 |
| **API Framework** | FastAPI 0.115 + Uvicorn |
| **Agent Orchestration** | Pydantic AI (tool-calling agent) |
| **LLM Providers** | OpenAI, Google Gemini, Anthropic, DeepSeek, Ollama |
| **Embedding Providers** | OpenAI, Gemini, Ollama, custom OpenAI-compatible |
| **Vector Store (integrated)** | PostgreSQL 17 + pgvector (HNSW index) |
| **Vector Store (dedicated)** | Qdrant |
| **Relational Database** | PostgreSQL 17 via asyncpg |
| **Full-Text Search** | PostgreSQL `tsvector` / BM25 |
| **Schema Validation** | Pydantic v2 + pydantic-settings |
| **DB Migrations** | Alembic |
| **Background Jobs** | arq + Redis |
| **Document Parsing** | docling (PDF/DOCX/HTML), python-docx, trafilatura |
| **OCR** | pytesseract (10+ languages) + Pillow |
| **Reranking** | sentence-transformers (BGE CrossEncoder), Cohere API |
| **Chunking** | LangChain RecursiveCharacterTextSplitter + semantic chunker |
| **Rate Limiting** | slowapi |
| **Evaluation** | RAGAS |
| **UI** | Streamlit |
| **Containerisation** | Docker + Docker Compose (multi-stage build) |

---

## ⚙️ How It Works

### Document Ingestion Pipeline

When a file is uploaded, the API creates a job record and pushes it to a **Redis queue** via arq. The background worker picks it up and runs a four-stage pipeline:

**1. Extract** — behaviour differs by file type:

- **PDF**: docling converts the file to clean Markdown, preserving tables and structure. SmolVLM (a small vision-language model bundled in docling) additionally generates natural-language descriptions for embedded images and figures. For large PDFs, image description is auto-disabled above a configurable page threshold to prevent memory pressure.
- **DOCX / HTML / Markdown / TXT**: parsed into plain text via python-docx, trafilatura, or a plain reader.
- **Images (JPG / PNG / GIF / WebP / BMP)**: two extractions run independently:
  - **SmolVLM description** (via Docling) — generates a natural-language description of the visual content. Effective for photos, diagrams, and charts where embedded text is absent.
  - **OCR** (pytesseract, 10+ languages configurable via `OCR_LANGUAGES` in `.env`) — extracts verbatim text character by character. Effective for screenshots, scanned text, and tables.
  
  The original image file is also copied to persistent storage — required so the LLM can receive it at query time.

**2. Chunk** — text documents are split by `RecursiveCharacterTextSplitter` (fixed overlap) or a semantic chunker. Images bypass the generic chunker entirely and always produce exactly **two targeted chunks**:

| Chunk type | Content | Best for |
|---|---|---|
| `image_description` | SmolVLM natural-language description | Scenic photos, diagrams, charts |
| `image_ocr` | pytesseract verbatim text | Screenshots, scanned tables, dense text |

Both chunks carry an `image_path` pointer to the stored original. Either may be absent if its extractor produced no output, but at least one must exist for ingestion to succeed.

**3. Embed** — all chunks are sent to the configured embedding provider (OpenAI / Gemini / Ollama) in a batch.

**4. Store** — document metadata is committed to PostgreSQL first; then chunk vectors are written to the active vector store (pgvector or Qdrant). This order is intentional: a failed vector upsert leaves detectable orphan metadata (recoverable by re-indexing) rather than silent orphan vectors.

The job status (`pending → processing → completed / failed`) and progress percentage are updated in real time so the UI can poll `GET /jobs/{job_id}`.

**At query time**, image chunks retrieved by search are handled differently based on the LLM mode (set `LLM_MULTIMODAL` in `.env`):

- **`LLM_MULTIMODAL=false`** — the description and OCR text are passed as plain context, like any other chunk. No vision capability required from the LLM.
- **`LLM_MULTIMODAL=true`** — in addition to the text chunks, the raw image bytes are loaded from disk and injected as `BinaryContent` into the LLM message, allowing a vision-capable model (e.g. GPT-4o, Gemini 1.5, LLaVA, Qwen2-VL) to reason directly over the image. Each unique image is attached at most once per query.

### Hybrid Search & Query Enhancement

Before retrieval, the query optionally passes through two enhancement steps:

- **HyDE (Hypothetical Document Embeddings)** — the LLM generates a short hypothetical answer, which is embedded and used as the search vector instead of the raw question embedding. This closes the vocabulary gap between short queries and long document passages.
- **Multi-Query decomposition** — the LLM rewrites the question into several sub-questions, each retrieved independently. Results are merged via **Reciprocal Rank Fusion (RRF)**, which re-ranks candidates by their harmonic position across multiple result lists without needing score calibration.

Hybrid retrieval then combines two signals:

| Signal | Backend | Strength |
|---|---|---|
| Dense (semantic) | pgvector / Qdrant HNSW | Captures meaning and paraphrase |
| Sparse (keyword) | PostgreSQL `tsvector` / BM25 | Exact term matching, rare tokens |

Both result sets are merged again with RRF before reranking.

### Neural Reranking

The top-*k* merged candidates are passed to a **cross-encoder reranker** (BGE CrossEncoder locally, or the Cohere Rerank API). Unlike bi-encoders that embed query and document separately, a cross-encoder attends to both simultaneously — producing more accurate relevance scores at the cost of higher latency. Only the top-*n* reranked chunks proceed to generation.

### Agentic Orchestration & Inline Citations

The core of the system is a **Pydantic AI agent** configured with a set of tools (retrieval, summarisation, direct answer, etc.). Given a user question, the agent decides at runtime which tools to call and in what order, rather than following a fixed retrieval-then-generate pipeline. This lets it handle multi-step questions, refuse out-of-scope queries, or fall back to a general answer when retrieval confidence is low.

After the LLM generates a response, a post-processing step scans the text for `[1]`, `[2]` markers, matches each back to the exact chunk that was retrieved, and returns a structured `citations` array alongside the answer — so every claim is traceable to its source passage.

---

## 🚀 Installation

### Option A — Docker Compose (recommended)

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/) or Docker Engine + Compose v2.

**1. Clone the repository**

```bash
git clone https://github.com/YikunHuang123/neat_rag.git
cd neat_rag
```

**2. Create your environment file**

```bash
cp .env.example .env
```

Open `.env` and fill in at minimum — choose **one** of the following LLM setups:

**Cloud provider (OpenAI / Gemini / Anthropic / DeepSeek)**

```bash
# Pick one key that matches your chosen LLM_MODEL
GEMINI_API_KEY=...
# DEEPSEEK_API_KEY=...
# OPENAI_API_KEY=sk-...
# GEMINI_API_KEY=...
# ANTHROPIC_API_KEY=...

LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash

EMBEDDING_PROVIDER=gemini
EMBEDDING_MODEL=gemini-embedding-001

# Admin bootstrap key — any secret string you choose
ADMIN_BOOTSTRAP_KEY=your-very-secret-admin-key
```

**Local models via Ollama (no API key required)**

```bash
# Point the app at your local Ollama instance
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:7b          # or any model you have pulled
LLM_BASE_URL=http://host.docker.internal:11434   # from inside Docker
# LLM_BASE_URL=http://localhost:11434            # for local dev (Option B)

EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
# EMBEDDING_BASE_URL=http://host.docker.internal:11434

# Admin bootstrap key — any secret string you choose
ADMIN_BOOTSTRAP_KEY=your-very-secret-admin-key
```

> Make sure Ollama is running and you have pulled the required models:
> ```bash
> ollama pull qwen2.5:7b
> ollama pull nomic-embed-text
> ```

**3. Build and start all services**

```bash
docker compose up --build
```

This starts six containers: `postgres`, `qdrant`, `redis`, `api` (port **8058**), `worker`, and `ui` (port **8501**).

**4. Run database migrations**

```bash
docker compose exec api alembic upgrade head
```

**5. Open the UI**

Navigate to `http://localhost:8501` in your browser.
The interactive API docs are at `http://localhost:8058/docs`.

---

### Option B — Local Development

**Prerequisites:** [Conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda / Anaconda), PostgreSQL 17 with pgvector, Redis, (optionally) Qdrant.

**1. Clone the repository**

```bash
git clone https://github.com/YikunHuang123/neat_rag.git
cd neat_rag
```

**2. Create and activate a Conda environment**

```bash
conda create -n neat_rag python=3.12 -y
conda activate neat_rag
```

**3. Install dependencies**

```bash
pip install -e ".[dev]"
```

**4. Configure environment**

```bash
cp .env.example .env
# Edit .env — see the LLM setup options in Option A above
# At minimum set: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, ADMIN_BOOTSTRAP_KEY
```

**5. Apply database migrations**

```bash
alembic upgrade head
```

**6. Start the API server**

```bash
uvicorn neat_rag.api:create_app --factory --host 0.0.0.0 --port 8058 --reload
```

**7. Start the background worker** (separate terminal)

```bash
arq neat_rag.ingestion.pipeline.WorkerSettings
```

**8. Start the Streamlit UI** (separate terminal, optional)

```bash
streamlit run src/neat_rag/ui.py
```

---

## 💡 Usage

### Workflow Overview

The end-to-end journey from first deployment to a working knowledge base:

```
1. Deploy the stack
   └─ docker compose up --build
      docker compose exec api alembic upgrade head

2. Admin onboards a new user
   └─ POST /admin/invites  →  { token }  →  shared out-of-band

3. User exchanges token for a personal API key  (one-time use)
   └─ POST /auth/redeem  →  { api_key: "nrag_..." }

4. User uploads documents
   └─ POST /documents/upload  →  { job_id, filename, message }

5. Background worker ingests and indexes documents
   └─ GET /jobs/{job_id}  →  { id, status: "completed", progress: 1.0 }

6. User queries the knowledge base
   ├─ POST /chat         →  { answer, citations }          (blocking)
   └─ POST /chat/stream  →  SSE delta stream + citations   (real-time)
```

All API calls require the `X-API-Key` header. The **admin bootstrap key** (set as `ADMIN_BOOTSTRAP_KEY` in `.env`) is active immediately after deployment; all other users must be onboarded through the invite flow described below.

---

### Uploading a document
<p align="center">
  <img width="80%" alt="neat_rag_upload" src="https://github.com/user-attachments/assets/6f6b8547-a66a-4765-af7c-13d8b265c9b6" />
</p>

```bash
curl -X POST http://localhost:8058/documents/upload \
  -H "X-API-Key: admin" \   # Defined as ADMIN_BOOTSTRAP_KEY in .env
  -F "file=@/path/to/report.pdf"
```

```json
{
  "job_id": "3f2a1b...",
  "filename": "report.pdf",
  "message": "File uploaded. Ingestion started."
}
```

### Checking ingestion progress
<p align="center">
  <img width="80%" alt="neat_rag_upload" src="https://github.com/user-attachments/assets/e963542e-12b7-4f74-90b0-b7cfc4d9aa50" />
</p>


```bash
curl http://localhost:8058/jobs/3f2a1b... \
  -H "X-API-Key: admin"   # Defined as ADMIN_BOOTSTRAP_KEY in .env
```

```json
{
  "id": "3f2a1b...",
  "filename": "report.pdf",
  "status": "completed",
  "progress": 1.0,
  "error": null,
  "created_at": "2026-05-20T10:00:00Z",
  "updated_at": "2026-05-20T10:02:30Z"
}
```

### Asking a question (blocking) - Support multi-round dialogue

<p align="center">
  <img width="80%" alt="neat_rag_chat" src="https://github.com/user-attachments/assets/f3bce90b-7042-49a6-810c-75726f0026ae" />
  <br>
  <sub><i>Users can conduct multiple rounds of dialogue. Click the reference icon to view the original source passage.</i></sub>
</p>

> **Note:** `session_id` is required and must reference an existing session. Create one first via `POST /sessions`, then use its `id` here.

```bash
curl -X POST http://localhost:8058/chat \
  -H "X-API-Key: admin" \   # Defined as ADMIN_BOOTSTRAP_KEY in .env
  -H "Content-Type: application/json" \
  -d '{"message": "What are the key findings in the report?", "session_id": "s1a2b3..."}'
```

```json
{
  "session_id": "s1a2b3...",
  "content": "The report highlights three main findings: ... [1][2]",
  "message_id": "m9x8y7...",
  "citations": [
    {"citation_number": 1, "document_title": "report.pdf", "document_source": "report.pdf", "content_snippet": "Revenue grew 12%..."},
    {"citation_number": 2, "document_title": "report.pdf", "document_source": "report.pdf", "content_snippet": "Operating margin..."}
  ]
}
```

### Streaming response (SSE)

```bash
curl -N -X POST http://localhost:8058/chat/stream \
  -H "X-API-Key: admin" \   # Defined as ADMIN_BOOTSTRAP_KEY in .env
  -H "Content-Type: application/json" \
  -d '{"message": "Summarise the methodology section.", "session_id": "s1a2b3..."}'
```

```
data: {"delta": "The methodology"}
data: {"delta": " section describes"}
data: {"delta": " a two-stage process..."}
data: {"done": true, "citations": [...]}
```

### Admin Operations

All admin endpoints require the `ADMIN_BOOTSTRAP_KEY` from `.env` (or an API key with `scopes: ["admin"]`).

> The default administrator key is "admin".

#### Invite Token Management

<p align="center">
  <img width="80%" alt="neat_rag_upload" src="https://github.com/user-attachments/assets/01240b78-8906-41f5-b52c-506838577107" />
  <br>
  <sub><i>Administrator generates invitation code</i></sub>
</p>

<p align="center">
  <img width="80%" alt="neat_rag_upload" src="https://github.com/user-attachments/assets/4104b4a3-4c1e-4f53-abad-f7db7894cb69" />
  <br>
  <sub><i>The user obtains their key by redeeming the invite (each token is single-use)</i></sub>
</p>

<p align="center">
  <img width="80%" alt="neat_rag_upload" src="https://github.com/user-attachments/assets/0ed38374-fb28-4ce0-bc9d-f48405a3eeeb" />
  <br>
  <sub><i>Users enter their API key in the UI to start a conversation</i></sub>
</p>

```bash
# Create a single-use invite (7-day expiry by default; range 1–90 days)
curl -X POST http://localhost:8058/admin/invites \
  -H "X-API-Key: admin"   # Defined as ADMIN_BOOTSTRAP_KEY in .env

# {"id": "...", "token": "inv_abc123...", "used": false, "expires_at": "2026-05-27T..."}

# Create an invite for a named user, expiring in 30 days
curl -X POST http://localhost:8058/admin/invites \
  -H "X-API-Key: admin" \
  -H "Content-Type: application/json" \
  -d '{"owner": "alice", "expires_in_days": 30}'

# List all invite tokens
curl http://localhost:8058/admin/invites \
  -H "X-API-Key: admin"

# Revoke an unused invite
curl -X DELETE "http://localhost:8058/admin/invites/{invite_id}" \
  -H "X-API-Key: admin"
```

The user redeems the token once to receive their personal API key:

```bash
curl -X POST http://localhost:8058/auth/redeem \
  -H "Content-Type: application/json" \
  -d '{"token": "inv_abc123..."}'

# {"owner": "alice", "raw_key": "nrag_xyz789..."}
```

> The raw key is shown **exactly once** and is never stored. If a user loses their key, revoke it and issue a new invite.

#### API Key Management

<p align="center">
  <img width="80%" alt="neat_rag_upload" src="https://github.com/user-attachments/assets/79470c73-4e48-4720-bf69-ee5a8b939aee" />
  <br>
  <sub><i>Administrators can check the key creation time and last usage time of the assigned user.</i></sub>
</p>


```bash
# Create a key directly (bypasses the invite flow)
curl -X POST http://localhost:8058/admin/keys \
  -H "X-API-Key: admin" \
  -H "Content-Type: application/json" \
  -d '{"owner": "bob", "scopes": []}'

# {"id": "...", "owner": "bob", "scopes": [], "raw_key": "nrag_..."}

# List all keys (optionally filter by owner: ?owner=alice)
curl http://localhost:8058/admin/keys \
  -H "X-API-Key: admin"

# Revoke a key permanently
curl -X DELETE "http://localhost:8058/admin/keys/{key_id}" \
  -H "X-API-Key: admin"

# Adjust a user's permissions (see Access Control & Permissions section)
curl -X PATCH "http://localhost:8058/admin/keys/{key_id}/permissions" \
  -H "X-API-Key: admin" \
  -H "Content-Type: application/json" \
  -d '{"can_upload": false, "can_delete": false}'
```

---

## 🔐 Access Control & Permissions

<p align="center">
  <img width="80%" alt="neat_rag_upload" src="https://github.com/user-attachments/assets/d5d3b57b-ff7c-4ef7-a350-7448c73c5290" />
  <br>
  <sub><i>Administrators can set the permissions to upload and delete the knowledge base and chat for the assigned users.</i></sub>
</p>

### Roles

Neat-RAG has three levels of access, checked on every protected request:

| Role | Credential | Capabilities |
|---|---|---|
| **Admin** | `ADMIN_BOOTSTRAP_KEY` or API key with `scopes: ["admin"]` | All admin endpoints + all user operations; bypasses all permission flags |
| **Regular user** | API key issued via invite or direct admin creation | Chat, upload, delete — governed by per-key permission flags |
| **Unauthenticated** | No `X-API-Key` header | Blocked on all protected endpoints (403) |

### Per-Key Permission Flags

Each regular API key carries four boolean flags, all `true` by default:

| Flag | Protects | Error message when denied |
|---|---|---|
| `is_active` | Every endpoint — master on/off switch | `"Account is disabled"` |
| `can_upload` | `POST /documents/upload` | `"Upload permission denied"` |
| `can_delete` | `DELETE /documents/{id}` | `"Delete permission denied"` |
| `can_chat` | `POST /chat` and `POST /chat/stream` | `"Chat permission denied"` |

Admin-scoped keys bypass all four flags entirely.

### Adjusting Permissions

Use `PATCH /admin/keys/{key_id}/permissions` to update any combination of flags in real time — no restart required:

```bash
# Disable a user (soft ban)
curl -X PATCH "http://localhost:8058/admin/keys/{key_id}/permissions" \
  -H "X-API-Key: admin" \
  -H "Content-Type: application/json" \
  -d '{"is_active": false}'

# Read-only access: chat allowed, no upload or delete
curl -X PATCH "http://localhost:8058/admin/keys/{key_id}/permissions" \
  -H "X-API-Key: admin" \
  -H "Content-Type: application/json" \
  -d '{"can_upload": false, "can_delete": false, "can_chat": true}'

# Restore full access
curl -X PATCH "http://localhost:8058/admin/keys/{key_id}/permissions" \
  -H "X-API-Key: admin" \
  -H "Content-Type: application/json" \
  -d '{"is_active": true, "can_upload": true, "can_delete": true, "can_chat": true}'
```

### Delegating Admin Access

To give a team member admin privileges without sharing the bootstrap key, create an admin-scoped API key:

```bash
curl -X POST http://localhost:8058/admin/keys \
  -H "X-API-Key: admin" \
  -H "Content-Type: application/json" \
  -d '{"owner": "ops-team", "scopes": ["admin"]}'
```

> **Security note:** Admin-scoped keys bypass all permission flags and have unrestricted access. Treat them with the same care as the bootstrap key itself.

### Data Isolation

**Chat sessions** are always user-scoped. Each API key owner has an independent conversation history — one user can never read or continue another user's sessions.


**Knowledge base (documents)** is shared across all users by default: any document uploaded by any user is searchable by everyone. To give each user their own isolated document namespace, set the following in `.env` before starting the stack:

```env
DATABASE_SHARED=false
```

With `DATABASE_SHARED=false`, each user's uploaded documents are stored and indexed under their own namespace and are invisible to other users. Admin-scoped keys retain cross-namespace visibility regardless of this setting.

> Changing `DATABASE_SHARED` after documents have already been ingested requires re-indexing existing data. Decide on this setting before the first document upload.

---

## 🗂 Project Structure

```
neat_rag/
├── docker-compose.yml          # Six-service stack (postgres, qdrant, redis, api, worker, ui)
├── docker/
│   └── Dockerfile              # Multi-stage build — targets: api | worker | ui
├── alembic.ini                 # Alembic migration configuration
├── migrations/
│   └── versions/
│       ├── 001_initial_schema.py      # Core tables + HNSW index
│       ├── 002_add_user_id.py
│       ├── 003_invite_tokens.py
│       ├── 004_migrate_to_qdrant.py
│       └── 005_add_user_permissions.py
├── src/neat_rag/
│   ├── config.py               # Pydantic Settings — all env vars with defaults
│   ├── models.py               # Domain models: Document, Chunk, Session, Message, Job…
│   ├── exceptions.py           # Custom exception hierarchy
│   ├── eval.py                 # RAGAS evaluation helpers
│   ├── ui.py                   # Streamlit frontend
│   ├── providers/
│   │   ├── llm.py              # LLM factory (OpenAI / Gemini / Anthropic / DeepSeek / Ollama)
│   │   ├── embedding.py        # Embedding provider factory
│   │   └── reranker.py         # BGE CrossEncoder + Cohere reranker
│   ├── ingestion/
│   │   ├── extractors.py       # PDF / DOCX / HTML / image text extraction
│   │   ├── chunkers.py         # Recursive and semantic chunking strategies
│   │   └── pipeline.py         # Extract → chunk → embed → store orchestration
│   ├── retrieval/
│   │   ├── retrievers.py       # VectorRetriever and HybridRetriever
│   │   ├── rerank.py           # Post-retrieval reranking
│   │   ├── rewrite.py          # HyDE, Multi-Query, RRF fusion
│   │   └── citation.py         # Inline citation extraction
│   ├── agent/
│   │   ├── orchestrator.py     # run_query() — main public entry point
│   │   ├── tools.py            # Agent tools + AgentContext (dependency injection)
│   │   ├── memory.py           # Conversation history management
│   │   ├── prompts.py          # System prompt templates
│   │   └── title.py            # Auto-generated session titles
│   ├── db/
│   │   ├── pool.py             # asyncpg connection pool
│   │   ├── vector_store.py     # Abstract VectorStoreBase + backend factory
│   │   ├── pgvector_store.py   # PostgreSQL + pgvector implementation
│   │   ├── qdrant.py           # Qdrant implementation
│   │   ├── documents.py        # Document & chunk CRUD
│   │   ├── sessions.py         # Session & message CRUD
│   │   ├── jobs.py             # Ingestion job tracking
│   │   ├── feedback.py         # User feedback CRUD
│   │   ├── api_keys.py         # API key management
│   │   └── invites.py          # Invite token CRUD
│   └── api/
│       ├── __init__.py         # FastAPI app factory with lifespan management
│       ├── schemas.py          # Request / response DTOs
│       ├── deps.py             # FastAPI Depends factories
│       ├── middleware.py       # Auth, rate limiting, request tracing
│       ├── health.py           # /health/live  &  /health/ready
│       ├── documents.py        # Document & job routes
│       ├── chat.py             # Blocking and SSE streaming chat
│       ├── sessions.py         # Session CRUD + message history
│       ├── feedback.py         # User feedback routes
│       ├── admin.py            # API key + invite management
│       └── auth.py             # Public redeem-invite endpoint
├── test_ingestion.py           # Phase 1 — ingestion pipeline tests
├── test_agent.py               # Phase 2 — agent orchestration tests
└── test_orchestrator_flow.py   # Phase 5 — advanced retrieval tests
```

---

## 🔮 Follow-up development plans

- **Frontend Overhaul** — Refactor the existing Streamlit UI into **React** application for improved interactivity, state management, and performance.
- **Enhanced Feedback Loop** — Currently, user "likes" and "dislikes" are collected and stored. Future updates will leverage this data for:
    - **Offline Quality Assessment**: Systematic analysis of user feedback to identify and fix failure modes in the RAG pipeline.
    - **RLHF Integration**: Utilizing user preferences to fine-tune response generation and alignment.
    - **Dynamic Re-ranking**: Implementing a feedback-aware retrieval layer that boosts the scores of document snippets that have historically received positive user ratings.
- **LangGraph-Powered Advanced Retrieval** — Introduce [LangGraph](https://github.com/langchain-ai/langgraph) to replace the current fixed-step pipeline with a stateful, graph-based workflow, enabling three advanced retrieval strategies:
    - **Adaptive RAG**: After each retrieval attempt, a dedicated judge node evaluates result quality. If the retrieved chunks are insufficient, the graph loops back and retries with an alternative strategy (e.g., expanded queries or a different search mode) before proceeding to reranking and generation.
    - **Plan-and-Execute (Multi-step Reasoning)**: For complex questions, a planner node first decomposes the query into sub-questions. Each sub-question is then retrieved and answered independently, and the results are synthesised into a final, coherent response — improving accuracy across multi-document reasoning tasks.
    - **Corrective RAG (CRAG)**: After generating an answer, a verification node checks whether the response is grounded in the retrieved documents. If confidence is low, the graph triggers a fallback (e.g., web search or broader retrieval) before returning the final answer, reducing hallucinations on knowledge-boundary queries.
- **The markdown output of current models occasionally appears incorrectly rendered**
- **Hallowed citation markers sometimes appear** — This is mainly depends on the model's ability to follow instructions, will be considered in the future to continue optimizing prompts。

---
## 🤝 Contributing

Contributions, bug reports, and feature requests are welcome.

1. **Fork** the repository and create a feature branch:
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Make your changes** — follow the existing code style (Ruff / Black formatting).

3. **Add or update tests** for any new behaviour:
   ```bash
   pytest -v
   ```

4. **Commit** with a descriptive message:
   ```bash
   git commit -m "feat: add XYZ retrieval strategy"
   ```

5. **Open a Pull Request** against `main`. Include:
   - A clear description of the problem solved
   - Steps to reproduce (for bug fixes)
   - Any relevant environment or config changes

**Reporting bugs:** Please open a GitHub Issue with the label `bug`, your Python version, and a minimal reproduction snippet.

---

## 📄 License & Contact

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

**Author:** Yikun Huang
**Email:** yikun.huang@hotmail.com
**GitHub:** [@YikunHuang123](https://github.com/YikunHuang123)

> Built as a full-stack RAG engineering showcase — covering async API design, agentic LLM orchestration, hybrid retrieval, and production-grade containerisation.
