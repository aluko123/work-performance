# Documentation Index

Quick reference to project documentation organized by category.

---

## 📋 Plans (`plans/`)

Current work items, planning documents, and TODO lists.

**Key file:** [plans/TODO.md](plans/TODO.md)

---

## 📊 Summary (`summary/`)

High-level summaries of features, migrations, and architectural changes.

**Recent:**
- [ARCHITECTURE_NEW.md](summary/ARCHITECTURE_NEW.md) - New OpenAI native agent system
- [FRONTEND_POLISH_COMPLETE.md](summary/FRONTEND_POLISH_COMPLETE.md) - UI improvements
- [FINAL_FIXES_SUMMARY.md](summary/FINAL_FIXES_SUMMARY.md) - Citations and markdown cleanup
- [FRONTEND_CLEANUP_SUMMARY.md](summary/FRONTEND_CLEANUP_SUMMARY.md) - Endpoint migration
- [CITATION_FIX_SUMMARY.md](summary/CITATION_FIX_SUMMARY.md) - Citation extraction

---

## 🔧 Implementation (`implementation/`)

Technical implementation details, debugging notes, and fixes.

**Recent:**
- [DEDUPLICATION_FIX.md](implementation/DEDUPLICATION_FIX.md) - Citation deduplication
- [INDEXING_ISSUE_FOUND.md](implementation/INDEXING_ISSUE_FOUND.md) - Embedding dimension mismatch
- [DEPRECATED_FILES.md](implementation/DEPRECATED_FILES.md) - Files to remove

---

## 🏗️ Core Documentation

- [README.md](README.md) - Main project README
- [AGENTS.md](AGENTS.md) - Agent guidelines and commands
- [PROFILING.md](PROFILING.md) - Performance profiling notes



# ARCHITECTURE
Architecture

  - Backend (FastAPI + ARQ worker): Uploads files,
  extracts structured utterances, runs BERT-based
  scoring, persists results, exposes analytics and a
  streaming chat endpoint.
  - Data/DB: SQLAlchemy models for Analysis and
  Utterance; Postgres + pgvector in Docker, SQLite for
  local dev.
  - Vector search: pgvector embeddings for semantic “what
  was said” queries (no longer using Chroma).
  - Agent: OpenAI-native function-calling agent with
  explicit tools (stats, comparisons, search, charting)
  and SSE streaming.
  - Frontends: A static prototype (frontend/) and a newer
  React app (frontend-v2/).
  - Infra: Docker Compose for Postgres, Redis, backend,
  worker, frontend and a separate migrate container
  (Alembic).

  Key Backend Pieces

  - API wiring and SSE chat
      - backend/main.py:102 POST /analyze_text/ enqueues
  an ARQ job and returns a correlation job_id.
      - backend/main.py:116 GET /analysis_status/{job_id}
  polls Redis for results and normalizes msgpack/JSON
  payloads.
      - backend/main.py:206 GET /api/trends builds
  speaker-time series for charting.
      - backend/main.py:214 POST /api/chat streams
  agent output tokens and then structured JSON (bullets,
  citations, charts).
  - Worker + pipeline
      - backend/worker.py:15 ARQ worker preloads models
  and config; pipeline extracts document → analyzes
  utterances (batched) → saves to DB → enqueues
  embedding.
      - backend/document_extractor.py:20 Unstructured
  + optional Chunkr chunking; LLM JSON-mode extraction
  with fallback and post-processing (date fill-forward,
  dedupe).
      - backend/services.py:138 BERT inference (batched)
  for all per-utterance metrics + SA multilabel;
  computes aggregate scores; DB persistence at backend/
  services.py:188.
      - backend/embeddings.py:18 Generates OpenAI
  embeddings, stores in pgvector, marks is_indexed=True.
  - Agent + Tools
      - backend/agent.py:18 OpenAI client; builds a
  metadata-aware system prompt (date coverage, speakers)
  via Redis-cached stats.
      - Iterative tool-calling loop, ensures a chart is
  generated for trend/comparison queries even if the LLM
  forgets (backend/agent.py:262).
      - Tools for speakers/metrics, stats, comparisons,
  retrieval, chart data (backend/tools.py:600 mapping;
  definitions starting backend/tools.py:460). Chart data
  produced by backend/services.py:292.
  - DB and models
      - SQLAlchemy setup with SQLite performance pragmas
  for local dev (backend/database.py:22).
      - Tables in backend/db_models.py:6: Analysis,
  Utterance (JSON columns for predictions/aggregates;
  embedding vector; is_indexed flag).
      - API schemas in backend/models.py with Pydantic
  from_attributes for ORM models.

  Frontends

  - Static prototype
      - frontend/index.html, frontend/script.js — upload,
  list analyses, show radar and trends; basic chat UI
  using SSE.
  - React app (preferred)
      - Vite app under frontend-v2/, typed API client
  (frontend-v2/src/lib/api.ts:1), chat UI with charts
  (frontend-v2/src/components/RAGQuery.tsx:1, frontend-
  v2/src/components/ChartRenderer.tsx:1), routes for
  upload/analyses/trends.

  Infra

  - Docker Compose (docker-compose.yml:1) brings up
  Postgres (pgvector), Redis, backend, worker, migrate
  job (alembic upgrade), and frontend-v2. Volumes mount
  data/ and bert_classification/.
  - Dockerfile installs Python deps and runs Uvicorn
  (Dockerfile:1).
  - Requirements include FastAPI, SQLAlchemy, pgvector,
  transformers/torch, arq, alembic (requirements.txt:1).

  What’s Good

  - Clear separation between API and worker; uploads are
  non-blocking with Redis-backed status.
  - Batched inference for speed; SA multilabel inference
  included.
  - Clean tool surface for the agent; automatic chart
  injection keeps answers visual and consistent.
  - pgvector replaces Chroma cleanly; search tool falls
  back to LIKE for SQLite.
  - React v2 frontend cleanly consumes SSE and charts
  via Recharts.

  Gaps and Improvements

  - Endpoint/docs/test drift
      - Static frontend still calls /api/get_insights
  but backend exposes /api/chat (streaming). Update the
  static app to use /api/chat (frontend/script.js:302).
  Likewise README and tests still reference old RAG graph
  and Chroma.
      - Tests import removed prompts (answer/
  metadata) and hit /api/get_insights (backend/tests/
  test_prompts.py:1, backend/tests/test_api.py:20). Align
  tests to the new agent and endpoints or mark/remove
  legacy tests.
  - DB portability for date grouping
      - func.strftime is SQLite-only. Weekly grouping
  and date truncation should be database-aware. Replace
  backend/services.py:226 with Postgres-friendly logic
  (e.g., func.date_trunc('week', ...)) when on Postgres;
  keep func.strftime fallback for SQLite. Same for time-
  series paths in get_chart_data.
  - Mutable default in Pydantic
      - backend/models.py:28 sets utterances:
  List[Utterance] = []. Use = Field(default_factory=list)
  to avoid shared mutable defaults.
  - Charting consistency
      - The React renderer supports
  'grouped_bar' (frontend-v2/src/components/
  ChartRenderer.tsx:22), but generate_chart only
  allows 'line'|'bar' (backend/tools.py:571-
  575). Consider adding grouped-bar generation
  (single speaker across multiple metrics) using
  services.get_chart_data(group_by='metric').
  - Embedding throughput
      - backend/embeddings.py:33 calls the embeddings API
  per utterance. Batch requests (OpenAI supports multiple
  inputs) to reduce latency/cost and leverage vectorized
  inserts.
  - Metric registry vs model outputs
      - The metrics list for validation is static
  (backend/config/chart_config.py:28). The worker
  dynamically discovers columns from training data.
  Consider surfacing the discovered columns (or storing
  in DB/config) and unifying validation, display names,
  and charts.
  - Postgres-first types
      - Utterance.date and timestamp are stored as String
  (backend/db_models.py:16). If Postgres is primary in
  Docker, consider DATE and TIME types for efficient
  grouping/comparisons (with migration).
  - Resilience and UX polish
      - Make model IDs env-configurable (e.g., GPT_MODEL,
  EMBEDDING_MODEL) instead of hardcoding "gpt-4o-
  mini" (backend/document_extractor.py:44, backend/
  agent.py:62) and "text-embedding-3-small" (backend/
  tools.py:37, backend/embeddings.py:37).
      - Add timeouts/retries to OpenAI calls and embed
  generation with backoff; return actionable errors in /
  analysis_status/{job_id}.
      - Replace print-based logs with structured logging
  and add a /healthz route.
  - Cleanup and alignment
      - Remove or quarantine deprecated files/notes
  (README still references Chroma/LangGraph); ensure
  agent_functions.md, DOCS.md reflect the new agent+tools
  approach.
      - backend/main.py:16 ml_models = {} is unused; can
  be removed.
      - Trim or convert backend/tests/
  test_chart_tool.py:1 from print-based smoke to
  assertions.

  How to Run

  - Local: python -m venv .venv && source .venv/bin/
  activate && pip install -r requirements.txt then
  uvicorn backend.main:app --reload --port 8000, and
  serve frontend-v2 with docker compose up frontend or
  static prototype via python -m http.server 8001 -d
  frontend.
  - Docker (recommended): docker compose up --build
  with .env providing OPENAI_API_KEY, DATABASE_URL, etc.