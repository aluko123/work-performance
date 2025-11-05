# System Architecture

**Last Updated:** 2025-11-03  
**Status:** Production-ready  
**Stack:** FastAPI + Postgres + pgvector + OpenAI SDK

---

## 🏗️ Current Architecture

### Database Layer
- **Postgres 16** with **pgvector** extension
- Single database for all data (analyses, utterances, embeddings)
- Fast semantic search via vector similarity (`<=>` operator)
- Alembic migrations for schema management

### Backend (FastAPI)
```
backend/
├── main.py              # API endpoints, startup logic
├── agent.py             # Conversational agent with tool calling
├── tools.py             # Tool implementations (search, stats, charts)
├── embeddings.py        # OpenAI embedding generation
├── metadata.py          # Redis-backed metadata cache
├── database.py          # DB connection & session management
├── db_models.py         # SQLAlchemy ORM models
├── models.py            # Pydantic API models
├── worker.py            # ARQ background worker
├── document_extractor.py # File upload processing
├── parsing.py           # Text extraction
├── services.py          # Business logic utilities
├── metrics.py           # Metric calculations
├── utils.py             # Helper functions
├── sanitization.py      # Input validation
└── config/              # Configuration (chart metrics, etc.)
```

### Frontend (React + TypeScript)
```
frontend-v2/
├── src/
│   ├── components/      # UI components (AnalysisCard, ChatInterface)
│   ├── lib/
│   │   ├── api.ts       # API client
│   │   └── types.ts     # TypeScript types
│   └── App.tsx          # Main app
└── Dockerfile           # Nginx-based production build
```

### Data Flow

```
1. Upload Flow:
   User uploads file → /analyze_text
   → ARQ worker processes async
   → BERT inference + parsing
   → Save to Postgres
   → Generate embeddings (background)

2. Chat Flow:
   User asks question → /api/chat
   → Agent (agent.py) with OpenAI SDK
   → Calls tools (tools.py):
      - search_utterances (pgvector semantic search)
      - get_metric_stats (aggregation queries)
      - generate_chart (matplotlib)
   → Streams response with charts + citations
   → Saves conversation to Redis

3. Chart Generation (auto-enforced):
   Comparison query detected
   → get_metric_stats called for N speakers
   → System auto-injects generate_chart if LLM forgets
   → Returns base64 chart image
```

---

## 📊 Key Components

### 1. Conversational Agent (`agent.py`)
- OpenAI native SDK with streaming
- Tool calling with automatic chart enforcement
- Redis conversation history
- Max 3 iterations to prevent loops

### 2. Tools System (`tools.py`)
- `list_speakers` - Get available speakers
- `list_metrics` - Get available metrics
- `search_utterances` - Semantic search via pgvector
- `get_metric_stats` - Aggregation queries
- `compare_periods` - Time-based comparisons
- `generate_chart` - Bar/line chart generation

### 3. Embeddings (`embeddings.py`)
- OpenAI `text-embedding-3-small` (1536 dimensions)
- Batch generation with rate limiting
- Stored in Postgres vector column
- IVFFlat index for fast similarity search

### 4. Background Worker (`worker.py`)
- ARQ (async Redis queue)
- Handles long-running inference tasks
- Startup indexing for new utterances
- Processes files asynchronously

---

## 🗄️ Database Schema

### Tables
```sql
analyses
├── id (PK)
├── source_filename
└── created_at

utterances
├── id (PK)
├── analysis_id (FK → analyses.id)
├── date, timestamp, speaker
├── text
├── predictions (JSON)      -- BERT scores
├── aggregated_scores (JSON) -- Aggregated metrics
├── sa_labels (JSON)        -- Situation awareness labels
├── is_indexed (BOOLEAN)    -- Embedding status
└── embedding (vector(1536)) -- Semantic search vector

Indexes:
- idx_utterances_embedding (IVFFlat for vector similarity)
- idx_utterance_date, idx_utterance_speaker (query optimization)
```

---

## 🔧 Infrastructure

### Docker Services
```yaml
postgres:    # Database (pgvector/pgvector:pg16)
migrate:     # Alembic migrations (runs on startup)
backend:     # FastAPI app (port 8000)
frontend:    # React app via Nginx (port 8001)
redis:       # Conversation cache + ARQ queue
arq_worker:  # Background task processor
```

### Environment Variables
```bash
DATABASE_URL               # Postgres connection
OPENAI_API_KEY            # OpenAI API access
REDIS_URL                 # Redis connection
MODEL_PATH                # BERT model path
SA_MODEL_PATH             # Situation awareness model
INFER_BATCH_SIZE=32       # Inference batch size
CORS_ORIGINS              # Allowed frontend origins
```

---

## 📦 Dependencies (Simplified)

**Core:**
- fastapi, uvicorn - Web framework
- sqlalchemy, psycopg2-binary, pgvector - Database
- openai - LLM & embeddings
- redis, arq - Caching & background jobs
- alembic - DB migrations

**ML/Processing:**
- torch, transformers - BERT inference
- scikit-learn - ML utilities
- pandas, numpy - Data manipulation

**Document Processing:**
- unstructured[pdf,xlsx] - File parsing
- chunkr_ai - Text chunking

**Removed (Nov 2025):**
- ~~langchain, langchain-community, langchain-openai, langchain-core~~
- ~~langgraph~~ (complex orchestration, replaced with simple tool calling)
- ~~chromadb~~ (replaced with pgvector)

---

## 🚀 Performance Characteristics

**Semantic Search:**
- ~10-20ms for top-K queries (pgvector + IVFFlat index)
- Supports 2K+ utterances with room to scale to 100K+

**Inference:**
- Batch size: 32 (configurable via `INFER_BATCH_SIZE`)
- ~50-100 utterances/min on CPU

**Embedding Generation:**
- ~50-100 embeddings/min (OpenAI rate limits)
- Background processing doesn't block uploads

**Chat Response:**
- <2s for simple queries
- 3-5s for complex queries with multiple tool calls
- Streaming tokens appear within 500ms

---

## 🎯 API Endpoints

### Active Endpoints
```
POST /analyze_text        # Upload & analyze document
GET  /analyses/           # List all analyses (paginated)
GET  /analyses/{id}       # Get specific analysis
GET  /api/trends          # Time-series data
POST /api/chat            # Conversational agent (primary)
```

### Removed Endpoints
```
POST /api/get_insights    # REMOVED (use /api/chat)
```

---

## 🔐 Security Notes

- All secrets in `.env` (never committed)
- CORS properly configured
- Database credentials: non-production defaults (change in prod)
- No exposed admin interfaces
- Input sanitization on all endpoints

---

## 📈 Recent Improvements (Nov 2025)

1. **SQLite → Postgres migration**
   - Production-ready database
   - Better concurrency & reliability
   - Transactional integrity

2. **ChromaDB → pgvector migration**
   - Single database (simpler ops)
   - Faster queries
   - Better scalability

3. **LangChain → OpenAI native SDK**
   - Removed ~1,100 lines of complex code
   - Simpler, more maintainable
   - Better streaming support
   - Automatic chart generation enforcement

4. **Codebase cleanup**
   - Removed 468 MB of unused data
   - Removed 6 deprecated dependencies
   - Cleaner file structure
   - Better documentation

---

## 🧪 Testing

**Run tests:**
```bash
docker compose run --rm backend pytest backend/tests/
```

**Key test files:**
- `test_agent.py` - Agent & tool calling
- `test_services.py` - Business logic
- `test_api.py` - API endpoints
- `test_prompts.py` - Prompt templates

---

## 📚 Documentation

- [AGENTS.md](AGENTS.md) - Dev guidelines & commands
- [README.md](README.md) - Project overview
- [POSTGRES_MIGRATION.md](POSTGRES_MIGRATION.md) - Migration guide
- [BACKLOG.md](BACKLOG.md) - Future features
- [DEPRECATED_FILES.md](implementation/DEPRECATED_FILES.md) - Cleanup history
- [PROFILING.md](PROFILING.md) - Performance profiling

---

## 🔄 Migration Status

| Component | From | To | Status |
|-----------|------|----|----|
| Database | SQLite | Postgres+pgvector | ✅ Complete |
| Vector Store | ChromaDB | pgvector | ✅ Complete |
| Agent Framework | LangGraph | OpenAI SDK | ✅ Complete |
| Embeddings | ChromaDB index | Postgres column | ✅ Complete |
| Frontend | frontend/ | frontend-v2/ | ✅ Complete |

---

## 🎨 Code Quality Metrics

**Backend:**
- 27 Python files
- ~4,500 total lines (down from ~5,600)
- Single responsibility principle
- Type hints throughout
- Comprehensive error handling

**Architecture Score:**
- ✅ Single database (no data fragmentation)
- ✅ Async everywhere (better concurrency)
- ✅ Streaming responses (better UX)
- ✅ Background processing (non-blocking uploads)
- ✅ Automatic failsafes (chart enforcement)
- ✅ Production-ready stack (Postgres, Redis, Docker)
