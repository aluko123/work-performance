# work-performance
AI powered workplace performance tracking platform

## Pipeline Diagram

```mermaid
graph TD
    subgraph User Interaction
        User("User")
        Frontend("Frontend (React)")
    end

    subgraph Backend Services
        BackendApi("FastAPI Backend")
        DocExtractor("Document Extractor")
        AnalysisEngine("Analysis Engine")
        RAG_Pipeline("RAG Pipeline")
    end

    subgraph Data & Models
        Database("SQL Database")
        VectorStore("Vector Store (ChromaDB)")
        BertModel("Multi-Task BERT Model")
        SAModel("SA Model")
        TrainingScripts("Offline Training Scripts")
        TrainingData("Labeled CSV Data")
    end

    subgraph External APIs
        OpenAI_Chunkr("OpenAI / ChunkrAI")
        OpenAI_LLM("OpenAI LLM")
    end

    %% Data Ingestion and Analysis Flow
    User -- 1. Uploads Transcript --> Frontend
    Frontend -- 2. POST /analyze_text/ --> BackendApi
    BackendApi -- 3. Raw Document --> DocExtractor
    DocExtractor -- 4. Extracts Text --> OpenAI_Chunkr
    OpenAI_Chunkr -- 5. Structured Transcript --> DocExtractor
    DocExtractor -- 6. Structured Transcript --> BackendApi
    BackendApi -- 7. Feeds Transcript to --> AnalysisEngine
    AnalysisEngine -- 8. Scores Utterances with --> BertModel
    AnalysisEngine -- " " --> SAModel
    AnalysisEngine -- 9. Saves Results --> Database
    Database -- 10. Indexes Utterances --> VectorStore

    %% RAG Query Flow
    User -- 11. Asks Question --> Frontend
    Frontend -- 12. POST /api/get_insights --> BackendApi
    BackendApi -- 13. Sends Query to --> RAG_Pipeline
    RAG_Pipeline -- 14. Retrieves Context from --> VectorStore
    RAG_Pipeline -- 15. Generates Answer with --> OpenAI_LLM
    OpenAI_LLM -- 16. Streams Answer --> RAG_Pipeline
    RAG_Pipeline -- 17. Streams to --> BackendApi
    BackendApi -- 18. Streams to --> Frontend
    Frontend -- 19. Displays Answer --> User

    %% Model Training Flow
    TrainingData -- Used by --> TrainingScripts
    TrainingScripts -- Creates --> BertModel
    TrainingScripts -- " " --> SAModel

    %% Styling
    classDef userStyle fill:#d4edda,stroke:#155724,stroke-width:2px
    class User userStyle

    classDef frontendStyle fill:#cce5ff,stroke:#004085,stroke-width:2px
    class Frontend frontendStyle

    classDef backendStyle fill:#f8d7da,stroke:#721c24,stroke-width:2px
    class BackendApi,DocExtractor,AnalysisEngine,RAG_Pipeline backendStyle

    classDef dataStyle fill:#fff3cd,stroke:#856404,stroke-width:2px
    class Database,VectorStore,BertModel,SAModel,TrainingScripts,TrainingData dataStyle

    classDef externalStyle fill:#e2e3e5,stroke:#383d41,stroke-width:2px
    class OpenAI_Chunkr,OpenAI_LLM externalStyle
```

# Design Document: AI-Powered Workplace Performance Tracking Platform

  1. High-Level Overview

  This document outlines the design and architecture of an 
  AI-powered workplace performance tracking platform. The 
  platform analyzes meeting transcripts to provide insights 
  into team performance, individual contributions, and overall 
  communication dynamics.

  The system is designed as a microservices architecture, 
  consisting of a React frontend, a FastAPI backend, a Redis 
  cache, and an arq worker for asynchronous task processing. 
  The backend leverages a Retrieval-Augmented Generation (RAG) 
  pipeline to answer user questions and provide insights from 
  the analyzed data.

  2. Frontend

  The frontend is a single-page application (SPA) built with 
  React and TypeScript. It provides a user-friendly interface 
  for uploading meeting transcripts, viewing analysis results, 
  and querying the RAG pipeline.

  Components

   - `FileUpload`: Allows users to upload a meeting transcript 
     file to the backend for analysis.
   - `AnalysisDisplay`: Displays the analysis results, including 
     speaker-specific metrics and a timeline of the conversation.

   - `RAGQuery`: Provides a chat-like interface for users to ask 
     questions about the analyzed data. It streams the answers 
     from the backend, providing a real-time experience.

  Interaction with Backend

  The frontend communicates with the backend via a REST API. It 
  uses the fetch API to make requests to the following endpoints:


   - POST /analyze_text/: Uploads a transcript for analysis.
   - GET /analysis_status/{job_id}: Polls for the status of an 
     analysis job.
   - GET /analyses/{analysis_id}: Retrieves the results of a 
     specific analysis.
   - POST /api/get_insights: Sends a query to the RAG pipeline 
     and streams the response.

  3. Backend

  The backend is a FastAPI application that provides the core 
  functionality of the platform. It exposes a REST API for the 
  frontend and interacts with the database, the arq worker, and
   the RAG pipeline.

  API Endpoints

   - POST /analyze_text/: Accepts a transcript file, saves it, 
     and enqueues a background job for processing.
   - GET /analysis_status/{job_id}: Checks the status of a 
     processing job.
   - GET /analyses/: Returns a list of all completed analyses.
   - GET /analyses/{analysis_id}: Returns the detailed results 
     of a single analysis.
   - GET /api/trends: Provides trend data for specific metrics.
   - POST /api/get_insights: The main endpoint for the RAG 
     pipeline, which streams answers to user queries.

  4. Asynchronous Task Processing

  The platform uses arq, a Python task queue, to process 
  analysis jobs asynchronously. This prevents the backend from 
  being blocked by long-running tasks and improves the overall 
  responsiveness of the application.

  When a user uploads a transcript, the backend enqueues a 
  process_document_task to the arq worker. The worker then 
  performs the following steps:

   1. Document Extraction: Extracts the text from the uploaded 
      file.
   2. Analysis: Analyzes the text using a multi-task BERT model 
      to score utterances on various metrics.
   3. Database Storage: Saves the analysis results to the 
      database.
   4. Indexing: Indexes the utterances in the Chroma vector store 
      for the RAG pipeline.

  5. RAG Pipeline

  The RAG pipeline is the core of the platform's insights 
  generation capabilities. It uses a combination of retrieval 
  and generation to answer user questions about the analyzed 
  data. The pipeline is implemented using langgraph and 
  consists of the following steps:

   1. Load History: Loads the conversation history from Redis to 
      provide context for the current query.
   2. Classify Query: Classifies the user's query to determine 
      the type of analysis required (e.g., trend analysis, 
      comparison, root cause analysis).
   3. Retrieve Docs: Retrieves relevant documents (utterances) 
      from the Chroma vector store based on the user's query.
   4. Compute Aggregates: Computes aggregate metrics from the 
      retrieved documents to provide a quantitative summary.
   5. Generate Draft: Generates a draft answer using a large 
      language model (LLM) based on the retrieved documents and 
      computed aggregates.
   6. Format Answer: Formats the final answer, including a 
      narrative summary, bullet points, and citations to the 
      source documents.
   7. Save History: Saves the current query and answer to the 
      conversation history in Redis.

  The RAG pipeline is designed to be streaming-first, providing a
   real-time experience for the user.

  6. Data Model

  The application uses a SQL database (PostgreSQL) to 
  store the analysis results. The database schema is managed 
  using Alembic migrations. The main table is the utterances 
  table, which stores the following information for each 
  utterance in a meeting:

   - id: The primary key.
   - text: The text of the utterance.
   - speaker: The speaker of the utterance.
   - timestamp: The timestamp of the utterance.
   - predictions: A JSON field containing the scores from the 
     analysis.
   - analysis_id: A foreign key to the analyses table.

  7. Deployment

  The application is deployed using Docker and docker-compose. 
  The docker-compose.yml file defines the following services:

   - frontend: The React frontend.
   - backend: The FastAPI backend.
   - migrate: A service to run database migrations.
   - redis: A Redis instance for caching and the arq task queue.
   - arq_worker: The arq worker for asynchronous task 
     processing.

  This multi-container setup allows for easy scaling and 
  management of the different components of the application.




## Granular Design Details

This section expands the high-level design with implementation-level details, mapping behaviors to concrete files, functions, and runtime flows.

### System Overview

- Components
  - `frontend-v2/`: React + Vite App for uploading files, viewing analyses, and chatting with the RAG assistant. `frontend/` contains an earlier static prototype.
  - `backend/`: FastAPI app exposing REST APIs; background jobs via `arq`; RAG using LangChain + Chroma; SQLite by default.
  - `data/`: SQLite database (`data/analysis.db`) and Chroma persistence (`data/chroma_db/`). Large models/data are mounted here in Docker.

- Startup wiring (API):
  - `backend/main.py:51` defines `lifespan` to initialize the DB, create an ARQ Redis pool, enqueue `startup_indexing_task`, and attach a `RAGGraph` instance to `app.state`.
  - CORS origins come from `CORS_ORIGINS`; defaults allow local dev at port 8001.

### Frontend (React SPA)

- File Upload
  - UI and submission logic in `frontend-v2/src/components/FileUpload.tsx`.
  - On submit, constructs `FormData` with key `text_file` and `POST`s to `${VITE_API_BASE_URL}/analyze_text/`.
  - Stores API response (an async `job_id`) in state for subsequent status polling/fetching.

- Analysis Display
  - Table rendering in `frontend-v2/src/components/AnalysisDisplay.tsx`.
  - Dynamically discovers metric keys from `utterances[*].predictions` and `utterances[*].aggregated_scores`.
  - User-friendly column labels loaded at app start from `frontend-v2/public/column_name_mapping.json` (see `frontend-v2/src/App.tsx`).

- RAG Assistant (Streaming)
  - Chat UI in `frontend-v2/src/components/RAGQuery.tsx`.
  - Generates/persists `session_id` in `localStorage` for conversation continuity.
  - `POST /api/get_insights` with `{ question, session_id }`; consumes Server-Sent Events (SSE):
    - Interim tokens arrive as `data: {"answer_token": "..."}` and are appended live.
    - Final message is a JSON object containing `bullets`, `metrics_summary`, `follow_ups`, `citations`, etc., used to enrich the UI and show follow-up buttons.

### Backend API (FastAPI)

- Initialization & CORS
  - `backend/main.py:51-88` initializes DB, ARQ pool, enqueues `startup_indexing_task`, and creates `RAGGraph`.
  - `backend/main.py:92-101` configures CORS (origins from `CORS_ORIGINS`).

- Endpoints
  - `POST /analyze_text/` → enqueues `process_document_task` with a correlation ID; returns `{ job_id }` (see `backend/main.py:108-114`).
  - `GET /analysis_status/{job_id}` → checks Redis for `job_result:{job_id}` and normalizes output; if done, includes `analysis_id` (see `backend/main.py:116-197`).
  - `GET /analyses/` and `GET /analyses/{analysis_id}` → list or fetch analyses with nested utterances.
  - `GET /api/trends?metric=...&period=daily|weekly` → aggregates over JSON columns to produce chart-ready data.
  - `POST /api/get_insights` → streams interim answer tokens and finally a structured payload for the RAG result (see `backend/main.py:214-236`).

### Asynchronous Processing (ARQ Worker)

- `process_document_task` (core pipeline) — `backend/worker.py:15`
  1) Load models and config
     - Discover metric columns from `DATA_PATH` CSV; build `MultiTaskBertModel` heads accordingly.
     - Load BERT base and multi-task state dict from `MODEL_PATH`.
     - Load SA multi-label model/tokenizer from `SA_MODEL_PATH`.
     - Load metric groupings JSON (`backend/config/metric_groups.json`).
  2) Document extraction
     - Write upload bytes to a temp file.
     - Run `RobustMeetingExtractor.process_any_document()` which orchestrates Unstructured + Chunkr + LLM (or a fallback) and returns a normalized list of `{date, timestamp, speaker, utterance}`.
  3) Utterance analysis
     - For each utterance, compute per-metric predictions (`predict_all_scores`) and SA labels, then compute group aggregates and totals (e.g., `Total_Comm_Score`).
  4) Persistence
     - Create `Analysis` and many `Utterance` rows with JSON `predictions`, `aggregated_scores`, `sa_labels` (see `backend/services.py:85-107`).
  5) Indexing enqueue
     - Enqueue `index_utterances_task` with new utterance IDs; persist `{status, analysis_id}` to Redis under `job_result:{corr_id}` for the API to poll.

- `index_utterances_task` — `backend/worker.py:120`
  - Load utterances by IDs; convert to LangChain `Document`s with rich page content (speaker/date/timestamp + scores inline).
  - Add to Chroma in batches (size from `BATCH_SIZE`), then `persist()`; mark `is_indexed=True` in DB.

- `startup_indexing_task` — `backend/worker.py:160`
  - On API startup, enqueue indexing for any historic utterances where `is_indexed == False` (safe backfill).

### Document Extraction Pipeline

- Primary path (if `CHUNKRAI_API_KEY` is set)
  - `unstructured.partition.auto.partition` extracts raw elements.
  - `Chunkr` semantically chunks the file; Unstructured + Chunkr results are merged.
  - LLM (`gpt-4o-mini`) extracts structured meeting rows per chunk with enforced JSON output.

- Fallback path (if Chunkr unavailable or errors)
  - Extract a global meeting date hint from the header via a dedicated LLM call; chunk Unstructured text by token budget.
  - Extract `{date, timestamp, speaker, utterance}` per chunk; fill-forward missing dates; deduplicate utterances.

### Data Model

- `backend/db_models.py`
  - `Analysis`: `id`, `source_filename`, `created_at` + relation `utterances`.
  - `Utterance`: FK `analysis_id`, core fields (`date`, `timestamp`, `speaker`, `text`), JSON `predictions`, JSON `aggregated_scores`, JSON `sa_labels`, and `is_indexed` flag.

- `backend/models.py` (API schemas)
  - Mirrors DB shapes for API responses, plus `RAGQuery`, `RAGCitation`, `RAGAnswer` contracts.

### Trends & Aggregation

- Endpoint: `GET /api/trends`
  - Computes average scores grouped by speaker and period using SQL over JSON columns; supports either raw `predictions[metric]` or `aggregated_scores[metric]`.
  - Shapes output for charting libraries: `labels` (periods) + `datasets[{ label: speaker, data: [...] }]`.

### RAG Pipeline (LangGraph + Chroma)

- Vector store & retriever
  - Chroma persisted at `./data/chroma_db`; embeddings via `OpenAIEmbeddings`.
  - Retriever uses MMR with `k=8`/`fetch_k=40` for diverse, relevant contexts.

- Graph nodes (see `backend/rag_graph.py`)
  - `load_history` → load last turns from Redis using `session_id` (opt-in continuity).
  - `classify` → rule-based query type classification (`facts`, `performance_trend`, `compare_entities`, `root_cause`).
  - `retrieve` → fetch docs from Chroma; apply `top_k` cap.
  - `aggregate` → DB query over `Utterance` with optional filters (`speaker`, `date_from`, `date_to`); compute per-metric averages and sample size.
  - `draft` → build prompt with question, aggregates, and references; JSON-mode LLM returns a draft structure.
  - `format` → validate/sanitize `metrics_summary`, resolve `source_ids` to final citations, attach metadata (e.g., data quality from sample size).
  - `save_history` → append Q/A to Redis history for conversational continuity.

- Streaming execution (`astream_run`)
  - Runs pre-nodes, then streams narrative `answer` as tokens.
  - Performs a second JSON-mode call to produce `bullets`, `metrics_summary`, `follow_ups`, `source_ids`; merges and yields final payload.

### Configuration & Secrets

- Environment variables
  - `OPENAI_API_KEY`, `CHUNKRAI_API_KEY`: model + Chunkr access.
  - `DATABASE_URL`: defaults to `sqlite:///./data/analysis.db`.
  - `MODEL_PATH`, `DATA_PATH`, `SA_MODEL_PATH`, `CONFIG_PATH`: model and training artifacts.
  - `ARQ_REDIS_URL` / `REDIS_URL`: for ARQ + history storage.
  - `CHROMA_DIR`, `BATCH_SIZE`, `CORS_ORIGINS`: vector store persistence, indexing batch size, and CORS list.

### End-to-End Flows

- Upload → Analyze → Index
  1) User uploads file via `FileUpload` → `POST /analyze_text/` returns `job_id`.
  2) Worker extracts utterances (Unstructured + Chunkr/LLM) → analyzes with multi-task BERT + SA → computes aggregates.
  3) Results saved as `Analysis` + `Utterance`s → Redis `job_result:{job_id}` is set.
  4) Indexing task embeds utterances and persists Chroma; marks `is_indexed=true`.
  5) Frontend polls `/analysis_status/{job_id}` → fetches `/analyses/{analysis_id}` for display.

- RAG Query (Streaming)
  1) Frontend submits `{ question, session_id }` to `/api/get_insights`.
  2) API streams `answer_token` chunks while composing final metadata.
  3) Final payload returns `bullets`, `metrics_summary`, `follow_ups`, and `citations`; UI renders follow-ups as buttons.

### Error Handling & Robustness

- Extraction
  - JSON-mode LLM with retry; chunk-level failures are skipped; missing dates are forward-filled.

- Job status durability
  - Worker stores `{status, analysis_id}` in Redis keyed by correlation ID; API polls Redis first, then ARQ fallback.

- RAG output normalization
  - Guards against malformed `metrics_summary`; validates `source_ids` and falls back to available citations.

### Deployment Notes

- Docker Compose includes services for API, worker, Redis, and frontend; mounts `bert_classification/` and `data/`.
- Local dev: `uvicorn backend.main:app --reload --port 8000`; Vite dev server in `frontend-v2/` or static server for `frontend/`.


# TO DO
- visualization chart /graph tool - 
- paragraph of work done so far -  
- hosting - 
- Observability
- chat history
- sanitize response
- optimize AI section, evals, streamlit setup
- getting data from Polaris



docker compose exec backend py-spy record -o /app/data/profile.svg --duration 60 --pid 1

for i in {1..50}; do curl -F "text-file=@sample_meeting.txt" http://localhost:8000/analyze_text/ & done


docker compose exec backend py-spy record -o /app/data/inference_profile.svg --duration 120 --pid 1

./load_test.sh