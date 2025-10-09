import pandas as pd
import torch
import json
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import tempfile
from openai import AsyncOpenAI
from sqlalchemy.orm import Session
from typing import List
from transformers import BertModel
import redis
from langchain.cache import RedisCache
from langchain.globals import set_llm_cache
from arq import create_pool
from arq.connections import RedisSettings
import msgpack

# Local imports
from . import db_models
from .database import SessionLocal, init_db
from .models import Analysis as AnalysisModel, MultiTaskBertModel, TrendsResponse, RAGQuery, RAGAnswer, AsyncTask
from .services import get_speaker_trends
from .document_extractor import RobustMeetingExtractor
from .rag_service import PerformanceRAG
from .rag_graph import RAGGraph
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Configuration ---
MODEL_PATH = os.getenv('MODEL_PATH', 'bert_classification/multi_task_bert_model.pth')
DATA_PATH = os.getenv('DATA_PATH', 'bert_classification/master_training_data.csv')
CONFIG_PATH = os.getenv('CONFIG_PATH', 'backend/config/metric_groups.json')
SA_MODEL_PATH = os.getenv('SA_MODEL_PATH', 'bert_classification/sa_bert_model_multilabel/')
ARQ_REDIS_URL = os.getenv("ARQ_REDIS_URL", "redis://localhost:6379")

# --- Application State ---
ml_models = {}

# --- Database Dependency ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- FastAPI Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing application...")
    init_db()  # Initialize database

    # Log configured CORS origins for visibility
    cors_env = os.getenv("CORS_ORIGINS", "http://localhost:8001,http://127.0.0.1:8001")
    print(f"CORS_ORIGINS loaded: {cors_env}")

    # --- Arq Redis Pool ---
    redis_settings = RedisSettings.from_dsn(ARQ_REDIS_URL)
    app.state.arq_pool = await create_pool(redis_settings)

    # --- Enqueue Startup Indexing Task (Non-blocking) ---
    await app.state.arq_pool.enqueue_job("startup_indexing_task")
    print("Enqueued startup indexing task.")

    # --- Caching, Model Loading, etc. (remains the same) ---
    # ... (rest of the original lifespan logic for model loading)
    print("Loading models and other resources...")
    # NOTE: The original blocking indexing logic is now removed from here.

    # --- Initialize RAG Graph for chatbot endpoint ---
    # Ensure attribute exists even if initialization fails, so endpoint returns a clear 500
    app.state.rag_graph = None
    try:
        app.state.rag_graph = RAGGraph()
        print("RAG graph initialized and attached to app state.")
    except Exception as e:
        # Keep None and log; endpoint will respond with a controlled 500
        print(f"Failed to initialize RAG graph: {e}")
    
    yield
    
    print("Closing application resources...")
    await app.state.arq_pool.close()
    ml_models.clear()

# --- FastAPI App Initialization ---
app = FastAPI(lifespan=lifespan)

# --- CORS Middleware ---
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8001,http://127.0.0.1:8001")
origins = [origin.strip() for origin in CORS_ORIGINS.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Dependency for Arq Pool ---
async def get_arq_pool(request: Request):
    return request.app.state.arq_pool

# --- API Endpoints ---
@app.post("/analyze_text/", response_model=AsyncTask)
async def analyze_text_endpoint(text_file: UploadFile = File(...), arq_pool = Depends(get_arq_pool)):
    contents = await text_file.read()
    corr_id = uuid.uuid4().hex
    # Enqueue with correlation id so worker can persist result under this key
    await arq_pool.enqueue_job('process_document_task', contents, text_file.filename, corr_id)
    return {"job_id": corr_id}

@app.get("/analysis_status/{job_id}")
async def get_analysis_status(job_id: str, arq_pool = Depends(get_arq_pool), db: Session = Depends(get_db)):
    import asyncio
    result = None
    # 1) Primary: read worker-persisted result by correlation id
    try:
        r = redis.from_url(ARQ_REDIS_URL)
        raw = r.get(f"job_result:{job_id}")
        if raw:
            try:
                result = json.loads(raw.decode('utf-8'))
            except Exception:
                # fallback to msgpack if needed
                try:
                    result = msgpack.unpackb(raw, raw=False)
                except Exception:
                    result = None
    except Exception:
        result = None

    # 2) Secondary (legacy IDs): try ARQ if available
    if result is None and hasattr(arq_pool, 'result'):
        try:
            result = await arq_pool.result(job_id)
        except Exception:
            result = None
    
    if result is None:
        return {"job_id": job_id, "status": "PENDING"}

    # Normalize task return into a stable response
    # If ARQ returned bytes, try msgpack first, then JSON as fallback
    if isinstance(result, (bytes, bytearray, memoryview)):
        raw_bytes = bytes(result)
        decoded = None
        # Try msgpack
        try:
            decoded = msgpack.unpackb(raw_bytes, raw=False)
        except Exception:
            decoded = None
        if decoded is None:
            # Try JSON
            try:
                decoded = json.loads(raw_bytes.decode("utf-8"))
            except Exception:
                decoded = None
        if isinstance(decoded, dict):
            result = decoded
    elif isinstance(result, str):
        try:
            parsed = json.loads(result)
            if isinstance(parsed, dict):
                result = parsed
        except Exception:
            pass

    if isinstance(result, dict):
        status = result.get("status", "COMPLETED")
        resp = {"job_id": job_id, "status": status}
        if status == "COMPLETED" and "analysis_id" in result:
            analysis_id = result.get("analysis_id")
            # Small retry to ensure the analysis row is visible to the API container/DB
            for _ in range(3):
                try:
                    from . import db_models
                    found = db.query(db_models.Analysis).filter(db_models.Analysis.id == analysis_id).first()
                    if found:
                        break
                except Exception:
                    pass
                await asyncio.sleep(0.1)
            resp["analysis_id"] = analysis_id
            try:
                print(f"Analysis job {job_id} completed; analysis_id={analysis_id}")
            except Exception:
                pass
        if status == "FAILED" and "error" in result:
            resp["error"] = result.get("error")
        return resp

    # Fallback if task returned a non-dict value
    return {"job_id": job_id, "status": "COMPLETED"}

@app.get("/analyses/", response_model=List[AnalysisModel])
def get_analyses(db: Session = Depends(get_db)):
    analyses = db.query(db_models.Analysis).order_by(db_models.Analysis.created_at.desc()).all()
    return analyses

@app.get("/analyses/{analysis_id}", response_model=AnalysisModel)
def get_analysis(analysis_id: int, db: Session = Depends(get_db)):
    analysis = db.query(db_models.Analysis).filter(db_models.Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return analysis

@app.get("/api/trends", response_model=TrendsResponse)
def get_trends(metric: str, period: str = 'daily', db: Session = Depends(get_db)):
    return get_speaker_trends(db=db, metric=metric, period=period)

@app.post("/api/get_insights")
async def rag_query_endpoint(query: RAGQuery, request: Request):
    # This endpoint does not need arq, but shows how to access shared state if needed
    rag_graph: RAGGraph = request.app.state.rag_graph
    if not rag_graph:
        raise HTTPException(status_code=500, detail="RAG graph is not available.")

    async def stream_generator():
        final_data = {}
        async for chunk in rag_graph.astream_run(
            question=query.question, session_id=query.session_id,
            filters={
                "speaker": query.speaker, "date_from": query.date_from,
                "date_to": query.date_to, "top_k": query.top_k,
            },
        ):
            if "answer_token" in chunk:
                yield f"data: {json.dumps({'answer_token': chunk['answer_token']})}\n\n"
            else:
                final_data = chunk
        yield f"data: {json.dumps(final_data)}\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")
