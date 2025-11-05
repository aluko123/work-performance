import pandas as pd
import torch
import json
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import tempfile
from openai import AsyncOpenAI
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List
from transformers import BertModel
import redis
# Removed: LangChain imports (not needed anymore)
from arq import create_pool
from arq.connections import RedisSettings
import msgpack

# Local imports
from . import db_models
from .database import SessionLocal, init_db
from .models import Analysis as AnalysisModel, AnalysesResponse, MultiTaskBertModel, TrendsResponse, AsyncTask, RAGQuery
from .services import get_speaker_trends
from .document_extractor import RobustMeetingExtractor
# Deprecated: RAG system replaced by simple pgvector search in tools.py
# Removed deprecated RAG imports
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Configuration ---
MODEL_PATH = os.getenv('MODEL_PATH', 'bert_classification/multi_task_bert_model.pth')
DATA_PATH = os.getenv('DATA_PATH', 'bert_classification/master_training_data.csv')
CONFIG_PATH = os.getenv('CONFIG_PATH', 'backend/config/metric_groups.json')
SA_MODEL_PATH = os.getenv('SA_MODEL_PATH', 'bert_classification/sa_bert_model_multilabel/')
ARQ_REDIS_URL = os.getenv("ARQ_REDIS_URL", "redis://localhost:6379")


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

    # Removed: LangChain LLM cache initialization (no longer using LangChain)

    print("Loading models and other resources...")
    # NOTE: The original blocking indexing logic is now removed from here.

    # RAG Graph removed - using OpenAI native SDK in /api/chat instead
    
    yield
    
    print("Closing application resources...")
    await app.state.arq_pool.close()

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

# Simple health check
@app.get("/healthz")
def healthz(db: Session = Depends(get_db)):
    from sqlalchemy import text as sa_text
    ok = {"status": "ok", "db": False, "redis": False}
    try:
        db.execute(sa_text("SELECT 1"))
        ok["db"] = True
    except Exception:
        ok["db"] = False
    try:
        r = redis.from_url(ARQ_REDIS_URL)
        ok["redis"] = bool(r.ping())
    except Exception:
        ok["redis"] = False
    return ok

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

@app.get("/analyses/", response_model=AnalysesResponse)
def get_analyses(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db)
):
    offset = (page - 1) * limit

    # Get total count
    total = db.query(func.count(db_models.Analysis.id)).scalar()

    # Get paginated analyses
    analyses = db.query(db_models.Analysis)\
        .order_by(db_models.Analysis.created_at.desc())\
        .offset(offset)\
        .limit(limit)\
        .all()

    has_more = offset + len(analyses) < total

    return {
        "items": analyses,
        "total": total,
        "page": page,
        "limit": limit,
        "has_more": has_more
    }

@app.get("/analyses/{analysis_id}", response_model=AnalysisModel)
def get_analysis(analysis_id: int, db: Session = Depends(get_db)):
    analysis = db.query(db_models.Analysis).filter(db_models.Analysis.id == analysis_id).first()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return analysis

@app.get("/api/trends", response_model=TrendsResponse)
def get_trends(metric: str, period: str = 'daily', db: Session = Depends(get_db)):
    return get_speaker_trends(db=db, metric=metric, period=period)

@app.post("/api/chat")
async def chat_endpoint(query: RAGQuery):
    """
    New conversational agent endpoint using OpenAI native SDK.
    Replaces complex LangGraph with simple tool calling.
    """
    from .agent import run_agent
    
    async def stream_generator():
        answer_text = ""
        tool_calls_made = 0
        charts = []
        citations = []
        seen_sources = set()  # Track unique sources across all tool calls
        
        async for chunk in run_agent(query.question, query.session_id or "default"):
            chunk_type = chunk.get("type")
            
            if chunk_type == "token":
                # Stream answer tokens
                answer_text += chunk["content"]
                yield f"data: {json.dumps({'answer_token': chunk['content']})}\n\n"
            
            elif chunk_type == "status":
                # Stream progress updates
                yield f"data: {json.dumps({'status': chunk['message']})}\n\n"
            
            elif chunk_type == "tool_result":
                # Extract chart data and citations from tool results
                result = chunk.get("result", {})
                tool_name = chunk.get("tool_name", "")
                
                print(f"🔍 Tool result: {tool_name}, has 'type': {result.get('type') if isinstance(result, dict) else 'N/A'}")
                
                # Extract charts
                if isinstance(result, dict) and result.get("type") in ["line", "bar"]:
                    print(f"✅ Adding chart: {result.get('type')} for {result.get('metric')}")
                    charts.append(result)
                
                # Extract citations from search_utterances
                if tool_name == "search_utterances" and isinstance(result, list):
                    for item in result:
                        if isinstance(item, dict) and not item.get("error"):
                            source_id = item.get("source_id")
                            # Fallback dedupe key if source_id is missing
                            fallback_key = f"{item.get('speaker','')}|{item.get('date','')}|{item.get('timestamp','')}|{hash((item.get('text') or '')[:100])}"
                            key = source_id or fallback_key
                            
                            # Only add if we haven't seen this source and under limit
                            if key not in seen_sources and len(citations) < 3:
                                seen_sources.add(key)
                                citations.append({
                                    "speaker": item.get("speaker"),
                                    "date": item.get("date"),
                                    "timestamp": item.get("timestamp"),
                                    "snippet": item.get("text", "")[:500]  # First 500 chars
                                })
            
            elif chunk_type == "final":
                tool_calls_made = chunk.get("tool_calls_made", 0)
                answer_text = chunk["answer"]
        
        # After streaming complete, send structured data
        # Strip markdown images from answer (LLM sometimes generates them)
        import re
        answer_text = re.sub(r'!\[.*?\]\(.*?\)', '', answer_text)
        
        # Helper to strip markdown formatting from bullets
        def strip_md(s):
            s = re.sub(r'\*\*(.*?)\*\*', r'\1', s)  # Bold
            s = re.sub(r'__(.*?)__', r'\1', s)  # Underline
            s = re.sub(r'\*([^\*]*)\*', r'\1', s)  # Italic (single *)
            s = re.sub(r'_([^_]*)_', r'\1', s)  # Italic (single _)
            s = re.sub(r'`([^`]*)`', r'\1', s)  # Code
            s = re.sub(r'\[(.*?)\]\([^\)]*\)', r'\1', s)  # Links
            s = re.sub(r'^#+\s*', '', s)  # Headers
            return s.strip()
        
        # Extract bullets and strip markdown
        bullets = []
        lines = answer_text.split('\n')
        for line in lines:
            raw = line.strip()
            # Match bullets (-, *, •) or numbered lists (1., 2., etc.)
            if re.match(r'^(\-|\*|•)\s+.+', raw) or re.match(r'^\d+\.\s+.+', raw):
                content = re.sub(r'^(\-|\*|•|\d+\.)\s+', '', raw)
                bullets.append(strip_md(content))
        bullets = bullets[:6]  # Keep it tight
        
        # Generate smart follow-ups based on conversation
        follow_ups = [
            "Tell me more about specific team members",
            "Show me trends over a different time period",
            "How do these metrics compare to benchmarks?"
        ]
        
        # Send final structured payload
        print(f"📤 Final payload: {len(charts)} charts, {len(citations)} citations, {len(bullets)} bullets")
        yield f"data: {json.dumps({'follow_ups': follow_ups})}\n\n"
        yield f"data: {json.dumps({'bullets': bullets})}\n\n"
        yield f"data: {json.dumps({'citations': citations})}\n\n"
        yield f"data: {json.dumps({'charts': charts})}\n\n"
        
        # Final complete payload
        final_data = {
            "answer": answer_text,
            "bullets": bullets,
            "follow_ups": follow_ups,
            "citations": citations,
            "charts": charts,
            "metadata": {
                "tool_calls": tool_calls_made,
                "analysis_type": "agent_based"
            }
        }
        yield f"data: {json.dumps(final_data)}\n\n"
    
    return StreamingResponse(stream_generator(), media_type="text/event-stream")


# REMOVED: /api/get_insights endpoint (deprecated, use /api/chat instead)
