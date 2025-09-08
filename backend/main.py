import pandas as pd
import torch
import json
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
from openai import AsyncOpenAI
from sqlalchemy.orm import Session
from typing import List
from .rag_service import PerformanceRAG
from .services import get_all_utterances
from .models  import RAGQuery

# Local imports
from . import db_models
from .database import SessionLocal, init_db
from .models import Analysis as AnalysisModel, MultiTaskBertModel, TrendsResponse
from .services import analyze_structured_transcript, save_analysis_results, get_speaker_trends
from .parsing import (
    parse_document_with_unstructured, 
    extract_transcript_single_shot, 
    extract_transcript_chunking
)
from transformers import BertTokenizer

# --- Configuration ---
MODEL_PATH = os.getenv('MODEL_PATH', 'bert_classification/multi_task_bert_model.pth')
DATA_PATH = os.getenv('DATA_PATH', 'bert_classification/master_training_data.csv')
CONFIG_PATH = os.getenv('CONFIG_PATH', 'backend/config/metric_groups.json')
SA_MODEL_PATH = os.getenv('SA_MODEL_PATH', 'bert_classification/sa_bert_model_multilabel/')

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
    print("Loading model and configuration...")
    init_db()  # Initialize database and create tables
    # Load local models and configs
    df = pd.read_csv(DATA_PATH)
    ml_models["metric_cols"] = [col for col in df.columns if col.startswith(('comm_', 'feedback_', 'deviation_', 'sqdcp_'))]
    n_classes = {col: 5 for col in ml_models["metric_cols"]}
    model = MultiTaskBertModel(n_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    ml_models["model"] = model
    ml_models["tokenizer"] = BertTokenizer.from_pretrained('bert-base-uncased')

    print("Loading SA model...")
    ml_models["sa_tokenizer"] = AutoTokenizer.from_pretrained(SA_MODEL_PATH)
    ml_models["sa_model"] = AutoModelForSequenceClassification.from_pretrained(SA_MODEL_PATH)
    ml_models["sa_model"].eval()
    print("SA model loaded successfully.")



    # Load metric groups
    with open(CONFIG_PATH, 'r') as f:
        ml_models["metric_groups"] = json.load(f)

    # Configure OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("WARNING: OPENAI_API_KEY environment variable not set.")
    ml_models["openai_client"] = AsyncOpenAI(api_key=api_key)

    # Configure parsing strategy
    ml_models["parsing_strategy"] = os.getenv("PARSING_STRATEGY", "single_shot")
    print(f"Using parsing strategy: {ml_models['parsing_strategy']}")

    print("Model and configuration loaded successfully.")


    #---RAG Service Init---
    print("Initializing RAG service...")
    rag_service = PerformanceRAG()

    #create temp DB session to load data
    db = SessionLocal()
    try:
        all_utterances = get_all_utterances(db)
        if all_utterances:
            print(f"Found {len(all_utterances)} utterances to index.")
            rag_service.index_utterances(all_utterances)
        else:
            print("No existing utterances found to index.")
    finally:
        db.close()
    
    ml_models["rag_service"] = rag_service
    print("RAG service initialized successfully.")


    yield
    ml_models.clear()

# --- FastAPI App Initialization ---
app = FastAPI(lifespan=lifespan)

# --- CORS Middleware ---
# Get CORS origins from environment variable, default to localhost
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8001,http://127.0.0.1:8001")
origins = [origin.strip() for origin in CORS_ORIGINS.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Endpoints ---
@app.post("/analyze_text/", response_model=AnalysisModel)
async def analyze_text_endpoint(text_file: UploadFile = File(...), db: Session = Depends(get_db)):
    contents = await text_file.read()
    raw_text = parse_document_with_unstructured(contents, text_file.content_type)
    if not raw_text.strip():
        raise HTTPException(status_code=422, detail="Could not extract any text from the provided file.")
    
    strategy = ml_models.get("parsing_strategy", "single_shot")
    if strategy == "chunking":
        structured_transcript = await extract_transcript_chunking(raw_text, ml_models["openai_client"])
    else:
        structured_transcript = await extract_transcript_single_shot(raw_text, ml_models["openai_client"])

    if not structured_transcript:
        raise HTTPException(status_code=422, detail="The AI model could not extract a valid transcript from the document.")

    final_results = analyze_structured_transcript(
        transcript=structured_transcript,
        model=ml_models["model"],
        tokenizer=ml_models["tokenizer"],
        metric_cols=ml_models["metric_cols"],
        metric_groups=ml_models["metric_groups"]
    )
    
    if not final_results:
        raise HTTPException(status_code=422, detail="Could not parse any valid utterances from the provided text. Please check the format.")

    db_analysis = save_analysis_results(db, text_file.filename, final_results)

    print("Updating RAG service with new utterances...")

    rag_service: PerformanceRAG = ml_models.get("rag_service")
    if rag_service:
        rag_service.index_utterances(db.analysis.utterances)

    return db_analysis

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
def rag_query_endpoint(query: RAGQuery):
    rag_service: PerformanceRAG = ml_models.get("rag_service")
    if not rag_service:
        raise HTTPException(status_code=500, detail="RAG service is not available.")
    
    answer = rag_service.query_insights(query.question)
    return {"answer": answer}