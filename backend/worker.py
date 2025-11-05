import os
import tempfile
import asyncio
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertModel
from sqlalchemy.orm import Session
from arq.connections import RedisSettings
from openai import AsyncOpenAI

# Assuming these modules and functions are available in the project's path
from .database import SessionLocal
from .document_extractor import RobustMeetingExtractor
from .services import analyze_structured_transcript, save_analysis_results, get_all_utterances
from .embeddings import generate_embeddings_for_utterances
from .models import MultiTaskBertModel
from .db_models import Utterance


async def startup(ctx):
    """
    ARQ startup hook - loads models once per worker and stores in context.
    This prevents reloading 500MB+ of models for every job.
    """
    print("🚀 ARQ Worker starting up - preloading models...")
    
    MODEL_PATH = os.getenv('MODEL_PATH', 'bert_classification/multi_task_bert_model.pth')
    DATA_PATH = os.getenv('DATA_PATH', 'bert_classification/master_training_data.csv')
    CONFIG_PATH = os.getenv('CONFIG_PATH', 'backend/config/metric_groups.json')
    SA_MODEL_PATH = os.getenv('SA_MODEL_PATH', 'bert_classification/sa_bert_model_multilabel/')
    
    df = pd.read_csv(DATA_PATH)
    metric_cols = [col for col in df.columns if col.startswith(('comm_', 'feedback_', 'deviation_', 'sqdcp_'))]
    n_classes = {col: 5 for col in metric_cols}
    
    base_bert_model = BertModel.from_pretrained('bert-base-uncased')
    model = MultiTaskBertModel(n_classes, base_bert_model)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    sa_tokenizer = AutoTokenizer.from_pretrained(SA_MODEL_PATH)
    sa_model = AutoModelForSequenceClassification.from_pretrained(SA_MODEL_PATH)
    sa_model.eval()
    
    with open(CONFIG_PATH, 'r') as f:
        metric_groups = json.load(f)
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_client = AsyncOpenAI(api_key=openai_api_key)
    
    ctx['model'] = model
    ctx['tokenizer'] = tokenizer
    ctx['sa_model'] = sa_model
    ctx['sa_tokenizer'] = sa_tokenizer
    ctx['metric_cols'] = metric_cols
    ctx['metric_groups'] = metric_groups
    ctx['openai_client'] = openai_client
    
    print(f"✅ Models preloaded: {len(metric_cols)} metrics configured")


async def shutdown(ctx):
    """
    ARQ shutdown hook - cleanup if needed.
    """
    print("🛑 ARQ Worker shutting down...")


async def process_document_task(ctx, file_contents: bytes, filename: str, corr_id: str):
    """
    Arq task to process a document, save analysis, and enqueue indexing.
    """
    print(f"Starting document processing for: {filename}")
    db: Session = SessionLocal()
    redis = ctx['redis']

    try:
        # --- 1. Get Preloaded Models from Context ---
        model = ctx['model']
        tokenizer = ctx['tokenizer']
        sa_model = ctx['sa_model']
        sa_tokenizer = ctx['sa_tokenizer']
        metric_cols = ctx['metric_cols']
        metric_groups = ctx['metric_groups']
        openai_client = ctx['openai_client']

        # --- 2. Extract Content from Document ---
        chunkr_api_key = os.getenv("CHUNKRAI_API_KEY")
        extractor = RobustMeetingExtractor(chunkr_api_key=chunkr_api_key, openai_client=openai_client)

        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}") as tmp:
            tmp.write(file_contents)
            tmp_path = tmp.name
        
        extraction_result = await extractor.process_any_document(tmp_path, filename)
        structured_transcript = extraction_result.get("extractions", [])
        os.remove(tmp_path)

        if not structured_transcript:
            print(f"Could not extract a valid transcript from {filename}.")
            return {"status": "FAILED", "error": "No transcript extracted."}

        # --- 3. Analyze Transcript ---
        final_results = analyze_structured_transcript(
            transcript=structured_transcript,
            model=model, tokenizer=tokenizer, sa_model=sa_model, sa_tokenizer=sa_tokenizer,
            metric_cols=metric_cols, metric_groups=metric_groups
        )

        if not final_results:
            print(f"Could not parse any valid utterances from {filename}.")
            return {"status": "FAILED", "error": "No utterances parsed."}

        # --- 4. Save Results to Database ---
        # Note: Utterances will be saved with is_indexed=False by default.
        db_analysis = save_analysis_results(db, filename, final_results)
        new_utterance_ids = [utt.id for utt in db_analysis.utterances]
        print(f"Saved analysis {db_analysis.id} with {len(new_utterance_ids)} new utterances.")

        # --- 5. Enqueue Indexing Task ---
        if new_utterance_ids:
            await redis.enqueue_job('index_utterances_task', new_utterance_ids)
            print(f"Enqueued indexing task for {len(new_utterance_ids)} utterances.")

        result_payload = {"status": "COMPLETED", "analysis_id": db_analysis.id}
        # Persist result explicitly for API polling robustness using correlation id
        try:
            if corr_id:
                await redis.set(f"job_result:{corr_id}", json.dumps(result_payload), ex=3600)
        except Exception:
            pass
        return result_payload

    except Exception as e:
        print(f"Error processing document {filename}: {e}")
        # Persist failure for API polling robustness using correlation id
        try:
            if corr_id:
                await redis.set(f"job_result:{corr_id}", json.dumps({"status": "FAILED", "error": str(e)}), ex=3600)
        except Exception:
            pass
        return {"status": "FAILED", "error": str(e)}
    finally:
        db.close()


async def index_utterances_task(ctx, utterance_ids: list[int]):
    """
    Arq task to generate embeddings for utterances and store in pgvector.
    """
    if not utterance_ids:
        return {"status": "SKIPPED", "message": "No utterance IDs provided."}

    print(f"Starting embedding generation for {len(utterance_ids)} utterances.")
    
    try:
        # Generate embeddings and update is_indexed flag
        indexed_count = generate_embeddings_for_utterances(utterance_ids)
        
        if indexed_count == 0:
            return {"status": "SKIPPED", "message": "No utterances needed indexing."}
        
        return {"status": "COMPLETED", "indexed_count": indexed_count}

    except Exception as e:
        print(f"Error during indexing task: {e}")
        return {"status": "FAILED", "error": str(e)}


async def startup_indexing_task(ctx):
    """
    Finds all un-indexed utterances and enqueues them for indexing.
    """
    print("Starting startup indexing task...")
    db: Session = SessionLocal()
    redis = ctx['redis']
    try:
        unindexed_utterance_ids = db.query(Utterance.id).filter(Utterance.is_indexed == False).all()
        unindexed_utterance_ids = [res[0] for res in unindexed_utterance_ids]

        if not unindexed_utterance_ids:
            print("No un-indexed utterances found.")
            return {"status": "SKIPPED", "message": "No un-indexed utterances found."}

        print(f"Found {len(unindexed_utterance_ids)} un-indexed utterances. Enqueuing for processing.")
        await redis.enqueue_job('index_utterances_task', unindexed_utterance_ids)
        
        return {"status": "COMPLETED", "enqueued_count": len(unindexed_utterance_ids)}
    except Exception as e:
        print(f"Error during startup indexing: {e}")
        return {"status": "FAILED", "error": str(e)}
    finally:
        db.close()


class WorkerSettings:
    redis_settings = RedisSettings.from_dsn(os.getenv("ARQ_REDIS_URL", "redis://localhost:6379"))
    keep_result = 3600
    functions = [process_document_task, index_utterances_task, startup_indexing_task]
    on_startup = startup
    on_shutdown = shutdown
