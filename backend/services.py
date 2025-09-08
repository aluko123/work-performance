from typing import Dict, List, Any
import torch
from sqlalchemy.orm import Session
from sqlalchemy import func, cast, JSON
from sqlalchemy.sql.expression import case
from . import db_models
from datetime import datetime

def _parse_date(date_str: str) -> str | None:
    """Parses a date string from common formats into YYYY-MM-DD."""
    if not date_str or not isinstance(date_str, str):
        return None
    # Add more formats here if the AI returns different ones
    for fmt in ("%A, %B %d, %Y", "%B %d, %Y", "%b %d, %Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return date_str
    except ValueError:
        pass
    print(f"Warning: Could not parse date string: {date_str}")
    return date_str

def _parse_time(time_str: str) -> str | None:
    """Parses a time string from common formats into HH:MM:SS."""
    if not time_str or not isinstance(time_str, str):
        return None
    for fmt in ("%I:%M:%S %p", "%I:%M %p", "%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(time_str, fmt).strftime("%H:%M:%S")
        except ValueError:
            continue
    print(f"Warning: Could not parse time string: {time_str}")
    return time_str

def predict_all_scores(text: str, model, tokenizer, metric_cols: List[str]) -> Dict[str, int]:
    inputs = tokenizer(
        text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)

    predictions = {}
    for col_name in metric_cols:
        pred_index = torch.argmax(outputs[col_name], dim=1).item()
        predictions[col_name] = pred_index + 1
    return predictions

def analyze_structured_transcript(transcript: List[Dict[str, Any]], model, tokenizer, metric_cols: List[str], metric_groups: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """
    Analyzes structured transcript by running the BERT model on each utterance
    and using a dynamic configuration for scoring.
    """
    print("Analyzing structured transcript with BERT model...")
    results = []
    for item in transcript:
        # Normalize keys to lowercase for consistent access
        item_lower = {k.lower(): v for k, v in item.items()}
        utterance_text = item_lower.get('utterance', '')
        if utterance_text:
            predicted_scores = predict_all_scores(utterance_text, model, tokenizer, metric_cols)
            
            aggregated_scores = {}
            for group_name, metric_list in metric_groups.items():
                total_score = sum(predicted_scores.get(metric, 0) for metric in metric_list)
                aggregated_scores[group_name.replace("_COLS", "_Score")] = total_score

            COMM_DIMS = [col for col in metric_cols if col.startswith('comm_')]
            FEEDBACK_DIMS = [col for col in metric_cols if col.startswith('feedback_')]
            DEV_DIMS = [col for col in metric_cols if col.startswith('deviation_')]
            
            aggregated_scores['Total_Deviation_Score'] = sum(predicted_scores.get(dim, 0) for dim in DEV_DIMS)
            aggregated_scores['Total_Comm_Score'] = sum(predicted_scores.get(dim, 0) for dim in COMM_DIMS)
            aggregated_scores['Feedback_Tier1_Score'] = sum(predicted_scores.get(dim, 0) for dim in FEEDBACK_DIMS[:6])
            aggregated_scores['Feedback_Tier2_Score'] = sum(predicted_scores.get(dim, 0) for dim in FEEDBACK_DIMS[6:])

            

            processed_item = {
                "date": _parse_date(item_lower.get("date")),
                "timestamp": _parse_time(item_lower.get("timestamp")),
                "speaker": item_lower.get("speaker"),
                "utterance": utterance_text,
                "predictions": predicted_scores,
                "aggregated_scores": aggregated_scores
            }
            results.append(processed_item)

    return results

def save_analysis_results(db: Session, filename: str, results: List[Dict[str, Any]]):
    db_analysis = db_models.Analysis(source_filename=filename)
    db.add(db_analysis)
    db.flush()

    for res in results:
        # Ensure keys are lowercase before saving to prevent case-sensitivity issues.
        res_lower = {k.lower(): v for k, v in res.items()}
        db_utterance = db_models.Utterance(
            analysis_id=db_analysis.id,
            date=res_lower.get("date"),
            timestamp=res_lower.get("timestamp"),
            speaker=res_lower.get("speaker"),
            text=res_lower.get("utterance"),
            predictions=res_lower.get("predictions"),
            aggregated_scores=res_lower.get("aggregated_scores")
        )
        db.add(db_utterance)
    
    db.commit()
    db.refresh(db_analysis)
    return db_analysis


def get_all_utterances(db: Session) -> List[db_models.Utterance]:
    return db.query(db_models.Utterance).all()



def get_speaker_trends(db: Session, metric: str, period: str):
    """
    Calculates the average score for a given metric, grouped by speaker and time period.
    """
    # Clean the metric name by removing '.1' suffix
    cleaned_metric = metric.replace('.1', '')

    if period == 'daily':
        date_trunc_func = func.date(db_models.Utterance.date)
    else: # weekly
        date_trunc_func = func.strftime('%Y-%W', db_models.Utterance.date)

    metric_value = case(
        (db_models.Utterance.predictions[cleaned_metric] != None, db_models.Utterance.predictions[cleaned_metric].as_float()),
        (db_models.Utterance.aggregated_scores[cleaned_metric] != None, db_models.Utterance.aggregated_scores[cleaned_metric].as_float()),
        else_=0
    )

    results = db.query(
        db_models.Utterance.speaker,
        date_trunc_func.label('period'),
        func.avg(metric_value).label('average_score')
    ).join(db_models.Analysis).filter(
        db_models.Utterance.date != None, 
        db_models.Utterance.date != ''
    ).group_by(
        db_models.Utterance.speaker,
        date_trunc_func
    ).order_by(
        date_trunc_func
    ).all()

    # Restructure data for Chart.js
    periods = set([res.period for res in results])
    periods.discard(None) # Safely remove None from the set
    labels = sorted(list(periods))

    speakers = sorted(list(set([res.speaker for res in results])))
    datasets = []

    for speaker in speakers:
        speaker_data = [None] * len(labels)
        for res in results:
            if res.speaker == speaker and res.period in labels:
                idx = labels.index(res.period)
                speaker_data[idx] = res.average_score
        
        datasets.append({
            'label': speaker,
            'data': speaker_data,
        })

    return {'labels': labels, 'datasets': datasets}
