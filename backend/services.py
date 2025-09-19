from typing import Dict, List, Any
import torch
from sqlalchemy.orm import Session
from sqlalchemy import func, cast, JSON
from sqlalchemy.sql.expression import case
from . import db_models

def predict_all_scores(text: str, model, tokenizer, metric_cols: List[str]) -> Dict[str, int]:
    inputs = tokenizer(
        text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    )
    with torch.no_grad():
        outputs = model(**inputs)

    predictions = {}
    for col_name in metric_cols:
        pred_index = torch.argmax(outputs[col_name], dim=1).item()
        predictions[col_name] = pred_index + 1
    return predictions

def predict_sa_labels(text, model, tokenizer) -> List[str]:
    """
    predicts situational awareness (SA) lables for a given text using a multi-label classification model
    """
    
    # Define the mapping from generic labels to meaningful names
    label_mapping = {
        "LABEL_0": "Perception",
        "LABEL_1": "Comprehension",
        "LABEL_2": "Projection",
        "LABEL_3": "Action"
    }

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    probabilities = torch.sigmoid(logits).squeeze()
    # 0.5 threshold to determine "on" labels
    predictions = (probabilities > 0.5).nonzero(as_tuple=True)[0]

    # First, get the generic labels from the model's config
    generic_labels = [model.config.id2label[idx.item()] for idx in predictions]
    
    # Then, map the generic labels to the meaningful names
    predicted_labels = [label_mapping.get(label, label) for label in generic_labels]

    return predicted_labels

def analyze_structured_transcript(transcript: List[Dict[str, Any]], model, tokenizer, sa_model, sa_tokenizer, metric_cols: List[str], metric_groups: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """
    Analyzes structured transcript by running the BERT model on each utterance
    and using a dynamic configuration for scoring.
    """
    print("Analyzing structured transcript with BERT model...")
    results = []
    for item in transcript:
        # The extractor now provides pre-formatted, lowercased keys.
        utterance_text = item.get('utterance', '')
        if utterance_text:

            #analyze number metrics for utterances
            predicted_scores = predict_all_scores(utterance_text, model, tokenizer, metric_cols)
            
            #analyze SA category for utterances
            sa_labels = predict_sa_labels(utterance_text, sa_model, sa_tokenizer)

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

            # item already contains standardized date, timestamp, speaker, utterance
            item['predictions'] = predicted_scores
            item['aggregated_scores'] = aggregated_scores
            item['sa_labels'] = sa_labels
            results.append(item)

    return results

def save_analysis_results(db: Session, filename: str, results: List[Dict[str, Any]]):
    db_analysis = db_models.Analysis(source_filename=filename)
    db.add(db_analysis)
    db.flush()

    for res in results:
        # The extractor and analysis functions now provide clean, lowercase keys.
        db_utterance = db_models.Utterance(
            analysis_id=db_analysis.id,
            date=res.get("date"),
            timestamp=res.get("timestamp"),
            speaker=res.get("speaker"),
            text=res.get("utterance"),
            predictions=res.get("predictions"),
            aggregated_scores=res.get("aggregated_scores"),
            sa_labels=res.get("sa_labels")
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
