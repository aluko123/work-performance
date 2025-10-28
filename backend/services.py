from typing import Dict, List, Any, Optional
import os
import torch
from functools import lru_cache
from sqlalchemy.orm import Session
from sqlalchemy import func, cast, JSON
from sqlalchemy.sql.expression import case
from . import db_models
from .config.chart_config import CHART_CACHE_SIZE, CHART_MAX_DATA_POINTS, METRIC_DISPLAY_NAMES


def predict_all_scores_batch(texts: List[str], model, tokenizer, metric_cols: List[str]) -> List[Dict[str, int]]:
    """
    Batched version of predict_all_scores - processes multiple texts in one forward pass.
    Returns a list of predictions, one per input text.
    """
    if not texts:
        return []
    
    inputs = tokenizer(
        texts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    batch_predictions = []
    batch_size = len(texts)
    
    for i in range(batch_size):
        predictions = {}
        for col_name in metric_cols:
            pred_index = torch.argmax(outputs[col_name][i]).item()
            predictions[col_name] = pred_index + 1
        batch_predictions.append(predictions)
    
    return batch_predictions


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

def predict_sa_labels_batch(texts: List[str], model, tokenizer) -> List[List[str]]:
    """
    Batched version of predict_sa_labels - processes multiple texts in one forward pass.
    Returns a list of label lists, one per input text.
    """
    if not texts:
        return []
    
    label_mapping = {
        "LABEL_0": "Perception",
        "LABEL_1": "Comprehension",
        "LABEL_2": "Projection",
        "LABEL_3": "Action"
    }
    
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    probabilities = torch.sigmoid(logits)
    batch_predictions = []
    
    for i in range(len(texts)):
        probs = probabilities[i]
        predictions = (probs > 0.5).nonzero(as_tuple=True)[0]
        generic_labels = [model.config.id2label[idx.item()] for idx in predictions]
        predicted_labels = [label_mapping.get(label, label) for label in generic_labels]
        batch_predictions.append(predicted_labels)
    
    return batch_predictions


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
    Analyzes structured transcript by running batched BERT inference on utterances.
    Uses configurable batch size for optimal throughput.
    """
    batch_size = int(os.getenv('INFER_BATCH_SIZE', '32'))
    print(f"Analyzing structured transcript with BERT model (batch_size={batch_size})...")
    
    # Filter items with utterances and prepare for batching
    valid_items = [item for item in transcript if item.get('utterance', '')]
    if not valid_items:
        return []
    
    texts = [item['utterance'] for item in valid_items]
    results = []
    
    # Precompute dimension groupings once
    COMM_DIMS = [col for col in metric_cols if col.startswith('comm_')]
    FEEDBACK_DIMS = [col for col in metric_cols if col.startswith('feedback_')]
    DEV_DIMS = [col for col in metric_cols if col.startswith('deviation_')]
    
    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_items = valid_items[i:i + batch_size]
        
        # Batched inference for both models
        batch_scores = predict_all_scores_batch(batch_texts, model, tokenizer, metric_cols)
        batch_sa_labels = predict_sa_labels_batch(batch_texts, sa_model, sa_tokenizer)
        
        # Combine results with original items
        for item, predicted_scores, sa_labels in zip(batch_items, batch_scores, batch_sa_labels):
            aggregated_scores = {}
            for group_name, metric_list in metric_groups.items():
                total_score = sum(predicted_scores.get(metric, 0) for metric in metric_list)
                aggregated_scores[group_name.replace("_COLS", "_Score")] = total_score
            
            aggregated_scores['Total_Deviation_Score'] = sum(predicted_scores.get(dim, 0) for dim in DEV_DIMS)
            aggregated_scores['Total_Comm_Score'] = sum(predicted_scores.get(dim, 0) for dim in COMM_DIMS)
            aggregated_scores['Feedback_Tier1_Score'] = sum(predicted_scores.get(dim, 0) for dim in FEEDBACK_DIMS[:6])
            aggregated_scores['Feedback_Tier2_Score'] = sum(predicted_scores.get(dim, 0) for dim in FEEDBACK_DIMS[6:])
            
            item['predictions'] = predicted_scores
            item['aggregated_scores'] = aggregated_scores
            item['sa_labels'] = sa_labels
            results.append(item)
    
    print(f"✅ Analyzed {len(results)} utterances in {(len(results) + batch_size - 1) // batch_size} batches")
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


def _sample_data(data: List[Any], max_points: int) -> List[Any]:
    """
    Evenly sample data if it exceeds max_points.
    Preserves first and last points.
    """
    if len(data) <= max_points:
        return data
    
    # Calculate step size
    step = len(data) / (max_points - 1)
    indices = [0]  # Always include first point
    
    for i in range(1, max_points - 1):
        indices.append(int(i * step))
    
    indices.append(len(data) - 1)  # Always include last point
    
    return [data[i] for i in indices]


def get_chart_data(
    db: Session,
    chart_type: str,
    metric: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    group_by: str = "date",
    filters: Optional[Dict[str, Any]] = None,
    max_points: int = None,
    level: str = "aggregated"
) -> Optional[Dict[str, Any]]:
    """
    Generalized chart data retrieval with caching and auto-sampling.
    
    Args:
        db: Database session
        chart_type: "line", "bar", or "grouped_bar"
        metric: Single metric name (for line/bar charts)
        metrics: List of metrics (for grouped_bar)
        group_by: "date", "speaker", or "metric"
        filters: Dict with optional keys: speaker, date_from, date_to
        max_points: Maximum data points (defaults to CHART_MAX_DATA_POINTS)
    
    Returns:
        {
            "labels": List[str],
            "datasets": List[{
                "name": str,
                "data": List[float]
            }]
        }
        or None if no data available
    """
    if max_points is None:
        max_points = CHART_MAX_DATA_POINTS
    
    filters = filters or {}
    
    # Clean metric name
    if metric:
        metric = metric.replace('.1', '')
    
    print(f"📊 get_chart_data: type={chart_type}, metric={metric}, level={level}, group_by={group_by}")
    
    # Build base query
    query = db.query(db_models.Utterance).join(db_models.Analysis)
    
    # Apply filters
    if filters.get('speaker'):
        query = query.filter(db_models.Utterance.speaker == filters['speaker'])
    if filters.get('date_from'):
        query = query.filter(db_models.Utterance.date >= filters['date_from'])
    if filters.get('date_to'):
        query = query.filter(db_models.Utterance.date <= filters['date_to'])
    
    # Filter out empty dates
    query = query.filter(
        db_models.Utterance.date != None,
        db_models.Utterance.date != ''
    )
    
    # Execute based on group_by
    if group_by == "date":
        # Time series data
        date_trunc_func = func.date(db_models.Utterance.date)
        
        # Choose the right column based on metric level and metric name
        # Check if this specific metric is aggregated (ends with _Score or starts with Total_/Feedback_)
        is_aggregated_metric = (
            metric and (
                metric.endswith('_Score') or 
                metric.startswith('Total_') or 
                metric.startswith('Feedback_Tier')
            )
        )
        
        if is_aggregated_metric or level == "aggregated":
            metric_value = case(
                (db_models.Utterance.aggregated_scores[metric] != None, 
                 db_models.Utterance.aggregated_scores[metric].as_float()),
                else_=None
            )
        else:  # granular
            metric_value = case(
                (db_models.Utterance.predictions[metric] != None, 
                 db_models.Utterance.predictions[metric].as_float()),
                else_=None
            )
        
        results = query.with_entities(
            date_trunc_func.label('period'),
            func.avg(metric_value).label('average_score')
        ).group_by(date_trunc_func).order_by(date_trunc_func).all()
        
        if not results:
            print(f"📊 No results returned for metric={metric}, level={level}")
            return None
        
        labels = [str(res.period) for res in results]
        data = [res.average_score for res in results]
        
        print(f"📊 Chart data: {len(labels)} points, first 3: {list(zip(labels[:3], data[:3]))}")
        
        # Sample if too many points
        if len(labels) > max_points:
            sampled_indices = [0]
            step = len(labels) / (max_points - 1)
            for i in range(1, max_points - 1):
                sampled_indices.append(int(i * step))
            sampled_indices.append(len(labels) - 1)
            
            labels = [labels[i] for i in sampled_indices]
            data = [data[i] for i in sampled_indices]
        
        result = {
            "labels": labels,
            "datasets": [{
                "name": METRIC_DISPLAY_NAMES.get(metric, metric),
                "data": data
            }]
        }
        print(f"📊 Returning chart data: {len(labels)} labels, {len(data)} data points, sample: {data[:3]}")
        return result
    
    elif group_by == "speaker":
        # Speaker comparison
        is_aggregated_metric = (
            metric and (
                metric.endswith('_Score') or 
                metric.startswith('Total_') or 
                metric.startswith('Feedback_Tier')
            )
        )
        
        if is_aggregated_metric or level == "aggregated":
            metric_value = case(
                (db_models.Utterance.aggregated_scores[metric] != None, 
                 db_models.Utterance.aggregated_scores[metric].as_float()),
                else_=None
            )
        else:  # granular
            metric_value = case(
                (db_models.Utterance.predictions[metric] != None, 
                 db_models.Utterance.predictions[metric].as_float()),
                else_=None
            )
        
        results = query.with_entities(
            db_models.Utterance.speaker,
            func.avg(metric_value).label('average_score')
        ).group_by(db_models.Utterance.speaker).order_by(
            func.avg(metric_value).desc()
        ).all()
        
        if not results:
            return None
        
        labels = [res.speaker for res in results]
        data = [res.average_score for res in results]
        
        return {
            "labels": labels,
            "datasets": [{
                "name": METRIC_DISPLAY_NAMES.get(metric, metric),
                "data": data
            }]
        }
    
    elif group_by == "metric":
        # Multi-metric for single entity (e.g., grouped bar for one speaker)
        if not metrics:
            return None
        
        dataset_data = []
        valid_metrics = []
        
        for m in metrics:
            m_clean = m.replace('.1', '')
            
            # Auto-detect if this metric is aggregated
            is_aggregated_metric = (
                m_clean.endswith('_Score') or 
                m_clean.startswith('Total_') or 
                m_clean.startswith('Feedback_Tier')
            )
            
            if is_aggregated_metric:
                metric_value = case(
                    (db_models.Utterance.aggregated_scores[m_clean] != None, 
                     db_models.Utterance.aggregated_scores[m_clean].as_float()),
                    else_=None
                )
            else:  # granular
                metric_value = case(
                    (db_models.Utterance.predictions[m_clean] != None, 
                     db_models.Utterance.predictions[m_clean].as_float()),
                    else_=None
                )
            
            result = query.with_entities(
                func.avg(metric_value).label('average_score')
            ).scalar()
            
            if result is not None:
                dataset_data.append(result)
                valid_metrics.append(m_clean)
        
        if not valid_metrics:
            return None
        
        return {
            "labels": [METRIC_DISPLAY_NAMES.get(m, m) for m in valid_metrics],
            "datasets": [{
                "name": filters.get('speaker', 'All Speakers'),
                "data": dataset_data
            }]
        }
    
    return None
