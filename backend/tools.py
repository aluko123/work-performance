"""
Simple tools for OpenAI function calling.
Each tool is a plain Python function with clear docstrings.
"""

import os
from typing import Optional, List, Dict, Any
from datetime import datetime
import difflib
import chromadb
from openai import OpenAI

from .database import SessionLocal
from . import db_models
from .config.chart_config import AGGREGATED_METRICS, GRANULAR_METRICS, METRIC_DISPLAY_NAMES

# Initialize clients
openai_client = OpenAI()
CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/chroma_db")

# Valid metrics
VALID_METRICS = set(AGGREGATED_METRICS.keys()) | set(GRANULAR_METRICS.keys())


def get_valid_speakers() -> List[str]:
    """Get list of all speakers in database (cached in memory)"""
    session = SessionLocal()
    try:
        speakers = session.query(db_models.Utterance.speaker)\
            .filter(db_models.Utterance.speaker.isnot(None))\
            .distinct()\
            .all()
        return sorted([s[0] for s in speakers if s[0]])
    finally:
        session.close()


def validate_speaker(speaker: str) -> Optional[str]:
    """
    Validate speaker name. Returns error message if invalid, None if valid.
    """
    if not speaker:
        return None
    
    valid_speakers = get_valid_speakers()
    if speaker not in valid_speakers:
        # Find close matches
        close_matches = difflib.get_close_matches(speaker, valid_speakers, n=3, cutoff=0.6)
        if close_matches:
            return f"Speaker '{speaker}' not found. Did you mean: {', '.join(close_matches)}? Available speakers: {', '.join(valid_speakers)}"
        else:
            return f"Speaker '{speaker}' not found. Available speakers: {', '.join(valid_speakers)}"
    
    return None


def validate_metric(metric: str) -> Optional[str]:
    """
    Validate metric name. Returns error message if invalid, None if valid.
    """
    if not metric:
        return None
    
    if metric not in VALID_METRICS:
        # Find close matches
        close_matches = difflib.get_close_matches(metric, VALID_METRICS, n=3, cutoff=0.6)
        if close_matches:
            return f"Metric '{metric}' not found. Did you mean: {', '.join(close_matches)}?"
        else:
            return f"Metric '{metric}' not valid. Check available metrics with list_metrics tool."
    
    return None


def search_utterances(
    query: str,
    speaker: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Search meeting transcripts semantically.
    
    Args:
        query: What to search for (e.g., "safety discussions", "feedback about quality")
        speaker: Filter by speaker name (e.g., "Tasha", "Mike")
        date_from: Start date YYYY-MM-DD
        date_to: End date YYYY-MM-DD
        top_k: Number of results to return
    
    Returns:
        List of relevant utterances with speaker, date, text, and scores
    """
    # VALIDATE SPEAKER
    speaker_error = validate_speaker(speaker)
    if speaker_error:
        return [{"error": speaker_error}]
    
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        collection = chroma_client.get_collection("utterances")
        
        # Build filters
        where_filter = {}
        if speaker:
            where_filter["speaker"] = speaker
        
        # Query ChromaDB
        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter if where_filter else None
        )
        
        # Format results
        utterances = []
        if results and results["documents"]:
            for i in range(len(results["documents"][0])):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                utterances.append({
                    "speaker": meta.get("speaker"),
                    "date": meta.get("date"),
                    "timestamp": meta.get("timestamp"),
                    "text": results["documents"][0][i],
                    "source_id": meta.get("source_id")
                })
        
        return utterances
    except Exception as e:
        return [{"error": f"Search failed: {str(e)}"}]


def list_speakers() -> Dict[str, Any]:
    """
    Get list of all available speakers in the dataset.
    Use this when you're unsure if a speaker exists or to discover team members.
    
    Returns:
        List of speaker names
    """
    return {
        "speakers": get_valid_speakers(),
        "count": len(get_valid_speakers())
    }


def list_metrics() -> Dict[str, Any]:
    """
    Get list of all available performance metrics.
    Use this when you're unsure which metrics are available.
    
    Returns:
        Categorized metrics with descriptions
    """
    return {
        "aggregated_metrics": {
            name: METRIC_DISPLAY_NAMES.get(name, name)
            for name in AGGREGATED_METRICS.keys()
        },
        "granular_metrics": {
            name: METRIC_DISPLAY_NAMES.get(name, name)
            for name in list(GRANULAR_METRICS.keys())[:20]  # Show sample
        }
    }


def get_metric_stats(
    metric: str,
    speaker: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get statistics for a performance metric.
    
    Args:
        metric: Metric name (e.g., "SAFETY_Score", "comm_Pausing", "QUALITY_Score")
        speaker: Filter by speaker name
        date_from: Start date YYYY-MM-DD
        date_to: End date YYYY-MM-DD
    
    Returns:
        Statistics including average, count, min, max, and trend direction
    """
    # VALIDATE INPUTS
    metric_error = validate_metric(metric)
    if metric_error:
        return {"error": metric_error}
    
    speaker_error = validate_speaker(speaker)
    if speaker_error:
        return {"error": speaker_error}
    
    session = SessionLocal()
    try:
        # Build query
        query = session.query(db_models.Utterance)
        
        if speaker:
            query = query.filter(db_models.Utterance.speaker == speaker)
        if date_from:
            query = query.filter(db_models.Utterance.date >= date_from)
        if date_to:
            query = query.filter(db_models.Utterance.date <= date_to)
        
        utterances = query.all()
        
        if not utterances:
            return {"error": "No data found for the given filters"}
        
        # Extract metric values
        values = []
        for u in utterances:
            # Check both predictions and aggregated_scores
            val = None
            if u.predictions and metric in u.predictions:
                val = u.predictions[metric]
            elif u.aggregated_scores and metric in u.aggregated_scores:
                val = u.aggregated_scores[metric]
            
            if val is not None:
                try:
                    values.append(float(val))
                except (ValueError, TypeError):
                    continue
        
        if not values:
            return {"error": f"Metric '{metric}' not found in data"}
        
        # Compute statistics
        avg = sum(values) / len(values)
        
        return {
            "metric": metric,
            "average": round(avg, 2),
            "count": len(values),
            "min": round(min(values), 2),
            "max": round(max(values), 2),
            "filters": {
                "speaker": speaker,
                "date_from": date_from,
                "date_to": date_to
            }
        }
    finally:
        session.close()


def compare_periods(
    metric: str,
    early_start: str,
    early_end: str,
    late_start: str,
    late_end: str,
    speaker: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compare a metric across two time periods.
    
    Args:
        metric: Metric name (e.g., "SAFETY_Score")
        early_start: Early period start date YYYY-MM-DD
        early_end: Early period end date YYYY-MM-DD
        late_start: Late period start date YYYY-MM-DD
        late_end: Late period end date YYYY-MM-DD
        speaker: Optional speaker filter
    
    Returns:
        Comparison showing early average, late average, and change
    """
    # VALIDATE INPUTS
    metric_error = validate_metric(metric)
    if metric_error:
        return {"error": metric_error}
    
    speaker_error = validate_speaker(speaker)
    if speaker_error:
        return {"error": speaker_error}
    
    # Get stats for early period
    early_stats = get_metric_stats(metric, speaker, early_start, early_end)
    
    # Get stats for late period
    late_stats = get_metric_stats(metric, speaker, late_start, late_end)
    
    if "error" in early_stats or "error" in late_stats:
        return {"error": "Could not compare periods - insufficient data"}
    
    # Calculate change
    early_avg = early_stats["average"]
    late_avg = late_stats["average"]
    absolute_change = late_avg - early_avg
    percent_change = (absolute_change / early_avg * 100) if early_avg > 0 else 0
    
    return {
        "metric": metric,
        "early_period": {
            "dates": f"{early_start} to {early_end}",
            "average": early_avg,
            "count": early_stats["count"]
        },
        "late_period": {
            "dates": f"{late_start} to {late_end}",
            "average": late_avg,
            "count": late_stats["count"]
        },
        "change": {
            "absolute": round(absolute_change, 2),
            "percent": round(percent_change, 2),
            "direction": "increase" if absolute_change > 0 else "decrease" if absolute_change < 0 else "stable"
        }
    }


# Tool definitions for OpenAI function calling
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "list_speakers",
            "description": "Get list of all speakers in the dataset. Use this when you're unsure if a speaker exists or to discover available team members.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_metrics",
            "description": "Get list of all available performance metrics. Use this when you're unsure which metrics exist.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_utterances",
            "description": "Search meeting transcripts semantically. Use this to find what people said about specific topics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for (e.g., 'safety discussions', 'feedback about quality')"
                    },
                    "speaker": {
                        "type": "string",
                        "description": "Filter by speaker name (e.g., 'Tasha', 'Mike')"
                    },
                    "date_from": {
                        "type": "string",
                        "description": "Start date in YYYY-MM-DD format"
                    },
                    "date_to": {
                        "type": "string",
                        "description": "End date in YYYY-MM-DD format"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5)"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_metric_stats",
            "description": "Get statistics for a performance metric (average, count, etc.). Use this for current state or overall performance questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "description": "Metric name: SAFETY_Score, QUALITY_Score, DELIVERY_Score, COST_Score, PEOPLE_Score, comm_Pausing, comm_Clarifying_Questions, etc."
                    },
                    "speaker": {
                        "type": "string",
                        "description": "Filter by speaker name"
                    },
                    "date_from": {
                        "type": "string",
                        "description": "Start date YYYY-MM-DD"
                    },
                    "date_to": {
                        "type": "string",
                        "description": "End date YYYY-MM-DD"
                    }
                },
                "required": ["metric"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_periods",
            "description": "Compare a metric between two time periods. Use this for 'over time', 'improved', 'changed' questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "description": "Metric name"
                    },
                    "early_start": {
                        "type": "string",
                        "description": "Early period start date YYYY-MM-DD"
                    },
                    "early_end": {
                        "type": "string",
                        "description": "Early period end date YYYY-MM-DD"
                    },
                    "late_start": {
                        "type": "string",
                        "description": "Late period start date YYYY-MM-DD"
                    },
                    "late_end": {
                        "type": "string",
                        "description": "Late period end date YYYY-MM-DD"
                    },
                    "speaker": {
                        "type": "string",
                        "description": "Optional speaker filter"
                    }
                },
                "required": ["metric", "early_start", "early_end", "late_start", "late_end"]
            }
        }
    }
]


# Tool execution mapping
TOOL_FUNCTIONS = {
    "list_speakers": list_speakers,
    "list_metrics": list_metrics,
    "search_utterances": search_utterances,
    "get_metric_stats": get_metric_stats,
    "compare_periods": compare_periods,
}
