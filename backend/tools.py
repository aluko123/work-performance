"""
Simple tools for OpenAI function calling.
Each tool is a plain Python function with clear docstrings.
"""

import os
from typing import Optional, List, Dict, Any
from datetime import datetime
import difflib
from sqlalchemy import text
from openai import OpenAI

from .database import SessionLocal, SQLALCHEMY_DATABASE_URL
from . import db_models
from .config.chart_config import AGGREGATED_METRICS, GRANULAR_METRICS
from . import metrics as metrics_registry

# Initialize clients
openai_client = OpenAI()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

def _available_metric_set() -> set:
    # Dynamic metrics from mapping + ensure aggregated are present
    return set(metrics_registry.get_available_metrics()) | set(AGGREGATED_METRICS.keys())


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
    
    # Normalize common suffix noise
    cleaned = metric.replace('.1', '') if metric else metric
    available = _available_metric_set()
    if cleaned not in available:
        # Find close matches
        close_matches = difflib.get_close_matches(cleaned or "", available, n=3, cutoff=0.6)
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
    Search meeting transcripts semantically using pgvector.
    
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
    
    session = SessionLocal()
    try:
        # Ensure Postgres
        if "postgresql" not in (SQLALCHEMY_DATABASE_URL or "").lower():
            return [{"error": "Semantic search requires Postgres + pgvector"}]

        # Generate OpenAI embedding for query
        embedding_response = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query[:8000]  # Limit text length
        )
        query_embedding = embedding_response.data[0].embedding

        # Set ivfflat probes for better recall
        session.execute(text("SET ivfflat.probes = 10"))

        # Build WHERE clause dynamically
        where_clauses = ["embedding IS NOT NULL"]
        params = {"embedding": query_embedding, "top_k": top_k}

        if speaker:
            where_clauses.append("speaker = :speaker")
            params["speaker"] = speaker
        if date_from:
            where_clauses.append("date >= :date_from")
            params["date_from"] = date_from
        if date_to:
            where_clauses.append("date <= :date_to")
            params["date_to"] = date_to

        where_clause = " AND ".join(where_clauses)

        # Query with pgvector similarity search
        sql = text(f"""
            SELECT id, speaker, date, "timestamp", text,
                   1 - (embedding <=> :embedding::vector) AS similarity
            FROM utterances
            WHERE {where_clause}
            ORDER BY embedding <=> :embedding::vector
            LIMIT :top_k
        """)

        rows = session.execute(sql, params).mappings().all()

        return [
            {
                "source_id": r["id"],
                "speaker": r["speaker"],
                "date": r["date"],
                "timestamp": r["timestamp"],
                "text": r["text"],
                "similarity": round(float(r["similarity"]), 3)
            }
            for r in rows
        ]

    except Exception as e:
        return [{"error": f"Search failed: {str(e)}"}]
    finally:
        session.close()


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
    # Full dynamic mapping
    all_keys = metrics_registry.get_available_metrics()
    agg_keys = set(AGGREGATED_METRICS.keys())
    return {
        "aggregated_metrics": {k: metrics_registry.get_metric_display_name(k) for k in all_keys if k in agg_keys},
        "granular_metrics": {k: metrics_registry.get_metric_display_name(k) for k in all_keys if k not in agg_keys},
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


def generate_chart(
    chart_type: str,
    metric: Optional[str] = None,
    speaker: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    speakers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Generate chart data for visualization.
    
    Args:
        chart_type: Type of chart - "line" for trends over time, "bar" for speaker comparison
        metric: Metric name (e.g., "SAFETY_Score", "comm_Pausing")
        speaker: Filter by speaker (for line chart) or compare multiple speakers (for bar chart, leave None)
        date_from: Start date YYYY-MM-DD
        date_to: End date YYYY-MM-DD
    
    Returns:
        Chart configuration with data ready for frontend rendering
    """
    from .services import get_chart_data
    
    # VALIDATE INPUTS
    if chart_type == "grouped_bar":
        # grouped_bar requires a list of metrics, and optionally a speaker filter
        if not metrics or not isinstance(metrics, list):
            return {"error": "'metrics' (list) is required for grouped_bar charts"}
        # Validate each metric
        for m in metrics:
            err = validate_metric(m)
            if err:
                return {"error": err}
    else:
        metric_error = validate_metric(metric)
        if metric_error:
            return {"error": metric_error}
    
    speaker_error = validate_speaker(speaker)
    if speaker_error:
        return {"error": speaker_error}
    
    if chart_type not in ["line", "bar", "grouped_bar"]:
        return {"error": "chart_type must be 'line', 'bar', or 'grouped_bar'"}
    
    session = SessionLocal()
    try:
        filters = {}
        if speakers:
            # For comparison charts, filter to specific speakers
            filters["speakers"] = speakers
        elif speaker:
            filters["speaker"] = speaker
        if date_from:
            filters["date_from"] = date_from
        if date_to:
            filters["date_to"] = date_to
        
        # Determine group_by based on chart_type
        if chart_type == "line":
            group_by = "date"
        elif chart_type == "bar":
            group_by = "speaker"
        else:  # grouped_bar
            group_by = "metric"
        
        chart_data = get_chart_data(
            db=session,
            chart_type=chart_type,
            metric=metric,
            metrics=metrics,
            group_by=group_by,
            filters=filters
        )
        
        if not chart_data:
            return {"error": "No data available for chart"}
        
        title_metric = metric if chart_type != "grouped_bar" else ", ".join(metrics or [])
        x_axis = "Date" if chart_type == "line" else ("Speaker" if chart_type == "bar" else "Metric")
        return {
            "type": chart_type,
            "metric": metric,
            "data": chart_data,
            "config": {
                "title": metrics_registry.get_metric_display_name(title_metric),
                "xAxisLabel": x_axis,
                "yAxisLabel": "Score",
                "filters": filters
            }
        }
    finally:
        session.close()


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
            "name": "suggest_metrics",
            "description": "Suggest relevant metrics for a natural-language question. Returns a ranked list of candidates (aggregates for overall questions, granular for behavior questions).",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "User's question in natural language"},
                    "top_n": {"type": "integer", "description": "Max number of metrics to return", "default": 5}
                },
                "required": ["question"]
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
    },
    {
        "type": "function",
        "function": {
            "name": "generate_chart",
            "description": "Generate a chart visualization for metrics. Use this when user wants to see trends or comparisons visually. Line charts show trends over time, bar charts compare speakers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "enum": ["line", "bar", "grouped_bar"],
                        "description": "Type of chart: 'line' for time trends, 'bar' for speaker comparison, 'grouped_bar' to compare multiple metrics for one entity"
                    },
                    "metric": {
                        "type": "string",
                        "description": "Metric name (e.g., SAFETY_Score, comm_Pausing)"
                    },
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "For grouped_bar: list of metrics to compare"
                    },
                    "speaker": {
                        "type": "string",
                        "description": "Optional: Filter by single speaker (for line charts)"
                    },
                    "speakers": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional: List of specific speakers to compare (for bar charts). Use this when comparing 2+ specific people. Example: ['Tasha', 'Devon', 'Mike']"
                    },
                    "date_from": {
                        "type": "string",
                        "description": "Optional: Start date YYYY-MM-DD"
                    },
                    "date_to": {
                        "type": "string",
                        "description": "Optional: End date YYYY-MM-DD"
                    }
                },
                "required": ["chart_type"]
            }
        }
    }
]


# Tool execution mapping
TOOL_FUNCTIONS = {
    "list_speakers": list_speakers,
    "list_metrics": list_metrics,
    "suggest_metrics": lambda **kwargs: suggest_metrics(**kwargs),
    "search_utterances": search_utterances,
    "get_metric_stats": get_metric_stats,
    "compare_periods": compare_periods,
    "generate_chart": generate_chart,
}


def suggest_metrics(question: str, top_n: int = 5) -> Dict[str, Any]:
    """
    Heuristic metric suggester based on keywords and dynamic registry.

    - Aggregate questions → SAFETY_Score, QUALITY_Score, DELIVERY_Score, COST_Score, PEOPLE_Score, Total_Comm_Score, Total_Deviation_Score
    - Communication/behavior questions → comm_* and feedback_* depending on keywords
    - Safety/Quality/Delivery/Cost/People keywords map to corresponding aggregates
    """
    q = (question or "").lower()
    all_metrics = metrics_registry.get_available_metrics()
    disp = metrics_registry.get_metric_display_name

    # Seed candidate sets
    agg_pref = [
        "PEOPLE_Score", "SAFETY_Score", "QUALITY_Score", "DELIVERY_Score", "COST_Score",
        "Total_Comm_Score", "Total_Deviation_Score",
    ]

    # Keyword → aggregates
    if any(k in q for k in ["overall", "aggregate", "summary", "performance", "trend", "improved", "change over time"]):
        cands = [m for m in agg_pref if m in all_metrics or m in AGGREGATED_METRICS]
    else:
        cands = []

    # Domain keywords
    if "safety" in q:
        cands.insert(0, "SAFETY_Score")
    if "quality" in q:
        cands.insert(0, "QUALITY_Score")
    if any(k in q for k in ["delivery", "on-time", "schedule"]):
        cands.insert(0, "DELIVERY_Score")
    if "cost" in q:
        cands.insert(0, "COST_Score")
    if any(k in q for k in ["people", "behavior", "engagement", "communication", "comm", "talking", "conversation"]):
        cands.insert(0, "PEOPLE_Score")
        cands.append("Total_Comm_Score")

    # Behavior-specific keywords → granular comm_*
    granular_additions: List[str] = []
    def add_if_exists(key: str):
        if key in all_metrics:
            granular_additions.append(key)

    if any(k in q for k in ["clarify", "clarifying", "question", "questions"]):
        add_if_exists("comm_Clarifying_Questions")
        add_if_exists("comm_Probing_Questions")
    if any(k in q for k in ["probe", "probing"]):
        add_if_exists("comm_Probing_Questions")
    if any(k in q for k in ["open-ended", "open ended"]):
        add_if_exists("comm_Open_Ended_Questions")
    if any(k in q for k in ["coach", "coaching"]):
        add_if_exists("comm_Coaching_Questions")
    if any(k in q for k in ["pause", "pausing"]):
        add_if_exists("comm_Pausing")
    if any(k in q for k in ["affirm", "affirmation"]):
        add_if_exists("comm_Verbal_Affirmation")
    if any(k in q for k in ["feedback"]):
        for k in all_metrics:
            if k.startswith("feedback_"):
                granular_additions.append(k)

    # Deduplicate while preserving order
    seen = set()
    ordered = []
    for m in cands + granular_additions:
        if m not in seen:
            seen.add(m)
            ordered.append(m)

    # Fallback if empty: default to commonly useful aggregates
    if not ordered:
        ordered = [m for m in agg_pref if m in all_metrics or m in AGGREGATED_METRICS]

    ordered = ordered[: max(1, top_n)]
    return {
        "candidates": [
            {"metric": m, "display_name": disp(m), "aggregated": bool(m in AGGREGATED_METRICS)}
            for m in ordered
        ]
    }
