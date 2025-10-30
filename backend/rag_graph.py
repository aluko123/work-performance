from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, TypedDict, AsyncGenerator

import redis
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI
import numpy as np

from . import db_models
from .database import SessionLocal
from .models import RAGAnswer, RAGCitation, Chart, ChartConfig
from .services import get_chart_data
from .config.chart_config import (
    ENABLE_CHARTS, 
    MAX_CHARTS_PER_RESPONSE, 
    CHART_DEFAULT_COLORS, 
    METRIC_DISPLAY_NAMES,
    AGGREGATED_METRICS,
    GRANULAR_METRICS,
    METRIC_DESCRIPTIONS
)
from .prompts import (
    answer_system,
    answer_user_template,
    metadata_system,
    metadata_user_template,
    verification_system,
    verification_user_template,
)

# --- Constants ---
REDIS_URL = os.getenv("REDIS_URL") or os.getenv("ARQ_REDIS_URL") or "redis://redis:6379/0"
LLM_MODEL_NAME = "gpt-4o-mini"
LLM_TEMPERATURE = 0.1  # Lower temperature for more faithful, deterministic responses
CHROMA_DIR = os.getenv("CHROMA_DIR", "./data/chroma_db")


class GraphState(TypedDict):
    question: str
    session_id: Optional[str]
    filters: Dict[str, Any]
    analysis_type: str
    history: List[Dict[str, str]]
    retrieved: List[Document]
    aggregates: Dict[str, Any]
    draft: Dict[str, Any]
    answer: Dict[str, Any]
    # Internal key for passing citations
    _citations: List[Dict[str, Any]]
    # Chart-related keys
    chart_specs: List[Dict[str, Any]]
    charts: List[Dict[str, Any]]


def _redis_client() -> Optional[redis.Redis]:
    """Return a Redis client if available; otherwise None."""
    try:
        client = redis.from_url(REDIS_URL)
        client.ping()
        return client
    except Exception:
        return None


def _normalize_history_to_turns(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Normalize history to per-turn format: [{"user": "...", "assistant": "..."}, ...]
    Handles both old role/content format and new user/assistant format.
    """
    if not history:
        return []
    
    # If already in turn format (has "user" or "assistant" keys)
    if history and ("user" in history[0] or "assistant" in history[0]):
        return history
    
    # Convert role/content pairs to turns
    turns = []
    current = {}
    
    for msg in history:
        role = msg.get("role")
        content = msg.get("content", "")
        
        if role == "user":
            if current:  # Save previous turn
                turns.append(current)
            current = {"user": content, "assistant": ""}
        elif role == "assistant":
            if "user" in current:
                current["assistant"] = content
                turns.append(current)
                current = {}
            else:
                # Orphan assistant message, create turn
                turns.append({"user": "", "assistant": content})
    
    if current:  # Save last incomplete turn
        turns.append(current)
    
    return turns


def load_history(state: GraphState) -> GraphState:
    sid = state.get("session_id")
    if not sid:
        state["history"] = []
        return state
    r = _redis_client()
    if not r:
        state["history"] = []
        return state
    try:
        raw = r.get(f"rag:hist:{sid}")
    except Exception:
        raw = None
    history: List[Dict[str, str]] = []
    if raw:
        try:
            history = json.loads(raw)
        except Exception:
            history = []
    
    # Normalize to turn format and keep last 10 turns
    state["history"] = _normalize_history_to_turns(history)[-10:]
    return state


def save_history(state: GraphState) -> GraphState:
    sid = state.get("session_id")
    if not sid:
        return state
    r = _redis_client()
    
    # Normalize history to turn format
    history: List[Dict[str, str]] = _normalize_history_to_turns(state.get("history", []))
    
    # Get question and answer
    q = state.get("question")
    # Prefer final answer, fall back to draft
    msg = state.get("answer") or state.get("draft") or {}
    a = msg.get("answer") if isinstance(msg, dict) else None
    
    # Append current turn
    if q and a:
        history = (history + [{"user": q, "assistant": a}])[-10:]
    
    # Save to Redis
    if r:
        try:
            r.set(f"rag:hist:{sid}", json.dumps(history))
        except Exception:
            pass
    
    state["history"] = history
    return state


def resolve_contextual_query(question: str, history: List[dict]) -> str:
    """
    Expand abbreviated queries using conversation context.
    Carries over speaker, metric, and intent from previous turn.
    
    Examples:
    - "What about Mike?" → "What about Mike's safety performance?"
    - "what about communication?" (after "Tasha's safety?") → "what about Tasha's communication?"
    - "and delivery?" (after trends query) → "and delivery trends?"
    """
    if not history:
        return question
    
    # Patterns indicating context dependency
    contextual_patterns = [
        r"^what about (.+?)\??$",
        r"^how about (.+?)\??$",
        r"^and (.+?)\??$",
        r"^show me (.+?)$",
        r"^(.+?) too\??$",
        r"^same for (.+?)\??$"
    ]
    
    question_lower = question.lower().strip()
    is_contextual = any(re.match(p, question_lower) for p in contextual_patterns)
    
    if not is_contextual:
        return question
    
    # Get last turn context
    last_turn = history[-1]
    last_question = last_turn.get("user", "")
    last_answer = last_turn.get("assistant", "")
    last_text = last_question + " " + last_answer
    
    # Extract what was discussed previously
    last_metrics = extract_metrics_from_text(last_text)
    last_speakers = extract_speakers_from_text(last_text)
    
    # Extract what's in current question
    current_metrics = extract_metrics_from_text(question)
    current_speakers = extract_speakers_from_text(question)
    
    # Start building expanded question
    expanded = question.rstrip(" ?").strip()
    
    # Carry over speaker if missing
    if not current_speakers and last_speakers:
        # "what about communication?" → "what about Tasha's communication?"
        expanded = expanded.replace("about", f"about {last_speakers[0]}'s", 1)
    
    # Carry over metric if missing
    if not current_metrics and last_metrics:
        metric_display = METRIC_DISPLAY_NAMES.get(last_metrics[0], last_metrics[0])
        if not current_speakers:
            # "What about Mike?" → "What about Mike safety performance?"
            expanded += f" {metric_display}"
        # else speaker was added above, metric goes after speaker
    
    # Carry over "trend" intent if previous query was temporal
    trend_words = ["trend", "trending", "over time", "increase", "decrease", "improve", "change"]
    last_is_temporal = any(w in last_text.lower() for w in trend_words)
    current_is_temporal = any(w in question_lower for w in trend_words)
    
    if last_is_temporal and not current_is_temporal:
        expanded += " trends"
    
    # Ensure proper punctuation
    if not expanded.endswith("?"):
        expanded += "?"
    
    print(f"🧠 Contextual expansion: '{question}' → '{expanded}'")
    return expanded


def classify_query(state: GraphState) -> GraphState:
    original_question = state["question"]
    history = state.get("history", [])
    
    # Expand contextual queries FIRST
    expanded_question = resolve_contextual_query(original_question, history)
    
    # Store both for transparency
    state["original_question"] = original_question
    state["expanded_question"] = expanded_question
    
    # CRITICAL: Replace question with expanded version for entire pipeline
    state["question"] = expanded_question
    
    print(f"🧠 Query expansion: '{original_question}' → '{expanded_question}'")
    
    # Use expanded question for classification
    q = expanded_question.lower()
    
    if any(w in q for w in ["trend", "trending", "over time", "increase", "decrease"]):
        state["analysis_type"] = "performance_trend"
    elif any(w in q for w in ["compare", "versus", "vs", "benchmark"]):
        state["analysis_type"] = "compare_entities"
    elif any(w in q for w in ["why", "cause", "factor", "contribute"]):
        state["analysis_type"] = "root_cause"
    else:
        state["analysis_type"] = "facts"
    return state


def build_retriever(vector_store: Chroma):
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 40,
            "lambda_mult": 0.7,  # Balance between relevance (1.0) and diversity (0.0)
        }
    )


def retrieve_docs(state: GraphState, retriever) -> GraphState:
    q = state["question"]
    analysis_type = state.get("analysis_type", "facts")
    top_k = int(state.get("filters", {}).get("top_k", 8))
    
    # For trend queries, retrieve more docs to get better temporal coverage
    fetch_k = 16 if analysis_type == "performance_trend" else top_k
    docs: List[Document] = retriever.invoke(q)[:fetch_k]
    
    # Filter out low-relevance documents based on basic heuristics
    filtered_docs = []
    for doc in docs:
        # Simple relevance check: document should have meaningful content
        content = doc.page_content.strip()
        if len(content) > 50 and doc.metadata.get("source_id") is not None:
            filtered_docs.append(doc)
    
    # For trend analysis, ensure temporal diversity
    if analysis_type == "performance_trend" and len(filtered_docs) > top_k:
        # Sort by date and sample evenly across time range
        dated_docs = [(doc.metadata.get("date", ""), doc) for doc in filtered_docs]
        dated_docs.sort(key=lambda x: x[0])
        
        # Sample evenly: keep first, last, and evenly spaced middle docs
        step = max(1, len(dated_docs) // top_k)
        sampled = [dated_docs[i][1] for i in range(0, len(dated_docs), step)][:top_k]
        filtered_docs = sampled
    
    state["retrieved"] = filtered_docs[:top_k] if filtered_docs else docs[:min(3, len(docs))]
    return state


def compute_aggregates(state: GraphState) -> GraphState:
    filters = state.get("filters", {})
    speaker = filters.get("speaker")
    date_from = filters.get("date_from")
    date_to = filters.get("date_to")

    session = SessionLocal()
    try:
        q = session.query(db_models.Utterance)
        if speaker:
            q = q.filter(db_models.Utterance.speaker == speaker)
        if date_from:
            q = q.filter(db_models.Utterance.date >= date_from)
        if date_to:
            q = q.filter(db_models.Utterance.date <= date_to)
        utts = q.all()

        metrics_totals: Dict[str, float] = {}
        metrics_counts: Dict[str, int] = {}
        
        # Collect from both predictions (granular) and aggregated_scores
        for u in utts:
            # Process granular metrics from predictions
            if u.predictions:
                for k, v in (u.predictions or {}).items():
                    try:
                        metrics_totals[k] = metrics_totals.get(k, 0.0) + float(v)
                        metrics_counts[k] = metrics_counts.get(k, 0) + 1
                    except Exception:
                        continue
            
            # Process aggregated metrics from aggregated_scores
            if u.aggregated_scores:
                for k, v in (u.aggregated_scores or {}).items():
                    try:
                        metrics_totals[k] = metrics_totals.get(k, 0.0) + float(v)
                        metrics_counts[k] = metrics_counts.get(k, 0) + 1
                    except Exception:
                        continue

        averages = {k: (metrics_totals[k] / metrics_counts[k]) for k in metrics_totals if metrics_counts.get(k)}
        
        print(f"📊 Computed aggregates: {len(utts)} utterances, sample metrics: {list(averages.keys())[:5]}")
        if 'SAFETY_Score' in averages:
            print(f"📊 SAFETY_Score average: {averages['SAFETY_Score']:.2f}")
        
        # For trend queries, compute temporal comparison
        temporal_comparison = None
        analysis_type = state.get("analysis_type", "facts")
        if analysis_type == "performance_trend" and utts:
            # Split into early and late periods
            dated_utts = [u for u in utts if u.date]
            if dated_utts:
                dated_utts.sort(key=lambda u: u.date)
                midpoint = len(dated_utts) // 2
                early_utts = dated_utts[:midpoint]
                late_utts = dated_utts[midpoint:]
                
                # Compute averages for each period
                early_totals, early_counts = {}, {}
                late_totals, late_counts = {}, {}
                
                for u in early_utts:
                    for k, v in {**(u.predictions or {}), **(u.aggregated_scores or {})}.items():
                        try:
                            early_totals[k] = early_totals.get(k, 0.0) + float(v)
                            early_counts[k] = early_counts.get(k, 0) + 1
                        except:
                            pass
                
                for u in late_utts:
                    for k, v in {**(u.predictions or {}), **(u.aggregated_scores or {})}.items():
                        try:
                            late_totals[k] = late_totals.get(k, 0.0) + float(v)
                            late_counts[k] = late_counts.get(k, 0) + 1
                        except:
                            pass
                
                early_avgs = {k: early_totals[k] / early_counts[k] for k in early_totals if early_counts.get(k)}
                late_avgs = {k: late_totals[k] / late_counts[k] for k in late_totals if late_counts.get(k)}
                
                temporal_comparison = {
                    "early_period": {"start": early_utts[0].date, "end": early_utts[-1].date, "averages": early_avgs},
                    "late_period": {"start": late_utts[0].date, "end": late_utts[-1].date, "averages": late_avgs}
                }
                
                print(f"📊 Temporal split: early {early_utts[0].date} to {early_utts[-1].date}, late {late_utts[0].date} to {late_utts[-1].date}")
        
        state["aggregates"] = {
            "count": len(utts),
            "averages": averages,
            "temporal_comparison": temporal_comparison
        }
        return state
    finally:
        session.close()


def select_metrics_with_llm(
    question: str, 
    analysis_type: str,
    filters: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Use LLM to intelligently select appropriate metrics for charting.
    
    Returns:
        {
            "metrics": List[str],  # Metric field names from DB
            "level": "aggregated" | "granular",
            "reasoning": str
        }
        or None if selection fails
    """
    try:
        # Prepare available metrics
        aggregated_list = list(AGGREGATED_METRICS.keys())
        granular_list = list(GRANULAR_METRICS.keys())
        
        # Build prompt
        prompt = f"""You are selecting performance metrics for data visualization.

User Query: "{question}"
Analysis Type: {analysis_type}
Filters: {filters}

Available Aggregated Metrics (high-level rollups):
{json.dumps({k: METRIC_DESCRIPTIONS.get(k, AGGREGATED_METRICS[k]) for k in aggregated_list}, indent=2)}

Available Granular Metrics (specific behaviors - sample):
{json.dumps({k: GRANULAR_METRICS[k] for k in list(GRANULAR_METRICS.keys())[:15]}, indent=2)}
... and {len(GRANULAR_METRICS) - 15} more granular metrics

Instructions:
1. Choose aggregated metrics for high-level trends (e.g., "How is safety improving?")
2. Choose granular metrics for specific behavior analysis (e.g., "How is pausing?")
3. You can select multiple related granular metrics for breakdown views
4. Maximum 5 metrics total
5. Only select metrics that exist in the lists above

Return JSON:
{{
    "metrics": ["metric_name1", "metric_name2"],
    "level": "aggregated" or "granular",
    "reasoning": "Brief explanation of why these metrics were chosen"
}}"""

        llm = make_llm(json_mode=True)
        response = llm.invoke(prompt)
        
        # Parse response
        if isinstance(response.content, str):
            result = json.loads(response.content)
        else:
            result = response.content
        
        # Validate metrics exist
        all_valid_metrics = set(aggregated_list + granular_list)
        selected_metrics = result.get("metrics", [])
        
        # Filter to only valid metrics
        valid_selected = [m for m in selected_metrics if m in all_valid_metrics]
        
        if not valid_selected:
            return None
        
        # Auto-detect actual level based on selected metrics
        has_aggregated = any(m in aggregated_list for m in valid_selected)
        has_granular = any(m in granular_list for m in valid_selected)
        
        # Prefer the dominant type, default to aggregated
        if has_aggregated and not has_granular:
            detected_level = "aggregated"
        elif has_granular and not has_aggregated:
            detected_level = "granular"
        else:
            # Mixed case - prefer aggregated for trend queries
            detected_level = "aggregated" if analysis_type == "performance_trend" else "granular"
        
        return {
            "metrics": valid_selected[:5],  # Max 5
            "level": detected_level,
            "reasoning": result.get("reasoning", "")
        }
        
    except Exception as e:
        print(f"LLM metric selection error: {e}")
        return None


def fallback_metric_selection(
    question: str,
    analysis_type: str,
    filters: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Simple keyword-based fallback if LLM selection fails.
    """
    question_lower = question.lower()
    
    # Check for specific aggregated metrics
    if any(word in question_lower for word in ['safety', 'hazard', 'ppe']):
        return {"metrics": ["SAFETY_Score"], "level": "aggregated", "reasoning": "Keyword match: safety"}
    
    if any(word in question_lower for word in ['quality', 'defect', 'root cause']):
        return {"metrics": ["QUALITY_Score"], "level": "aggregated", "reasoning": "Keyword match: quality"}
    
    if any(word in question_lower for word in ['communication', 'comm ', 'speaking']):
        return {"metrics": ["Total_Comm_Score"], "level": "aggregated", "reasoning": "Keyword match: communication"}
    
    if any(word in question_lower for word in ['people', 'feedback', 'coaching']):
        return {"metrics": ["PEOPLE_Score"], "level": "aggregated", "reasoning": "Keyword match: people"}
    
    if any(word in question_lower for word in ['delivery', 'deviation']):
        return {"metrics": ["DELIVERY_Score"], "level": "aggregated", "reasoning": "Keyword match: delivery"}
    
    # Check for specific granular behaviors
    if 'pausing' in question_lower or 'pause' in question_lower:
        return {"metrics": ["comm_Pausing"], "level": "granular", "reasoning": "Keyword match: pausing"}
    
    if 'question' in question_lower:
        return {
            "metrics": ["comm_Clarifying_Questions", "comm_Probing_Questions", "comm_Open_Ended_Questions"],
            "level": "granular",
            "reasoning": "Keyword match: questions"
        }
    
    # Default: return all major aggregated scores for overview
    if analysis_type == "facts" and filters.get("speaker"):
        return {
            "metrics": ["SAFETY_Score", "QUALITY_Score", "DELIVERY_Score", "PEOPLE_Score"],
            "level": "aggregated",
            "reasoning": "Default: speaker overview"
        }
    
    return None


def generate_chart_specs(state: GraphState) -> GraphState:
    """
    LLM-powered chart specification generation with rule-based fallback.
    Uses LLM to intelligently select metrics, then applies chart type rules.
    """
    if not ENABLE_CHARTS:
        state["chart_specs"] = []
        return state
    
    analysis_type = state.get("analysis_type", "facts")
    question = state.get("question", "")
    filters = state.get("filters", {})
    
    specs = []
    telemetry = []
    
    # Try LLM-based metric selection first
    metric_selection = select_metrics_with_llm(question, analysis_type, filters)
    
    # Fallback to keyword matching if LLM fails
    if not metric_selection:
        metric_selection = fallback_metric_selection(question, analysis_type, filters)
    
    if not metric_selection:
        state["chart_specs"] = []
        return state
    
    selected_metrics = metric_selection["metrics"]
    level = metric_selection["level"]
    reasoning = metric_selection.get("reasoning", "")
    
    print(f"📊 Chart spec generation: metrics={selected_metrics}, level={level}, analysis_type={analysis_type}")
    
    # Determine chart type based on analysis_type and question
    question_lower = question.lower()
    
    # Rule 1: Trend analysis → Line chart
    if analysis_type == "performance_trend" or any(word in question_lower for word in ["trend", "over time", "improve", "changed", "progress"]):
        for metric in selected_metrics[:1]:  # Only first metric for line charts to avoid clutter
            specs.append({
                "type": "line",
                "metric": metric,
                "metrics": None,
                "group_by": "date",
                "filters": filters,
                "level": level
            })
            telemetry.append({
                "chart_type": "line",
                "metric": metric,
                "level": level,
                "reason": f"trend_analysis: {reasoning}"
            })
    
    # Rule 2: Comparison → Bar chart
    elif analysis_type == "compare_entities" or any(word in question_lower for word in ["compare", "who has", "rank", "best", "worst"]):
        for metric in selected_metrics[:1]:  # Only first metric for comparisons
            specs.append({
                "type": "bar",
                "metric": metric,
                "metrics": None,
                "group_by": "speaker",
                "filters": filters,
                "level": level
            })
            telemetry.append({
                "chart_type": "bar",
                "metric": metric,
                "level": level,
                "reason": f"comparison: {reasoning}"
            })
    
    # Rule 3: Facts / Overview → Grouped bar (multiple metrics)
    elif analysis_type == "facts":
        specs.append({
            "type": "grouped_bar",
            "metric": None,
            "metrics": selected_metrics,
            "group_by": "metric",
            "filters": filters,
            "level": level
        })
        telemetry.append({
            "chart_type": "grouped_bar",
            "metrics": selected_metrics,
            "level": level,
            "reason": f"overview: {reasoning}"
        })
    
    # Limit to MAX_CHARTS_PER_RESPONSE
    state["chart_specs"] = specs[:MAX_CHARTS_PER_RESPONSE]
    
    # Add telemetry to metadata
    if telemetry:
        metadata = state.get("draft", {}).get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        metadata["chart_telemetry"] = telemetry
        if "draft" not in state:
            state["draft"] = {}
        state["draft"]["metadata"] = metadata
    
    return state


def execute_chart_queries(state: GraphState) -> GraphState:
    """
    Execute chart data queries based on chart specs.
    Handles errors gracefully - charts are optional enhancements.
    """
    chart_specs = state.get("chart_specs", [])
    if not chart_specs:
        state["charts"] = []
        return state
    
    charts = []
    db = SessionLocal()
    
    try:
        for spec in chart_specs:
            try:
                chart_type = spec.get("type")
                metric = spec.get("metric")
                metrics = spec.get("metrics")
                group_by = spec.get("group_by", "date")
                filters = spec.get("filters", {})
                level = spec.get("level", "aggregated")
                
                # Get chart data from database
                chart_data = get_chart_data(
                    db=db,
                    chart_type=chart_type,
                    metric=metric,
                    metrics=metrics,
                    group_by=group_by,
                    filters=filters,
                    level=level
                )
                
                if not chart_data:
                    # Track skipped charts
                    metadata = state.get("draft", {}).get("metadata", {})
                    if isinstance(metadata, dict):
                        skipped = metadata.get("charts_skipped", [])
                        skipped.append({"type": chart_type, "reason": "no_data"})
                        metadata["charts_skipped"] = skipped
                    continue
                
                # Build chart config
                config = _build_chart_config(chart_type, metric, metrics, filters)
                
                # Assemble chart object
                chart = {
                    "type": chart_type,
                    "data": chart_data,
                    "config": config
                }
                
                charts.append(chart)
                
            except Exception as e:
                # Log error but continue - charts are optional
                print(f"Chart generation error: {e}")
                metadata = state.get("draft", {}).get("metadata", {})
                if isinstance(metadata, dict):
                    errors = metadata.get("chart_errors", [])
                    errors.append({"type": spec.get("type"), "error": str(e)})
                    metadata["chart_errors"] = errors
                continue
    
    finally:
        db.close()
    
    state["charts"] = charts
    return state


def _build_chart_config(chart_type: str, metric: Optional[str], metrics: Optional[List[str]], filters: Dict[str, Any]) -> Dict[str, Any]:
    """Build chart configuration with titles and labels."""
    # Ensure colors have # prefix
    colors = [f"#{c.lstrip('#')}" for c in CHART_DEFAULT_COLORS]
    
    config = {
        "colors": colors
    }
    
    # Determine if we're using aggregated metrics (which have 0-50 range)
    is_aggregated = False
    check_metric = metric if metric else (metrics[0] if metrics else None)
    if check_metric and check_metric in AGGREGATED_METRICS:
        is_aggregated = True
    
    if chart_type == "line":
        metric_name = METRIC_DISPLAY_NAMES.get(metric, metric)
        y_label = "Score (0-50)" if is_aggregated else "Score (1-5)"
        config.update({
            "title": f"{metric_name} Over Time",
            "xAxisLabel": "Date",
            "yAxisLabel": y_label,
            "yDomain": [0, 50] if is_aggregated else [0, 5]
        })
    
    elif chart_type == "bar":
        metric_name = METRIC_DISPLAY_NAMES.get(metric, metric)
        y_label = "Average Score (0-50)" if is_aggregated else "Average Score (1-5)"
        config.update({
            "title": f"Speaker Comparison - {metric_name}",
            "xAxisLabel": "Speaker",
            "yAxisLabel": y_label,
            "yDomain": [0, 50] if is_aggregated else [0, 5]
        })
    
    elif chart_type == "grouped_bar":
        speaker = filters.get("speaker", "Team")
        # For grouped bar, check if any metric is aggregated
        has_aggregated = any(m in AGGREGATED_METRICS for m in (metrics or []))
        y_label = "Score (0-50)" if has_aggregated else "Score (1-5)"
        config.update({
            "title": f"{speaker} - Performance Metrics",
            "xAxisLabel": "Metric",
            "yAxisLabel": y_label,
            "yDomain": [0, 50] if has_aggregated else [0, 5]
        })
    
    return config


def make_llm(json_mode: bool = False) -> ChatOpenAI:
    model_kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    return ChatOpenAI(
        model_name=LLM_MODEL_NAME,
        temperature=LLM_TEMPERATURE,
        model_kwargs=model_kwargs,
    )


def filter_relevant_metrics(
    question: str, 
    averages: Dict[str, float], 
    top_k: int = 10
) -> Dict[str, float]:
    """
    Use embedding similarity to keep only relevant metrics.
    Reduces prompt bloat from 50+ metrics to ~10.
    
    Args:
        question: The user's question
        averages: Dict of all metric averages
        top_k: Number of top metrics to keep
    
    Returns:
        Filtered dict with only top-k relevant metrics
    """
    if not averages or len(averages) <= top_k:
        return averages
    
    try:
        client = OpenAI()
        
        # Embed question
        q_response = client.embeddings.create(
            model="text-embedding-3-small",
            input=question
        )
        q_emb = np.array(q_response.data[0].embedding)
        
        # Embed metric names + descriptions
        metric_texts = []
        metric_keys = list(averages.keys())
        
        for key in metric_keys:
            # Use display name + description if available
            display = METRIC_DISPLAY_NAMES.get(key, key)
            desc = METRIC_DESCRIPTIONS.get(key, "")
            metric_texts.append(f"{display}: {desc}")
        
        m_response = client.embeddings.create(
            model="text-embedding-3-small",
            input=metric_texts
        )
        
        # Compute cosine similarity
        scores = []
        for i, key in enumerate(metric_keys):
            m_emb = np.array(m_response.data[i].embedding)
            # Cosine similarity
            similarity = np.dot(q_emb, m_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(m_emb))
            scores.append((key, float(similarity)))
        
        # Keep top-k most relevant
        top_metrics = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
        filtered = {k: averages[k] for k, _ in top_metrics}
        
        print(f"📊 Filtered metrics: {len(averages)} → {len(filtered)}")
        print(f"   Top 3 relevant: {[k for k, _ in top_metrics[:3]]}")
        
        return filtered
        
    except Exception as e:
        print(f"⚠️ Metric filtering failed: {e}, using all metrics")
        return averages


def _prepare_context(state: GraphState) -> Dict[str, Any]:
    # Deduplicate citations by source_id
    seen_ids = set()
    citations = []
    for d in state.get("retrieved", []):
        if not d:
            continue
        source_id = d.metadata.get("source_id")
        if source_id and source_id not in seen_ids:
            seen_ids.add(source_id)
            citations.append({
                "source_id": source_id,
                "speaker": d.metadata.get("speaker"),
                "date": d.metadata.get("date"),
                "timestamp": d.metadata.get("timestamp"),
                "snippet": d.page_content[:200],
            })
    
    # Format conversation history for LLM context
    history = state.get("history", [])
    conversation_context = ""
    
    if history:
        recent_turns = history[-5:]  # Last 5 turns
        conversation_context = "CONVERSATION HISTORY:\n"
        for i, turn in enumerate(recent_turns, 1):
            user_q = turn.get("user", "")
            assistant_a = turn.get("assistant", "")
            conversation_context += f"Turn {i}:\n"
            conversation_context += f"  User: {user_q}\n"
            # Truncate assistant response to 150 chars for context
            assistant_preview = assistant_a[:150] + "..." if len(assistant_a) > 150 else assistant_a
            conversation_context += f"  Assistant: {assistant_preview}\n\n"
        
        print(f"💬 Including {len(recent_turns)} conversation turns in context")
    
    # Filter metrics to reduce context bloat
    aggregates_raw = state.get("aggregates", {})
    averages = aggregates_raw.get("averages", {})
    
    # Filter to relevant metrics using embedding similarity
    filtered_averages = filter_relevant_metrics(
        state['question'], 
        averages,
        top_k=10
    )
    
    # Rebuild aggregates with filtered averages
    aggregates_filtered = {
        "count": aggregates_raw.get("count"),
        "averages": filtered_averages,
        "temporal_comparison": aggregates_raw.get("temporal_comparison")
    }
    
    agg_json = json.dumps(aggregates_filtered)
    print(f"🔍 Aggregates JSON (first 500 chars): {agg_json[:500]}")
    
    return {
        "question": state['question'],
        "conversation_history": conversation_context,  # NEW: conversation memory
        "analysis_type": state.get("analysis_type", "facts"),
        "aggregates": agg_json,
        "citations": json.dumps(citations),
        "valid_source_ids": json.dumps([c["source_id"] for c in citations if c.get("source_id") is not None]),
        "_citations": citations, # Pass this through internally
    }


def _get_answer_chain(json_mode: bool = False) -> Runnable:
    system_msg = answer_system(json_mode=json_mode)
    user_tmpl = answer_user_template()
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("user", user_tmpl),
    ])
    return prompt | make_llm(json_mode=json_mode)


def _parse_llm_output(content: str | dict, state: GraphState) -> GraphState:
    if isinstance(content, str):
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = {"answer": content, "bullets": [], "metrics_summary": [], "follow_ups": []}
    else:
        data = content

    # Basic validation and fallback for source IDs
    valid_source_ids = [c["source_id"] for c in state.get("_citations", []) if c.get("source_id") is not None]
    if not data.get("source_ids"):
        data["source_ids"] = valid_source_ids[:min(3, len(valid_source_ids))]

    state["draft"] = data
    return state


def generate_draft_sync(state: GraphState) -> GraphState:
    """Synchronous draft generation for the non-streaming `run` method."""
    context = _prepare_context(state)
    state["_citations"] = context.pop("_citations", []) # Keep citations for formatting
    chain = _get_answer_chain(json_mode=True)
    resp = chain.invoke(context)
    return _parse_llm_output(resp.content, state)

def extract_metrics_from_text(text: str) -> List[str]:
    """Extract metric names mentioned in text"""
    metrics = []
    all_metrics = list(AGGREGATED_METRICS.keys()) + list(GRANULAR_METRICS.keys())
    
    text_lower = text.lower()
    for metric in all_metrics:
        display_name = METRIC_DISPLAY_NAMES.get(metric, metric)
        # Check if metric or display name appears in text
        if metric.lower() in text_lower or display_name.lower() in text_lower:
            metrics.append(metric)
    
    return list(set(metrics))  # Deduplicate


def extract_speakers_from_text(text: str) -> List[str]:
    """Extract speaker names from text"""
    speakers = []
    text_lower = text.lower()
    
    # Get known speakers from database
    session = SessionLocal()
    try:
        known_speakers_db = session.query(db_models.Utterance.speaker)\
            .filter(db_models.Utterance.speaker.isnot(None))\
            .distinct()\
            .limit(20)\
            .all()
        known_speakers = [s[0] for s in known_speakers_db if s[0]]
    finally:
        session.close()
    
    for speaker in known_speakers:
        if speaker.lower() in text_lower:
            speakers.append(speaker)
    
    return list(set(speakers))  # Deduplicate


def generate_smart_follow_ups(
    question: str,
    answer_text: str,
    history: List[dict],
    aggregates: dict
) -> List[str]:
    """Generate context-aware follow-up questions"""
    
    follow_ups = []
    
    # Extract what was discussed in current exchange
    combined_text = question + " " + answer_text
    mentioned_metrics = extract_metrics_from_text(combined_text)
    mentioned_speakers = extract_speakers_from_text(combined_text)
    
    # 1. Speaker comparison follow-ups
    if mentioned_speakers and len(mentioned_speakers) == 1:
        # They asked about one speaker, suggest comparing to others
        session = SessionLocal()
        try:
            all_speakers_db = session.query(db_models.Utterance.speaker)\
                .filter(db_models.Utterance.speaker.isnot(None))\
                .distinct()\
                .limit(10)\
                .all()
            all_speakers = [s[0] for s in all_speakers_db if s[0]]
        finally:
            session.close()
        
        other_speakers = [s for s in all_speakers if s not in mentioned_speakers]
        
        if other_speakers and mentioned_metrics:
            metric_display = METRIC_DISPLAY_NAMES.get(mentioned_metrics[0], mentioned_metrics[0])
            follow_ups.append(f"How does {mentioned_speakers[0]} compare to {other_speakers[0]} on {metric_display}?")
    
    # 2. Temporal follow-ups
    is_temporal_query = any(word in question.lower() for word in ["trend", "over time", "improved", "changed", "increased", "decreased"])
    
    if is_temporal_query and mentioned_metrics:
        # They asked about trends, suggest drilling into causes
        metric_display = METRIC_DISPLAY_NAMES.get(mentioned_metrics[0], mentioned_metrics[0])
        follow_ups.append(f"What specific behaviors drove the change in {metric_display}?")
    elif mentioned_metrics and not is_temporal_query:
        # They asked about current state, suggest temporal view
        metric_display = METRIC_DISPLAY_NAMES.get(mentioned_metrics[0], mentioned_metrics[0])
        follow_ups.append(f"How has {metric_display} trended over time?")
    
    # 3. Proactive insights from data anomalies
    temporal = aggregates.get("temporal_comparison")
    if temporal:
        early = temporal.get("early_period", {}).get("averages", {})
        late = temporal.get("late_period", {}).get("averages", {})
        
        # Find metrics with big changes (>10%)
        big_changes = []
        for metric in set(early.keys()) & set(late.keys()):
            if early[metric] > 0:  # Avoid division by zero
                change_pct = ((late[metric] - early[metric]) / early[metric]) * 100
                if abs(change_pct) > 10:
                    big_changes.append((metric, change_pct))
        
        # Sort by magnitude
        big_changes.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Suggest if not already discussed
        for metric, pct in big_changes[:2]:
            if metric not in mentioned_metrics:
                direction = "improved" if pct > 0 else "declined"
                display_name = METRIC_DISPLAY_NAMES.get(metric, metric)
                follow_ups.append(f"I noticed {display_name} {direction} by {abs(pct):.1f}% - want to explore that?")
                break  # Only add one proactive insight
    
    # 4. Conversation continuity
    if history and mentioned_metrics:
        last_turn = history[-1]
        last_question = last_turn.get("user", "")
        last_metrics = extract_metrics_from_text(last_question)
        
        # If they're asking about a different topic, offer to connect
        if last_metrics and set(last_metrics) != set(mentioned_metrics):
            metric1_display = METRIC_DISPLAY_NAMES.get(mentioned_metrics[0], mentioned_metrics[0])
            metric2_display = METRIC_DISPLAY_NAMES.get(last_metrics[0], last_metrics[0])
            follow_ups.append(f"Want to compare {metric1_display} with {metric2_display}?")
    
    # Return top 3, ensuring variety
    return follow_ups[:3]


def verify_faithfulness(state: GraphState) -> GraphState:
    """Optional verification step to check answer faithfulness."""
    draft = state.get("draft", {})
    answer_text = draft.get("answer", "")
    citations = state.get("_citations", [])
    
    if not answer_text or not citations:
        return state
    
    try:
        verification_prompt = ChatPromptTemplate.from_messages([
            ("system", verification_system()),
            ("user", verification_user_template()),
        ])
        chain = verification_prompt | make_llm(json_mode=True)
        
        result = chain.invoke({
            "question": state.get("question", ""),
            "answer": answer_text,
            "citations": json.dumps([c.get("snippet", "") for c in citations]),
        })
        
        verification = json.loads(result.content)
        
        # Add verification metadata to state for debugging/monitoring
        if not verification.get("is_faithful", True) or verification.get("confidence", 1.0) < 0.7:
            if "metadata" not in state.get("draft", {}):
                state["draft"]["metadata"] = {}
            state["draft"]["metadata"]["faithfulness_warning"] = True
            state["draft"]["metadata"]["unsupported_claims"] = verification.get("unsupported_claims", [])
            state["draft"]["metadata"]["faithfulness_confidence"] = verification.get("confidence", 0.0)
    except Exception as e:
        # Non-blocking - if verification fails, continue without it
        print(f"Verification step failed: {e}")
    
    return state


def format_answer(state: GraphState) -> GraphState:
    data = state.get("draft", {}) or {}
    all_citations = state.get("_citations", [])
    raw_ids = data.get("source_ids", []) or []
    used_source_ids: set[int] = set()
    if isinstance(raw_ids, list):
        for x in raw_ids:
            try:
                used_source_ids.add(int(x))
            except (ValueError, TypeError):
                continue
    
    if not used_source_ids and all_citations:
        used_source_ids = {int(c.get("source_id")) for c in all_citations if c.get("source_id") is not None}
    
    final_citations = [c for c in all_citations if c.get("source_id") in used_source_ids]

    # Defensive check for metrics_summary to handle incorrect LLM output
    metrics_summary_raw = data.get("metrics_summary", []) or []
    if isinstance(metrics_summary_raw, str):
        # Wrap a rogue string in the required List[Dict] format
        metrics_summary = [{"summary": metrics_summary_raw}]
    elif isinstance(metrics_summary_raw, dict):
        # Wrap a single dict in a list
        metrics_summary = [metrics_summary_raw]
    elif isinstance(metrics_summary_raw, list):
        # Ensure all items in the list are dicts
        metrics_summary = [item if isinstance(item, dict) else {"item": item} for item in metrics_summary_raw]
    else:
        metrics_summary = [] # Default to an empty list

    # Generate smart, context-aware follow-ups
    smart_follow_ups = generate_smart_follow_ups(
        question=state["question"],
        answer_text=data.get("answer", ""),
        history=state.get("history", []),
        aggregates=state.get("aggregates", {})
    )
    
    # Use smart follow-ups if generated, otherwise fall back to LLM's suggestions
    follow_ups = smart_follow_ups if smart_follow_ups else (data.get("follow_ups", []) or [])

    # Build metadata, including verification warnings if present
    metadata_dict = {
        "analysis_type": state.get("analysis_type"),
        "count": state.get("aggregates", {}).get("count"),
        "data_quality": "low" if (state.get("aggregates", {}).get("count", 0) < 5) else "normal",
    }
    
    # Merge verification metadata if present
    draft_metadata = data.get("metadata", {})
    if isinstance(draft_metadata, dict):
        metadata_dict.update(draft_metadata)

    # Include charts in the answer
    charts = state.get("charts", []) or []
    
    answer = RAGAnswer(
        answer=data.get("answer", ""),
        bullets=data.get("bullets", []) or [],
        metrics_summary=metrics_summary,
        citations=[RAGCitation(**c) for c in final_citations if c.get("source_id") is not None],
        follow_ups=follow_ups,
        charts=[Chart(**c) for c in charts],
        metadata=metadata_dict,
    )
    state["answer"] = json.loads(answer.model_dump_json())
    return state


class RAGGraph:
    def __init__(self, vector_store: Optional[Chroma] = None):
        self.vector_store = vector_store or Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=OpenAIEmbeddings(),
        )
        self.retriever = build_retriever(self.vector_store)

        graph = StateGraph(GraphState)
        graph.add_node("load_history", load_history)
        graph.add_node("classify", classify_query)
        graph.add_node("retrieve", lambda s: retrieve_docs(s, self.retriever))
        graph.add_node("aggregate", compute_aggregates)
        graph.add_node("chart_specs", generate_chart_specs)  # NEW: Generate chart specifications
        graph.add_node("chart_data", execute_chart_queries)  # NEW: Execute chart queries
        graph.add_node("draft", generate_draft_sync)
        graph.add_node("verify", verify_faithfulness)
        graph.add_node("format", format_answer)
        graph.add_node("save_history", save_history)

        graph.set_entry_point("load_history")
        graph.add_edge("load_history", "classify")
        graph.add_edge("classify", "retrieve")
        graph.add_edge("retrieve", "aggregate")
        graph.add_edge("aggregate", "chart_specs")  # Generate chart specs after aggregates
        graph.add_edge("chart_specs", "chart_data")  # Execute chart queries
        graph.add_edge("chart_data", "draft")  # Then continue to draft
        graph.add_edge("draft", "verify")
        graph.add_edge("verify", "format")
        graph.add_edge("format", "save_history")

        self.app = graph.compile()

    async def astream_run(self, question: str, session_id: Optional[str] = None, filters: Optional[Dict[str, Any]] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Runs the RAG process with streaming for the answer."""
        state: GraphState = {
            "question": question, "session_id": session_id, "filters": filters or {},
            "analysis_type": "facts", "history": [], "retrieved": [], "aggregates": {}, "draft": {}, "answer": {}, "_citations": [],
            "chart_specs": [], "charts": []
        }
        
        # 1. Run pre-processing steps
        state = load_history(state)
        state = classify_query(state)
        state = retrieve_docs(state, self.retriever)
        state = compute_aggregates(state)
        state = generate_chart_specs(state)  # Generate chart specs
        state = execute_chart_queries(state)  # Execute chart queries

        # 2. Prepare context and stream the narrative answer
        context = _prepare_context(state)
        state["_citations"] = context.pop("_citations", [])
        
        # Debug: Check what aggregates are being passed
        agg_data = state.get("aggregates", {})
        print(f"🔍 Aggregates being passed to LLM: count={agg_data.get('count')}, has SAFETY_Score={('SAFETY_Score' in agg_data.get('averages', {}))}")
        if 'SAFETY_Score' in agg_data.get('averages', {}):
            print(f"🔍 SAFETY_Score value: {agg_data['averages']['SAFETY_Score']:.2f}")
        print(f"🔍 Context keys: {list(context.keys())}")
        print(f"🔍 Aggregates string length: {len(context.get('aggregates', ''))}")
        
        answer_chain = _get_answer_chain(json_mode=False)
        
        full_answer = ""
        async for chunk in answer_chain.astream(context):
            content_chunk = chunk.content
            if content_chunk:
                full_answer += content_chunk
                yield {"answer_token": content_chunk}

        # 3. Get the metadata in a second, non-streaming call
        metadata_chain = _get_answer_chain(json_mode=True)
        # Add the streamed answer to the context for the metadata call
        context["answer"] = full_answer
        
        # Refine the prompt for the metadata call
        metadata_prompt = ChatPromptTemplate.from_messages([
            ("system", metadata_system()),
            ("user", metadata_user_template()),
        ])
        
        final_chain = metadata_prompt | make_llm(json_mode=True)
        resp = await final_chain.ainvoke(context)
        state = _parse_llm_output(resp.content, state)
        state["draft"]["answer"] = full_answer # Add the streamed answer to the draft

        # 4. Verify faithfulness
        state = verify_faithfulness(state)

        # 5. Format and save
        state = format_answer(state)
        state = save_history(state)

        # 5. Yield final payload WITHOUT the full answer (already streamed)
        final_payload = state.get("answer", {}).copy()
        final_payload.pop("answer", None)  # Remove duplicate answer text
        yield final_payload

    def run(self, question: str, session_id: Optional[str] = None, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Synchronous, non-streaming execution of the graph."""
        state: GraphState = {
            "question": question, "session_id": session_id, "filters": filters or {},
            "analysis_type": "facts", "history": [], "retrieved": [], "aggregates": {}, "draft": {}, "answer": {}, "_citations": [],
            "chart_specs": [], "charts": []
        }
        final_state = self.app.invoke(state)
        return final_state.get("answer", {})
