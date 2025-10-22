from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, TypedDict, AsyncGenerator

import redis
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.prompts import ChatPromptTemplate

from . import db_models
from .database import SessionLocal
from .models import RAGAnswer, RAGCitation
from .prompts import (
    answer_system,
    answer_user_template,
    metadata_system,
    metadata_user_template,
)

# --- Constants ---
REDIS_URL = os.getenv("REDIS_URL") or os.getenv("ARQ_REDIS_URL") or "redis://redis:6379/0"
LLM_MODEL_NAME = "gpt-4o-mini"
LLM_TEMPERATURE = 0.2
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


def _redis_client() -> Optional[redis.Redis]:
    """Return a Redis client if available; otherwise None."""
    try:
        client = redis.from_url(REDIS_URL)
        client.ping()
        return client
    except Exception:
        return None


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
    state["history"] = history[-10:]
    return state


def save_history(state: GraphState) -> GraphState:
    sid = state.get("session_id")
    if not sid:
        return state
    r = _redis_client()
    history: List[Dict[str, str]] = state.get("history", [])
    draft = state.get("answer") or {}
    q = state.get("question")
    a = draft.get("answer") if isinstance(draft, dict) else None
    if q and a:
        history = (history + [{"role": "user", "content": q}, {"role": "assistant", "content": a}])[-10:]
    if r:
        try:
            r.set(f"rag:hist:{sid}", json.dumps(history))
        except Exception:
            pass
    state["history"] = history
    return state


def classify_query(state: GraphState) -> GraphState:
    q = state["question"].lower()
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
    return vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 40})


def retrieve_docs(state: GraphState, retriever) -> GraphState:
    q = state["question"]
    top_k = int(state.get("filters", {}).get("top_k", 8))
    docs: List[Document] = retriever.invoke(q)
    state["retrieved"] = docs[:top_k]
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
        for u in utts:
            if not u.predictions:
                continue
            for k, v in (u.predictions or {}).items():
                try:
                    metrics_totals[k] = metrics_totals.get(k, 0.0) + float(v)
                    metrics_counts[k] = metrics_counts.get(k, 0) + 1
                except Exception:
                    continue

        averages = {k: (metrics_totals[k] / metrics_counts[k]) for k in metrics_totals if metrics_counts.get(k)}
        state["aggregates"] = {
            "count": len(utts),
            "averages": averages,
        }
        return state
    finally:
        session.close()


def make_llm(json_mode: bool = False) -> ChatOpenAI:
    model_kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    return ChatOpenAI(
        model_name=LLM_MODEL_NAME,
        temperature=LLM_TEMPERATURE,
        model_kwargs=model_kwargs,
    )


def _prepare_context(state: GraphState) -> Dict[str, Any]:
    citations = [
        {
            "source_id": d.metadata.get("source_id"),
            "speaker": d.metadata.get("speaker"),
            "date": d.metadata.get("date"),
            "timestamp": d.metadata.get("timestamp"),
            "snippet": d.page_content[:200],
        }
        for d in state.get("retrieved", [])
        if d
    ]
    return {
        "question": state['question'],
        "analysis_type": state.get("analysis_type", "facts"),
        "aggregates": json.dumps(state.get("aggregates", {})),
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

    answer = RAGAnswer(
        answer=data.get("answer", ""),
        bullets=data.get("bullets", []) or [],
        metrics_summary=metrics_summary,
        citations=[RAGCitation(**c) for c in final_citations if c.get("source_id") is not None],
        follow_ups=data.get("follow_ups", []) or [],
        metadata={
            "analysis_type": state.get("analysis_type"),
            "count": state.get("aggregates", {}).get("count"),
            "data_quality": "low" if (state.get("aggregates", {}).get("count", 0) < 5) else "normal",
        },
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
        graph.add_node("draft", generate_draft_sync)
        graph.add_node("format", format_answer)
        graph.add_node("save_history", save_history)

        graph.set_entry_point("load_history")
        graph.add_edge("load_history", "classify")
        graph.add_edge("classify", "retrieve")
        graph.add_edge("retrieve", "aggregate")
        graph.add_edge("aggregate", "draft")
        graph.add_edge("draft", "format")
        graph.add_edge("format", "save_history")

        self.app = graph.compile()

    async def astream_run(self, question: str, session_id: Optional[str] = None, filters: Optional[Dict[str, Any]] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Runs the RAG process with streaming for the answer."""
        state: GraphState = {
            "question": question, "session_id": session_id, "filters": filters or {},
            "analysis_type": "facts", "history": [], "retrieved": [], "aggregates": {}, "draft": {}, "answer": {}, "_citations": []
        }
        
        # 1. Run pre-processing steps
        state = load_history(state)
        state = classify_query(state)
        state = retrieve_docs(state, self.retriever)
        state = compute_aggregates(state)

        # 2. Prepare context and stream the narrative answer
        context = _prepare_context(state)
        state["_citations"] = context.pop("_citations", [])
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

        # 4. Format and save
        state = format_answer(state)
        state = save_history(state)

        # 5. Yield final payload
        yield state.get("answer", {})

    def run(self, question: str, session_id: Optional[str] = None, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Synchronous, non-streaming execution of the graph."""
        state: GraphState = {
            "question": question, "session_id": session_id, "filters": filters or {},
            "analysis_type": "facts", "history": [], "retrieved": [], "aggregates": {}, "draft": {}, "answer": {}, "_citations": []
        }
        final_state = self.app.invoke(state)
        return final_state.get("answer", {})
