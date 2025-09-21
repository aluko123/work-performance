from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, TypedDict

import redis
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langgraph.graph import StateGraph

from . import db_models
from .database import SessionLocal
from .models import RAGAnswer, RAGCitation


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


def _redis_client() -> redis.Redis:
    url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    return redis.from_url(url)


def load_history(state: GraphState) -> GraphState:
    sid = state.get("session_id")
    if not sid:
        state["history"] = []
        return state
    r = _redis_client()
    raw = r.get(f"rag:hist:{sid}")
    history: List[Dict[str, str]] = json.loads(raw) if raw else []
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
    r.set(f"rag:hist:{sid}", json.dumps(history))
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
    # Use MMR for diversity
    return vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 8, "fetch_k": 40})


def retrieve_docs(state: GraphState, retriever) -> GraphState:
    q = state["question"]
    top_k = int(state.get("filters", {}).get("top_k", 8))
    # Basic retrieval
    docs: List[Document] = retriever.get_relevant_documents(q)
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

        # Aggregate simple averages per metric across utterances
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


def make_llm() -> ChatOpenAI:
    return ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)


def generate_draft(state: GraphState) -> GraphState:
    llm = make_llm()
    analysis_type = state.get("analysis_type", "facts")
    aggregates = state.get("aggregates", {})
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

    system = (
        "You are an assistant producing concise, evidence-backed performance insights. "
        "Return ONLY valid JSON matching the schema keys: answer, bullets, metrics_summary, follow_ups."
    )
    user = (
        f"Question: {state['question']}\n"
        f"Analysis type: {analysis_type}\n"
        f"Aggregates (sample size, averages): {json.dumps(aggregates) }\n"
        f"Citations (for reference): {json.dumps(citations)}\n"
        "Constraints: <=120 words in 'answer'; 3-5 bullets; 2-4 follow_ups."
    )
    msg = [
        ("system", system),
        ("user", user),
    ]

    resp = llm.invoke(msg)
    try:
        data = json.loads(resp.content)
    except Exception:
        data = {"answer": resp.content, "bullets": [], "metrics_summary": [], "follow_ups": []}

    state["draft"] = data
    # Carry citations forward for formatting
    state["_citations"] = citations
    return state


def format_answer(state: GraphState) -> GraphState:
    data = state.get("draft", {}) or {}
    citations = state.get("_citations", [])

    # Defensive check: ensure metrics_summary is a list, as the model expects.
    # The LLM may return a single dict instead of a list of dicts.
    metrics_summary_raw = data.get("metrics_summary", []) or []
    if isinstance(metrics_summary_raw, dict):
        metrics_summary = [metrics_summary_raw]
    else:
        metrics_summary = metrics_summary_raw

    answer = RAGAnswer(
        answer=data.get("answer", ""),
        bullets=data.get("bullets", []) or [],
        metrics_summary=metrics_summary,
        citations=[RAGCitation(**c) for c in citations if c.get("source_id") is not None],
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
            persist_directory=os.getenv("CHROMA_DIR", "./data/chroma_db"),
            embedding_function=OpenAIEmbeddings(),
        )
        self.retriever = build_retriever(self.vector_store)

        graph = StateGraph(GraphState)
        graph.add_node("load_history", load_history)
        graph.add_node("classify", classify_query)
        graph.add_node("retrieve", lambda s: retrieve_docs(s, self.retriever))
        graph.add_node("aggregate", compute_aggregates)
        graph.add_node("draft", generate_draft)
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

    def run(self, question: str, session_id: Optional[str] = None, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        state: GraphState = {
            "question": question,
            "session_id": session_id,
            "filters": filters or {},
            "analysis_type": "facts",
            "history": [],
            "retrieved": [],
            "aggregates": {},
            "draft": {},
            "answer": {},
        }
        final = self.app.invoke(state)
        return final.get("answer", {})

