import json
from unittest.mock import MagicMock, patch

import pytest
from langchain.schema import Document

from backend import rag_graph as rg
from backend import db_models
from backend.rag_graph import RAGGraph


def test_classify_query_types():
    state = {"question": "Show me trends over time", "filters": {}}
    out = rg.classify_query(state)
    assert out["analysis_type"] == "performance_trend"

    state = {"question": "Compare Alice vs Bob", "filters": {}}
    out = rg.classify_query(state)
    assert out["analysis_type"] == "compare_entities"

    state = {"question": "Why did scores drop?", "filters": {}}
    out = rg.classify_query(state)
    assert out["analysis_type"] == "root_cause"

    state = {"question": "What happened yesterday?", "filters": {}}
    out = rg.classify_query(state)
    assert out["analysis_type"] == "facts"


def test_format_answer_shapes():
    draft = {
        "answer": "Scores improved.",
        "bullets": ["Clarity up", "Feedback timely"],
        "metrics_summary": [{"comm_Pausing": 3.2}],
        "follow_ups": ["Which speakers?"],
    }
    citations = [
        {"source_id": 1, "speaker": "Alice", "date": "2024-09-01", "timestamp": "08:00:00", "snippet": "..."}
    ]
    state = {"draft": draft, "_citations": citations, "analysis_type": "facts", "aggregates": {"count": 2}}
    out = rg.format_answer(state)
    ans = out["answer"]
    assert set(ans.keys()) == {"answer", "bullets", "metrics_summary", "citations", "follow_ups", "metadata"}
    assert ans["citations"][0]["source_id"] == 1


def test_history_load_save(fake_redis_client, monkeypatch):
    monkeypatch.setattr(rg, "_redis_client", lambda: fake_redis_client)
    state = {"session_id": "abc", "question": "Q?", "answer": {"answer": "A!"}, "history": []}
    out = rg.save_history(state)
    assert out["history"]  # stored
    out2 = rg.load_history({"session_id": "abc"})
    assert isinstance(out2["history"], list)


def test_compute_aggregates(temp_db_session, monkeypatch):
    # Patch SessionLocal used inside rag_graph to our temp session factory
    monkeypatch.setattr(rg, "SessionLocal", temp_db_session)

    # Seed DB
    session = temp_db_session()
    try:
        a = db_models.Analysis(source_filename="t.txt")
        session.add(a)
        session.flush()
        u1 = db_models.Utterance(
            analysis_id=a.id,
            speaker="Alice",
            date="2024-09-01",
            predictions={"comm_Pausing": 3},
        )
        u2 = db_models.Utterance(
            analysis_id=a.id,
            speaker="Bob",
            date="2024-09-02",
            predictions={"comm_Pausing": 5},
        )
        session.add_all([u1, u2])
        session.commit()
    finally:
        session.close()

    state = {"question": "", "filters": {}}
    out = rg.compute_aggregates(state)
    assert out["aggregates"]["count"] == 2
    avg = out["aggregates"]["averages"]["comm_Pausing"]
    assert 3.9 < avg < 4.1


def test_retrieve_docs():
    # Mock the retriever
    mock_retriever = MagicMock()
    mock_retriever.get_relevant_documents.return_value = [
        Document(page_content="doc1"),
        Document(page_content="doc2"),
        Document(page_content="doc3"),
    ]

    state = {"question": "test", "filters": {"top_k": 2}}
    out = rg.retrieve_docs(state, mock_retriever)
    
    assert len(out["retrieved"]) == 2
    mock_retriever.get_relevant_documents.assert_called_once_with("test")


@pytest.mark.parametrize("utterances, expected_count, expected_avg", [
    ([], 0, 0),  # No utterances
    ([db_models.Utterance(predictions=None)], 1, 0), # Utterance with no predictions
    ([db_models.Utterance(predictions={"comm_Pausing": "invalid"})], 1, 0), # Prediction is not a valid number
])
def test_compute_aggregates_edge_cases(temp_db_session, monkeypatch, utterances, expected_count, expected_avg):
    monkeypatch.setattr(rg, "SessionLocal", temp_db_session)
    session = temp_db_session()
    try:
        a = db_models.Analysis(source_filename="t.txt")
        session.add(a)
        session.flush()
        for utt in utterances:
            utt.analysis_id = a.id
            session.add(utt)
        session.commit()
    finally:
        session.close()

    state = {"question": "", "filters": {}}
    out = rg.compute_aggregates(state)
    
    assert out["aggregates"]["count"] == expected_count
    avg = out["aggregates"]["averages"].get("comm_Pausing", 0)
    assert avg == expected_avg


@patch("backend.rag_graph.make_llm")
@patch("backend.rag_graph.build_retriever")
def test_rag_graph_integration(mock_build_retriever, mock_make_llm, temp_db_session, fake_redis_client, monkeypatch):
    # Mock external dependencies
    monkeypatch.setattr(rg, "SessionLocal", temp_db_session)
    monkeypatch.setattr(rg, "_redis_client", lambda: fake_redis_client)

    # Mock the retriever
    mock_retriever = MagicMock()
    mock_retriever.get_relevant_documents.return_value = [
        Document(page_content="Alice said hello.", metadata={"source_id": 1, "speaker": "Alice"})
    ]
    mock_build_retriever.return_value = mock_retriever

    # Mock the LLM
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = json.dumps({
        "answer": "This is a test answer.",
        "bullets": ["Test bullet 1"],
        "metrics_summary": [],
        "follow_ups": ["Test follow up"],
    })
    mock_make_llm.return_value = mock_llm

    # Initialize the graph
    graph = RAGGraph(vector_store=MagicMock())

    # Run the graph
    result = graph.run(question="Who said hello?", session_id="test_session")

    # Assertions
    assert result["answer"] == "This is a test answer."
    assert result["bullets"] == ["Test bullet 1"]
    assert len(result["citations"]) == 1
    assert result["citations"][0]["speaker"] == "Alice"
    assert result["metadata"]["analysis_type"] == "facts"

    # Verify that the LLM and retriever were called
    mock_retriever.get_relevant_documents.assert_called_once_with("Who said hello?")
    mock_llm.invoke.assert_called_once()


@pytest.mark.parametrize("question, expected_analysis_type", [
    ("How is Alice trending over time?", "performance_trend"),
    ("Compare Alice vs Bob", "compare_entities"),
    ("Why are scores so low?", "root_cause"),
])
@patch("backend.rag_graph.make_llm")
@patch("backend.rag_graph.build_retriever")
def test_rag_graph_analysis_paths(mock_build_retriever, mock_make_llm, temp_db_session, fake_redis_client, monkeypatch, question, expected_analysis_type):
    # Mock external dependencies
    monkeypatch.setattr(rg, "SessionLocal", temp_db_session)
    monkeypatch.setattr(rg, "_redis_client", lambda: fake_redis_client)

    # Mock the retriever
    mock_retriever = MagicMock()
    mock_retriever.get_relevant_documents.return_value = []
    mock_build_retriever.return_value = mock_retriever

    # Mock the LLM
    mock_llm = MagicMock()
    mock_llm.invoke.return_value.content = json.dumps({
        "answer": "Test answer", "bullets": [], "metrics_summary": [], "follow_ups": [],
    })
    mock_make_llm.return_value = mock_llm

    # Initialize and run the graph
    graph = RAGGraph(vector_store=MagicMock())
    result = graph.run(question=question, session_id="test_session")

    # Assert that the correct analysis path was taken
    assert result["metadata"]["analysis_type"] == expected_analysis_type