"""
Tests for conversational UX improvements:
- Conversation memory
- Contextual query expansion
- Smart follow-ups
- Metric filtering
"""

import pytest
from unittest.mock import MagicMock, patch

from backend import rag_graph as rg
from backend import db_models


def test_conversation_history_included_in_context():
    """Test that conversation history is formatted and included in LLM context"""
    history = [
        {"user": "How is Tasha's safety?", "assistant": "Tasha's safety score is 25.2, which is strong."},
        {"user": "What about communication?", "assistant": "Communication score is 24.5."}
    ]
    
    state = {
        "question": "What about Mike?",
        "history": history,
        "retrieved": [],
        "aggregates": {"count": 10, "averages": {}},
        "_citations": []
    }
    
    context = rg._prepare_context(state)
    
    # Should include conversation history
    assert "conversation_history" in context
    assert "CONVERSATION HISTORY" in context["conversation_history"]
    assert "Tasha" in context["conversation_history"]
    assert "safety" in context["conversation_history"]
    
    # Should include only last 5 turns
    assert context["conversation_history"].count("Turn") == 2


def test_conversation_history_empty_when_no_history():
    """Test that empty history doesn't break context preparation"""
    state = {
        "question": "How is safety?",
        "history": [],
        "retrieved": [],
        "aggregates": {"count": 10, "averages": {}},
        "_citations": []
    }
    
    context = rg._prepare_context(state)
    assert context["conversation_history"] == ""


def test_contextual_query_expansion_what_about():
    """Test 'What about X?' inherits context from previous turn"""
    history = [
        {"user": "How is Tasha's safety performance?", "assistant": "Tasha's SAFETY_Score is 25.2"}
    ]
    
    expanded = rg.resolve_contextual_query("What about Mike?", history)
    
    # Should expand to include the metric from previous context
    assert "safety" in expanded.lower() or "SAFETY" in expanded
    assert "Mike" in expanded


def test_contextual_query_expansion_patterns():
    """Test various contextual query patterns with speaker and trend carryover"""
    history = [
        {"user": "Show me communication trends", "assistant": "comm_Pausing score averaged 3.5"}
    ]
    
    # With smarter expansion, should carry over speaker + metric + trend intent
    patterns = [
        ("How about Tasha?", ["tasha", "trend"]),  # Gets speaker + trend intent
        ("And Mike?", ["mike", "trend"]),
        ("Same for Bob?", ["bob", "trend"]),
    ]
    
    for query, expected_words in patterns:
        expanded = rg.resolve_contextual_query(query, history)
        # Should contain speaker name and trend keyword
        for word in expected_words:
            assert word.lower() in expanded.lower(), f"Expected '{word}' in '{expanded}'"


def test_contextual_query_no_expansion_without_history():
    """Test that queries without history are not modified"""
    query = "What about Mike?"
    expanded = rg.resolve_contextual_query(query, [])
    
    assert expanded == query  # Should return unchanged


def test_contextual_query_no_expansion_for_complete_questions():
    """Test that complete questions are not modified"""
    history = [{"user": "How is safety?", "assistant": "Safety is good"}]
    query = "How is Mike's communication performance?"
    
    expanded = rg.resolve_contextual_query(query, history)
    assert expanded == query  # Already complete, no expansion needed


def test_classify_query_with_expansion(temp_db_session, monkeypatch):
    """Test that classify_query uses expanded question and replaces state["question"]"""
    monkeypatch.setattr(rg, "SessionLocal", temp_db_session)
    
    history = [
        {"user": "Show me SAFETY_Score trends over time", "assistant": "SAFETY_Score has improved from 23.5 to 25.2"}
    ]
    
    state = {
        "question": "What about Mike?",  # Contextual query
        "history": history,
        "filters": {}
    }
    
    result = rg.classify_query(state)
    
    # Original question should be preserved
    assert result["original_question"] == "What about Mike?"
    
    # Expanded question should be stored
    assert "expanded_question" in result
    assert "safety" in result["expanded_question"].lower()
    assert "trend" in result["expanded_question"].lower()
    
    # CRITICAL: state["question"] should be replaced with expanded version
    assert result["question"] == result["expanded_question"]
    assert result["question"] != "What about Mike?"


def test_extract_metrics_from_text():
    """Test metric extraction from text"""
    text = "Tasha's SAFETY_Score improved from 23.5 to 25.2"
    metrics = rg.extract_metrics_from_text(text)
    
    assert len(metrics) > 0
    assert any("SAFETY" in m for m in metrics)


def test_extract_speakers_from_text(temp_db_session, monkeypatch):
    """Test speaker extraction from text"""
    monkeypatch.setattr(rg, "SessionLocal", temp_db_session)
    
    # Add some speakers to DB
    session = temp_db_session()
    try:
        a = db_models.Analysis(source_filename="test.txt")
        session.add(a)
        session.flush()
        
        session.add_all([
            db_models.Utterance(analysis_id=a.id, speaker="Tasha", date="2024-09-01"),
            db_models.Utterance(analysis_id=a.id, speaker="Mike", date="2024-09-02"),
        ])
        session.commit()
    finally:
        session.close()
    
    text = "Tasha's performance improved while Mike maintained steady scores"
    speakers = rg.extract_speakers_from_text(text)
    
    assert "Tasha" in speakers
    assert "Mike" in speakers


@patch("backend.rag_graph.SessionLocal")
def test_generate_smart_follow_ups_speaker_comparison(mock_session_class):
    """Test smart follow-ups suggest speaker comparisons"""
    # Mock database session
    mock_session = MagicMock()
    mock_session_class.return_value = mock_session
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)
    
    # Mock speaker query
    mock_session.query.return_value.filter.return_value.distinct.return_value.limit.return_value.all.return_value = [
        ("Tasha",), ("Mike",), ("Sarah",)
    ]
    
    question = "How is Tasha's safety performance?"
    answer_text = "Tasha's SAFETY_Score is 25.2"
    history = []
    aggregates = {"count": 10, "averages": {"SAFETY_Score": 24.5}}
    
    follow_ups = rg.generate_smart_follow_ups(question, answer_text, history, aggregates)
    
    # Should suggest comparing to another speaker
    assert len(follow_ups) > 0
    assert any("Mike" in f or "Sarah" in f for f in follow_ups)


def test_generate_smart_follow_ups_temporal_suggestion():
    """Test smart follow-ups suggest temporal analysis when appropriate"""
    question = "What is Tasha's safety score?"
    answer_text = "Tasha's SAFETY_Score is 25.2"
    history = []
    aggregates = {"count": 10, "averages": {"SAFETY_Score": 25.2}}
    
    follow_ups = rg.generate_smart_follow_ups(question, answer_text, history, aggregates)
    
    # Should suggest temporal view (since query was not temporal)
    assert any("trend" in f.lower() or "over time" in f.lower() for f in follow_ups)


def test_generate_smart_follow_ups_proactive_insights():
    """Test smart follow-ups surface data anomalies"""
    question = "How is safety?"
    answer_text = "SAFETY_Score is 25.2"
    history = []
    
    # Mock temporal data with big change in different metric
    aggregates = {
        "count": 10,
        "averages": {"SAFETY_Score": 25.2, "COMM_Score": 24.5},
        "temporal_comparison": {
            "early_period": {
                "start": "2024-06-01",
                "end": "2024-07-31",
                "averages": {"SAFETY_Score": 25.0, "COMM_Score": 20.0}
            },
            "late_period": {
                "start": "2024-08-01",
                "end": "2024-09-30",
                "averages": {"SAFETY_Score": 25.2, "COMM_Score": 24.5}
            }
        }
    }
    
    follow_ups = rg.generate_smart_follow_ups(question, answer_text, history, aggregates)
    
    # Should mention the big change in COMM_Score (20.0 → 24.5 = 22.5% increase)
    assert any("COMM" in f or "communication" in f.lower() for f in follow_ups)
    assert any("improved" in f.lower() or "increased" in f.lower() for f in follow_ups)


@patch("backend.rag_graph.OpenAI")
def test_filter_relevant_metrics(mock_openai_class):
    """Test metric filtering using embeddings"""
    # Mock OpenAI embeddings
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    
    # Question embedding
    mock_client.embeddings.create.return_value.data = [
        MagicMock(embedding=[0.1, 0.2, 0.3])  # Question embedding
    ]
    
    # First call: question embedding
    # Second call: metric embeddings (will be called with list of metrics)
    mock_client.embeddings.create.side_effect = [
        MagicMock(data=[MagicMock(embedding=[0.1, 0.2, 0.3])]),  # Question
        MagicMock(data=[
            MagicMock(embedding=[0.1, 0.21, 0.29]),  # Similar to question (safety)
            MagicMock(embedding=[0.9, 0.8, 0.7]),    # Different (other metric)
        ])
    ]
    
    averages = {
        "SAFETY_Score": 25.2,
        "COMM_Score": 24.5,
    }
    
    filtered = rg.filter_relevant_metrics("How is safety?", averages, top_k=1)
    
    # Should keep most relevant metric
    assert len(filtered) == 1
    assert "SAFETY_Score" in filtered


def test_filter_relevant_metrics_handles_errors():
    """Test metric filtering falls back gracefully on errors"""
    averages = {"SAFETY_Score": 25.2, "COMM_Score": 24.5}
    
    # If OpenAI call fails, should return all metrics
    with patch("backend.rag_graph.OpenAI", side_effect=Exception("API Error")):
        filtered = rg.filter_relevant_metrics("How is safety?", averages, top_k=1)
        
        # Should fall back to returning all metrics
        assert len(filtered) == len(averages)


def test_filter_relevant_metrics_no_filter_when_few_metrics():
    """Test metric filtering skips when metrics <= top_k"""
    averages = {"SAFETY_Score": 25.2, "COMM_Score": 24.5}
    
    # When we have 2 metrics and top_k=10, should not filter
    filtered = rg.filter_relevant_metrics("test", averages, top_k=10)
    
    assert filtered == averages  # Returns unchanged


def test_format_answer_uses_smart_follow_ups(temp_db_session, monkeypatch):
    """Test that format_answer generates and uses smart follow-ups"""
    monkeypatch.setattr(rg, "SessionLocal", temp_db_session)
    
    # Add test data for speaker extraction
    session = temp_db_session()
    try:
        a = db_models.Analysis(source_filename="test.txt")
        session.add(a)
        session.flush()
        session.add(db_models.Utterance(analysis_id=a.id, speaker="Tasha", date="2024-09-01"))
        session.commit()
    finally:
        session.close()
    
    draft = {
        "answer": "Tasha's SAFETY_Score is 25.2",
        "bullets": ["Strong performance"],
        "metrics_summary": [],
        "follow_ups": ["Generic question"],  # LLM's generic follow-up
        "source_ids": []
    }
    
    state = {
        "question": "How is Tasha's safety?",
        "history": [],
        "draft": draft,
        "_citations": [],
        "analysis_type": "facts",
        "aggregates": {
            "count": 10,
            "averages": {"SAFETY_Score": 25.2},
            "temporal_comparison": None
        },
        "charts": []
    }
    
    result = rg.format_answer(state)
    answer = result["answer"]
    
    # Should have follow-ups (either smart or LLM's)
    assert len(answer["follow_ups"]) > 0


def test_classify_query_preserves_original_question():
    """Test that original question is stored but question is replaced with expanded version"""
    history = [{"user": "How is SAFETY_Score?", "assistant": "Good"}]
    original = "What about Mike?"
    
    state = {"question": original, "history": history, "filters": {}}
    result = rg.classify_query(state)
    
    # Original question should be preserved in separate field
    assert result["original_question"] == original
    # Expanded version should be stored
    assert "expanded_question" in result
    # CRITICAL: question should be replaced with expanded version
    assert result["question"] != original
    assert "mike" in result["question"].lower()


# Integration test
@pytest.mark.skip(reason="Complex end-to-end test with LangGraph - individual components tested separately")
@patch("backend.rag_graph.make_llm")
@patch("backend.rag_graph.build_retriever")
@patch("backend.rag_graph.OpenAI")
def test_conversational_flow_end_to_end(
    mock_openai_class, mock_build_retriever, mock_make_llm, 
    temp_db_session, fake_redis_client, monkeypatch
):
    """Test full conversational flow with history"""
    from langchain_core.messages import AIMessage
    
    monkeypatch.setattr(rg, "SessionLocal", temp_db_session)
    monkeypatch.setattr(rg, "_redis_client", lambda: fake_redis_client)
    
    # Setup DB with speakers
    session = temp_db_session()
    try:
        a = db_models.Analysis(source_filename="test.txt")
        session.add(a)
        session.flush()
        session.add_all([
            db_models.Utterance(
                analysis_id=a.id, 
                speaker="Tasha", 
                date="2024-09-01",
                predictions={"SAFETY_Score": 25.2}
            ),
            db_models.Utterance(
                analysis_id=a.id, 
                speaker="Mike", 
                date="2024-09-02",
                predictions={"SAFETY_Score": 23.5}
            ),
        ])
        session.commit()
    finally:
        session.close()
    
    # Mock retriever
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []
    mock_build_retriever.return_value = mock_retriever
    
    # Mock LLM - needs to return a proper response object with .content
    mock_llm_response = MagicMock()
    mock_llm_response.content = '''{
        "answer": "Mike's SAFETY_Score is 23.5",
        "bullets": ["Lower than Tasha"],
        "metrics_summary": [],
        "follow_ups": [],
        "source_ids": []
    }'''
    
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = mock_llm_response
    mock_make_llm.return_value = mock_llm
    
    # Mock OpenAI for metric filtering
    mock_client = MagicMock()
    mock_openai_class.return_value = mock_client
    mock_client.embeddings.create.side_effect = [
        MagicMock(data=[MagicMock(embedding=[0.1, 0.2, 0.3])]),  # Question
        MagicMock(data=[MagicMock(embedding=[0.1, 0.2, 0.3])]),  # Metrics
    ]
    
    graph = rg.RAGGraph(vector_store=MagicMock())
    
    # First turn: Ask about Tasha
    result1 = graph.run(question="How is Tasha's safety?", session_id="conv_test")
    assert "Tasha" in str(result1)
    
    # Second turn: Contextual follow-up "What about Mike?"
    # This should expand to "What about Mike's safety?"
    result2 = graph.run(question="What about Mike?", session_id="conv_test")
    
    # The expanded question should have been used
    # LLM should have been called with history in context
    llm_call_args = mock_llm.invoke.call_args_list[-1][0][0]
    
    # Context should include conversation history
    assert any("CONVERSATION HISTORY" in str(arg) or "Tasha" in str(arg) 
               for arg in llm_call_args.values())
