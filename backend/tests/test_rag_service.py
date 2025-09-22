import os
from unittest.mock import MagicMock, patch
from backend import rag_service as rs
from backend.rag_service import PerformanceRAG


class Obj:
    pass


def make_utterance(i, speaker="Alice"):
    u = Obj()
    u.id = i
    u.speaker = speaker
    u.date = "2024-09-01"
    u.timestamp = "08:00:00"
    u.text = f"Hello {i}"
    u.predictions = {"comm_Pausing": 3}
    u.aggregated_scores = {"Total_Comm_Score": 12}
    return u


@patch("backend.rag_service.RAGGraph")
@patch("backend.rag_service.Chroma")
@patch("backend.rag_service.OpenAIEmbeddings")
def test_query_insights(MockEmbeddings, MockChroma, MockRAGGraph):
    # Arrange
    mock_graph_instance = MockRAGGraph.return_value
    mock_graph_instance.run.return_value = {"answer": "Hello from graph"}

    # Act
    pr = PerformanceRAG(persist_directory="./data/chroma_db_test_ignore")
    result = pr.query_insights("What improved?")

    # Assert
    assert result["answer"] == "Hello from graph"
    mock_graph_instance.run.assert_called_once_with(
        question="What improved?", session_id=None, filters={}
    )


@patch("backend.rag_service.Chroma")
@patch("backend.rag_service.OpenAIEmbeddings")
def test_index_utterances_batches(MockEmbeddings, MockChroma, monkeypatch):
    # Arrange
    mock_store = MockChroma.return_value
    monkeypatch.setenv("BATCH_SIZE", "2")
    
    # Act
    pr = PerformanceRAG(persist_directory="./data/chroma_db_test_ignore")
    utterances = [make_utterance(i) for i in range(5)]
    pr.index_utterances(utterances)

    # Assert
    assert mock_store.add_documents.call_count == 3  # 5 docs in batches of 2 => 3 calls
    assert mock_store.persist.call_count == 1