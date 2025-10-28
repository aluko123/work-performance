"""
Unit tests for chart generation functionality
"""

import pytest
from backend.services import extract_metric_from_question, METRIC_SYNONYMS
from backend.rag_graph import (
    get_top_metrics_from_aggregates,
    generate_chart_specs,
    _build_chart_config,
    GraphState
)


class TestMetricExtraction:
    """Test metric extraction from user questions."""
    
    def test_extract_communication_clarity(self):
        """Should extract comm_clarity from various phrasings."""
        assert extract_metric_from_question("How is communication improving?") == "comm_clarity"
        assert extract_metric_from_question("Show me clarity scores") == "comm_clarity"
        assert extract_metric_from_question("What's the comm quality like?") == "comm_clarity"
    
    def test_extract_pausing(self):
        """Should extract comm_pausing from pausing-related questions."""
        assert extract_metric_from_question("How is pausing performance?") == "comm_pausing"
        assert extract_metric_from_question("Show me pacing trends") == "comm_pausing"
    
    def test_extract_situation_awareness(self):
        """Should extract sa_recognition from SA-related questions."""
        assert extract_metric_from_question("How is situation awareness?") == "sa_recognition"
        assert extract_metric_from_question("Show me SA trends") == "sa_recognition"
        assert extract_metric_from_question("Check awareness scores") == "sa_recognition"
    
    def test_no_metric_match(self):
        """Should return None when no metric matches."""
        assert extract_metric_from_question("Hello world") is None
        assert extract_metric_from_question("Random text") is None


class TestAggregateMetrics:
    """Test extraction of top metrics from aggregates."""
    
    def test_get_top_metrics_basic(self):
        """Should extract top N metrics by value."""
        aggregates = {
            "comm_clarity": 0.85,
            "comm_pausing": 0.72,
            "sa_recognition": 0.91,
            "comm_vocab_usage": 0.68,
            "count": 100  # Should be skipped
        }
        result = get_top_metrics_from_aggregates(aggregates, top_n=3)
        assert len(result) == 3
        assert "sa_recognition" in result  # Highest
        assert "count" not in result  # Should be excluded
    
    def test_get_top_metrics_with_zeros(self):
        """Should skip zero values."""
        aggregates = {
            "comm_clarity": 0.85,
            "comm_pausing": 0,  # Should be skipped
            "sa_recognition": 0.91,
        }
        result = get_top_metrics_from_aggregates(aggregates, top_n=5)
        assert "comm_pausing" not in result
        assert len(result) == 2
    
    def test_get_top_metrics_empty(self):
        """Should handle empty aggregates."""
        result = get_top_metrics_from_aggregates({}, top_n=5)
        assert result == []


class TestChartSpecGeneration:
    """Test chart specification generation logic."""
    
    def test_trend_analysis_generates_line_chart(self):
        """Should generate line chart for trend queries."""
        state: GraphState = {
            "question": "How did communication improve over time?",
            "analysis_type": "performance_trend",
            "filters": {},
            "aggregates": {},
            "session_id": None,
            "history": [],
            "retrieved": [],
            "draft": {},
            "answer": {},
            "_citations": [],
            "chart_specs": [],
            "charts": []
        }
        result = generate_chart_specs(state)
        assert len(result["chart_specs"]) == 1
        assert result["chart_specs"][0]["type"] == "line"
        assert result["chart_specs"][0]["metric"] == "comm_clarity"
    
    def test_comparison_generates_bar_chart(self):
        """Should generate bar chart for comparison queries."""
        state: GraphState = {
            "question": "Compare all speakers on communication",
            "analysis_type": "compare_entities",
            "filters": {},
            "aggregates": {},
            "session_id": None,
            "history": [],
            "retrieved": [],
            "draft": {},
            "answer": {},
            "_citations": [],
            "chart_specs": [],
            "charts": []
        }
        result = generate_chart_specs(state)
        assert len(result["chart_specs"]) == 1
        assert result["chart_specs"][0]["type"] == "bar"
        assert result["chart_specs"][0]["group_by"] == "speaker"
    
    def test_speaker_facts_generates_grouped_bar(self):
        """Should generate grouped bar for speaker-specific queries."""
        state: GraphState = {
            "question": "How is John performing?",
            "analysis_type": "facts",
            "filters": {"speaker": "John"},
            "aggregates": {
                "comm_clarity": 0.85,
                "comm_pausing": 0.72,
                "sa_recognition": 0.91
            },
            "session_id": None,
            "history": [],
            "retrieved": [],
            "draft": {},
            "answer": {},
            "_citations": [],
            "chart_specs": [],
            "charts": []
        }
        result = generate_chart_specs(state)
        assert len(result["chart_specs"]) == 1
        assert result["chart_specs"][0]["type"] == "grouped_bar"
        assert "metrics" in result["chart_specs"][0]
    
    def test_chart_feature_disabled(self):
        """Should return empty specs when charts are disabled."""
        import backend.rag_graph as rag_module
        original_enable = rag_module.ENABLE_CHARTS
        try:
            rag_module.ENABLE_CHARTS = False
            state: GraphState = {
                "question": "How did communication improve over time?",
                "analysis_type": "performance_trend",
                "filters": {},
                "aggregates": {},
                "session_id": None,
                "history": [],
                "retrieved": [],
                "draft": {},
                "answer": {},
                "_citations": [],
                "chart_specs": [],
                "charts": []
            }
            result = generate_chart_specs(state)
            assert result["chart_specs"] == []
        finally:
            rag_module.ENABLE_CHARTS = original_enable


class TestChartConfig:
    """Test chart configuration building."""
    
    def test_line_chart_config(self):
        """Should build proper config for line charts."""
        config = _build_chart_config("line", "comm_clarity", None, {})
        assert config["title"] == "Communication Clarity Over Time"
        assert config["xAxisLabel"] == "Date"
        assert config["yAxisLabel"] == "Score"
        assert "colors" in config
    
    def test_bar_chart_config(self):
        """Should build proper config for bar charts."""
        config = _build_chart_config("bar", "sa_recognition", None, {})
        assert "Speaker Comparison" in config["title"]
        assert config["xAxisLabel"] == "Speaker"
        assert config["yAxisLabel"] == "Average Score"
    
    def test_grouped_bar_config_with_speaker(self):
        """Should include speaker name in grouped bar title."""
        config = _build_chart_config("grouped_bar", None, ["comm_clarity"], {"speaker": "John"})
        assert "John" in config["title"]
        assert config["xAxisLabel"] == "Metric"
        assert config["yAxisLabel"] == "Score"
    
    def test_grouped_bar_config_without_speaker(self):
        """Should use default name when speaker is missing."""
        config = _build_chart_config("grouped_bar", None, ["comm_clarity"], {})
        assert "Team" in config["title"]
