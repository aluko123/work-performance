"""
Comprehensive tests for the new OpenAI-native agent system.
Tests tools, agent loop, conversation memory, and end-to-end flows.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import json

from backend import tools, agent
from backend import db_models


class TestTools:
    """Test individual tool functions"""
    
    def test_get_metric_stats_basic(self, temp_db_session, monkeypatch):
        """Test get_metric_stats returns correct statistics"""
        monkeypatch.setattr(tools, "SessionLocal", temp_db_session)
        
        # Add test data
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
                    aggregated_scores={"SAFETY_Score": 25.0}
                ),
                db_models.Utterance(
                    analysis_id=a.id,
                    speaker="Tasha",
                    date="2024-09-02",
                    aggregated_scores={"SAFETY_Score": 27.0}
                ),
            ])
            session.commit()
        finally:
            session.close()
        
        # Test tool
        result = tools.get_metric_stats("SAFETY_Score", speaker="Tasha")
        
        assert "error" not in result
        assert result["metric"] == "SAFETY_Score"
        assert result["average"] == 26.0
        assert result["count"] == 2
        assert result["min"] == 25.0
        assert result["max"] == 27.0
    
    def test_get_metric_stats_with_date_filter(self, temp_db_session, monkeypatch):
        """Test get_metric_stats with date filtering"""
        monkeypatch.setattr(tools, "SessionLocal", temp_db_session)
        
        session = temp_db_session()
        try:
            a = db_models.Analysis(source_filename="test.txt")
            session.add(a)
            session.flush()
            
            session.add_all([
                db_models.Utterance(
                    analysis_id=a.id,
                    speaker="Tasha",
                    date="2024-06-15",
                    aggregated_scores={"SAFETY_Score": 20.0}
                ),
                db_models.Utterance(
                    analysis_id=a.id,
                    speaker="Tasha",
                    date="2024-09-15",
                    aggregated_scores={"SAFETY_Score": 30.0}
                ),
            ])
            session.commit()
        finally:
            session.close()
        
        # Filter to only September
        result = tools.get_metric_stats("SAFETY_Score", speaker="Tasha", date_from="2024-09-01")
        
        assert result["average"] == 30.0
        assert result["count"] == 1
    
    def test_compare_periods(self, temp_db_session, monkeypatch):
        """Test compare_periods calculates change correctly"""
        monkeypatch.setattr(tools, "SessionLocal", temp_db_session)
        
        session = temp_db_session()
        try:
            a = db_models.Analysis(source_filename="test.txt")
            session.add(a)
            session.flush()
            
            # Early period: avg = 20.0
            session.add(db_models.Utterance(
                analysis_id=a.id,
                speaker="Tasha",
                date="2024-06-15",
                aggregated_scores={"SAFETY_Score": 20.0}
            ))
            
            # Late period: avg = 25.0
            session.add(db_models.Utterance(
                analysis_id=a.id,
                speaker="Tasha",
                date="2024-09-15",
                aggregated_scores={"SAFETY_Score": 25.0}
            ))
            session.commit()
        finally:
            session.close()
        
        result = tools.compare_periods(
            metric="SAFETY_Score",
            early_start="2024-06-01",
            early_end="2024-06-30",
            late_start="2024-09-01",
            late_end="2024-09-30"
        )
        
        assert "error" not in result
        assert result["early_period"]["average"] == 20.0
        assert result["late_period"]["average"] == 25.0
        assert result["change"]["absolute"] == 5.0
        assert result["change"]["percent"] == 25.0
        assert result["change"]["direction"] == "increase"
    
    def test_search_utterances_no_chroma(self):
        """Test search_utterances handles missing ChromaDB gracefully"""
        # Without ChromaDB collection, should return error
        result = tools.search_utterances("safety discussions")
        
        assert isinstance(result, list)
        if result and "error" in result[0]:
            assert "does not exist" in result[0]["error"].lower() or "failed" in result[0]["error"].lower()


class TestAgentConversation:
    """Test agent conversation handling"""
    
    @pytest.mark.asyncio
    async def test_load_history_creates_system_prompt(self):
        """Test that load_history creates system prompt with metadata"""
        with patch('backend.agent.redis.from_url') as mock_redis:
            mock_client = MagicMock()
            mock_client.get.return_value = None
            mock_redis.return_value = mock_client
            
            with patch('backend.agent.get_corpus_metadata') as mock_meta:
                mock_meta.return_value = {
                    "date_range": {"min": "2024-06-10", "max": "2024-09-30"},
                    "speakers": ["Tasha", "Mike"],
                    "total_utterances": 2000
                }
                
                messages = await agent.load_history("test-session")
                
                # Should have system prompt
                assert len(messages) == 1
                assert messages[0]["role"] == "system"
                # Should include metadata
                assert "2024-06-10" in messages[0]["content"]
                assert "2024-09-30" in messages[0]["content"]
                assert "Tasha" in messages[0]["content"]
    
    @pytest.mark.asyncio
    async def test_load_history_with_existing_conversation(self):
        """Test loading existing conversation preserves user/assistant messages"""
        with patch('backend.agent.redis.from_url') as mock_redis:
            mock_client = MagicMock()
            existing_history = [
                {"role": "system", "content": "old system prompt"},
                {"role": "user", "content": "How is safety?"},
                {"role": "assistant", "content": "Safety is good"}
            ]
            mock_client.get.return_value = json.dumps(existing_history)
            mock_redis.return_value = mock_client
            
            with patch('backend.agent.get_corpus_metadata') as mock_meta:
                mock_meta.return_value = {
                    "date_range": {"min": "2024-06-10", "max": "2024-09-30"},
                    "speakers": ["Tasha"],
                    "total_utterances": 100
                }
                
                messages = await agent.load_history("test-session")
                
                # Should have 3 messages
                assert len(messages) == 3
                # System prompt should be updated
                assert messages[0]["role"] == "system"
                assert "2024-06-10" in messages[0]["content"]
                # User/assistant preserved
                assert messages[1]["role"] == "user"
                assert messages[1]["content"] == "How is safety?"
                assert messages[2]["role"] == "assistant"
                assert messages[2]["content"] == "Safety is good"
    
    @pytest.mark.asyncio
    async def test_save_history_removes_tool_messages(self):
        """Test that save_history only keeps user/assistant, removes tool calls"""
        with patch('backend.agent.redis.from_url') as mock_redis:
            mock_client = MagicMock()
            mock_redis.return_value = mock_client
            
            messages = [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "test question"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "123", "function": {"name": "get_metric_stats"}}]
                },
                {"role": "tool", "tool_call_id": "123", "content": "tool result"},
                {"role": "assistant", "content": "final answer"}
            ]
            
            await agent.save_history("test-session", messages)
            
            # Check what was saved
            saved_data = mock_client.set.call_args[0][1]
            saved_messages = json.loads(saved_data)
            
            # Should have system, user, assistant (tool execution), assistant (final)
            assert len(saved_messages) == 4
            assert saved_messages[0]["role"] == "system"
            assert saved_messages[1]["role"] == "user"
            assert saved_messages[2]["role"] == "assistant"
            assert "tool_calls" not in saved_messages[2]  # Tool calls removed
            assert saved_messages[3]["role"] == "assistant"
            assert saved_messages[3]["content"] == "final answer"
            
            # Tool message should be gone
            roles = [m["role"] for m in saved_messages]
            assert "tool" not in roles


class TestAgentIntegration:
    """Integration tests for full agent workflows"""
    
    @pytest.mark.asyncio
    @patch('backend.agent.client.chat.completions.create')
    @patch('backend.agent.redis.from_url')
    @patch('backend.agent.get_corpus_metadata')
    async def test_simple_query_with_tool(self, mock_meta, mock_redis, mock_create):
        """Test agent handles simple query with one tool call"""
        # Setup mocks
        mock_meta.return_value = {
            "date_range": {"min": "2024-06-10", "max": "2024-09-30"},
            "speakers": ["Tasha"],
            "total_utterances": 100
        }
        
        redis_client = MagicMock()
        redis_client.get.return_value = None
        mock_redis.return_value = redis_client
        
        # Mock OpenAI response - tool call then answer
        async def mock_stream_1():
            # First call: returns tool call
            yield MagicMock(choices=[MagicMock(delta=MagicMock(
                content=None,
                tool_calls=[MagicMock(
                    index=0,
                    id="call_123",
                    function=MagicMock(name="get_metric_stats", arguments='{"metric": "SAFETY_Score"}')
                )]
            ))])
        
        async def mock_stream_2():
            # Second call: returns answer
            for token in ["Safety", " is", " good"]:
                yield MagicMock(choices=[MagicMock(delta=MagicMock(content=token, tool_calls=None))])
        
        mock_create.side_effect = [
            mock_stream_1(),
            mock_stream_2()
        ]
        
        # Mock tool execution
        with patch('backend.agent.TOOL_FUNCTIONS', {
            "get_metric_stats": lambda **kwargs: {"average": 25.0}
        }):
            # Run agent
            chunks = []
            async for chunk in agent.run_agent("How is safety?", "test-session"):
                chunks.append(chunk)
            
            # Should have tool execution and final answer
            assert any(c.get("type") == "status" for c in chunks)
            assert any(c.get("type") == "token" for c in chunks)
            assert any(c.get("type") == "final" for c in chunks)
            
            # Final answer should exist
            final = [c for c in chunks if c.get("type") == "final"][0]
            assert "safety" in final["answer"].lower() or "good" in final["answer"].lower()


class TestMetadata:
    """Test corpus metadata caching"""
    
    def test_get_corpus_metadata(self, temp_db_session, monkeypatch):
        """Test metadata computation from database"""
        from backend.metadata import get_corpus_metadata
        
        monkeypatch.setattr("backend.metadata.SessionLocal", temp_db_session)
        
        # Add test data
        session = temp_db_session()
        try:
            a = db_models.Analysis(source_filename="test.txt")
            session.add(a)
            session.flush()
            
            session.add_all([
                db_models.Utterance(
                    analysis_id=a.id,
                    speaker="Tasha",
                    date="2024-06-10"
                ),
                db_models.Utterance(
                    analysis_id=a.id,
                    speaker="Mike",
                    date="2024-09-30"
                ),
            ])
            session.commit()
        finally:
            session.close()
        
        # Mock Redis
        with patch('backend.metadata.redis.from_url') as mock_redis:
            mock_client = MagicMock()
            mock_client.get.return_value = None  # Force recompute
            mock_redis.return_value = mock_client
            
            meta = get_corpus_metadata(force_refresh=True)
            
            # Should have computed metadata
            assert meta["date_range"]["min"] == "2024-06-10"
            assert meta["date_range"]["max"] == "2024-09-30"
            assert "Tasha" in meta["speakers"]
            assert "Mike" in meta["speakers"]
            assert meta["total_utterances"] == 2
            
            # Should have cached it
            assert mock_client.setex.called


class TestToolDefinitions:
    """Test OpenAI tool definitions are valid"""
    
    def test_tool_definitions_structure(self):
        """Test that tool definitions follow OpenAI schema"""
        for tool in tools.TOOL_DEFINITIONS:
            assert tool["type"] == "function"
            assert "function" in tool
            assert "name" in tool["function"]
            assert "description" in tool["function"]
            assert "parameters" in tool["function"]
            
            params = tool["function"]["parameters"]
            assert params["type"] == "object"
            assert "properties" in params
    
    def test_all_tools_have_implementations(self):
        """Test that all defined tools have corresponding functions"""
        for tool in tools.TOOL_DEFINITIONS:
            func_name = tool["function"]["name"]
            assert func_name in tools.TOOL_FUNCTIONS
            assert callable(tools.TOOL_FUNCTIONS[func_name])


class TestConversationalScenarios:
    """End-to-end conversation scenarios"""
    
    @pytest.mark.asyncio
    async def test_conversation_memory_persists(self):
        """Test that conversations are saved and loaded correctly"""
        with patch('backend.agent.redis.from_url') as mock_redis:
            mock_client = MagicMock()
            stored_history = None
            
            def mock_get(key):
                return stored_history
            
            def mock_set(key, value):
                nonlocal stored_history
                stored_history = value
            
            mock_client.get = mock_get
            mock_client.set = mock_set
            mock_redis.return_value = mock_client
            
            with patch('backend.agent.get_corpus_metadata') as mock_meta:
                mock_meta.return_value = {
                    "date_range": {"min": "2024-06-10", "max": "2024-09-30"},
                    "speakers": ["Tasha"],
                    "total_utterances": 100
                }
                
                # First query
                messages1 = await agent.load_history("session-1")
                messages1.append({"role": "user", "content": "How is Tasha?"})
                messages1.append({"role": "assistant", "content": "Tasha is good"})
                await agent.save_history("session-1", messages1)
                
                # Second query - should load previous
                messages2 = await agent.load_history("session-1")
                
                # Should have system + previous turn
                assert len(messages2) == 3
                assert messages2[0]["role"] == "system"
                assert messages2[1]["role"] == "user"
                assert messages2[1]["content"] == "How is Tasha?"
                assert messages2[2]["role"] == "assistant"
                assert messages2[2]["content"] == "Tasha is good"
    
    @pytest.mark.asyncio
    async def test_context_maintained_across_turns(self):
        """Test that OpenAI maintains context when history is provided"""
        # This is more of a sanity check that our message structure is correct
        with patch('backend.agent.redis.from_url') as mock_redis:
            mock_client = MagicMock()
            mock_client.get.return_value = json.dumps([
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "How is Tasha safety?"},
                {"role": "assistant", "content": "Tasha's safety is 29.64"}
            ])
            mock_redis.return_value = mock_client
            
            with patch('backend.agent.get_corpus_metadata') as mock_meta:
                mock_meta.return_value = {
                    "date_range": {"min": "2024-06-10", "max": "2024-09-30"},
                    "speakers": ["Tasha", "Mike"],
                    "total_utterances": 100
                }
                
                messages = await agent.load_history("session-1")
                
                # Add new question
                messages.append({"role": "user", "content": "what about Mike?"})
                
                # Verify conversation structure is valid for OpenAI
                assert messages[0]["role"] == "system"
                assert messages[-2]["role"] == "assistant"  # Last assistant message
                assert messages[-1]["role"] == "user"  # Current question
                
                # No empty tool_calls
                for msg in messages:
                    if "tool_calls" in msg:
                        assert msg["tool_calls"], "tool_calls should not be empty array"


# Smoke tests for quick validation
class TestSmoke:
    """Quick smoke tests"""
    
    def test_tool_functions_dict_complete(self):
        """Ensure TOOL_FUNCTIONS has all defined tools"""
        defined_names = {t["function"]["name"] for t in tools.TOOL_DEFINITIONS}
        implemented_names = set(tools.TOOL_FUNCTIONS.keys())
        
        assert defined_names == implemented_names
    
    def test_system_prompt_builder_doesnt_crash(self):
        """Test that build_system_prompt doesn't crash with minimal data"""
        with patch('backend.agent.get_corpus_metadata') as mock_meta:
            mock_meta.return_value = {
                "date_range": {"min": None, "max": None},
                "speakers": [],
                "total_utterances": 0
            }
            
            prompt = agent.build_system_prompt()
            
            assert isinstance(prompt, str)
            assert len(prompt) > 100
            assert "SAFETY_Score" in prompt
