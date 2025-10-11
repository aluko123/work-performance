import json
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.main import app as real_app


class FakeRAGGraph:
    async def astream_run(self, question, session_id=None, filters=None):
        final_data = {
            "answer": "Test answer",
            "bullets": ["b1", "b2"],
            "metrics_summary": [{"k": 1}],
            "citations": [],
            "follow_ups": ["f1"],
            "metadata": {"analysis_type": "facts", "count": 0, "data_quality": "low"},
        }
        for token in "Test answer".split():
            yield {"answer_token": token}
        yield final_data


def make_test_app():
    # Build a lightweight app that reuses the real routes but avoids running lifespan
    app = FastAPI()
    for r in real_app.router.routes:
        app.router.routes.append(r)
    return app


def test_rag_endpoint_offline():
    app = make_test_app()
    app.state.rag_graph = FakeRAGGraph()
    client = TestClient(app)

    resp = client.post("/api/get_insights", json={"question": "What improved?"})
    assert resp.status_code == 200
    
    lines = resp.text.split('\n\n')
    data_lines = [line.replace('data: ', '') for line in lines if line.startswith('data: ')]
    
    tokens = []
    final_data = None
    for line in data_lines:
        if not line:
            continue
        payload = json.loads(line)
        if 'answer_token' in payload:
            tokens.append(payload['answer_token'])
        else:
            final_data = payload

    assert "".join(tokens) == "Testanswer"
    assert final_data is not None
    assert final_data["bullets"][0] == "b1"