from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.main import app as real_app, ml_models


class FakeRAGGraph:
    def run(self, question, session_id=None, filters=None):
        return {
            "answer": "Test answer",
            "bullets": ["b1", "b2"],
            "metrics_summary": [{"k": 1}],
            "citations": [],
            "follow_ups": ["f1"],
            "metadata": {"analysis_type": "facts", "count": 0, "data_quality": "low"},
        }


def make_test_app():
    # Build a lightweight app that reuses the real routes but avoids running lifespan
    app = FastAPI()
    for r in real_app.router.routes:
        app.router.routes.append(r)
    return app


def test_rag_endpoint_offline(monkeypatch):
    ml_models["rag_graph"] = FakeRAGGraph()
    app = make_test_app()
    client = TestClient(app)

    resp = client.post("/api/get_insights", json={"question": "What improved?"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "Test answer"
    assert data["bullets"][0] == "b1"

