import json
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.main import app as real_app


def make_test_app():
    """Build a lightweight app that reuses the real routes but avoids running lifespan."""
    app = FastAPI()
    for r in real_app.router.routes:
        app.router.routes.append(r)
    return app


@pytest.mark.asyncio
async def test_chat_endpoint_offline(monkeypatch):
    """Validate /api/chat SSE shape by mocking the agent run loop."""
    # Mock the agent.run_agent to yield token chunks then a final message
    async def fake_run_agent(question: str, session_id: str):
        yield {"type": "token", "content": "Hello "}
        yield {"type": "token", "content": "world"}
        yield {"type": "final", "answer": "- bullet one\nAnswer body", "tool_calls_made": 0}

    # Patch where the endpoint imports from
    import backend.agent as agent_mod
    monkeypatch.setattr(agent_mod, "run_agent", fake_run_agent)

    app = make_test_app()
    client = TestClient(app)

    resp = client.post("/api/chat", json={"question": "What improved?", "session_id": "t1"})
    assert resp.status_code == 200

    # Parse SSE stream
    lines = [l for l in resp.text.split("\n") if l.startswith("data: ")]
    tokens = []
    final_payload = None
    for line in lines:
        payload = json.loads(line[len("data: "):])
        if "answer_token" in payload:
            tokens.append(payload["answer_token"])
        # The endpoint sends multiple structured chunks; the final one has "answer"
        if "answer" in payload:
            final_payload = payload

    assert "".join(tokens) == "Hello world"
    assert final_payload is not None
    # Bullets are extracted from the answer by the endpoint
    assert "bullets" in final_payload
    assert any("bullet one" in b for b in final_payload["bullets"]) 
