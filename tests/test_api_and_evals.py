from fastapi.testclient import TestClient

from ragdemo.api import create_app
from ragdemo.evals import load_golden, run_evals
from ragdemo.pipeline import build_pipeline


def test_health_reports_index_size():
    client = TestClient(create_app(build_pipeline()))

    body = client.get("/health").json()

    assert body["status"] == "ok" and body["chunks"] > 10


def test_ask_returns_answer_with_citations():
    client = TestClient(create_app(build_pipeline()))

    body = client.post("/ask", json={"question": "How many minutes does the password reset link stay valid?"}).json()

    assert "30 minutes" in body["answer"]
    assert body["citations"][0]["doc_id"] == "account-security"


def test_ask_validates_input():
    client = TestClient(create_app(build_pipeline()))

    assert client.post("/ask", json={"question": "hi"}).status_code == 422
    assert client.post("/ask", json={"question": "a valid question", "k": 99}).status_code == 422


def test_golden_set_baseline_meets_quality_floor():
    """A regression gate: the offline pipeline must keep retrieving and abstaining correctly."""
    report = run_evals(build_pipeline(), load_golden())

    assert report.hit_at_k == 1.0
    assert report.hit_at_1 >= 0.9
    assert report.abstain_rate == 1.0
    assert report.groundedness == 1.0
