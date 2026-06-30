from fastapi.testclient import TestClient

from app.main import app


def test_chat_rejects_blank_message() -> None:
    payload = {
        "session_id": "test-session",
        "message": "   ",
        "user_id": "local-user",
    }

    with TestClient(app) as client:
        response = client.post("/api/chat", json=payload)

    assert response.status_code == 400
    assert response.json()["detail"] == "Message is required"
