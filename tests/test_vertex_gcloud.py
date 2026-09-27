"""Vertex request-shape checks for the voice simulator's Gemini fallback."""

from unittest.mock import Mock

from tau2.utils import vertex_gcloud


def test_gemini_38_continuation_uses_supported_request(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")
    monkeypatch.setattr(vertex_gcloud, "_access_token", lambda: "test-token")
    calls = []

    def post(url, *, json, headers, timeout):
        calls.append((url, json))
        return Mock(
            status_code=200,
            json=lambda: {
                "candidates": [
                    {"content": {"parts": [{"text": "Ready"}]}, "finishReason": "STOP"}
                ]
            },
        )

    monkeypatch.setattr(vertex_gcloud.httpx, "post", post)
    result = vertex_gcloud.completion_with_gcloud(
        model="gemini-3.8-flash",
        messages=[
            {"role": "system", "content": "You are the customer."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ],
        temperature=0.0,
    )

    url, body = calls[0]
    assert "/locations/global/" in url
    assert body["contents"][-1]["role"] == "user"
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "LOW"}
    assert "temperature" not in body["generationConfig"]
    assert result.choices[0].message.content == "Ready"
