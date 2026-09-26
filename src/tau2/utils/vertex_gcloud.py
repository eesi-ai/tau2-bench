"""Text chat on Vertex using the active gcloud login instead of ADC.

The voice benchmark uses this for its simulated user and optional reviewer.
The Nur audio agent remains a separate provider. No credentials are saved or
logged: gcloud supplies a short-lived access token when needed.
"""

import os
import subprocess
import threading
import time
from typing import Any

import httpx
from litellm.main import ModelResponse

_token = ""
_token_acquired_at = 0.0
_token_lock = threading.Lock()


def _access_token() -> str:
    global _token, _token_acquired_at
    with _token_lock:
        if _token and time.monotonic() - _token_acquired_at < 2400:
            return _token
        _token = subprocess.check_output(
            ["gcloud", "auth", "print-access-token"], text=True
        ).strip()
        _token_acquired_at = time.monotonic()
        if not _token:
            raise RuntimeError("gcloud returned an empty Vertex access token")
        return _token


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "text":
                raise ValueError("vertex_gcloud accepts text message parts only")
            parts.append(part.get("text", ""))
        return "\n".join(parts)
    raise ValueError("vertex_gcloud accepts text messages only")


def completion_with_gcloud(
    *, model: str, messages: list[dict[str, Any]], **kwargs: Any
) -> ModelResponse:
    """Return a LiteLLM-shaped response for a Vertex text conversation."""
    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    system = []
    contents = []
    for message in messages:
        role = message["role"]
        text = _text(message.get("content"))
        if role in ("system", "developer"):
            system.append(text)
        elif role in ("user", "assistant"):
            contents.append(
                {
                    "role": "model" if role == "assistant" else "user",
                    "parts": [{"text": text}],
                }
            )
        else:
            raise ValueError(f"vertex_gcloud does not support {role} messages")
    if not contents:
        raise ValueError("vertex_gcloud requires a user or assistant message")

    generation = {
        "temperature": kwargs.get("temperature", 0.2),
        "maxOutputTokens": kwargs.get("max_tokens") or 4096,
    }
    response_format = kwargs.get("response_format") or {}
    if isinstance(response_format, dict) and response_format.get("type") in (
        "json_object",
        "json_schema",
    ):
        generation["responseMimeType"] = "application/json"
    if model.startswith("gemini-2.5-flash"):
        generation["thinkingConfig"] = {"thinkingBudget": 0}
    body: dict[str, Any] = {"contents": contents, "generationConfig": generation}
    if system:
        body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system)}]}

    host = (
        "aiplatform.googleapis.com"
        if location == "global"
        else f"{location}-aiplatform.googleapis.com"
    )
    url = (
        f"https://{host}/v1/projects/{project}/locations/{location}"
        f"/publishers/google/models/{model}:generateContent"
    )
    response = httpx.post(
        url,
        json=body,
        headers={"Authorization": f"Bearer {_access_token()}"},
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    candidate = data["candidates"][0]
    answer = "".join(
        part.get("text", "") for part in candidate["content"].get("parts", [])
    )
    usage = data.get("usageMetadata", {})
    prompt_tokens = usage.get("promptTokenCount", 0)
    completion_tokens = usage.get("candidatesTokenCount", 0)
    return ModelResponse(
        model=f"vertex_ai/{model}",
        choices=[
            {
                "index": 0,
                "finish_reason": "length"
                if candidate.get("finishReason") == "MAX_TOKENS"
                else "stop",
                "message": {"role": "assistant", "content": answer},
            }
        ],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": usage.get(
                "totalTokenCount", prompt_tokens + completion_tokens
            ),
        },
    )
