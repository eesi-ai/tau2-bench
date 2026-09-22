"""EESI Nur Live realtime provider.

Nur Live speaks the OpenAI Realtime GA protocol, so this is the OpenAI
provider with three differences: EESI's endpoint and key, a handshake that
opens with the gateway's own frames, and turn detection left to the server's
tuned defaults unless asked otherwise.
"""

import json
from typing import Any, Dict, Optional

import websockets
from loguru import logger
from pydantic import BaseModel

from tau2.config import DEFAULT_EESI_REALTIME_MODEL, EESI_REQUEST_SOURCE
from tau2.utils.retry import websocket_retry
from tau2.voice.audio_native.eesi.events import parse_eesi_event
from tau2.voice.audio_native.openai.provider import OpenAIRealtimeProvider
from tau2.voice.utils.eesi_utils import eesi_api_key, eesi_realtime_url


class EesiVADConfig(BaseModel):
    """Server VAD overrides for Nur Live. ``None`` keeps the server default.

    The server reads only ``threshold`` and ``silence_duration_ms`` here; its
    end-of-turn model and turn-reopen window apply either way.
    """

    threshold: Optional[float] = None
    silence_duration_ms: Optional[int] = None


class EesiRealtimeProvider(OpenAIRealtimeProvider):
    """OpenAI Realtime client for ``wss://…/v1/realtime`` on EESI.

    Reads ``EESI_API_KEY`` and ``EESI_BASE_URL`` (the HTTP base, e.g.
    ``https://api.dev.eesi.ai/v1``) unless given explicitly.
    """

    DEFAULT_MODEL = DEFAULT_EESI_REALTIME_MODEL
    parse_event = staticmethod(parse_eesi_event)

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        super().__init__(
            api_key=eesi_api_key(api_key),
            model=model or self.DEFAULT_MODEL,
            reasoning_effort=reasoning_effort,
        )
        self.base_url = eesi_realtime_url(base_url)

    @websocket_retry
    async def connect(self) -> None:
        """Open the socket and read past the gateway's frames to session.created.

        ``eesi.synthetic_audio``, ``eesi.recording_disclosure`` and
        ``eesi.session`` arrive before the model's ``session.created``.
        """
        if self.is_connected:
            return

        url = f"{self.base_url}?model={self.model}&source={EESI_REQUEST_SOURCE}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        self.ws = await websockets.connect(url, additional_headers=headers)

        async for raw in self.ws:
            data = json.loads(raw)
            event_type = data.get("type")
            if event_type == "session.created":
                break
            if event_type == "error":
                error = data.get("error", {})
                raise RuntimeError(
                    f"EESI refused the session: {error.get('code') or error.get('type')}"
                    f" — {error.get('message', 'no message')}"
                )
        else:
            raise RuntimeError("EESI closed the socket before session.created")

        self.session_id = data.get("session", {}).get("id")
        logger.info(f"EESI realtime: session created (session_id={self.session_id})")

    def _build_turn_detection_config(self, vad_config: Any) -> Dict:
        if not isinstance(vad_config, EesiVADConfig):
            raise TypeError(f"Expected EesiVADConfig, got {type(vad_config).__name__}")
        return {"type": "server_vad", **vad_config.model_dump(exclude_none=True)}
