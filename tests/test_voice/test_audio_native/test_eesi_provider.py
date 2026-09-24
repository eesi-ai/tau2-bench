"""Offline tests for the EESI Nur Live provider and adapter.

The live contract (connect, replies, tool round trip, barge-in) is covered by
test_provider_suite.py with EESI_API_KEY set.
"""

import asyncio
import base64
import json
from unittest.mock import AsyncMock, Mock

import pytest
from websockets.protocol import State

from tau2.config import TELEPHONY_ULAW_SILENCE
from tau2.voice.audio_native.adapter import create_adapter
from tau2.voice.audio_native.eesi import provider as eesi_provider_module
from tau2.voice.audio_native.eesi.discrete_time_adapter import (
    DiscreteTimeEesiAdapter,
)
from tau2.voice.audio_native.eesi.events import (
    LateSpeechStartedEvent,
    parse_eesi_event,
)
from tau2.voice.audio_native.eesi.provider import EesiRealtimeProvider, EesiVADConfig
from tau2.voice.audio_native.openai.events import (
    AudioDeltaEvent,
    ResponseDoneEvent,
    SpeechStartedEvent,
)
from tau2.voice.audio_native.openai.provider import OpenAIVADConfig
from tau2.voice.audio_native.tick_result import TickResult
from tau2.voice.pricing import compute_record_cost


class FakeSocket:
    """Enough of a websockets client connection for the provider."""

    def __init__(self, frames):
        self.state = State.OPEN
        self._frames = [json.dumps(frame) for frame in frames]
        self.sent = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._frames:
            raise StopAsyncIteration
        return self._frames.pop(0)

    async def recv(self):
        return await self.__anext__()

    async def send(self, message):
        self.sent.append(json.loads(message))

    async def close(self):
        self.state = State.CLOSED


def connect_to(monkeypatch, frames):
    socket = FakeSocket(frames)
    calls = []

    async def fake_connect(url, additional_headers):
        calls.append((url, additional_headers))
        return socket

    monkeypatch.setattr(eesi_provider_module.websockets, "connect", fake_connect)
    return socket, calls


HANDSHAKE = [
    {"type": "eesi.synthetic_audio"},
    {"type": "eesi.recording_disclosure"},
    {"type": "eesi.session", "session_id": "gw-1"},
    {"type": "session.created", "session": {"id": "sess_1", "model": "nur-live-v1"}},
]


def test_provider_reads_eesi_environment(monkeypatch):
    monkeypatch.setenv("EESI_API_KEY", "eesi-key")
    monkeypatch.setenv("EESI_BASE_URL", "https://api.dev.eesi.ai/v1/")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    provider = EesiRealtimeProvider()

    assert provider.api_key == "eesi-key"
    assert provider.base_url == "wss://api.dev.eesi.ai/v1/realtime"
    assert provider.model == "nur-live-v1"


def test_provider_requires_a_key(monkeypatch):
    monkeypatch.delenv("EESI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="EESI_API_KEY"):
        EesiRealtimeProvider()


def test_connect_reads_past_gateway_frames(monkeypatch):
    socket, calls = connect_to(monkeypatch, HANDSHAKE)
    provider = EesiRealtimeProvider(api_key="k", base_url="https://api.eesi.ai/v1")

    asyncio.run(provider.connect())

    url, headers = calls[0]
    assert url == ("wss://api.eesi.ai/v1/realtime?model=nur-live-v1&source=tau2-bench")
    assert headers == {"Authorization": "Bearer k"}
    assert provider.session_id == "sess_1"
    assert provider.is_connected


def test_connect_names_the_session_by_the_gateway_id_when_the_server_gives_none(
    monkeypatch,
):
    """Nur's session.created carries no session id; the gateway's eesi.session
    does, and it is what the server's debug log names the call by."""
    handshake = [
        frame
        if frame.get("type") != "session.created"
        else {**frame, "session": {"type": "realtime"}}
        for frame in HANDSHAKE
    ]
    connect_to(monkeypatch, handshake)
    provider = EesiRealtimeProvider(api_key="k", base_url="https://api.eesi.ai/v1")

    asyncio.run(provider.connect())

    assert provider.session_id == "gw-1"


def test_connect_surfaces_a_refusal(monkeypatch):
    connect_to(
        monkeypatch,
        [
            {"type": "eesi.synthetic_audio"},
            {
                "type": "error",
                "error": {"code": "session_limit_reached", "message": "pool full"},
            },
        ],
    )
    provider = EesiRealtimeProvider(api_key="k")

    with pytest.raises(RuntimeError, match="session_limit_reached"):
        asyncio.run(provider.connect())


def test_session_keeps_telephony_audio_and_server_turn_detection(monkeypatch):
    socket, _ = connect_to(monkeypatch, [*HANDSHAKE, {"type": "session.updated"}])
    provider = EesiRealtimeProvider(api_key="k")

    async def configure():
        await provider.connect()
        await provider.configure_session(
            system_prompt="policy",
            tools=[],
            vad_config=EesiVADConfig(),
            modality="audio",
        )

    asyncio.run(configure())

    session = socket.sent[0]["session"]
    assert session["audio"]["input"]["format"] == {"type": "audio/pcmu"}
    assert session["audio"]["output"]["format"] == {"type": "audio/pcmu"}
    assert session["audio"]["input"]["turn_detection"] == {"type": "server_vad"}
    assert session["instructions"] == "policy"


def test_turn_detection_sends_only_set_overrides():
    provider = EesiRealtimeProvider(api_key="k")

    assert provider._build_turn_detection_config(
        EesiVADConfig(silence_duration_ms=300)
    ) == {"type": "server_vad", "silence_duration_ms": 300}
    with pytest.raises(TypeError, match="EesiVADConfig"):
        provider._build_turn_detection_config(OpenAIVADConfig())


@pytest.mark.parametrize("extra", [{"eesi_late": True}, {"interrupt_response": False}])
def test_late_speech_start_is_its_own_event(extra):
    raw = {"type": "input_audio_buffer.speech_started", "audio_start_ms": 1200}

    assert type(parse_eesi_event(raw)) is SpeechStartedEvent
    late = parse_eesi_event({**raw, **extra})
    assert isinstance(late, LateSpeechStartedEvent)
    assert late.audio_start_ms == 1200


def make_tick() -> TickResult:
    return TickResult(
        tick_number=1,
        audio_sent_bytes=1600,
        audio_sent_duration_ms=200,
        user_audio_data=TELEPHONY_ULAW_SILENCE * 1600,
        cumulative_user_audio_at_tick_start_ms=1000,
        bytes_per_tick=1600,
        bytes_per_second=8000,
        silence_byte=TELEPHONY_ULAW_SILENCE,
    )


def make_adapter():
    provider = Mock(model="nur-live-v1", truncate_item=AsyncMock())
    return DiscreteTimeEesiAdapter(tick_duration_ms=200, provider=provider), provider


def agent_audio(n: int) -> AudioDeltaEvent:
    return AudioDeltaEvent(
        type="response.output_audio.delta",
        delta=base64.b64encode(b"\x10" * n).decode(),
        item_id="item_1",
    )


def test_late_speech_start_keeps_the_reply_playing():
    adapter, provider = make_adapter()
    tick = make_tick()

    async def run():
        await adapter._process_event(tick, agent_audio(800))
        await adapter._process_event(
            tick,
            LateSpeechStartedEvent(
                type="input_audio_buffer.speech_started", audio_start_ms=1100
            ),
        )

    asyncio.run(run())

    assert tick.vad_events == ["speech_started"]
    assert tick.agent_audio_bytes == 800
    assert tick.skip_item_id is None
    provider.truncate_item.assert_not_awaited()


def test_barge_in_still_truncates_the_reply():
    adapter, provider = make_adapter()
    tick = make_tick()

    async def run():
        await adapter._process_event(tick, agent_audio(800))
        await adapter._process_event(
            tick,
            SpeechStartedEvent(
                type="input_audio_buffer.speech_started", audio_start_ms=1100
            ),
        )

    asyncio.run(run())

    assert tick.skip_item_id == "item_1"
    provider.truncate_item.assert_awaited_once()


def test_truncate_reports_the_whole_reply_played_not_one_tick():
    """Upstream sends one tick (200 ms) as audio_end_ms on every barge-in.
    Nur keeps only the heard part of a cut reply, so that would leave a word
    of a reply the caller heard for a second and a half."""
    adapter, provider = make_adapter()
    # Two full ticks of item_1 already played: 3200 bytes = 400 ms at 8 kHz.
    adapter._played_bytes["item_1"] = 3200
    tick = make_tick()

    async def run():
        await adapter._process_event(tick, agent_audio(1600))
        # Barge-in 100 ms into this tick: 800 more bytes played.
        await adapter._process_event(
            tick,
            SpeechStartedEvent(
                type="input_audio_buffer.speech_started", audio_start_ms=1100
            ),
        )

    asyncio.run(run())

    provider.truncate_item.assert_awaited_once()
    assert provider.truncate_item.await_args.kwargs["audio_end_ms"] == 500


def test_played_audio_is_counted_per_reply_across_ticks():
    tick = make_tick()
    tick.agent_audio_chunks = [(b"\x10" * 600, "item_1"), (b"\x10" * 1000, "item_2")]
    assert DiscreteTimeEesiAdapter._played_this_tick(tick) == [
        ("item_1", 600),
        ("item_2", 1000),
    ]

    tick.truncate_agent_audio("item_2", 1100, 1000, 1600)  # 100 ms = 800 bytes in
    assert DiscreteTimeEesiAdapter._played_this_tick(tick) == [
        ("item_1", 600),
        ("item_2", 200),
    ]


def test_usage_is_labelled_eesi_and_billed_per_minute():
    adapter, _ = make_adapter()
    tick = make_tick()

    asyncio.run(
        adapter._process_event(
            tick,
            ResponseDoneEvent(
                type="response.done",
                response_id="resp_1",
                usage={"input_tokens": 120, "output_tokens": 30},
            ),
        )
    )
    adapter._cumulative_user_audio_ms = 90_000
    tokens, meter = adapter.get_usage_records()

    assert (tokens.provider, tokens.billable) == ("eesi", False)
    assert tokens.input_tokens == 120
    assert meter.audio_input_seconds == 90
    assert compute_record_cost(meter) == pytest.approx(0.0015)


def test_factory_builds_the_eesi_adapter(monkeypatch):
    monkeypatch.setenv("EESI_API_KEY", "k")

    adapter, model = create_adapter("eesi", tick_duration_ms=200)

    assert isinstance(adapter, DiscreteTimeEesiAdapter)
    assert model == "nur-live-v1"
