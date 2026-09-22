"""Offline tests for EESI TTS and its wiring into the user simulator."""

import io
import wave

import httpx
import numpy as np
import pytest

from tau2.data_model.simulation import AudioNativeConfig, VoiceRunConfig
from tau2.data_model.voice import EesiTTSConfig, ElevenLabsTTSConfig, SynthesisConfig
from tau2.data_model.voice_personas import (
    ALL_PERSONA_NAMES,
    _resolve_eesi_voice,
    get_elevenlabs_voice_id,
    get_voice_id,
)
from tau2.runner.batch import make_voice_run_settings
from tau2.user_simulation_voice_presets import sample_voice_config
from tau2.utils.retry import _is_retryable_tts_error
from tau2.voice.utils import eesi_utils
from tau2.voice.utils.eesi_utils import EesiAPIError, eesi_realtime_url, tts_eesi

BASE = "https://api.dev.eesi.ai/v1"
VOICES = {
    "data": [
        {
            "voice_id": "ev_orion",
            "name": "Orion",
            "is_builtin": True,
            "status": "ready",
        },
        {
            "voice_id": "ev_clone",
            "name": "orion",
            "is_builtin": False,
            "status": "ready",
        },
        {"voice_id": "ev_nova", "name": "Nova", "is_builtin": True, "status": "ready"},
        {
            "voice_id": "ev_new",
            "name": "Draft",
            "is_builtin": False,
            "status": "processing",
        },
    ]
}


def wav_bytes(seconds: float, rate: int = 24000) -> bytes:
    samples = (np.sin(np.arange(int(seconds * rate)) / 10) * 8000).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(samples.tobytes())
    return buffer.getvalue()


@pytest.fixture
def api(monkeypatch):
    """Fake GET /voices and POST /audio/speech; records every request."""
    eesi_utils._voice_ids_by_name.cache_clear()
    monkeypatch.setenv("EESI_API_KEY", "k")
    monkeypatch.setenv("EESI_BASE_URL", BASE)
    calls = {"get": [], "post": []}
    replies = {"post": httpx.Response(200, content=wav_bytes(0.5))}

    def fake_get(url, headers, timeout):
        calls["get"].append(url)
        return httpx.Response(200, json=VOICES)

    def fake_post(url, headers, json, timeout):
        calls["post"].append((url, headers, json))
        return replies["post"]

    monkeypatch.setattr(eesi_utils.httpx, "get", fake_get)
    monkeypatch.setattr(eesi_utils.httpx, "post", fake_post)
    yield calls, replies
    eesi_utils._voice_ids_by_name.cache_clear()


def test_tts_names_a_builtin_and_returns_pipeline_pcm(api):
    calls, _ = api

    audio = tts_eesi(
        "Change my flight [pause] please.", EesiTTSConfig(voice_id="Orion")
    )
    tts_eesi("Thanks.", EesiTTSConfig(voice_id="orion"))

    url, headers, body = calls["post"][0]
    assert url == f"{BASE}/audio/speech"
    assert headers == {"Authorization": "Bearer k"}
    assert body == {
        "model": "nur-tts-v1",
        "voice": "ev_orion",
        "input": "Change my flight ... please.",
        "response_format": "wav",
        "source": "tau2-bench",
        "language": "en",
    }
    assert calls["post"][1][2]["voice"] == "ev_orion"  # the built-in beats a clone
    assert calls["get"] == [f"{BASE}/voices"]  # looked up once per process
    assert audio.format.is_pcm16
    assert audio.format.sample_rate == 16000
    assert audio.duration == pytest.approx(0.5, abs=0.01)


def test_tts_passes_voice_ids_through(api):
    calls, _ = api

    tts_eesi("Hello.", EesiTTSConfig(voice_id="ev_custom"))

    assert calls["post"][0][2]["voice"] == "ev_custom"
    assert calls["get"] == []


def test_tts_drops_audio_tags_and_refuses_a_tag_alone(api):
    calls, _ = api

    tts_eesi("I [cough] need help.", EesiTTSConfig(voice_id="Nova"))

    assert calls["post"][0][2]["input"] == "I  need help."
    with pytest.raises(ValueError, match="No speakable text"):
        tts_eesi(".[cough][cough][cough]", EesiTTSConfig(voice_id="Nova"))


def test_tts_asks_again_after_an_empty_clip(api, monkeypatch):
    calls, _ = api
    replies = [wav_bytes(0), wav_bytes(0.3)]

    def fake_post(url, headers, json, timeout):
        calls["post"].append(json)
        return httpx.Response(200, content=replies.pop(0))

    monkeypatch.setattr(eesi_utils.httpx, "post", fake_post)

    audio = tts_eesi("uh-huh", EesiTTSConfig(voice_id="ev_custom"))

    assert len(calls["post"]) == 2
    assert audio.duration == pytest.approx(0.3, abs=0.01)


def test_tts_gives_up_on_a_voice_that_stays_silent(api):
    calls, replies = api
    replies["post"] = httpx.Response(200, content=wav_bytes(0))

    with pytest.raises(ValueError, match="empty audio"):
        tts_eesi("mm-hmm", EesiTTSConfig(voice_id="ev_custom"))

    assert len(calls["post"]) == eesi_utils.EMPTY_AUDIO_ATTEMPTS


def test_tts_names_voices_it_cannot_find(api):
    with pytest.raises(ValueError, match="Available: \\['nova', 'orion'\\]"):
        tts_eesi("Hello.", EesiTTSConfig(voice_id="Draft"))


@pytest.mark.parametrize("status,retried", [(429, True), (502, True), (400, False)])
def test_api_errors_carry_status_for_the_tts_retry(api, status, retried):
    _, replies = api
    replies["post"] = httpx.Response(
        status, json={"error": {"message": "nope", "code": "x"}}
    )

    with pytest.raises(EesiAPIError, match="nope") as caught:
        tts_eesi("Hello.", EesiTTSConfig(voice_id="ev_custom"))

    assert caught.value.status_code == status
    assert _is_retryable_tts_error(caught.value) is retried


def test_realtime_url_follows_the_http_base():
    assert eesi_realtime_url(f"{BASE}/") == "wss://api.dev.eesi.ai/v1/realtime"
    assert eesi_realtime_url("http://localhost:8000/v1") == (
        "ws://localhost:8000/v1/realtime"
    )


def test_every_persona_has_an_eesi_voice():
    for name in ALL_PERSONA_NAMES:
        assert get_voice_id(name, "eesi")
        assert get_voice_id(name) == get_elevenlabs_voice_id(name)
    assert get_voice_id("matt_delaney", "eesi") == "Orion"
    with pytest.raises(KeyError):
        get_voice_id("nobody", "eesi")


def test_eesi_voice_env_override(monkeypatch):
    monkeypatch.setenv("TAU2_EESI_VOICE_MATT_DELANEY", "ev_accented")

    assert _resolve_eesi_voice("matt_delaney", "Orion") == "ev_accented"
    assert _resolve_eesi_voice("lisa_brenner", "Nova") == "Nova"


def test_synthesis_config_for_provider():
    eesi = SynthesisConfig.for_provider("eesi")
    assert eesi.provider == "eesi"
    assert isinstance(eesi.provider_config, EesiTTSConfig)
    assert isinstance(
        SynthesisConfig.for_provider("elevenlabs").provider_config,
        ElevenLabsTTSConfig,
    )
    with pytest.raises(ValueError):
        SynthesisConfig.for_provider("acme")


def test_voice_run_uses_the_chosen_tts_provider():
    config = VoiceRunConfig(
        domain="mock",
        audio_native_config=AudioNativeConfig(provider="eesi", model="nur-live-v1"),
        speech_complexity="control",
        voice_synthesis_provider="eesi",
    )

    user_voice_settings, _ = make_voice_run_settings(config)
    synthesis = user_voice_settings.synthesis_config
    environment = sample_voice_config(7, synthesis, "control").to_speech_environment(
        7, provider=synthesis.provider
    )

    assert synthesis.provider == "eesi"
    assert environment.voice_id == get_voice_id(environment.persona_name, "eesi")
