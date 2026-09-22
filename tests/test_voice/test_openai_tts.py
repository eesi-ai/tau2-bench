"""Offline tests for OpenAI TTS and its wiring into the user simulator."""

import httpx
import numpy as np
import pytest

from tau2.data_model.simulation import AudioNativeConfig, VoiceRunConfig
from tau2.data_model.voice import EesiTTSConfig, OpenAITTSConfig, SynthesisConfig
from tau2.data_model.voice_personas import (
    ALL_PERSONA_NAMES,
    ALL_PERSONAS,
    _resolve_openai_voice,
    get_voice_id,
)
from tau2.runner.batch import make_voice_run_settings
from tau2.user_simulation_voice_presets import sample_voice_config
from tau2.utils.retry import _is_retryable_tts_error
from tau2.voice.synthesis.synthesize import synthesize_voice
from tau2.voice.utils import openai_utils
from tau2.voice.utils.openai_utils import (
    DICTATION_INSTRUCTIONS,
    OpenAIAPIError,
    tts_openai,
)

# Built-in voices by gender, so each persona keeps the gender of its
# ElevenLabs voice.
FEMALE_VOICES = {"coral", "marin", "nova", "sage", "shimmer"}
MALE_VOICES = {"ash", "ballad", "cedar", "echo", "onyx", "verse"}
MALE_PERSONAS = {"matt_delaney", "arjun_roy", "mamadou_diallo"}


def pcm_bytes(seconds: float, rate: int = 24000) -> bytes:
    """Raw PCM16 mono, as ``response_format: pcm`` answers."""
    samples = (np.sin(np.arange(int(seconds * rate)) / 10) * 8000).astype(np.int16)
    return samples.tobytes()


@pytest.fixture
def api(monkeypatch):
    """Fake POST /v1/audio/speech; records every request."""
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    calls = []
    replies = {"post": httpx.Response(200, content=pcm_bytes(0.5))}

    def fake_post(url, headers, json, timeout):
        calls.append((url, headers, json))
        return replies["post"]

    monkeypatch.setattr(openai_utils.httpx, "post", fake_post)
    return calls, replies


def test_tts_speaks_as_the_persona_and_returns_pipeline_pcm(api):
    calls, _ = api
    config = SynthesisConfig.for_provider("openai").persona_provider_config("arjun_roy")

    audio = tts_openai("Change my flight [pause] please.", config)

    url, headers, body = calls[0]
    assert url == "https://api.openai.com/v1/audio/speech"
    assert headers == {"Authorization": "Bearer k"}
    assert body == {
        "model": "gpt-4o-mini-tts",
        "voice": "ash",
        "input": "Change my flight ... please.",
        "instructions": (
            f"{ALL_PERSONAS['arjun_roy'].openai_instructions} {DICTATION_INSTRUCTIONS}"
        ),
        "response_format": "pcm",
    }
    assert audio.format.is_pcm16
    assert audio.format.sample_rate == 16000
    assert audio.duration == pytest.approx(0.5, abs=0.01)


def test_tts_without_a_persona_still_asks_for_clear_dictation(api):
    calls, _ = api

    tts_openai("E, H, G, L, P, three.", OpenAITTSConfig(voice_id="cedar"))

    assert calls[0][2]["instructions"] == DICTATION_INSTRUCTIONS


def test_tts_drops_audio_tags_and_refuses_a_tag_alone(api):
    calls, _ = api

    tts_openai("I [cough] need help.", OpenAITTSConfig(voice_id="coral"))

    assert calls[0][2]["input"] == "I  need help."
    with pytest.raises(ValueError, match="No speakable text"):
        tts_openai(".[cough][cough][cough]", OpenAITTSConfig(voice_id="coral"))
    assert len(calls) == 1


def test_tts_needs_a_key_and_a_voice(api, monkeypatch):
    calls, _ = api

    with pytest.raises(ValueError, match="needs a voice"):
        tts_openai("Hello.", OpenAITTSConfig())
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        tts_openai("Hello.", OpenAITTSConfig(voice_id="cedar"))
    assert calls == []


def test_tts_refuses_an_empty_answer(api):
    _, replies = api
    replies["post"] = httpx.Response(200, content=b"")

    with pytest.raises(ValueError, match="empty audio"):
        tts_openai("Hello.", OpenAITTSConfig(voice_id="cedar"))


@pytest.mark.parametrize("status,retried", [(429, True), (502, True), (400, False)])
def test_api_errors_carry_status_for_the_tts_retry(api, status, retried):
    _, replies = api
    replies["post"] = httpx.Response(
        status, json={"error": {"message": "nope", "type": "x"}}
    )

    with pytest.raises(OpenAIAPIError, match="nope") as caught:
        tts_openai("Hello.", OpenAITTSConfig(voice_id="cedar"))

    assert caught.value.status_code == status
    assert _is_retryable_tts_error(caught.value) is retried


def test_synthesize_voice_routes_openai(api):
    calls, _ = api

    audio = synthesize_voice(
        text="Hello.",
        provider="openai",
        provider_config=OpenAITTSConfig(voice_id="cedar"),
    )

    assert len(calls) == 1
    assert audio.format.sample_rate == 16000


def test_every_persona_has_an_openai_voice_of_its_gender():
    voices = [get_voice_id(name, "openai") for name in ALL_PERSONA_NAMES]
    for name, voice in zip(ALL_PERSONA_NAMES, voices):
        expected = MALE_VOICES if name in MALE_PERSONAS else FEMALE_VOICES
        assert voice in expected, name
        # Every persona names its accent; the accents are what "regular" tests.
        assert "accent" in ALL_PERSONAS[name].openai_instructions, name
    assert len(set(voices)) == len(voices)  # each persona sounds different


def test_openai_voice_env_override(monkeypatch):
    monkeypatch.setenv("TAU2_OPENAI_VOICE_MATT_DELANEY", "verse")

    assert _resolve_openai_voice("matt_delaney", "cedar") == "verse"
    assert _resolve_openai_voice("lisa_brenner", "marin") == "marin"


def test_persona_provider_config_leaves_the_shared_config_alone():
    openai = SynthesisConfig.for_provider("openai")
    eesi = SynthesisConfig.for_provider("eesi")

    wei = openai.persona_provider_config("wei_lin")
    matt = eesi.persona_provider_config("matt_delaney")

    assert isinstance(openai.provider_config, OpenAITTSConfig)
    assert (wei.voice_id, wei.instructions) == (
        "nova",
        ALL_PERSONAS["wei_lin"].openai_instructions,
    )
    assert openai.provider_config.voice_id is None
    assert openai.provider_config.instructions is None
    assert isinstance(matt, EesiTTSConfig)
    assert matt.voice_id == "Orion"


def test_voice_run_uses_openai_tts():
    config = VoiceRunConfig(
        domain="mock",
        audio_native_config=AudioNativeConfig(provider="eesi", model="nur-live-v1"),
        speech_complexity="regular",
        voice_synthesis_provider="openai",
    )

    user_voice_settings, _ = make_voice_run_settings(config)
    synthesis = user_voice_settings.synthesis_config
    environment = sample_voice_config(7, synthesis, "regular").to_speech_environment(
        7, provider=synthesis.provider
    )

    assert synthesis.provider == "openai"
    assert isinstance(synthesis.provider_config, OpenAITTSConfig)
    assert environment.voice_id == get_voice_id(environment.persona_name, "openai")
