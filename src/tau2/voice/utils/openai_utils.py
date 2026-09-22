"""OpenAI audio utilities: Realtime API formats and text to speech.

This module provides conversion functions between tau2's AudioFormat
and OpenAI Realtime API format objects, and the user simulator's OpenAI TTS.

OpenAI GA API audio formats:
- audio/pcmu: G.711 μ-law (telephony)
- audio/pcma: G.711 A-law (telephony)
- audio/pcm: 24kHz, 16-bit signed PCM (requires rate: 24000)
"""

import os

import httpx
from loguru import logger

from tau2.config import DEFAULT_OPENAI_OUTPUT_SAMPLE_RATE
from tau2.data_model.audio import (
    TELEPHONY_SAMPLE_RATE,
    AudioData,
    AudioEncoding,
    AudioFormat,
)
from tau2.data_model.voice import OpenAITTSConfig
from tau2.voice.utils.audio_preprocessing import resample_audio
from tau2.voice.utils.text_effects import strip_audio_tags

OPENAI_PCM16_SAMPLE_RATE = DEFAULT_OPENAI_OUTPUT_SAMPLE_RATE

OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENAI_SPEECH_URL = "https://api.openai.com/v1/audio/speech"

# The whole clip is read before returning, so allow for a long turn.
TTS_TIMEOUT_SECONDS = 120.0

# The simulated caller dictates names, IDs and codes. Rushed, spelled letters
# are ambiguous to every listener (nur-tts-v1 said "M, E, I, P, A, T, E, L" in
# 1.5 s and no recognizer recovered it), and the run then scores the caller's
# voice instead of the agent. A real caller spells one character at a time.
DICTATION_INSTRUCTIONS = (
    "When spelling or dictating letters, digits or codes, say each character "
    "clearly and separately, one by one, at a natural pace."
)


def audio_format_to_openai(audio_format: AudioFormat) -> dict:
    """Convert AudioFormat to OpenAI GA API format object.

    Args:
        audio_format: The AudioFormat to convert.

    Returns:
        Dict with 'type' and optionally 'rate' for the GA API.

    Raises:
        ValueError: If the format is not supported by OpenAI Realtime API.
    """
    if audio_format.encoding == AudioEncoding.ULAW:
        if audio_format.sample_rate != TELEPHONY_SAMPLE_RATE:
            raise ValueError(
                f"OpenAI audio/pcmu requires {TELEPHONY_SAMPLE_RATE}Hz, "
                f"got {audio_format.sample_rate}Hz"
            )
        return {"type": "audio/pcmu"}
    elif audio_format.encoding == AudioEncoding.ALAW:
        if audio_format.sample_rate != TELEPHONY_SAMPLE_RATE:
            raise ValueError(
                f"OpenAI audio/pcma requires {TELEPHONY_SAMPLE_RATE}Hz, "
                f"got {audio_format.sample_rate}Hz"
            )
        return {"type": "audio/pcma"}
    elif audio_format.encoding == AudioEncoding.PCM_S16LE:
        if audio_format.sample_rate != OPENAI_PCM16_SAMPLE_RATE:
            raise ValueError(
                f"OpenAI audio/pcm requires {OPENAI_PCM16_SAMPLE_RATE}Hz, "
                f"got {audio_format.sample_rate}Hz"
            )
        return {"type": "audio/pcm", "rate": 24000}
    else:
        raise ValueError(
            f"Unsupported encoding for OpenAI: {audio_format.encoding}. "
            "Supported: ULAW (audio/pcmu), ALAW (audio/pcma), PCM_S16LE (audio/pcm)"
        )


def openai_format_to_audio_format(fmt: dict) -> AudioFormat:
    """Create AudioFormat from OpenAI GA API format object.

    Args:
        fmt: Dict with 'type' key (e.g. {"type": "audio/pcmu"}).

    Returns:
        AudioFormat configured for the specified format.

    Raises:
        ValueError: If the format type is not recognized.
    """
    fmt_type = fmt.get("type", "")
    if fmt_type == "audio/pcmu":
        return AudioFormat(
            encoding=AudioEncoding.ULAW, sample_rate=TELEPHONY_SAMPLE_RATE
        )
    elif fmt_type == "audio/pcma":
        return AudioFormat(
            encoding=AudioEncoding.ALAW, sample_rate=TELEPHONY_SAMPLE_RATE
        )
    elif fmt_type == "audio/pcm":
        return AudioFormat(
            encoding=AudioEncoding.PCM_S16LE, sample_rate=OPENAI_PCM16_SAMPLE_RATE
        )
    else:
        raise ValueError(
            f"Unknown OpenAI format: {fmt_type}. "
            "Supported: audio/pcmu, audio/pcma, audio/pcm"
        )


class OpenAIAPIError(Exception):
    """A non-2xx answer from the OpenAI API.

    Carries ``status_code`` so ``tts_retry`` retries 429 and 5xx, as it does
    for ElevenLabs.
    """

    def __init__(self, status_code: int, message: str):
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code


def tts_openai(text: str, config: OpenAITTSConfig) -> AudioData:
    """Text to speech with ``POST /v1/audio/speech``.

    The persona's ``instructions`` and the dictation rule go to the model as
    voice instructions. Returns mono PCM16 at
    ``config.output_audio_format.sample_rate``, the rate the synthesis
    pipeline expects (the API answers raw PCM at 24 kHz).
    """
    # gpt-4o-mini-tts has no audio tags.
    spoken = strip_audio_tags(text)
    api_key = config.api_key or os.environ.get(OPENAI_API_KEY_ENV)
    if not api_key:
        raise ValueError(f"OpenAI API key not provided. Set {OPENAI_API_KEY_ENV}.")
    if not config.voice_id:
        raise ValueError("OpenAI TTS needs a voice, e.g. 'cedar' or 'marin'.")

    instructions = " ".join(filter(None, [config.instructions, DICTATION_INSTRUCTIONS]))
    body = {
        "model": config.model_id,
        "voice": config.voice_id,
        "input": spoken,
        "instructions": instructions,
        "response_format": "pcm",
    }

    preview = spoken[:50] + "..." if len(spoken) > 50 else spoken
    logger.debug(
        f"OpenAI TTS: '{preview}' (voice={body['voice']}, model={body['model']})"
    )
    response = httpx.post(
        OPENAI_SPEECH_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json=body,
        timeout=TTS_TIMEOUT_SECONDS,
    )
    if not response.is_success:
        try:
            message = response.json()["error"]["message"]
        except (ValueError, KeyError, TypeError):
            message = response.text[:200]
        raise OpenAIAPIError(response.status_code, message)
    if not response.content:
        raise ValueError(f"OpenAI TTS returned empty audio for text: '{spoken}'")

    audio = AudioData(
        data=response.content,
        format=AudioFormat(
            encoding=AudioEncoding.PCM_S16LE, sample_rate=OPENAI_PCM16_SAMPLE_RATE
        ),
    )
    return resample_audio(audio, config.output_audio_format.sample_rate)
