"""EESI API helpers: credentials, endpoints, and text to speech.

EESI serves an OpenAI-compatible API. One key (``EESI_API_KEY``) and one HTTP
base (``EESI_BASE_URL``, default https://api.eesi.ai/v1) cover both the
realtime socket and ``POST /v1/audio/speech``.
"""

import io
import os
import re
import wave
from functools import lru_cache
from typing import Optional

import httpx
from loguru import logger

from tau2.config import DEFAULT_EESI_BASE_URL, EESI_REQUEST_SOURCE
from tau2.data_model.audio import AudioData, AudioEncoding, AudioFormat
from tau2.data_model.voice import EesiTTSConfig
from tau2.voice.utils.audio_preprocessing import resample_audio

EESI_API_KEY_ENV = "EESI_API_KEY"
EESI_BASE_URL_ENV = "EESI_BASE_URL"
EESI_VOICE_ID_PREFIX = "ev_"

# Synthesis finishes before the first byte arrives, so allow for a long turn.
TTS_TIMEOUT_SECONDS = 120.0

# nur-tts-v1 sometimes answers a very short input ("uh-huh", "mm-hmm") with a
# 200 and a clip of zero frames; asking again usually returns speech.
EMPTY_AUDIO_ATTEMPTS = 5

# nur-tts-v1 has no audio tags: a pause is read as an ellipsis, the rest dropped.
AUDIO_TAG_PATTERN = re.compile(r"\[(cough|sneeze|sniffle)\]", re.IGNORECASE)
PAUSE_TAG_PATTERN = re.compile(r"\[pause\]", re.IGNORECASE)


class EesiAPIError(Exception):
    """A non-2xx answer from the EESI API.

    Carries ``status_code`` so ``tts_retry`` retries 429 and 5xx, as it does
    for ElevenLabs.
    """

    def __init__(self, status_code: int, message: str):
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code


def eesi_api_key(api_key: Optional[str] = None) -> str:
    """The explicit key, else ``EESI_API_KEY``."""
    key = api_key or os.environ.get(EESI_API_KEY_ENV)
    if not key:
        raise ValueError(f"EESI API key not provided. Set {EESI_API_KEY_ENV}.")
    return key


def eesi_base_url(base_url: Optional[str] = None) -> str:
    """The HTTP API base, e.g. ``https://api.dev.eesi.ai/v1``."""
    base = base_url or os.environ.get(EESI_BASE_URL_ENV) or DEFAULT_EESI_BASE_URL
    return base.rstrip("/")


def eesi_realtime_url(base_url: Optional[str] = None) -> str:
    """The realtime socket for an HTTP base: ``https://…/v1`` → ``wss://…/v1/realtime``."""
    return re.sub(r"^http", "ws", eesi_base_url(base_url)) + "/realtime"


def _read_wav(content: bytes) -> tuple[int, bytes]:
    """Sample rate and PCM16 frames of a mono WAV answer."""
    with wave.open(io.BytesIO(content), "rb") as wav:
        if wav.getsampwidth() != 2 or wav.getnchannels() != 1:
            raise ValueError(
                f"EESI TTS returned {wav.getnchannels()}ch "
                f"{8 * wav.getsampwidth()}-bit audio; expected mono PCM16"
            )
        return wav.getframerate(), wav.readframes(wav.getnframes())


def _raise_for_status(response: httpx.Response) -> None:
    if response.is_success:
        return
    try:
        message = response.json()["error"]["message"]
    except (ValueError, KeyError, TypeError):
        message = response.text[:200]
    raise EesiAPIError(response.status_code, message)


@lru_cache(maxsize=None)
def _voice_ids_by_name(base_url: str, api_key: str) -> dict[str, str]:
    """Ready voices in the key's library, by lower-cased name.

    Built-in voice ids differ between deployments, so personas name built-ins
    and the id is looked up once per process. A built-in wins a name clash
    with a clone.
    """
    response = httpx.get(
        f"{base_url}/voices",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30.0,
    )
    _raise_for_status(response)
    voices = [v for v in response.json()["data"] if v.get("status") == "ready"]
    voices.sort(key=lambda v: bool(v.get("is_builtin")))
    return {v["name"].lower(): v["voice_id"] for v in voices}


def resolve_eesi_voice(voice: Optional[str], base_url: str, api_key: str) -> str:
    """An ``ev_…`` id as given, or the id of the ready voice with that name."""
    if not voice:
        raise ValueError("EESI TTS needs a voice: an ev_… id or a voice name.")
    if voice.startswith(EESI_VOICE_ID_PREFIX):
        return voice
    voices = _voice_ids_by_name(base_url, api_key)
    try:
        return voices[voice.lower()]
    except KeyError:
        raise ValueError(
            f"No ready EESI voice named {voice!r}. Available: {sorted(voices)}"
        ) from None


def tts_eesi(text: str, config: EesiTTSConfig) -> AudioData:
    """Text to speech with ``POST /v1/audio/speech``.

    Returns mono PCM16 at ``config.output_audio_format.sample_rate``, the rate
    the synthesis pipeline expects (the API answers at 24 kHz).
    """
    spoken = AUDIO_TAG_PATTERN.sub("", PAUSE_TAG_PATTERN.sub("...", text)).strip()
    if not re.search(r"\w", spoken):
        # A vocal tic alone ("[cough]") has nothing for this model to say.
        raise ValueError(f"No speakable text for EESI TTS in {text!r}")

    base_url = eesi_base_url(config.base_url)
    api_key = eesi_api_key(config.api_key)
    body = {
        "model": config.model_id,
        "voice": resolve_eesi_voice(config.voice_id, base_url, api_key),
        "input": spoken,
        "response_format": "wav",
        "source": EESI_REQUEST_SOURCE,
    }
    if config.language:
        body["language"] = config.language

    preview = spoken[:50] + "..." if len(spoken) > 50 else spoken
    logger.debug(
        f"EESI TTS: '{preview}' (voice={body['voice']}, model={body['model']})"
    )
    for attempt in range(1, EMPTY_AUDIO_ATTEMPTS + 1):
        response = httpx.post(
            f"{base_url}/audio/speech",
            headers={"Authorization": f"Bearer {api_key}"},
            json=body,
            timeout=TTS_TIMEOUT_SECONDS,
        )
        _raise_for_status(response)
        sample_rate, frames = _read_wav(response.content)
        if frames:
            break
        logger.warning(
            f"EESI TTS returned an empty clip for '{preview}' "
            f"(attempt {attempt}/{EMPTY_AUDIO_ATTEMPTS})"
        )
    else:
        raise ValueError(f"EESI TTS returned empty audio for text: '{spoken}'")

    audio = AudioData(
        data=frames,
        format=AudioFormat(encoding=AudioEncoding.PCM_S16LE, sample_rate=sample_rate),
    )
    return resample_audio(audio, config.output_audio_format.sample_rate)
