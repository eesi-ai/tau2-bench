"""Core voice synthesis (TTS) functions."""

from dotenv import load_dotenv

from tau2.data_model.audio import AudioData
from tau2.data_model.voice import ProviderConfig
from tau2.utils.retry import tts_retry
from tau2.voice.utils.audio_preprocessing import match_speech_reference_level
from tau2.voice.utils.eesi_utils import tts_eesi
from tau2.voice.utils.elevenlabs_utils import tts_elevenlabs
from tau2.voice.utils.openai_utils import tts_openai

load_dotenv()


@tts_retry
def synthesize_voice(
    text: str,
    provider: str,
    provider_config: ProviderConfig,
) -> AudioData:
    """Synthesize voice from text using the specified configuration."""
    if provider == "elevenlabs":
        audio_data = tts_elevenlabs(text=text, config=provider_config)
    elif provider == "eesi":
        audio_data = match_speech_reference_level(
            tts_eesi(text=text, config=provider_config)
        )
    elif provider == "openai":
        # Noise is scaled against a fixed speech level the ElevenLabs voices
        # sit near; these voices range from 15 dB under it to about level.
        audio_data = match_speech_reference_level(
            tts_openai(text=text, config=provider_config)
        )
    else:
        raise ValueError(f"Unsupported synthesis provider: {provider}")

    if not audio_data.format.is_pcm16:
        raise ValueError(
            f"TTS must output PCM_S16LE, got {audio_data.format.encoding}. "
            "Configure the provider to use PCM output format."
        )

    return audio_data
