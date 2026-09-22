"""Core voice synthesis (TTS) functions."""

from dotenv import load_dotenv

from tau2.data_model.audio import AudioData
from tau2.data_model.voice import ProviderConfig
from tau2.utils.retry import tts_retry
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
        audio_data = tts_eesi(text=text, config=provider_config)
    elif provider == "openai":
        audio_data = tts_openai(text=text, config=provider_config)
    else:
        raise ValueError(f"Unsupported synthesis provider: {provider}")

    if not audio_data.format.is_pcm16:
        raise ValueError(
            f"TTS must output PCM_S16LE, got {audio_data.format.encoding}. "
            "Configure the provider to use PCM output format."
        )

    return audio_data
