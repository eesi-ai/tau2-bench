# Copyright Sierra
"""Text manipulation utilities for speech effects.

These functions modify text before TTS synthesis.
"""

import random
import re

from loguru import logger

from tau2.data_model.audio_effects import UserSpeechInsert
from tau2.voice_config import MIN_WORDS_FOR_VOCAL_TICS

# TTS models without audio tags (EESI, OpenAI) read a pause as an ellipsis and
# would say a vocal tic as a word, so tics are dropped.
AUDIO_TAG_PATTERN = re.compile(r"\[(cough|sneeze|sniffle)\]", re.IGNORECASE)
PAUSE_TAG_PATTERN = re.compile(r"\[pause\]", re.IGNORECASE)


def strip_audio_tags(text: str) -> str:
    """Text for a TTS model that has no audio tags.

    ``[pause]`` becomes an ellipsis; ``[cough]``, ``[sneeze]`` and
    ``[sniffle]`` are dropped. Raises ``ValueError`` when nothing speakable is
    left: a vocal tic alone ("[cough]") has nothing for such a model to say.
    """
    spoken = AUDIO_TAG_PATTERN.sub("", PAUSE_TAG_PATTERN.sub("...", text)).strip()
    if not re.search(r"\w", spoken):
        raise ValueError(f"No speakable text in {text!r}")
    return spoken


def insert_speech_text(
    text: str,
    speech_insert: UserSpeechInsert,
    rng: random.Random,
    min_words: int = MIN_WORDS_FOR_VOCAL_TICS,
    in_turn: bool = True,
) -> str:
    """Insert speech text at a random position.

    For in-turn insertion, only vocal tics are appropriate. Non-directed phrases
    should only be used out-of-turn (pre-rendered audio during silence).
    """
    if in_turn and speech_insert.type == "non_directed_phrase":
        logger.warning(
            f"Non-directed phrase '{speech_insert.text}' used in-turn. "
            "Non-directed phrases should only be used out-of-turn."
        )

    words = text.split()
    if len(words) < min_words:
        return text

    insert_position = rng.randint(1, len(words) - 1)
    words.insert(insert_position, speech_insert.text)
    return " ".join(words)
