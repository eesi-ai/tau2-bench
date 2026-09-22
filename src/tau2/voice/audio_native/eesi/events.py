"""EESI additions to the OpenAI Realtime event models.

Nur Live's events are OpenAI Realtime GA events; the one difference the
adapter has to act on is a ``speech_started`` that is not a barge-in.
"""

from tau2.voice.audio_native.openai.events import (
    BaseRealtimeEvent,
    SpeechStartedEvent,
    parse_realtime_event,
)


class LateSpeechStartedEvent(SpeechStartedEvent):
    """``speech_started`` for speech that does not cancel the reply.

    EESI announces some speech after the fact — a short segment, or one held
    until the recognizer vouched for it — with ``eesi_late: true``, and marks
    speech that must not cancel the reply with ``interrupt_response: false``.
    Either way the server keeps the reply streaming, so the listener must keep
    playing it.
    """


def parse_eesi_event(raw_data: dict) -> BaseRealtimeEvent:
    """Parse a Nur Live frame, separating late speech starts from barge-ins."""
    event = parse_realtime_event(raw_data)
    if isinstance(event, SpeechStartedEvent) and (
        raw_data.get("eesi_late") or raw_data.get("interrupt_response") is False
    ):
        return LateSpeechStartedEvent.model_validate(event.model_dump())
    return event
