"""Discrete-time adapter for EESI Nur Live.

Nur Live speaks the OpenAI Realtime GA protocol and takes telephony audio
(G.711 μ-law, 8 kHz) natively, so this is the OpenAI adapter with EESI's
provider and two EESI-specific rules:

- A late ``speech_started`` (see ``LateSpeechStartedEvent``) is recorded but
  does not truncate the agent's audio: the server keeps that reply playing.
- EESI bills per connected minute, so token counts are informational and a
  session-audio meter carries the cost, as for xAI.
- A truncate reports how much of the reply was played across every tick, not
  one tick's worth: Nur keeps only the heard part of an interrupted reply in
  its history, so a flat 200 ms would cut every interrupted reply to a word.
"""

from typing import Any, List, Optional

from tau2.data_model.usage import UsageRecord
from tau2.environment.tool import Tool
from tau2.voice.audio_native.eesi.events import LateSpeechStartedEvent
from tau2.voice.audio_native.eesi.provider import EesiRealtimeProvider, EesiVADConfig
from tau2.voice.audio_native.openai.discrete_time_adapter import (
    DiscreteTimeOpenAIAdapter,
)
from tau2.voice.audio_native.openai.events import ResponseDoneEvent
from tau2.voice.audio_native.tick_result import TickResult


class DiscreteTimeEesiAdapter(DiscreteTimeOpenAIAdapter):
    """Adapter for discrete-time simulation with EESI Nur Live."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._session_counter = 0
        # Bytes of each reply played in completed ticks.
        self._played_bytes: dict[str, int] = {}

    @staticmethod
    def _played_this_tick(result: TickResult) -> list[tuple[str, int]]:
        """(item_id, bytes) played this tick, stopping at an interruption."""
        limit = len(result.agent_audio_data)
        if result.was_truncated and result.interruption_audio_start_ms is not None:
            into_tick_ms = (
                result.interruption_audio_start_ms
                - result.cumulative_user_audio_at_tick_start_ms
            )
            tick_ms = result.bytes_per_tick / result.bytes_per_second * 1000
            into_tick_ms = max(0, min(into_tick_ms, tick_ms))
            limit = int(into_tick_ms * result.bytes_per_second / 1000)
        played, total = [], 0
        for data, item_id in result.agent_audio_chunks:
            take = min(len(data), max(0, limit - total))
            total += take
            if item_id and take:
                played.append((item_id, take))
        return played

    def _played_audio_ms(self, result: TickResult, item_id: str) -> int:
        heard = self._played_bytes.get(item_id, 0) + sum(
            n for played, n in self._played_this_tick(result) if played == item_id
        )
        return int(heard / self.audio_format.bytes_per_second * 1000)

    async def _async_run_tick(self, user_audio: bytes, tick_number: int) -> TickResult:
        result = await super()._async_run_tick(user_audio, tick_number)
        for item_id, n in self._played_this_tick(result):
            self._played_bytes[item_id] = self._played_bytes.get(item_id, 0) + n
        return result

    @property
    def provider(self) -> EesiRealtimeProvider:
        if self._provider is None:
            self._provider = EesiRealtimeProvider(
                model=self.model,
                reasoning_effort=self.reasoning_effort,
            )
        return self._provider

    def connect(
        self,
        system_prompt: str,
        tools: List[Tool],
        vad_config: Any = None,
        modality: str = "audio",
    ) -> None:
        super().connect(system_prompt, tools, vad_config or EesiVADConfig(), modality)
        self._session_counter += 1
        self._played_bytes.clear()

    def _make_audio_usage_record(self) -> Optional[UsageRecord]:
        """Cumulative session-audio meter, EESI's billing basis.

        User audio streams every tick, so it tracks connected time.
        """
        if self._cumulative_user_audio_ms <= 0:
            return None
        return UsageRecord(
            provider="eesi",
            model=self.provider.model,
            component="realtime",
            semantics="cumulative",
            scope_id=f"audio-session-{self._session_counter}",
            audio_input_seconds=self._cumulative_user_audio_ms / 1000,
        )

    def get_usage_records(self) -> List[UsageRecord]:
        """Ledger records plus the live session-audio meter (if connected)."""
        records = super().get_usage_records()
        live_audio = self._make_audio_usage_record()
        if live_audio is not None:
            records.append(live_audio)
        return records

    def disconnect(self) -> None:
        # Flush the meter before the parent resets the audio counter.
        if self._connected:
            audio_record = self._make_audio_usage_record()
            if audio_record is not None:
                self._usage_records.append(audio_record)
        super().disconnect()

    async def _process_event(self, result: TickResult, event: Any) -> None:
        if isinstance(event, LateSpeechStartedEvent):
            result.events.append(event)
            result.vad_events.append("speech_started")
            return

        if isinstance(event, ResponseDoneEvent):
            result.events.append(event)
            if event.usage:
                record = UsageRecord.from_openai_realtime_usage(
                    event.usage,
                    provider="eesi",
                    model=self.provider.model,
                    scope_id=event.response_id,
                )
                record.billable = False
                self.record_usage(record)
            return

        await super()._process_event(result, event)
