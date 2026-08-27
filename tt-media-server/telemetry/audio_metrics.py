# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Request-level (OpenAI-usage-style) metrics for the audio endpoints.

STT covers /v1/audio/transcriptions and /v1/audio/translations (Whisper);
TTS covers /v1/audio/speech (SpeechT5). These complement two existing
families rather than replacing them: the FastAPI instrumentator's generic
per-handler HTTP metrics (which cannot see audio seconds or characters), and
the pipeline-stage metrics in ``telemetry_client`` (VAD, chunking, encoder),
which have no request-level rollup.

Recorded from the endpoint handlers in ``open_ai_api/audio.py`` and
``open_ai_api/text_to_speech.py``, so durations include queueing, not just
device time. Usage totals and realtime factor are success-only — a request
that failed after 2s of a 60s file did not transcribe 60 audio seconds, and
letting it land in the RTF histogram would drag the quantiles.
"""

import time

from prometheus_client import Counter, Histogram
from utils.logger import TTLogger

logger = TTLogger()

STATUS_SUCCESS = "success"
STATUS_ERROR = "error"

# ``voice`` label fallbacks: a client-supplied raw embedding has no id, and a
# request naming no speaker uses the model's built-in default voice.
VOICE_DEFAULT = "default"
VOICE_CUSTOM = "custom"

# Wall time of the whole request, queue included. STT is bounded by input
# length (a 30-minute file legitimately takes minutes), so the tail runs to
# 10 minutes; the sub-second grid keeps short-clip latency readable.
_REQUEST_DURATION_BUCKETS = (
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    3.5,
    5.0,
    7.5,
    10.0,
    15.0,
    30.0,
    60.0,
    120.0,
    300.0,
    600.0,
    float("inf"),
)

# Submitted audio length. Whisper chunks internally, so requests span short
# voice commands to hour-long recordings.
_INPUT_AUDIO_BUCKETS = (
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
    300.0,
    600.0,
    1800.0,
    3600.0,
    float("inf"),
)

# Wall seconds per audio second (1.0 = realtime). Shared by STT (per input
# second) and TTS (per generated second); both should sit well below 1 on
# hardware, so the grid is dense there with room for cold-start outliers.
_REALTIME_FACTOR_BUCKETS = (
    0.01,
    0.025,
    0.05,
    0.1,
    0.2,
    0.35,
    0.5,
    0.75,
    1.0,
    1.5,
    2.5,
    5.0,
    10.0,
    float("inf"),
)

# Generated speech length; SpeechT5 output is typically seconds to a few
# minutes for the 20k-character request cap.
_OUTPUT_AUDIO_BUCKETS = (
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    30.0,
    60.0,
    120.0,
    300.0,
    float("inf"),
)

# Request arrival at the handler until the first transcript update reaches the
# client: audio decode/VAD plus the first chunk's encode-and-first-decode
# (measured mean 0.45s), so the grid is dense from 0.25s to 2.5s with room for
# long preprocessing (diarization) and cold starts.
_FIRST_PARTIAL_BUCKETS = (
    0.05,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    float("inf"),
)

# Wall time between successive transcript updates within one stream. Updates
# arrive per decode step (tens of ms) with occasional chunk-boundary gaps of
# seconds, so the grid spans both regimes.
_PARTIAL_INTERVAL_BUCKETS = (
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    float("inf"),
)

# Client-facing bound is DEFAULT_MAX_TTS_TEXT_LENGTH (20000 characters).
_TTS_CHARACTER_BUCKETS = (
    10.0,
    50.0,
    100.0,
    250.0,
    500.0,
    1000.0,
    2500.0,
    5000.0,
    10000.0,
    20000.0,
    float("inf"),
)

# --- STT (transcriptions / translations) --------------------------------------
# ``task`` is "transcription" or "translation" and ``language`` is the
# configured input language (settings.audio_task / settings.audio_language) —
# both single-valued per deployment, so they add series only across
# deployments, where the mix panels need them. ``streaming`` is "true"/"false"
# on usage metrics too, so streaming mix is computable by audio seconds and
# characters, not just request count.
_STT_REQUEST_LABELS = ["model_type", "task", "language", "streaming", "status"]
_STT_USAGE_LABELS = ["model_type", "task", "language", "streaming"]
_STT_STREAM_LABELS = ["model_type", "task", "language"]

stt_requests_total = Counter(
    "tt_media_server_audio_stt_requests_total",
    "Finished speech-to-text requests",
    _STT_REQUEST_LABELS,
)

stt_request_duration = Histogram(
    "tt_media_server_audio_stt_request_duration_seconds",
    "Wall-clock duration of a speech-to-text request (queue + inference)",
    _STT_REQUEST_LABELS,
    buckets=_REQUEST_DURATION_BUCKETS,
)

stt_input_audio_seconds_total = Counter(
    "tt_media_server_audio_stt_input_audio_seconds_total",
    "Total seconds of audio successfully transcribed or translated",
    _STT_USAGE_LABELS,
)

stt_input_audio_duration = Histogram(
    "tt_media_server_audio_stt_input_audio_duration_seconds",
    "Audio duration of one successfully processed speech-to-text request",
    _STT_USAGE_LABELS,
    buckets=_INPUT_AUDIO_BUCKETS,
)

stt_output_characters_total = Counter(
    "tt_media_server_audio_stt_output_characters_total",
    "Total characters of text produced by speech-to-text",
    _STT_USAGE_LABELS,
)

stt_realtime_factor = Histogram(
    "tt_media_server_audio_stt_realtime_factor",
    "Wall-clock seconds spent per second of input audio (1.0 = realtime)",
    _STT_USAGE_LABELS,
    buckets=_REALTIME_FACTOR_BUCKETS,
)

# Streaming-only: one observation per stream for the first update, one per
# subsequent update for cadence. Wall time as the client experiences it — the
# emitting generator only resumes after the client consumes the previous
# update, so a slow consumer shows up here by design (matches the philosophy
# of audio_chunk_processing_seconds).
stt_first_partial_duration = Histogram(
    "tt_media_server_audio_stt_first_partial_seconds",
    "Time from request arrival until the first transcript update is emitted",
    _STT_STREAM_LABELS,
    buckets=_FIRST_PARTIAL_BUCKETS,
)

stt_partial_interval = Histogram(
    "tt_media_server_audio_stt_partial_interval_seconds",
    "Wall time between successive transcript updates within one stream",
    _STT_STREAM_LABELS,
    buckets=_PARTIAL_INTERVAL_BUCKETS,
)

# --- TTS (speech) --------------------------------------------------------------
# ``response_format`` is bounded by TTS_RESPONSE_FORMATS (wav/mp3/ogg/json/
# verbose_json). ``voice`` is the speaker id (client-supplied, so its
# cardinality is the deployment's speaker catalog — see tts_voice_label); it
# goes on the counters and the realtime factor, which the voice-mix and
# per-voice-RTF panels need, but not on the wide duration histograms, where
# it would multiply every bucket.
_TTS_REQUEST_LABELS = ["model_type", "response_format", "voice", "status"]

tts_requests_total = Counter(
    "tt_media_server_audio_tts_requests_total",
    "Finished text-to-speech requests",
    _TTS_REQUEST_LABELS,
)

tts_request_duration = Histogram(
    "tt_media_server_audio_tts_request_duration_seconds",
    "Wall-clock duration of a text-to-speech request (queue + inference)",
    ["model_type", "response_format", "status"],
    buckets=_REQUEST_DURATION_BUCKETS,
)

tts_input_characters_total = Counter(
    "tt_media_server_audio_tts_input_characters_total",
    "Total input characters successfully converted to speech",
    ["model_type"],
)

tts_input_characters = Histogram(
    "tt_media_server_audio_tts_input_characters_per_request",
    "Input characters of one successful text-to-speech request",
    ["model_type"],
    buckets=_TTS_CHARACTER_BUCKETS,
)

tts_output_audio_seconds_total = Counter(
    "tt_media_server_audio_tts_output_audio_seconds_total",
    "Total seconds of speech audio generated",
    ["model_type", "voice"],
)

tts_output_audio_duration = Histogram(
    "tt_media_server_audio_tts_output_audio_duration_seconds",
    "Audio duration of one successful text-to-speech generation",
    ["model_type"],
    buckets=_OUTPUT_AUDIO_BUCKETS,
)

tts_realtime_factor = Histogram(
    "tt_media_server_audio_tts_realtime_factor",
    "Wall-clock seconds spent per second of generated audio (1.0 = realtime)",
    ["model_type", "voice"],
    buckets=_REALTIME_FACTOR_BUCKETS,
)


def char_count(text) -> int | None:
    """``len`` for metric call sites: None for anything that is not a str.

    Handlers compute usage inside ``finally`` blocks; a non-str (None, or a
    Mock in tests) must degrade to "unknown" rather than raise there and mask
    the real response or exception.
    """
    return len(text) if isinstance(text, str) else None


def _positive_float(value) -> float | None:
    """Coerce to float, returning None for non-numeric or non-positive input."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _label_str(value, fallback: str) -> str:
    """A non-empty str verbatim; anything else becomes the fallback."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def tts_voice_label(request, result=None) -> str:
    """Resolve the ``voice`` label for one TTS request.

    The id the runner reports back wins over the one asked for; a request
    carrying a raw speaker embedding has no id at all and is labelled
    ``custom``; everything else used the model's default voice. Truncated so
    a pathological client id cannot bloat the exposition. Never raises —
    handlers call this from ``finally`` blocks.
    """
    try:
        speaker_id = getattr(result, "speaker_id", None)
        if not isinstance(speaker_id, str) or not speaker_id.strip():
            speaker_id = getattr(request, "speaker_id", None)
        if isinstance(speaker_id, str) and speaker_id.strip():
            return speaker_id.strip()[:64]
        if getattr(request, "speaker_embedding", None) is not None:
            return VOICE_CUSTOM
        return VOICE_DEFAULT
    except Exception:  # pragma: no cover - telemetry must not break serving
        return VOICE_DEFAULT


class SttStreamProgress:
    """First-partial latency and inter-update cadence for one STT stream.

    Construct when the stream request arrives, call :meth:`on_update` once per
    transcript update emitted to the client (partial chunks and the final
    result alike). The first call observes first-partial latency; every later
    call observes the gap since the previous one. Never raises.
    """

    def __init__(
        self,
        model_type: str,
        task: str,
        language: str,
        start: float | None = None,
    ) -> None:
        self.model_type = model_type
        self.task = task
        self.language = language
        self._start = start if start is not None else time.perf_counter()
        self._last: float | None = None

    def on_update(self) -> None:
        try:
            now = time.perf_counter()
            labels = dict(
                model_type=self.model_type, task=self.task, language=self.language
            )
            if self._last is None:
                stt_first_partial_duration.labels(**labels).observe(now - self._start)
            else:
                stt_partial_interval.labels(**labels).observe(now - self._last)
            self._last = now
        except Exception as exc:  # pragma: no cover - telemetry must not break serving
            logger.warning(f"Failed to record STT stream progress: {exc}")


def record_stt_request(
    *,
    model_type: str,
    task: str,
    language: str,
    streaming: bool,
    status: str,
    duration_seconds: float,
    audio_seconds: float | None = None,
    characters: int | None = None,
) -> None:
    """Export one finished STT request. Never raises into the request path.

    ``audio_seconds`` and ``characters`` may be passed on any outcome; usage
    totals and realtime factor are only recorded for ``status="success"``.
    """
    try:
        streaming_label = str(bool(streaming)).lower()
        language = _label_str(language, "unknown")
        duration_seconds = _positive_float(duration_seconds)
        request_labels = dict(
            model_type=model_type,
            task=task,
            language=language,
            streaming=streaming_label,
            status=status,
        )
        stt_requests_total.labels(**request_labels).inc()
        if duration_seconds is not None:
            stt_request_duration.labels(**request_labels).observe(duration_seconds)

        if status != STATUS_SUCCESS:
            return

        usage_labels = dict(
            model_type=model_type,
            task=task,
            language=language,
            streaming=streaming_label,
        )
        audio_seconds = _positive_float(audio_seconds)
        if audio_seconds is not None:
            stt_input_audio_seconds_total.labels(**usage_labels).inc(audio_seconds)
            stt_input_audio_duration.labels(**usage_labels).observe(audio_seconds)
            if duration_seconds is not None:
                stt_realtime_factor.labels(**usage_labels).observe(
                    duration_seconds / audio_seconds
                )
        if isinstance(characters, int) and characters > 0:
            stt_output_characters_total.labels(**usage_labels).inc(characters)
    except Exception as exc:  # pragma: no cover - telemetry must not break serving
        logger.warning(f"Failed to record STT request metrics: {exc}")


def record_tts_request(
    *,
    model_type: str,
    response_format: str,
    status: str,
    duration_seconds: float,
    voice: str = VOICE_DEFAULT,
    characters: int | None = None,
    audio_seconds: float | None = None,
) -> None:
    """Export one finished TTS request. Never raises into the request path.

    ``characters`` is the input text length; ``audio_seconds`` the generated
    speech duration. Both are only recorded for ``status="success"``.
    ``voice`` comes from :func:`tts_voice_label`.
    """
    try:
        duration_seconds = _positive_float(duration_seconds)
        response_format = (
            response_format.lower() if isinstance(response_format, str) else "unknown"
        )
        voice = _label_str(voice, VOICE_DEFAULT)
        tts_requests_total.labels(
            model_type=model_type,
            response_format=response_format,
            voice=voice,
            status=status,
        ).inc()
        if duration_seconds is not None:
            tts_request_duration.labels(
                model_type=model_type,
                response_format=response_format,
                status=status,
            ).observe(duration_seconds)

        if status != STATUS_SUCCESS:
            return

        if isinstance(characters, int) and characters > 0:
            tts_input_characters_total.labels(model_type=model_type).inc(characters)
            tts_input_characters.labels(model_type=model_type).observe(characters)
        audio_seconds = _positive_float(audio_seconds)
        if audio_seconds is not None:
            tts_output_audio_seconds_total.labels(
                model_type=model_type, voice=voice
            ).inc(audio_seconds)
            tts_output_audio_duration.labels(model_type=model_type).observe(
                audio_seconds
            )
            if duration_seconds is not None:
                tts_realtime_factor.labels(model_type=model_type, voice=voice).observe(
                    duration_seconds / audio_seconds
                )
    except Exception as exc:  # pragma: no cover - telemetry must not break serving
        logger.warning(f"Failed to record TTS request metrics: {exc}")
