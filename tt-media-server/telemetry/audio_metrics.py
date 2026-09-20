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

import math
import time
import zlib

from prometheus_client import Counter, Histogram
from utils.logger import TTLogger

logger = TTLogger()

STATUS_SUCCESS = "success"
STATUS_ERROR = "error"

# ``voice`` label fallbacks: a client-supplied raw embedding has no id, a
# request naming no speaker uses the model's built-in default voice, and a
# failed request gets "unknown" (see record_tts_request).
VOICE_DEFAULT = "default"
VOICE_CUSTOM = "custom"
VOICE_UNKNOWN = "unknown"

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

# Audio seconds per inference chunk. The merge targets
# settings.audio_chunk_duration_seconds and Whisper's frame caps useful length
# at 30s, but a single uninterrupted VAD segment is never split, so >30s
# chunks are real (the generator truncates them to one frame downstream).
_CHUNK_AUDIO_BUCKETS = (
    0.25,
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    7.5,
    10.0,
    15.0,
    20.0,
    30.0,
    60.0,
    120.0,
    float("inf"),
)

# Wait between the last partial and the final transcript: segment assembly
# and speaker sorting, usually milliseconds, so the grid bottoms out at 1ms.
_FINALIZATION_BUCKETS = (
    0.001,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    float("inf"),
)

# Whisper's mean per-token log-probability; ≤ 0 by construction. The grid is
# dense around -1.0 (Whisper's conventional low-confidence line) and the
# request-default logprob_threshold of -2.0.
_AVG_LOGPROB_BUCKETS = (
    -3.0,
    -2.0,
    -1.5,
    -1.25,
    -1.0,
    -0.8,
    -0.6,
    -0.5,
    -0.4,
    -0.3,
    -0.2,
    -0.1,
    0.0,
    float("inf"),
)

# Probability of the <|nospeech|> token; the request-default gate is 0.6.
_NO_SPEECH_PROB_BUCKETS = (
    0.01,
    0.02,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    float("inf"),
)

# zlib ratio of the transcript; > 2.4 is Whisper's conventional
# repetition-loop (hallucination) signal, so the grid is dense around it.
_COMPRESSION_RATIO_BUCKETS = (
    0.8,
    1.0,
    1.2,
    1.4,
    1.6,
    1.8,
    2.0,
    2.2,
    2.4,
    2.8,
    3.5,
    5.0,
    float("inf"),
)

# On-device generation of one text chunk's mel spectrogram: an autoregressive
# decode of up to ~300 steps, so sub-second through tens of seconds.
_TTS_CHUNK_BUCKETS = (
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
    20.0,
    30.0,
    60.0,
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

# English averages ~5 characters/word, so the 20000-character cap is ~4000
# words; same grid shape as characters, one order of magnitude down.
_TTS_WORD_BUCKETS = (
    2.0,
    5.0,
    10.0,
    25.0,
    50.0,
    100.0,
    250.0,
    500.0,
    1000.0,
    2000.0,
    4000.0,
    float("inf"),
)

# SpeechT5's tokenizer is near character-level, so per-request encoder tokens
# track the character count; the cap is ~20k with headroom for markup-heavy
# text.
_TTS_TOKEN_BUCKETS = (
    10.0,
    50.0,
    100.0,
    250.0,
    500.0,
    1000.0,
    2500.0,
    5000.0,
    10000.0,
    25000.0,
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

# Per-chunk audio length, one observation per chunk at the VAD-merge site in
# audio_manager (where audio_chunks_per_request is already observed). This is
# the distribution the count alone cannot give — it's bimodal by design:
# diarization yields short speaker-bounded chunks, VAD-only longer merged
# ones, and chunk length drives effective throughput (fixed cost per padded
# 30s frame). VAD-path only: with preprocessing skipped there are no chunks
# to measure and the series is legitimately absent. The configured merge
# target is exported on tt_media_server_info as audio_chunk_duration_seconds.
stt_chunk_audio_seconds = Histogram(
    "tt_media_server_audio_stt_chunk_audio_seconds",
    "Audio seconds of one inference chunk produced by VAD-segment merging",
    ["model_type", "mode"],
    buckets=_CHUNK_AUDIO_BUCKETS,
)

# Format mix BY REQUEST — deliberately distinct from the pre-existing
# audio_feature_extraction_input_seconds_total{sample_rate, channels}, which
# is denominated in audio seconds (and is the numerator of that stage's
# throughput calculation, so it stays). Ten hour-long files can dwarf a
# thousand voice commands there; this counter answers "what share of
# *requests* arrive at which operating point". Values are the submitted
# rate/channels stamped by preprocessing, before the mono/default-rate
# resample — same source as the feature-extraction labels.
stt_requests_by_audio_format = Counter(
    "tt_media_server_audio_stt_requests_by_audio_format_total",
    "Successful speech-to-text requests by submitted sample rate and channels",
    ["model_type", "task", "language", "sample_rate", "channels"],
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

# PROXY for finalization latency ("end-of-speech until the final transcript").
# The true metric needs live audio input with an end-of-audio marker, which
# the current protocol does not have — clients upload the whole file, so
# end-of-input coincides with request arrival and true finalization collapses
# into total request duration (stt_request_duration_seconds). What IS
# observable is the streaming-side tail: how long the client waits after the
# last partial before the transcript is finalized (segment assembly, speaker
# sorting). Only non-text-format streams yield a final result, so text-format
# streams contribute no samples.
stt_finalization_duration = Histogram(
    "tt_media_server_audio_stt_finalization_seconds",
    "Wait between the last partial and the final transcript of one stream",
    _STT_STREAM_LABELS,
    buckets=_FINALIZATION_BUCKETS,
)

# PROXY for word error rate. True WER needs reference transcripts, which
# production traffic does not have — the offline evals in
# test_module/eval_tests/ remain the source of truth. These are Whisper's own
# per-generation quality signals (the same ones its temperature-fallback loop
# gates on), exported as drift detectors: a WER regression moves avg_logprob
# down and compression_ratio/no_speech up before anyone files a ticket. They
# are recorded from the whisper runner, once per generate() call (per chunk),
# not per request. Low avg_logprob is *difficulty* from any cause — noise,
# accent, domain mismatch — so shifts say "harder or worse", not why.
_STT_CONFIDENCE_LABELS = ["model_type", "language"]

stt_avg_logprob = Histogram(
    "tt_media_server_audio_stt_avg_logprob",
    "Mean per-token log-probability of one Whisper generation",
    _STT_CONFIDENCE_LABELS,
    buckets=_AVG_LOGPROB_BUCKETS,
)

stt_no_speech_probability = Histogram(
    "tt_media_server_audio_stt_no_speech_probability",
    "Probability of the no-speech token for one Whisper generation",
    _STT_CONFIDENCE_LABELS,
    buckets=_NO_SPEECH_PROB_BUCKETS,
)

stt_compression_ratio = Histogram(
    "tt_media_server_audio_stt_compression_ratio",
    "zlib compression ratio of one generated transcript (>2.4 suggests loops)",
    _STT_CONFIDENCE_LABELS,
    buckets=_COMPRESSION_RATIO_BUCKETS,
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

# Whitespace word count — approximate, but honest for an English-only model.
tts_input_words_total = Counter(
    "tt_media_server_audio_tts_input_words_total",
    "Total input words successfully converted to speech",
    ["model_type"],
)

tts_input_words = Histogram(
    "tt_media_server_audio_tts_input_words_per_request",
    "Input words of one successful text-to-speech request",
    ["model_type"],
    buckets=_TTS_WORD_BUCKETS,
)

# Exact encoder token counts, recorded from the runner: the device worker
# already tokenizes every chunk (real_seq_len in _generate_mel_for_chunk), so
# this is the model's true input size, not a handler-side approximation.
tts_input_tokens_total = Counter(
    "tt_media_server_audio_tts_input_tokens_total",
    "Total encoder input tokens successfully converted to speech",
    ["model_type"],
)

tts_input_tokens = Histogram(
    "tt_media_server_audio_tts_input_tokens_per_request",
    "Encoder input tokens of one successful text-to-speech generation",
    ["model_type"],
    buckets=_TTS_TOKEN_BUCKETS,
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

# PROXY for time-to-first-audio and chunk cadence. The /v1/audio/speech
# endpoint has no streaming mode — the response is one complete utterance, so
# client-observed TTFA today is simply total request duration. What IS
# observable is what streaming *would* deliver: the SpeechT5 runner splits
# long text into chunks and generates each chunk's mel spectrogram in turn
# (the vocoder runs once at the end). First-chunk time is a lower bound on
# achievable streaming TTFA (a streaming implementation would add one
# per-chunk vocoder pass), and per-chunk generation time is the achievable
# chunk cadence. Recorded inside the runner, once per chunk; single-chunk
# requests contribute one observation to each.
tts_first_chunk_duration = Histogram(
    "tt_media_server_audio_tts_first_chunk_seconds",
    "Time from generation start until the first text chunk's mel is complete",
    ["model_type"],
    buckets=_TTS_CHUNK_BUCKETS,
)

tts_chunk_generation_duration = Histogram(
    "tt_media_server_audio_tts_chunk_generation_seconds",
    "Wall time to generate one text chunk's mel spectrogram",
    ["model_type"],
    buckets=_TTS_CHUNK_BUCKETS,
)


def char_count(text) -> int | None:
    """``len`` for metric call sites: None for anything that is not a str.

    Handlers compute usage inside ``finally`` blocks; a non-str (None, or a
    Mock in tests) must degrade to "unknown" rather than raise there and mask
    the real response or exception.
    """
    return len(text) if isinstance(text, str) else None


def word_count(text) -> int | None:
    """Whitespace word count with the same non-str contract as char_count."""
    return len(text.split()) if isinstance(text, str) else None


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


def _count_label(value) -> str:
    """A positive integer as a label value, else "unknown".

    Matches the value convention of the pre-existing feature-extraction
    labels (str of the submitted number, "unknown" when preprocessing could
    not determine it).
    """
    try:
        value = int(value)
    except (TypeError, ValueError):
        return "unknown"
    return str(value) if value > 0 else "unknown"


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
    """First-partial latency, cadence and finalization for one STT stream.

    Construct when the stream request arrives, call :meth:`on_update` once per
    partial transcript update emitted to the client — the first call observes
    first-partial latency, every later call the gap since the previous one —
    and :meth:`on_final` when the final transcript is emitted. Never raises.
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

    def _labels(self) -> dict:
        return dict(model_type=self.model_type, task=self.task, language=self.language)

    def on_update(self) -> None:
        try:
            now = time.perf_counter()
            if self._last is None:
                stt_first_partial_duration.labels(**self._labels()).observe(
                    now - self._start
                )
            else:
                stt_partial_interval.labels(**self._labels()).observe(now - self._last)
            self._last = now
        except Exception as exc:  # pragma: no cover - telemetry must not break serving
            logger.warning(f"Failed to record STT stream progress: {exc}")

    def on_final(self) -> None:
        """The final transcript arrived: observe the wait since the last
        partial (or since the start, for a stream with no partials). Does not
        touch the cadence clock — the final result is not a partial update.
        """
        try:
            now = time.perf_counter()
            since = self._last if self._last is not None else self._start
            stt_finalization_duration.labels(**self._labels()).observe(now - since)
        except Exception as exc:  # pragma: no cover - telemetry must not break serving
            logger.warning(f"Failed to record STT finalization: {exc}")


class TtsChunkProgress:
    """First-chunk latency and per-chunk cadence for one TTS generation.

    Construct when generation starts, call :meth:`on_chunk` as each text
    chunk's mel spectrogram completes. Every call observes the wall time
    since the previous mark into the cadence histogram; the first call also
    observes it as first-chunk latency (for the first chunk the two spans
    are the same interval by construction). Never raises.
    """

    def __init__(self, model_type: str, start: float | None = None) -> None:
        self.model_type = model_type
        self._last = start if start is not None else time.perf_counter()
        self._first_seen = False

    def on_chunk(self) -> None:
        try:
            now = time.perf_counter()
            elapsed = now - self._last
            tts_chunk_generation_duration.labels(model_type=self.model_type).observe(
                elapsed
            )
            if not self._first_seen:
                tts_first_chunk_duration.labels(model_type=self.model_type).observe(
                    elapsed
                )
                self._first_seen = True
            self._last = now
        except Exception as exc:  # pragma: no cover - telemetry must not break serving
            logger.warning(f"Failed to record TTS chunk progress: {exc}")


def _scalar_mean(value) -> float | None:
    """A finite float from a number, a 0-d/1-element tensor, or the mean of a
    sequence of them; None for anything unrecognisable."""
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except Exception:
        pass
    try:
        values = [float(v) for v in value]
    except Exception:
        return None
    values = [v for v in values if math.isfinite(v)]
    if not values:
        return None
    return sum(values) / len(values)


def confidence_from_generator_output(item) -> tuple[float | None, float | None]:
    """Pull ``(avg_logprob, no_speech_prob)`` off a whisper_generator result.

    Both the streaming yields and the non-streaming return tuples place
    avg_logprob(s) at index 1 and no_speech_prob(s) at index 2 (tt-metal
    ``WhisperGenerator``); values arrive as floats, 0-d tensors, or per-batch
    sequences, and are coerced without importing torch — matched by position
    the way image_metrics matches tt_dit events by name, so tt-metal never
    has to be importable here.

    A ``(0.0, 0.0)`` pair is treated as unavailable, not observed: the traced
    decode path never computes these (no-speech extraction is un-traced-path
    only, and avg_logprob falls back to ``torch.zeros``), and the
    all-temperatures-failed path returns zero tensors. Recording those would
    pile a fake spike at 0 on both histograms. A genuine simultaneous 0.0/0.0
    (perfect confidence and exactly-zero no-speech) is not realistically
    observable.
    """
    if not isinstance(item, tuple) or len(item) < 3:
        return None, None
    avg_logprob = _scalar_mean(item[1])
    no_speech_prob = _scalar_mean(item[2])
    if avg_logprob == 0.0 and no_speech_prob == 0.0:
        return None, None
    return avg_logprob, no_speech_prob


def transcript_compression_ratio(text) -> float | None:
    """zlib compression ratio of a transcript — the same formula the
    generator's temperature-fallback uses; >2.4 conventionally signals a
    repetition loop. None for empty or non-str input."""
    if not isinstance(text, str) or not text:
        return None
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))


def record_stt_chunk_sizes(*, model_type: str, mode: str, chunks) -> None:
    """Export the audio length of each merged inference chunk. Never raises.

    ``chunks`` is the VAD-merge output: mappings with ``start``/``end`` times
    in seconds. Unreadable entries are skipped rather than aborting the batch.
    """
    try:
        # Lazy labels(): resolving them up front would materialise an empty
        # series for a request whose chunks are all unreadable or absent.
        histogram = None
        for chunk in chunks or ():
            try:
                seconds = float(chunk["end"]) - float(chunk["start"])
            except Exception:
                continue
            if seconds > 0:
                if histogram is None:
                    histogram = stt_chunk_audio_seconds.labels(
                        model_type=model_type, mode=mode
                    )
                histogram.observe(seconds)
    except Exception as exc:  # pragma: no cover - telemetry must not break serving
        logger.warning(f"Failed to record STT chunk sizes: {exc}")


def record_stt_confidence(
    *,
    model_type: str,
    language: str,
    avg_logprob: float | None = None,
    no_speech_prob: float | None = None,
    compression_ratio: float | None = None,
) -> None:
    """Export Whisper's quality signals for one generation. Never raises.

    WER proxies, not WER — see the histogram definitions above. Each signal
    is recorded independently so a path that only knows some of them still
    contributes the ones it has.
    """
    try:
        labels = dict(model_type=model_type, language=_label_str(language, "unknown"))
        avg_logprob = _scalar_mean(avg_logprob)
        if avg_logprob is not None:
            stt_avg_logprob.labels(**labels).observe(avg_logprob)
        no_speech_prob = _scalar_mean(no_speech_prob)
        if no_speech_prob is not None:
            stt_no_speech_probability.labels(**labels).observe(no_speech_prob)
        compression_ratio = _positive_float(compression_ratio)
        if compression_ratio is not None:
            stt_compression_ratio.labels(**labels).observe(compression_ratio)
    except Exception as exc:  # pragma: no cover - telemetry must not break serving
        logger.warning(f"Failed to record STT confidence signals: {exc}")


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
    sample_rate=None,
    channels=None,
) -> None:
    """Export one finished STT request. Never raises into the request path.

    ``audio_seconds``, ``characters``, ``sample_rate`` and ``channels`` may be
    passed on any outcome; usage totals, realtime factor and the audio-format
    counter are only recorded for ``status="success"``.
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
        stt_requests_by_audio_format.labels(
            model_type=model_type,
            task=task,
            language=language,
            sample_rate=_count_label(sample_rate),
            channels=_count_label(channels),
        ).inc()
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
    words: int | None = None,
    audio_seconds: float | None = None,
) -> None:
    """Export one finished TTS request. Never raises into the request path.

    ``characters`` and ``words`` are the input text length; ``audio_seconds``
    the generated speech duration. Usage values are only recorded for
    ``status="success"``. ``voice`` comes from :func:`tts_voice_label`.
    (Input *tokens* are recorded by the runner via
    :func:`record_tts_input_tokens`, where the text is actually tokenized.)
    """
    try:
        duration_seconds = _positive_float(duration_seconds)
        response_format = (
            response_format.lower() if isinstance(response_format, str) else "unknown"
        )
        # Success is what proves a client-named speaker exists in the catalog;
        # on the error path ``voice`` may be an arbitrary string from a
        # request that failed precisely because the id was unknown, so error
        # rows are collapsed to "unknown" to keep label cardinality bounded.
        if status != STATUS_SUCCESS:
            voice = VOICE_UNKNOWN
        else:
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
        if isinstance(words, int) and words > 0:
            tts_input_words_total.labels(model_type=model_type).inc(words)
            tts_input_words.labels(model_type=model_type).observe(words)
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


def record_tts_input_tokens(*, model_type: str, tokens) -> None:
    """Export the encoder token count of one successful TTS generation.

    Called from the runner, which already computes the exact count while
    tokenizing each chunk — never from the handler, which has no tokenizer.
    Never raises.
    """
    try:
        if isinstance(tokens, int) and tokens > 0:
            tts_input_tokens_total.labels(model_type=model_type).inc(tokens)
            tts_input_tokens.labels(model_type=model_type).observe(tokens)
    except Exception as exc:  # pragma: no cover - telemetry must not break serving
        logger.warning(f"Failed to record TTS input tokens: {exc}")
