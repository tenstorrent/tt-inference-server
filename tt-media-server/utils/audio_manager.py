# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import atexit
import base64
import json
import os
import selectors
import struct
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

import numpy as np
from config.constants import SupportedModels
from config.settings import settings
from domain.audio_text_response import AudioTextResponse, AudioTextSegment

from utils.decorators import log_execution_time
from utils.ffmpeg_utils import decode_to_wav as ffmpeg_decode_to_wav
from utils.logger import TTLogger

# Path to the audio venv Python interpreter (set by Dockerfile).
# whisperx + pyannote + silero_vad live there with torch 2.7.x; the main venv
# runs torch 2.10 for vLLM 0.18.1. See Dockerfile for setup.
AUDIO_VENV_PYTHON = os.getenv(
    "AUDIO_VENV_PYTHON", "/home/container_app_user/tt-metal/audio_venv/bin/python"
)

# Path to the diarize.py script invoked inside the audio venv.
DIARIZE_SCRIPT = Path(__file__).parent / "diarize.py"


class AudioVenvWorker:
    """Long-lived subprocess bridge to `diarize.py --serve` in the audio venv.

    The old design spawned a fresh Python interpreter for every audio
    request, which paid the torch import (~1.5s) and pyannote/silero model
    load (~500-800ms) cost on every single call. This class replaces that
    with **one persistent subprocess per instance**: the subprocess stays
    alive, torch and the models are lazy-loaded on first use, and every
    subsequent :meth:`run` is a JSON round-trip over stdin/stdout on the
    already-warm worker.

    Concurrency contract: each :class:`AudioVenvWorker` is expected to have
    a **single caller** (the parent :class:`AudioManager` living inside one
    :mod:`CpuWorkloadHandler` worker process). There is no internal lock;
    the media server already fans concurrent requests out to N independent
    AudioManagers, so parallelism is achieved at the outer layer.

    Failure recovery: if the worker dies (SIGKILL, segfault, unhandled
    exception), the next call to :meth:`run` detects the dead pipe and
    respawns automatically. Callers observe a single failed request; the
    next request succeeds.
    """

    # Time budget for the child to signal ready. The server no longer
    # eager-loads models here (that happens lazily on first request, driven
    # by the CpuWorkloadHandler warmup task), so all we're waiting for is
    # Python interpreter startup + a tiny script import. 30s is plenty of
    # headroom for a cold filesystem.
    _READY_TIMEOUT_SECONDS = 30

    def __init__(
        self,
        logger: TTLogger,
        model_name: Optional[str] = None,
        hf_token: Optional[str] = None,
        python_executable: str = AUDIO_VENV_PYTHON,
        script_path: Path = DIARIZE_SCRIPT,
        popen_factory=None,
    ):
        self._logger = logger
        self._model_name = model_name
        self._hf_token = hf_token
        self._python = python_executable
        self._script = script_path
        # Injectable for tests (fake Popen). Default is subprocess.Popen.
        self._popen_factory = popen_factory or subprocess.Popen
        self._proc: Optional[subprocess.Popen] = None

    def is_available(self) -> bool:
        """Return True if both the audio venv Python and diarize script exist."""
        return os.path.exists(self._python) and self._script.exists()

    def assert_available(self) -> None:
        """Raise FileNotFoundError if the audio venv or script is missing."""
        if not os.path.exists(self._python):
            raise FileNotFoundError(f"Audio venv Python not found at {self._python}")
        if not self._script.exists():
            raise FileNotFoundError(f"Diarize script not found at {self._script}")

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> None:
        """Spawn the worker and block until it signals ready.

        Raises RuntimeError on spawn failure, ready timeout, or if the
        worker exits before signalling ready.
        """
        if self.is_running():
            return

        self.assert_available()

        cmd: List[str] = [self._python, str(self._script), "--serve"]
        if self._model_name:
            cmd.extend(["--model-name", self._model_name])
        if self._hf_token:
            cmd.extend(["--hf-token", self._hf_token])

        try:
            self._proc = self._popen_factory(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # line-buffered
            )
        except Exception as e:
            self._proc = None
            raise RuntimeError(f"Failed to spawn audio worker: {e}") from e

        ready = self._read_response(timeout_seconds=self._READY_TIMEOUT_SECONDS)
        if ready is None:
            self._terminate_proc()
            raise RuntimeError(
                f"Audio worker did not signal ready within "
                f"{self._READY_TIMEOUT_SECONDS}s"
            )
        if ready.get("status") != "ready":
            self._terminate_proc()
            raise RuntimeError(
                f"Audio worker returned unexpected ready payload: {ready}"
            )

        self._logger.info("Audio venv worker is ready")

    def run(
        self,
        mode: str,
        audio_array: np.ndarray,
        timeout_seconds: int,
    ) -> Optional[List[dict]]:
        """Send one request, wait for the matching response.

        Returns the segments list on success, or ``None`` on any failure
        (worker not startable, dead worker, timeout, error payload, or id
        mismatch). ``None`` is what the existing callers already treat as
        "audio preprocessing unavailable for this request, skip it".
        """
        if not self.is_running():
            try:
                self.start()
            except Exception as e:
                self._logger.error(f"Cannot start audio worker: {e}")
                return None

        request_id = str(uuid.uuid4())

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "audio.npy")
            np.save(audio_path, audio_array)

            request = {"id": request_id, "mode": mode, "audio_path": audio_path}

            if not self._send_request(request, mode):
                return None

            response = self._read_response(timeout_seconds=timeout_seconds)

        if response is None:
            self._logger.error(f"{mode} response timed out after {timeout_seconds}s")
            self._terminate_proc()
            return None

        if response.get("id") != request_id:
            self._logger.error(
                f"{mode} response id mismatch (expected {request_id}, "
                f"got {response.get('id')}); dropping worker"
            )
            self._terminate_proc()
            return None

        if response.get("status") != "success":
            self._logger.error(f"{mode} error: {response.get('error')}")
            return None

        return response.get("segments", [])

    def close(self) -> None:
        """Cleanly shut down the worker: close stdin (EOF), wait briefly,
        then hard-terminate if it didn't exit on its own."""
        if self._proc is None:
            return
        try:
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=5.0)
        except Exception:
            pass
        self._terminate_proc()

    # -------------------------------------------------------------------------
    # Low-level pipe I/O
    # -------------------------------------------------------------------------

    def _send_request(self, request: dict, mode_for_log: str) -> bool:
        """Write one JSON line to the worker's stdin. Returns False on
        failure (and drops the worker so the next call respawns)."""
        try:
            assert self._proc is not None and self._proc.stdin is not None
            self._proc.stdin.write(json.dumps(request) + "\n")
            self._proc.stdin.flush()
            return True
        except Exception as e:
            self._logger.error(f"Failed to send {mode_for_log} request: {e}")
            self._terminate_proc()
            return False

    def _read_response(self, timeout_seconds: float) -> Optional[dict]:
        """Read one JSON line from the worker's stdout, blocking at most
        ``timeout_seconds``. Returns:

        * the parsed dict on success
        * ``None`` on timeout
        * ``None`` on EOF or malformed JSON (worker gets terminated so the
          next call respawns)
        """
        if self._proc is None or self._proc.stdout is None:
            return None

        stdout = self._proc.stdout
        sel = selectors.DefaultSelector()
        sel.register(stdout, selectors.EVENT_READ)
        try:
            events = sel.select(timeout=timeout_seconds)
        finally:
            sel.unregister(stdout)

        if not events:
            return None  # timeout — caller decides whether to terminate

        try:
            # The server writes exactly one JSON payload per line and
            # flushes, so once stdout is readable a full line is imminent.
            line = stdout.readline()
        except Exception as e:
            self._logger.error(f"Error reading from audio worker stdout: {e}")
            self._terminate_proc()
            return None

        if not line:
            self._logger.error(
                "Audio worker closed stdout unexpectedly (EOF); dropping worker"
            )
            self._terminate_proc()
            return None

        try:
            return json.loads(line)
        except json.JSONDecodeError as e:
            self._logger.error(
                f"Audio worker sent malformed JSON: {e!r}; line={line!r}"
            )
            self._terminate_proc()
            return None

    def _terminate_proc(self) -> None:
        """Best-effort hard-stop. Safe to call multiple times."""
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except Exception:
            pass
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2.0)
            except Exception:
                pass
        if proc.poll() is None:
            try:
                proc.kill()
                proc.wait(timeout=1.0)
            except Exception:
                pass
        self._proc = None


class AudioManager:
    _whisperx_device: str = "cpu"

    # Per-request timeouts on the persistent worker. Diarization is heavier
    # than VAD because it runs the pyannote pipeline end-to-end. These are
    # generous ceilings — steady-state requests complete in well under a
    # second now that models stay hot in the worker.
    _DIARIZATION_TIMEOUT_SECONDS = 300
    _VAD_TIMEOUT_SECONDS = 120

    def __init__(self):
        self._logger = TTLogger()
        # `_diarization_model` and `_vad_model` are legacy sentinels that the
        # rest of AudioManager checks as "is diarization/VAD available?".
        # We set them to True after the worker is up and running.
        self._diarization_model = None
        self._diarization_model_name: Optional[str] = None
        self._vad_model = None
        self._audio_venv_worker: Optional[AudioVenvWorker] = None
        self._atexit_registered = False

        if settings.allow_audio_preprocessing:
            self._initialize_audio_worker()
        else:
            self._logger.info("Audio preprocessing disabled")

    def close(self) -> None:
        """Terminate the audio venv worker. Safe to call multiple times.

        Automatically registered as an ``atexit`` handler when the worker is
        successfully started so the child process doesn't outlive the
        parent under normal shutdown.
        """
        worker = self._audio_venv_worker
        self._audio_venv_worker = None
        if worker is not None:
            worker.close()

    def to_audio_array(self, file, should_preprocess):
        """Convert audio file (base64 string or raw bytes) to numpy array for audio model inference."""
        try:
            if isinstance(file, str):
                # Base64-encoded string
                audio_bytes = base64.b64decode(file)
            elif isinstance(file, bytes):
                # Raw audio bytes (from file upload)
                audio_bytes = file
            else:
                self._logger.error(
                    f"Unsupported file input type: {type(file).__name__}"
                )
                raise ValueError(f"Unsupported file input type: {type(file).__name__}")

            self._validate_file_size(audio_bytes)
            audio_array = self._convert_to_audio_array(audio_bytes)
            return self._validate_and_truncate_duration(audio_array, should_preprocess)
        except Exception as e:
            self._logger.error(f"Failed to decode audio data: {e}")
            raise ValueError(f"Failed to process audio data: {str(e)}")

    @log_execution_time("Applying VAD and optional diarization")
    def apply_diarization_with_vad(self, audio_array, enable_diarization):
        """Apply VAD first, then optionally speaker diarization on speech segments, then create speaker-aware chunks for Whisper processing.

        This method provides flexible audio preprocessing with two modes:
        1. VAD + Diarization (default): Detects speech segments and identifies speakers
        2. VAD-only: Detects speech segments without speaker identification (faster)

        Args:
            audio_array (np.ndarray): The audio data as numpy array
            enable_diarization (bool): If True, applies speaker diarization. If False, uses only VAD for speech detection.

        Returns:
            list: List of audio chunks with timing and speaker information for Whisper processing

        Raises:
            RuntimeError: If VAD model is not available and no fallback is possible
        """
        # Step 1: Apply VAD to detect speech segments
        vad_speech_segments = self._apply_vad(audio_array)

        if enable_diarization:
            # Step 2: Apply diarization (will run on entire audio but can be filtered by VAD results)
            if self._diarization_model is None:
                self._logger.warning(
                    "Diarization requested but model not available - falling back to VAD-only mode"
                )
                enable_diarization = False
            else:
                diarization_segments = self._apply_diarization(audio_array)

                # Step 4: Filter diarization segments using VAD results (if VAD was successful)
                if vad_speech_segments is not None and len(vad_speech_segments) > 0:
                    filtered_segments = self._filter_diarization_with_vad(
                        diarization_segments, vad_speech_segments
                    )
                    vad_segments = filtered_segments
                else:
                    # Use diarization results directly if VAD failed or found no speech
                    vad_segments = diarization_segments

        if not enable_diarization:
            # VAD-only mode: use VAD segments without speaker identification
            self._logger.info("Using VAD-only mode (no speaker diarization)")
            vad_segments = []
            if vad_speech_segments is not None and len(vad_speech_segments) > 0:
                for i, vad_seg in enumerate(vad_speech_segments):
                    vad_segments.append(
                        {
                            "start": vad_seg.get("start", 0),
                            "end": vad_seg.get("end", 0),
                            "text": "",  # TT-Metal will fill this
                            "speaker": "SPEAKER_00",  # Default speaker for VAD-only mode
                        }
                    )

        if not vad_segments:
            # Fallback: create single segment for entire audio
            vad_segments = [
                {
                    "start": 0.0,
                    "end": len(audio_array) / settings.default_sample_rate,
                    "text": "",
                    "speaker": "SPEAKER_00",
                }
            ]

        normalized_segments = self._normalize_speaker_ids(vad_segments)

        whisper_chunks = self._merge_vad_segments_by_speaker_and_duration(
            normalized_segments
        )

        if enable_diarization:
            self._logger.info(
                f"VAD + Diarization detected {len(vad_segments)} segments, created {len(whisper_chunks)} speaker-aware chunks for Whisper"
            )
        else:
            self._logger.info(
                f"VAD-only detected {len(vad_segments)} speech segments, created {len(whisper_chunks)} chunks for Whisper"
            )

        return whisper_chunks

    @log_execution_time("Merging VAD segments by speaker and duration")
    def _merge_vad_segments_by_speaker_and_duration(
        self, vad_segments, target_chunk_duration=None
    ):
        """
        Create speaker-aware chunks for Whisper processing that balance speaker boundaries with optimal chunk sizes.
        Respects speaker boundaries while ensuring reasonable chunk durations for Whisper performance.
        """
        if not vad_segments:
            return []

        if target_chunk_duration is None:
            target_chunk_duration = settings.audio_chunk_duration_seconds

        chunks = []
        current_chunk_start = vad_segments[0]["start"]
        current_chunk_end = vad_segments[0]["end"]
        current_speaker = vad_segments[0]["speaker"]

        for segment in vad_segments[1:]:
            potential_end = segment["end"]
            potential_duration = potential_end - current_chunk_start

            # Finalize chunk if:
            # 1. Speaker changes (always respect speaker boundaries), OR
            # 2. Would exceed target duration
            should_finalize = (
                segment["speaker"] != current_speaker
                or potential_duration > target_chunk_duration
            )

            if should_finalize:
                chunks.append(
                    {
                        "start": current_chunk_start,
                        "end": current_chunk_end,
                        "text": "",
                        "speaker": current_speaker,
                    }
                )

                # Start new chunk
                current_chunk_start = segment["start"]
                current_chunk_end = segment["end"]
                current_speaker = segment["speaker"]
            else:
                # Add segment to current chunk (same speaker only)
                current_chunk_end = segment["end"]

        # Add final chunk
        if current_chunk_start < current_chunk_end:
            chunks.append(
                {
                    "start": current_chunk_start,
                    "end": current_chunk_end,
                    "text": "",
                    "speaker": current_speaker,
                }
            )

        self._logger.info(f"Created {len(chunks)} Whisper chunks")
        return chunks

    def _initialize_audio_worker(self):
        """Spawn the persistent audio venv worker.

        Called once from :meth:`__init__` when audio preprocessing is
        enabled. Blocks until the worker signals ready (models fully
        loaded) so the first real request doesn't pay the cold-start cost.

        On failure we degrade gracefully: leave ``_diarization_model`` and
        ``_vad_model`` as ``None`` so downstream code falls back to
        VAD-only or no-preprocessing paths, and log actionable next steps.
        """
        self._diarization_model_name = (
            settings.preprocessing_model_weights_path
            or SupportedModels.PYANNOTE_SPEAKER_DIARIZATION.value
        )

        try:
            self._logger.info(
                f"Starting audio venv worker (model: {self._diarization_model_name})..."
            )
            worker = AudioVenvWorker(
                logger=self._logger,
                model_name=self._diarization_model_name,
                hf_token=os.getenv("HF_TOKEN"),
            )
            worker.start()
        except Exception as e:
            self._logger.warning(
                f"Failed to start audio worker: {e}. "
                "Continuing without audio preprocessing"
            )
            self._audio_venv_worker = None
            self._diarization_model = None
            self._diarization_model_name = None
            self._vad_model = None

            self._logger.info("To enable audio preprocessing:")
            self._logger.info(
                "1. Ensure HF_TOKEN is set or set HF_HOME to your Hugging Face cache directory."
            )
            self._logger.info(
                "2. Accept model terms at: https://hf.co/pyannote/speaker-diarization-3.0 and https://hf.co/pyannote/segmentation-3.0"
            )
            return

        self._audio_venv_worker = worker
        self._diarization_model = True
        self._vad_model = True

        # Best-effort cleanup on interpreter shutdown so the child doesn't
        # outlive the parent. `CpuWorkloadHandler.stop_workers` terminates
        # the outer worker process anyway; this is belt-and-braces for
        # direct-instantiation paths (tests, notebooks).
        if not self._atexit_registered:
            atexit.register(self.close)
            self._atexit_registered = True

        self._logger.info(
            f"Audio venv worker ready (diarization: {self._diarization_model_name})"
        )

    @log_execution_time("Applying diarization")
    def _apply_diarization(self, audio_array):
        """Run diarization on the persistent audio venv worker."""
        if self._audio_venv_worker is None:
            self._logger.warning(
                "Diarization requested but audio worker not available, returning no segments"
            )
            return []

        self._logger.info("Performing speaker diarization via audio venv worker...")

        segments = self._audio_venv_worker.run(
            mode="diarize",
            audio_array=audio_array,
            timeout_seconds=self._DIARIZATION_TIMEOUT_SECONDS,
        )

        if segments is None:
            return []

        diarization_segments = [
            {
                "start": seg.get("start", 0),
                "end": seg.get("end", 0),
                "text": "",
                "speaker": seg.get("speaker", "SPEAKER_00"),
            }
            for seg in segments
        ]
        self._logger.info(f"Diarization found {len(diarization_segments)} segments")
        return diarization_segments

    @log_execution_time("Applying VAD")
    def _apply_vad(self, audio_array):
        """Run VAD on the persistent audio venv worker."""
        if self._vad_model is None or self._audio_venv_worker is None:
            self._logger.warning("VAD model not available, skipping VAD step")
            return None

        self._logger.info("Applying VAD via audio venv worker...")

        segments = self._audio_venv_worker.run(
            mode="vad",
            audio_array=audio_array,
            timeout_seconds=self._VAD_TIMEOUT_SECONDS,
        )

        if segments is None:
            return None

        self._logger.info(f"VAD detected {len(segments)} speech segments")
        return segments

    def _filter_diarization_with_vad(self, diarization_segments, vad_segments):
        """Filter diarization segments to only include those that overlap with VAD-detected speech."""
        filtered_segments = []

        for diar_seg in diarization_segments:
            diar_start, diar_end = diar_seg["start"], diar_seg["end"]

            # Check if this diarization segment overlaps with any VAD segment
            for vad_seg in vad_segments:
                vad_start = vad_seg.get("start", 0)
                vad_end = vad_seg.get("end", 0)

                # Check for overlap
                overlap_start = max(diar_start, vad_start)
                overlap_end = min(diar_end, vad_end)

                if overlap_start < overlap_end:  # There is overlap
                    # Create a new segment with the overlapping region
                    filtered_segments.append(
                        {
                            "start": overlap_start,
                            "end": overlap_end,
                            "text": diar_seg["text"],
                            "speaker": diar_seg["speaker"],
                        }
                    )
                    break  # Move to next diarization segment

        self._logger.info(
            f"Filtered {len(diarization_segments)} diarization segments to {len(filtered_segments)} using VAD"
        )
        return filtered_segments

    def _validate_file_size(self, audio_bytes):
        if len(audio_bytes) > settings.max_audio_size_bytes:
            raise ValueError(
                f"Audio file too large: {len(audio_bytes)} bytes. Maximum allowed: {settings.max_audio_size_bytes} bytes"
            )

    @log_execution_time("Converting to audio array")
    def _convert_to_audio_array(self, audio_bytes):
        """Convert audio file bytes (WAV/MP3) to numpy array."""

        # Detect file format based on headers
        if (
            len(audio_bytes) >= 12
            and audio_bytes[:4] == b"RIFF"
            and audio_bytes[8:12] == b"WAVE"
        ):
            self._logger.info("Processing WAV file format")
            return self._decode_wav_file(audio_bytes)
        elif len(audio_bytes) >= 3 and (
            audio_bytes[:3] == b"ID3"
            or audio_bytes[:2] == b"\xff\xfb"
            or audio_bytes[:2] == b"\xff\xf3"
            or audio_bytes[:2] == b"\xff\xf2"
        ):
            self._logger.info("Processing MP3 file format")
            return self._decode_audio_file(audio_bytes, "MP3")
        else:
            raise ValueError(
                "Unsupported audio format. Only WAV and MP3 files are supported."
            )

    @log_execution_time("Decoding WAV file")
    def _decode_wav_file(self, audio_bytes):
        try:
            # Parse WAV file manually
            if len(audio_bytes) < 44:
                raise ValueError("WAV file too short")

            # Read WAV header
            riff = audio_bytes[0:4]
            wave = audio_bytes[8:12]
            fmt = audio_bytes[12:16]

            if riff != b"RIFF" or wave != b"WAVE" or fmt != b"fmt ":
                raise ValueError("Invalid WAV file format")

            # Parse format chunk
            fmt_size = struct.unpack("<I", audio_bytes[16:20])[0]
            audio_format = struct.unpack("<H", audio_bytes[20:22])[0]
            num_channels = struct.unpack("<H", audio_bytes[22:24])[0]
            sample_rate = struct.unpack("<I", audio_bytes[24:28])[0]
            bits_per_sample = struct.unpack("<H", audio_bytes[34:36])[0]

            self._logger.info(
                f"WAV format: {num_channels} channels, {sample_rate} Hz, {bits_per_sample} bits"
            )

            # Find data chunk
            pos = 20 + fmt_size
            while pos < len(audio_bytes) - 8:
                chunk_id = audio_bytes[pos : pos + 4]
                chunk_size = struct.unpack("<I", audio_bytes[pos + 4 : pos + 8])[0]

                if chunk_id == b"data":
                    # Found audio data
                    audio_data = audio_bytes[pos + 8 : pos + 8 + chunk_size]
                    break

                pos += 8 + chunk_size
            else:
                raise ValueError("No audio data found in WAV file")

            # Convert audio data to numpy array
            if bits_per_sample == 16:
                audio_array = (
                    np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )
            elif bits_per_sample == 24:
                # Handle 24-bit audio
                audio_ints = []
                # WAV files are typically little-endian
                endian = "<"
                for i in range(0, len(audio_data), 3):
                    if i + 3 <= len(audio_data):  # Ensure all three bytes are available
                        # Convert 3 bytes to int24 with proper endianness
                        if endian == "<":
                            byte_data = (
                                audio_data[i : i + 3] + b"\x00"
                            )  # Pad to 4 bytes (LSB)
                        else:
                            byte_data = (
                                b"\x00" + audio_data[i : i + 3]
                            )  # Pad to 4 bytes (MSB)
                        val = (
                            struct.unpack(endian + "i", byte_data)[0] >> 8
                        )  # Shift back
                        audio_ints.append(val)
                audio_array = np.array(audio_ints, dtype=np.float32) / 8388608.0  # 2^23
            elif bits_per_sample == 32:
                if audio_format == 3:  # IEEE float
                    audio_array = np.frombuffer(audio_data, dtype=np.float32)
                else:  # 32-bit PCM
                    audio_array = (
                        np.frombuffer(audio_data, dtype=np.int32).astype(np.float32)
                        / 2147483648.0
                    )
            else:
                raise ValueError(f"Unsupported bit depth: {bits_per_sample}")

            # Convert stereo to mono if needed
            if num_channels == 2:
                audio_array = audio_array.reshape(-1, 2).mean(axis=1)
            elif num_channels > 2:
                audio_array = audio_array.reshape(-1, num_channels).mean(axis=1)

            # Resample to default sample rate if needed (simple linear interpolation)
            if sample_rate != settings.default_sample_rate:
                target_length = int(
                    len(audio_array) * settings.default_sample_rate / sample_rate
                )
                indices = np.linspace(0, len(audio_array) - 1, target_length)
                audio_array = np.interp(
                    indices, np.arange(len(audio_array)), audio_array
                )

            self._logger.info(
                f"Loaded WAV: {len(audio_array)} samples, duration: {len(audio_array) / settings.default_sample_rate:.2f}s"
            )
            return audio_array.astype(np.float32)

        except Exception as e:
            self._logger.error(f"Failed to decode WAV file: {e}")
            raise ValueError(f"Could not decode WAV file: {str(e)}")

    @log_execution_time("Decoding audio file")
    def _decode_audio_file(self, audio_bytes, format_name):
        """Convert ffmpeg-supported audio formats (MP3, MP4, FLAC, OGG, etc.) to WAV using shared ffmpeg_utils, then decode with _decode_wav_file."""
        try:
            self._logger.info(
                f"Converting {format_name} to WAV using ffmpeg (in-memory)..."
            )
            wav_bytes = ffmpeg_decode_to_wav(
                audio_bytes, sample_rate=settings.default_sample_rate
            )
            self._logger.info(
                f"{format_name} to WAV conversion completed successfully (in-memory)"
            )
            return self._decode_wav_file(wav_bytes)
        except subprocess.CalledProcessError as e:
            self._logger.error(f"ffmpeg conversion failed: {e}")
            raise ValueError(
                f"{format_name} conversion failed. Ensure ffmpeg is installed and accessible."
            )
        except Exception as e:
            self._logger.error(f"Failed to decode {format_name} file: {e}")
            raise ValueError(f"Could not decode {format_name} file: {str(e)}")

    def _validate_and_truncate_duration(self, audio_array, should_preprocess):
        duration_seconds = len(audio_array) / settings.default_sample_rate

        # Use extended duration limit when preprocessing is allowed and requested
        max_duration = (
            settings.max_audio_duration_with_preprocessing_seconds
            if should_preprocess and self._diarization_model is not None
            else settings.max_audio_duration_seconds
        )

        if duration_seconds > max_duration:
            max_samples = int(max_duration * settings.default_sample_rate)
            self._logger.warning(
                f"Audio truncated from {duration_seconds:.2f}s to {max_duration}s"
            )
            return audio_array[:max_samples], max_duration
        return audio_array, duration_seconds

    @log_execution_time("Normalizing speaker IDs")
    def _normalize_speaker_ids(self, segments):
        """
        Normalize speaker IDs to ensure consistency across audio formats.
        Maps speakers based on chronological order of first appearance.
        """
        # Sort segments by start time to ensure chronological processing
        segments = sorted(segments, key=lambda x: x.get("start", 0))

        # Assign normalized IDs in order of first appearance
        speaker_mapping = {}
        next_speaker_id = 0

        for segment in segments:
            speaker = segment["speaker"]

            # First time we see this speaker - assign normalized ID
            if speaker not in speaker_mapping:
                speaker_mapping[speaker] = f"SPEAKER_{next_speaker_id:02d}"
                next_speaker_id += 1

        self._logger.info(f"Speaker mapping: {speaker_mapping}")

        # Apply the normalized IDs to all segments
        normalized_segments = []
        for segment in segments:
            normalized_segment = segment.copy()
            original_speaker = segment["speaker"]
            normalized_segment["speaker"] = speaker_mapping[original_speaker]
            normalized_segments.append(normalized_segment)

        return normalized_segments


def combine_transcription_responses(
    responses: List[AudioTextResponse],
) -> AudioTextResponse:
    """
    Combine multiple AudioTextResponse objects into a single response.
    Returns combined response with summed duration and merged content
    """
    if not responses:
        raise ValueError("No transcription responses to combine")

    if len(responses) == 1:
        return responses[0]

    logger = TTLogger()

    # Combine text from all responses
    combined_text = " ".join(
        response.text.strip() for response in responses if response.text.strip()
    )

    # Sum up all durations
    total_duration = sum(response.duration for response in responses)

    # Combine segments if available
    combined_segments = []
    segment_id_counter = 1
    all_speakers = set()

    # Flatten all segments from responses into a single list
    all_segments = [
        segment
        for response in responses
        for segment in response.segments
        if response.segments
    ]

    for segment in all_segments:
        # Create new segment with updated ID to maintain sequence
        combined_segment = AudioTextSegment(
            id=segment_id_counter,
            speaker=segment.speaker,
            start_time=segment.start_time,
            end_time=segment.end_time,
            text=segment.text,
        )
        combined_segments.append(combined_segment)
        all_speakers.add(segment.speaker)
        segment_id_counter += 1

    # Combine speaker information
    combined_speakers = sorted(all_speakers) if all_speakers else None
    combined_speaker_count = len(all_speakers) if all_speakers else None

    # Create combined response
    combined_response = AudioTextResponse(
        text=combined_text,
        duration=total_duration,
        segments=combined_segments if combined_segments else None,
        speaker_count=combined_speaker_count,
        speakers=combined_speakers,
    )

    logger.info(
        f"Combined {len(responses)} transcription responses: "
        f"total_duration={total_duration:.2f}s, "
        f"total_segments={len(combined_segments)}, "
        f"speaker_count={combined_speaker_count}"
    )

    return combined_response
