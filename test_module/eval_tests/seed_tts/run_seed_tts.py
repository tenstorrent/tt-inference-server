# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Seed-TTS-Eval test-en against a Qwen3-TTS vLLM-Omni server.

Runs in the EVALS_SEED_TTS venv, launched by tts_eval_tests' seed_tts_eval task.
EvalScope's seed_tts_eval benchmark generates every clip and computes its own
WER, transcribing through an OpenAI-style /audio/transcriptions endpoint; this
process serves that endpoint with Whisper-large-v3, called exactly as
seed-tts-eval's run_wer.py calls it. seed-tts-eval's own scoring then runs over
the same transcripts and clips: WER under its text normalization, and SIM with
its WavLM-large speaker-verification model. Writes a JSON summary to --out.
"""

import argparse
import base64
import io
import json
import os
import string
import sys
import tempfile
import threading
import time
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ASR_MODEL = "openai/whisper-large-v3"
SV_MODEL = "wavlm_large"
MODEL_NAME = "qwen3-tts"
SUBSET = "en"


def official_wer(hypo: str, truth: str) -> float:
    """seed-tts-eval run_wer.py's process_one for English, on jiwer 4."""
    from jiwer import process_words
    from zhon.hanzi import punctuation

    for x in punctuation + string.punctuation:
        if x == "'":
            continue
        truth = truth.replace(x, "")
        hypo = hypo.replace(x, "")
    truth = truth.replace("  ", " ").lower()
    hypo = hypo.replace("  ", " ").lower()
    return process_words(truth, hypo).wer


class WhisperASR:
    """Whisper-large-v3 English transcription as run_wer.py runs it."""

    def __init__(self):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.processor = WhisperProcessor.from_pretrained(ASR_MODEL)
        self.model = WhisperForConditionalGeneration.from_pretrained(ASR_MODEL).eval()
        # One call at a time: torch already spreads each call over every core.
        self.lock = threading.Lock()
        self.calls = 0
        self.seconds = 0.0

    def transcribe(self, wav_bytes: bytes) -> str:
        import scipy.signal
        import soundfile as sf
        import torch

        wav, sr = sf.read(io.BytesIO(wav_bytes))
        if wav.ndim > 1:
            wav = wav[:, 0]
        if sr != 16000:
            wav = scipy.signal.resample(wav, int(len(wav) * 16000 / sr))
        features = self.processor(
            wav, sampling_rate=16000, return_tensors="pt"
        ).input_features
        forced = self.processor.get_decoder_prompt_ids(
            language="english", task="transcribe"
        )
        with self.lock:
            start = time.monotonic()
            with torch.no_grad():
                ids = self.model.generate(features, forced_decoder_ids=forced)
            self.calls += 1
            self.seconds += time.monotonic() - start
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0]


def _parse_form(handler) -> tuple:
    body = handler.rfile.read(int(handler.headers["Content-Length"]))
    head = f"Content-Type: {handler.headers['Content-Type']}\r\n\r\n".encode()
    message = BytesParser(policy=default).parsebytes(head + body)
    fields, audio = {}, None
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if name == "file":
            audio = part.get_payload(decode=True)
        else:
            fields[name] = part.get_content().strip()
    return fields, audio


def serve_asr(asr: WhisperASR) -> ThreadingHTTPServer:
    """Serve /v1/audio/transcriptions on a free localhost port, in a thread."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, payload):
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if not self.path.endswith("/audio/transcriptions"):
                self._send(404, {"error": f"no route {self.path}"})
                return
            fields, audio = _parse_form(self)
            language = fields.get("language", "en")
            if not language.startswith("en") or audio is None:
                self._send(
                    400, {"error": f"English audio only; got language={language!r}"}
                )
                return
            self._send(200, {"text": asr.transcribe(audio)})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def run_evalscope(api_url: str, work_dir: Path, concurrency: int, limit) -> None:
    import omni_tts_api
    from evalscope import run_task
    from evalscope.config import TaskConfig

    run_task(
        task_cfg=TaskConfig(
            model=MODEL_NAME,
            api_url=api_url,
            api_key="EMPTY",
            eval_type=omni_tts_api.EVAL_TYPE,
            datasets=["seed_tts_eval"],
            dataset_args={"seed_tts_eval": {"subset_list": [SUBSET]}},
            eval_batch_size=concurrency,
            limit=limit,
            work_dir=str(work_dir),
            no_timestamp=True,
        )
    )


def score_official(rows: list, seed_tts_eval_dir: Path, checkpoint: Path) -> tuple:
    """(official WERs, SIMs), one per row, with seed-tts-eval's scorers."""
    sys.path.insert(
        0,
        str(
            seed_tts_eval_dir / "thirdparty/UniSpeech/downstreams/speaker_verification"
        ),
    )
    from verification import init_model, verification

    model = init_model(SV_MODEL, str(checkpoint)).eval()
    wers, sims = [], []
    with tempfile.TemporaryDirectory() as tmp:
        prompt = Path(tmp) / "prompt.wav"
        for row in rows:
            score, meta = (
                row["sample_score"]["score"],
                row["sample_score"]["sample_metadata"],
            )
            wers.append(
                official_wer(score["metadata"].get("transcription", ""), meta["text"])
            )
            # The sample's prompt clip, byte-identical to seed-tts-eval's
            # prompt-wavs file, rides along in the review as a data URI.
            clip = next(
                c["audio"]
                for c in row["messages"][0]["content"]
                if c["type"] == "audio"
            )
            prompt.write_bytes(base64.b64decode(clip.split(",", 1)[1]))
            sim, model = verification(
                SV_MODEL,
                meta["generated_audio_path"],
                str(prompt),
                use_gpu=False,
                checkpoint=str(checkpoint),
                model=model,
            )
            sims.append(sim.item())
    return wers, sims


def _mean(values: list):
    return sum(values) / len(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", required=True, help="server base URL with /v1")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-tts-eval-dir", type=Path, required=True)
    parser.add_argument("--sv-checkpoint", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--limit",
        type=lambda v: float(v) if "." in v else int(v),
        default=None,
        help="first N samples, or a fraction of them (EvalScope's limit)",
    )
    args = parser.parse_args()

    asr = WhisperASR()
    server = serve_asr(asr)
    os.environ["SEED_TTS_EVAL_ASR_API_BASE"] = (
        f"http://127.0.0.1:{server.server_address[1]}/v1"
    )
    os.environ["SEED_TTS_EVAL_ASR_API_KEY"] = "EMPTY"
    start = time.monotonic()
    run_evalscope(args.api_url, args.work_dir, args.concurrency, args.limit)
    evalscope_seconds = time.monotonic() - start
    server.shutdown()

    reviews = args.work_dir / "reviews" / MODEL_NAME / f"seed_tts_eval_{SUBSET}.jsonl"
    rows = [json.loads(line) for line in reviews.read_text().splitlines() if line]
    evalscope_wers = [r["sample_score"]["score"]["value"]["audio_wer"] for r in rows]
    start = time.monotonic()
    wers, sims = score_official(rows, args.seed_tts_eval_dir, args.sv_checkpoint)
    args.out.write_text(
        json.dumps(
            {
                "subset": f"test-{SUBSET}",
                "sample_count": len(rows),
                "concurrency": args.concurrency,
                "wer_official": _mean(wers),
                "wer_evalscope": _mean(evalscope_wers),
                "mean_sim": _mean(sims),
                "asr_model": ASR_MODEL,
                "evalscope_seconds": round(evalscope_seconds, 1),
                "asr_seconds": round(asr.seconds, 1),
                "sim_seconds": round(time.monotonic() - start, 1),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
