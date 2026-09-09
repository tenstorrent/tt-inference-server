# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from __future__ import annotations

import base64
import os
from typing import TYPE_CHECKING

import aiohttp

from report_module.schema import Block

from .._test_common import BaseTest, TestConfig

if TYPE_CHECKING:
    from ..context import MediaContext

DEFAULT_API_KEY = "your-secret-key"

HEADERS = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.getenv('API_KEY', DEFAULT_API_KEY)}",
}

REQUEST_TIMEOUT_S = 120


class XttsV2TTSTest(BaseTest):
    """XTTS-v2 functionality: basic synthesis, seed reproducibility, language handling.

    XTTS-v2 differs from SpeechT5 in the contract points this test pins down:
    sampling is stochastic unless ``seed`` is fixed (same seed + text must reproduce
    identical audio), and the ``language`` request field selects one of 17 languages
    (region variants normalize, unsupported codes are rejected at the API with 422).
    """

    KIND = "xtts_v2_tts"
    TASK_TYPE = "integration"

    async def _run_specific_test_async(self):
        results = {"success": False}
        subtests = (
            ("basic_tts", self._test_basic_tts),
            ("seed_reproducibility", self._test_seed_reproducibility),
            ("language_synthesis", self._test_language_synthesis),
            ("language_rejection", self._test_language_rejection),
            ("voice_cloning", self._test_voice_cloning),
        )
        ok = True
        for name, fn in subtests:
            try:
                results[name] = await fn()
                ok = ok and results[name].get("status") == "success"
            except Exception as e:
                results[name] = {"error": str(e)}
                ok = False
        results["success"] = ok
        return results

    async def _post(self, session, payload):
        url = f"{self.base_url}/v1/audio/speech"
        async with session.post(url, json=payload) as response:
            body = await response.json() if response.status == 200 else None
            return response.status, body

    def _decoded_audio(self, result):
        """Validate the JSON response shape and return the decoded audio bytes."""
        for field in ("audio", "duration", "sample_rate", "format"):
            assert field in result, f"Response should contain {field!r} field"
        audio_bytes = base64.b64decode(result["audio"])
        assert len(audio_bytes) > 0, "Decoded audio should not be empty"
        assert result["duration"] > 0, (
            f"Duration should be positive, got {result['duration']}"
        )
        return audio_bytes

    async def _test_basic_tts(self):
        payload = {
            "text": "Hello world, this is a test of XTTS text to speech synthesis.",
            "response_format": "json",
            "seed": 0,
        }
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)
        async with aiohttp.ClientSession(headers=HEADERS, timeout=timeout) as session:
            status, result = await self._post(session, payload)
            assert status == 200, f"Expected status 200, got {status}"
            audio_bytes = self._decoded_audio(result)
            assert result["duration"] < 30, (
                f"Duration seems too long for test text, got {result['duration']}s"
            )
            return {
                "status": "success",
                "duration": result["duration"],
                "sample_rate": result["sample_rate"],
                "format": result["format"],
                "audio_size_bytes": len(audio_bytes),
            }

    async def _test_seed_reproducibility(self):
        """Identical text + seed must produce byte-identical audio; XTTS samples, so
        this is the contract that makes its output testable at all."""
        payload = {
            "text": "Reproducibility is the foundation of every regression test.",
            "response_format": "json",
            "seed": 1234,
        }
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)
        async with aiohttp.ClientSession(headers=HEADERS, timeout=timeout) as session:
            status_a, result_a = await self._post(session, payload)
            status_b, result_b = await self._post(session, payload)
            assert status_a == 200 and status_b == 200, (
                f"Expected 200/200, got {status_a}/{status_b}"
            )
            self._decoded_audio(result_a)
            self._decoded_audio(result_b)
            assert result_a["audio"] == result_b["audio"], (
                "Same text and seed produced different audio"
            )
            return {
                "status": "success",
                "audio_size_bytes": len(base64.b64decode(result_a["audio"])),
            }

    async def _test_language_synthesis(self):
        """One non-Latin language (zh: pypinyin romanizer + CJK chunking) and one
        region-variant code (pt-BR must normalize to pt and be accepted)."""
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)
        sizes = {}
        async with aiohttp.ClientSession(headers=HEADERS, timeout=timeout) as session:
            for lang, text in (
                ("zh", "港口的灯光一盏一盏地亮了起来。"),
                ("pt-BR", "As luzes do porto acenderam uma a uma."),
            ):
                status, result = await self._post(
                    session,
                    {
                        "text": text,
                        "language": lang,
                        "response_format": "json",
                        "seed": 0,
                    },
                )
                assert status == 200, f"language={lang!r}: expected 200, got {status}"
                sizes[lang] = len(self._decoded_audio(result))
        return {"status": "success", "audio_size_bytes": sizes}

    async def _test_voice_cloning(self):
        """Per-request voice cloning via reference_audio: synthesize once with the
        default voice, then feed that WAV back as the reference clip (self-clone).
        Same text+seed with a different voice must produce different audio, and a
        repeat with the same reference must reproduce it exactly (voice cache +
        seeded sampling)."""
        text = "The lighthouse keeper kept a careful log of every passing ship."
        base = {"text": text, "response_format": "json", "seed": 99}
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)
        async with aiohttp.ClientSession(headers=HEADERS, timeout=timeout) as session:
            status, default_result = await self._post(session, base)
            assert status == 200, f"default-voice request failed: {status}"
            self._decoded_audio(default_result)

            cloned_payload = {**base, "reference_audio": default_result["audio"]}
            status_a, clone_a = await self._post(session, cloned_payload)
            assert status_a == 200, f"cloned-voice request failed: {status_a}"
            self._decoded_audio(clone_a)
            assert clone_a["audio"] != default_result["audio"], (
                "cloned voice produced identical audio to the default voice"
            )

            status_b, clone_b = await self._post(session, cloned_payload)
            assert status_b == 200, f"repeat cloned-voice request failed: {status_b}"
            assert clone_b["audio"] == clone_a["audio"], (
                "same reference clip + seed did not reproduce identical audio"
            )
        return {
            "status": "success",
            "audio_size_bytes": len(base64.b64decode(clone_a["audio"])),
        }

    async def _test_language_rejection(self):
        """An unsupported language code must 422 at the API, before any device work."""
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)
        async with aiohttp.ClientSession(headers=HEADERS, timeout=timeout) as session:
            url = f"{self.base_url}/v1/audio/speech"
            async with session.post(
                url, json={"text": "hello", "language": "xx", "response_format": "json"}
            ) as response:
                assert response.status == 422, (
                    f"Unsupported language should 422, got {response.status}"
                )
        return {"status": "success"}


def run_xtts_v2_tts(ctx: MediaContext) -> Block:
    """Run XttsV2TTSTest under ``ctx`` and return its Block."""
    test_config = TestConfig(
        {
            "timeout": 300,
            "retry_attempts": 2,
            "retry_delay": 5,
            "break_on_failure": False,
        }
    )
    return XttsV2TTSTest(test_config, targets={}, ctx=ctx).run_tests()


__all__ = ["XttsV2TTSTest", "run_xtts_v2_tts"]
