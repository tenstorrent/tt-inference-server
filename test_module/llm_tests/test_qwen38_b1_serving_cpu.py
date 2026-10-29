# SPDX-License-Identifier: Apache-2.0
"""CPU checks for token timing completeness and cold-prefill accounting."""

import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    "b1_serving", Path(__file__).with_name("qwen38_b1_serving.py")
)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class StreamingEvidenceTest(unittest.TestCase):
    def run_stream(self, *, count=128, cached=0):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        instance = probe.Probe("http://unused", "model", directory.name)
        events = [dict(choices=[dict(token_ids=[i], text="x")]) for i in range(count)]
        events.append(
            dict(
                choices=[],
                usage=dict(
                    prompt_tokens=128,
                    completion_tokens=count,
                    prompt_tokens_details=dict(cached_tokens=cached),
                ),
            )
        )
        raw = b"".join(
            b"data: " + json.dumps(event).encode() + b"\n\n" for event in events
        )
        raw += b"data: [DONE]\n\n"
        with patch.object(instance, "request", return_value=io.BytesIO(raw)):
            return instance.timed([1] * 128, 0)

    def test_complete_stream_measures127_intervals(self):
        row = self.run_stream()
        self.assertEqual(len(row["tokens"]), 128)
        self.assertEqual(len(row["events"]), 128)
        self.assertGreater(row["ttft_ms"], 0)
        self.assertGreater(row["tpot_ms"], 0)
        self.assertAlmostEqual(
            row["decode_tokens_per_second_per_user"], 1000 / row["tpot_ms"]
        )

    def test_short_stream_is_not_timing_evidence(self):
        with self.assertRaisesRegex(ValueError, "Incomplete streamed"):
            self.run_stream(count=127)

    def test_cache_hit_rejects_cold_prefill_row(self):
        with self.assertRaisesRegex(ValueError, "prefix cache disabled"):
            self.run_stream(cached=64)

    def test_shared_qualitative_suite_is_complete(self):
        self.assertEqual(len(probe.QUALITATIVE_PROMPTS), 6)
        self.assertTrue(any("French" in prompt for prompt in probe.QUALITATIVE_PROMPTS))
        self.assertTrue(any("Python" in prompt for prompt in probe.QUALITATIVE_PROMPTS))


if __name__ == "__main__":
    unittest.main()
