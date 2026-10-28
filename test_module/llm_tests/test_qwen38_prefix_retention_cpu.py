# SPDX-License-Identifier: Apache-2.0
"""CPU checks for prefix identity, exact accounting, and explicit unknowns."""

import importlib.util
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "retention", Path(__file__).with_name("qwen38_prefix_retention.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
CONFIG = dict(
    schema_version=1,
    block_size=512,
    base_lengths=[16384],
    streams=32,
    concurrency=32,
    append_tokens=512,
    churn_streams=8,
    tool_gap_s=1,
    request_budget_s=3600,
)


class ProbeTest(unittest.TestCase):
    def test_lcp_counts_token_identity_not_label_or_total_length(self):
        self.assertEqual(module.lcp([1, 2, 3], [1, 2, 4, 5]), 2)
        self.assertEqual(module.lcp([1, 2, 3], [1, 2, 3, 5]), 3)
        self.assertEqual(module.lcp([7], [8]), 0)

    def test_prometheus_keeps_engine_labels_and_histogram_names(self):
        rows = module.parse_metrics(
            'vllm:num_requests_running{engine="0"} 3\n'
            'vllm:num_requests_running{engine="1"} 5\n'
            "vllm:request_queue_time_seconds_sum 2.5\n"
            "unrelated 9\n# HELP ignored\n"
        )
        self.assertEqual(len(rows), 3)
        self.assertNotEqual(rows[0]["series"], rows[1]["series"])
        self.assertEqual(rows[2]["value"], 2.5)

    def test_cached_denominator_uses_aligned_common_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            probe = module.Probe(
                base_url="http://unused", model="model", config=CONFIG, output_dir=tmp
            )
            previous = list(range(1030))
            current = previous[:1029] + [90000] * 513
            probe.http = lambda *args, **kwargs: dict(
                usage=dict(
                    prompt_tokens=len(current),
                    completion_tokens=1,
                    prompt_tokens_details=dict(cached_tokens=1024),
                ),
                choices=[dict(token_ids=[42])],
            )
            row = probe.call(
                panel=0,
                stream=0,
                turn=1,
                tokens=current,
                salt="same-stream",
                previous=previous,
                phase="warm",
            )
            self.assertEqual(row["lcp_tokens"], 1029)
            self.assertEqual(row["reusable_aligned_tokens"], 1024)
            self.assertEqual(row["retention_fraction_of_eligible"], 1)
            self.assertEqual(row["recomputed_prompt_tokens"], len(current) - 1024)

    def test_missing_cached_count_fails_instead_of_reporting_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            probe = module.Probe(
                base_url="http://unused", model="model", config=CONFIG, output_dir=tmp
            )
            probe.http = lambda *args, **kwargs: dict(
                usage=dict(prompt_tokens=3, completion_tokens=1),
                choices=[dict(token_ids=[42])],
            )
            with self.assertRaisesRegex(RuntimeError, "cached-prompt"):
                probe.call(
                    panel=0,
                    stream=0,
                    turn=0,
                    tokens=[1, 2, 3],
                    salt="private",
                    phase="cold",
                )
            self.assertEqual(probe.outstanding, 0)
            self.assertIn("error", probe.calls[0])

    def test_real_http_fixed_work_records_nine_calls_and_exact_identity(self):
        cached = {}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def respond(self, body, raw=False):
                body = body.encode() if raw else json.dumps(body).encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/v1/models":
                    self.respond(dict(data=[dict(id="model", max_model_len=262144)]))
                else:
                    self.respond(
                        "vllm:num_requests_running 0\nvllm:num_requests_waiting 0\nvllm:num_preemptions_total 0\n",
                        True,
                    )

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/tokenize":
                    words = body["messages"][0]["content"].split()
                    ids = (
                        [1, 2]
                        + [
                            100 + int(hashlib.sha256(w.encode()).hexdigest()[:8], 16)
                            for w in words
                        ]
                        + [3, 4]
                    )
                    self.respond(dict(tokens=ids))
                else:
                    ids, salt = body["prompt"], body["cache_salt"]
                    reuse = module.lcp(cached.get(salt, []), ids) // 512 * 512
                    cached[salt] = ids
                    self.respond(
                        dict(
                            usage=dict(
                                prompt_tokens=len(ids),
                                completion_tokens=1,
                                prompt_tokens_details=dict(cached_tokens=reuse),
                            ),
                            choices=[dict(token_ids=[42])],
                        )
                    )

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                config = dict(
                    CONFIG,
                    streams=2,
                    concurrency=2,
                    base_lengths=[1024],
                    churn_streams=1,
                    tool_gap_s=0,
                )
                probe = module.Probe(
                    base_url=f"http://127.0.0.1:{server.server_port}",
                    model="model",
                    config=config,
                    output_dir=tmp,
                )
                result = probe.run()
                self.assertEqual(result["status"], "pass")
                self.assertEqual(result["request_count"], 9)
                self.assertEqual(result["completion_tokens"], 9)
                saved = json.loads((probe.root / "inputs.json").read_text())
                self.assertEqual(len(saved["prompts"]), 7)
                self.assertEqual(
                    {p["sha256"] for p in saved["prompts"]},
                    {c["input_sha256"] for c in probe.calls},
                )
                self.assertTrue(
                    all(
                        c["cached_prompt_tokens"] == 0
                        for c in probe.calls
                        if c["phase"] in ("cold", "churn")
                    )
                )
                self.assertTrue(
                    all(
                        c["lcp_tokens"] > 1000
                        for c in probe.calls
                        if c["phase"] == "warm_after_gap"
                    )
                )
                self.assertEqual(
                    [c["prompt_tokens"] for c in probe.calls].count(1024), 3
                )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_rejects_unbounded_configuration(self):
        with self.assertRaises(ValueError):
            module.validate_config(dict(CONFIG, request_budget_s=3601))


if __name__ == "__main__":
    unittest.main()
