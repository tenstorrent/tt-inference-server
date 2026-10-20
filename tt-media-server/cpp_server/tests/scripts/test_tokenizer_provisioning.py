# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Exercise tokenizer fetching and both image startup paths without network access."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[4]
FETCH = REPO / "tt-media-server/cpp_server/scripts/fetch_tokenizers.sh"
MODEL = "meta-llama/Llama-3.2-1B-Instruct"
ASSETS = ("tokenizer.json", "tokenizer_config.json", "config.json")


class TokenizerProvisioningTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.tokens = self.root / "cpp_server/tokenizers"
        self.events = self.root / "events"
        # A minimal environment prevents host credentials/configuration leaking
        # into these tests. The fake downloader cannot access the network.
        self.env = {
            "PATH": str(self.bin) + os.pathsep + os.defpath,
            "HF_TOKEN": "test-token",
            "TEST_EVENTS": str(self.events),
            "HTTP_PORT": "8000",
            "DYN_DISCOVERY_BACKEND": "etcd",
            "ETCD_ENDPOINTS": "http://test-etcd:2379",
            "DYN_REQUEST_PLANE": "tcp",
            "DYN_EVENT_PLANE": "zmq",
            "MODEL_NAME": "test-model",
            "TOKENIZER_DIR": str(self.tokens),
        }
        self.executable(
            self.bin / "wget",
            f"#!{sys.executable}\n"
            "import os, pathlib, sys\n"
            "with open(os.environ['TEST_EVENTS'], 'a') as log:\n"
            "    log.write(sys.argv[-1] + '\\n')\n"
            "if os.environ.get('FAIL_DOWNLOAD'):\n"
            "    pathlib.Path(sys.argv[sys.argv.index('-O') + 1]).write_text('partial')\n"
            "    sys.exit(1)\n"
            "pathlib.Path(sys.argv[sys.argv.index('-O') + 1]).write_text('{}')\n",
        )
        launch = '#!/bin/bash\necho launched >> "$TEST_EVENTS"\n'
        self.executable(self.bin / "python", launch)
        self.executable(self.root / "cpp_server/build/tt_media_server_cpp", launch)
        worker_fetch = self.root / "cpp_server/scripts/fetch_tokenizers.sh"
        worker_fetch.parent.mkdir(parents=True)
        shutil.copy2(FETCH, worker_fetch)
        shutil.copy2(FETCH, self.bin / "fetch_tokenizers.sh")

    @staticmethod
    def executable(path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        path.chmod(0o755)

    def run_script(self, script, *args):
        return subprocess.run(
            ["/bin/bash", str(script), *map(str, args)],
            cwd=self.root,
            env=self.env,
            text=True,
            capture_output=True,
            timeout=20,
        )

    def event_lines(self):
        return self.events.read_text().splitlines() if self.events.exists() else []

    def cache_assets(self):
        target = self.tokens / MODEL
        target.mkdir(parents=True)
        for name in ASSETS:
            (target / name).write_text("{}")

    def assert_success(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_selected_model_downloads_only_its_assets(self):
        result = self.run_script(FETCH, self.tokens, MODEL)
        self.assert_success(result)
        self.assertEqual(len(self.event_lines()), 4)
        self.assertTrue(all(MODEL in line for line in self.event_lines()))
        for name in ASSETS:
            self.assertTrue((self.tokens / MODEL / name).stat().st_size)

    def test_complete_cache_works_offline_without_credentials(self):
        self.cache_assets()
        self.env.pop("HF_TOKEN")
        self.env["FAIL_DOWNLOAD"] = "1"
        self.assert_success(self.run_script(FETCH, self.tokens, MODEL))
        self.assertEqual(self.event_lines(), [])

    def test_missing_selected_assets_require_credentials(self):
        self.env.pop("HF_TOKEN")
        result = self.run_script(FETCH, self.tokens, MODEL)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("set HF_TOKEN", result.stderr)
        self.assertEqual(self.event_lines(), [])

    def test_empty_cached_asset_is_downloaded_again(self):
        self.cache_assets()
        (self.tokens / MODEL / "tokenizer.json").write_text("")
        self.assert_success(self.run_script(FETCH, self.tokens, MODEL))
        self.assertTrue((self.tokens / MODEL / "tokenizer.json").stat().st_size)
        self.assertTrue(any("/tokenizer.json" in line for line in self.event_lines()))

    def test_generic_build_does_not_download_optional_llama_assets(self):
        self.assert_success(self.run_script(FETCH, self.tokens))
        self.assertTrue(self.event_lines())
        self.assertFalse(any(MODEL in line for line in self.event_lines()))
        self.assertFalse((self.tokens / MODEL).exists())

    def test_unknown_selected_model_fails_without_downloads(self):
        result = self.run_script(FETCH, self.tokens, "unknown/model")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unsupported tokenizer model", result.stderr)
        self.assertEqual(self.event_lines(), [])

    def test_image_startup_provisions_selected_model_before_launch(self):
        for script in ("tt-media-server/run_cpp.sh", "dynamo_frontend/entrypoint.sh"):
            for selector in ("MODEL", "HF_MODEL_ID"):
                with self.subTest(script=script, selector=selector):
                    if self.tokens.exists():
                        shutil.rmtree(self.tokens)
                    self.events.write_text("")
                    self.env.pop("MODEL", None)
                    self.env.pop("HF_MODEL_ID", None)
                    self.env[selector] = MODEL
                    self.assert_success(self.run_script(REPO / script))
                    events = self.event_lines()
                    self.assertEqual(events[-1], "launched")
                    self.assertEqual(len(events), 5)
                    self.assertTrue(all(MODEL in line for line in events[:-1]))

    def test_image_startup_reuses_cache_without_auth(self):
        self.cache_assets()
        self.env.pop("HF_TOKEN")
        self.env.update(MODEL=MODEL, FAIL_DOWNLOAD="1")
        for script in ("tt-media-server/run_cpp.sh", "dynamo_frontend/entrypoint.sh"):
            with self.subTest(script=script):
                self.events.write_text("")
                self.assert_success(self.run_script(REPO / script))
                self.assertEqual(self.event_lines(), ["launched"])

    def test_image_startup_fails_before_launch_when_download_fails(self):
        self.env.update(MODEL=MODEL, FAIL_DOWNLOAD="1")
        for script in ("tt-media-server/run_cpp.sh", "dynamo_frontend/entrypoint.sh"):
            with self.subTest(script=script):
                result = self.run_script(REPO / script)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("launched", self.event_lines())
                self.assertFalse(
                    (self.tokens / MODEL / "tokenizer_config.json").exists()
                )
                self.assertEqual(list(self.tokens.rglob("*.tmp.*")), [])

    def test_unrelated_model_startup_does_not_download(self):
        self.env["MODEL"] = "deepseek-ai/DeepSeek-R1-0528"
        for script in ("tt-media-server/run_cpp.sh", "dynamo_frontend/entrypoint.sh"):
            with self.subTest(script=script):
                self.events.write_text("")
                self.assert_success(self.run_script(REPO / script))
                self.assertEqual(self.event_lines(), ["launched"])


if __name__ == "__main__":
    unittest.main()
