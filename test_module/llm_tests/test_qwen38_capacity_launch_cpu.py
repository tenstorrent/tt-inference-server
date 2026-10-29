# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the pre-serving campaign lifecycle and exact image attribution."""

import hashlib
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "capacity_launch", ROOT / "vllm-tt-metal/src/qwen38_capacity_screen.py"
)
launch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launch)


class CampaignTest(unittest.TestCase):
    def test_diagnostic_startup_failure_is_not_automatically_retried(self):
        from workflow_module.commands import ServerCommand, ServerLaunchSpec, ServerMode

        spec = ServerLaunchSpec(
            mode=ServerMode.DOCKER,
            model_spec=SimpleNamespace(env_vars={"QWEN38_RUN_CAPACITY_SCREEN": "1"}),
            runtime_config=SimpleNamespace(),
            setup_config=None,
        )
        command = ServerCommand(spec)
        with patch.object(
            command, "_launch_once", return_value={}
        ) as launch_once, patch(
            "workflow_module.commands._server_boot_attempts", return_value=2
        ), patch(
            "workflow_module.commands._wait_until_ready", return_value="server exited"
        ), patch("workflow_module.commands._teardown_server"):
            result = command.execute()
        self.assertEqual(result.return_code, 1)
        self.assertEqual(launch_once.call_count, 1)

    def test_source_manifest_rejects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("tt", "tests"):
                (root / name).mkdir()
                (root / name / "file.py").write_text("tested bytes")
            digest = hashlib.sha256(b"tested bytes").hexdigest()
            manifest = {
                key: {"file.py": digest} for key in ("sha256", "diagnostics_sha256")
            }
            launch.verify_sources(root, manifest)
            (root / "tt/file.py").write_text("untested bytes")
            with self.assertRaisesRegex(RuntimeError, "Source hash mismatch"):
                launch.verify_sources(root, manifest)

    def run_campaign(self, failure=None):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name) / "metal/models/autoports/model"
        root.mkdir(parents=True)
        output = Path(directory.name) / "evidence"
        manifest = Path(directory.name) / "manifest.json"
        manifest.write_text("{}")
        calls = []

        def next_trials(rows):
            if any(r["status"] != "pass" for r in rows):
                return {"blocked": {"status": "needs_owner_review"}}
            return {
                shared: (
                    {
                        "status": "next_native_trial",
                        "ids": 1368,
                        "cycles": 1,
                        "argv": ["--shared-test", shared],
                    }
                    if not any(r["shared"] == shared for r in rows)
                    else {"status": "native_max_safe_verified_at_one_ID_boundary"}
                )
                for shared in ("0", "1")
            }

        def run_trial(command, env, cwd, log, timeout, stem):
            calls.append(env["QWEN38_SHARED_MLP_WEIGHTS"])
            if failure == "timeout":
                raise subprocess.TimeoutExpired(command, timeout)
            if failure == "no_receipt":
                return 1
            receipt = Path(command[command.index("--output") + 1])
            receipt.write_text(
                json.dumps(
                    {
                        "shared": calls[-1],
                        "status": "fail" if failure else "pass",
                        "close_mesh_device_returned": failure != "unclosed",
                    }
                )
            )
            return 1 if failure else 0

        planner = SimpleNamespace(
            next_trials=next_trials, known_guard_failure=lambda row: False
        )
        with patch.object(
            launch.importlib.metadata, "version", return_value="0.30.0"
        ), patch.object(launch, "verify_sources"), patch.object(
            launch, "load_planner", return_value=planner
        ), patch.object(launch, "run_trial", side_effect=run_trial):
            if failure:
                with self.assertRaises((RuntimeError, subprocess.TimeoutExpired)):
                    launch.screen(
                        root, output, "image@sha256:fixed", root, manifest_path=manifest
                    )
                self.assertEqual(
                    calls, ["0"], "Failure must stop both modes immediately"
                )
            else:
                launch.screen(
                    root, output, "image@sha256:fixed", root, manifest_path=manifest
                )
                self.assertEqual(calls, ["0", "1"])
                self.assertEqual(
                    set(json.loads((output / "selection.json").read_text())["arms"]),
                    {"0", "1"},
                )
            if failure == "timeout":
                self.assertEqual(len(list(output.glob("*.failure.json"))), 1)

    def test_both_anchors_close_before_serving(self):
        self.run_campaign()

    def test_unknown_failure_stops_without_retry(self):
        self.run_campaign("unknown")

    def test_missing_receipt_stops(self):
        self.run_campaign("no_receipt")

    def test_timeout_preserves_failure_and_stops(self):
        self.run_campaign("timeout")

    def test_no_opt_in_never_runs(self):
        with patch.dict(launch.os.environ, {}, clear=True), patch.object(
            launch, "screen"
        ) as screen:
            launch.maybe_run({})
            screen.assert_not_called()

    def test_exact_new_digest_allowed_but_mutation_rejected(self):
        spec = importlib.util.spec_from_file_location(
            "final_audits", ROOT / "workflows/qwen38_final_audits.py"
        )
        audit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(audit)
        plan = {
            "volume": "volume_id_qwen38_autoport-Qwen3.8-27B",
            "image": audit.CAPACITY_OVERLAY_IMAGE,
            "host_aliases": ["runner"],
            "source_owner": "goal12-123-1-1",
            "source_run_id": "123",
            "filenames": ["qwen38-http-1.jsonl", "qwen38-goal12-2.jsonl"],
        }
        audit.validate_plan(plan, "runner")
        plan["image"] = plan["image"][:-1] + "3"
        with self.assertRaisesRegex(ValueError, "unexpected immutable image"):
            audit.validate_plan(plan, "runner")


if __name__ == "__main__":
    unittest.main()
