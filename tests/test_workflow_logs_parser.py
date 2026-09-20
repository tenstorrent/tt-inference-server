# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
"""Report selection in ``workflow_logs_parser``.

``parse_workflow_logs_dir`` discards the whole bundle when this returns None,
so "no report matched" and "the model vanished from the release" are the same
event. These tests pin that a report is only ever dropped when there is none.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# The module reaches its own dependencies through ``scripts/release/_bootstrap``,
# which is only importable with that directory on the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "release"))

from scripts.release.workflow_logs_parser import (  # noqa: E402
    load_report_data_json,
    report_identity_matches,
)

MODEL_ID = "id_tt-transformers_Mistral-7B-Instruct-v0.3_n150"
MODEL_REPO = "mistralai/Mistral-7B-Instruct-v0.3"


def write_report(reports_root: Path, workflow: str, name: str, metadata: dict) -> Path:
    data_dir = reports_root / workflow / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"report_data_{name}.json"
    path.write_text(json.dumps({"metadata": metadata, "marker": name}))
    return path


class TestReportIdentityMatches:
    def test_exact_repo(self):
        assert report_identity_matches({"model_repo": MODEL_REPO}, None, MODEL_REPO)

    def test_bare_repo_matches_prefixed_spec(self):
        """The regression: a bundle built before the identity migration records
        the bare name while the spec now holds the full repo id."""
        assert report_identity_matches(
            {"model_repo": "Mistral-7B-Instruct-v0.3"}, None, MODEL_REPO
        )

    def test_spec_id_matches_exactly(self):
        assert report_identity_matches({"model_id": MODEL_ID}, MODEL_ID, None)

    def test_different_model_does_not_match(self):
        assert not report_identity_matches(
            {"model_repo": "Qwen/Qwen3-32B"}, MODEL_ID, MODEL_REPO
        )

    def test_identityless_report_does_not_match(self):
        assert not report_identity_matches({}, MODEL_ID, MODEL_REPO)


class TestLoadReportDataJson:
    def test_prefers_the_matching_report(self, tmp_path):
        write_report(tmp_path, "release", "other", {"model_repo": "Qwen/Qwen3-32B"})
        write_report(tmp_path, "release", "mine", {"model_repo": MODEL_REPO})
        data = load_report_data_json(tmp_path, MODEL_ID, MODEL_REPO)
        assert data["marker"] == "mine"

    def test_bare_spelling_still_matches(self, tmp_path):
        write_report(
            tmp_path, "release", "bare", {"model_repo": "Mistral-7B-Instruct-v0.3"}
        )
        data = load_report_data_json(tmp_path, MODEL_ID, MODEL_REPO)
        assert data["marker"] == "bare"

    def test_identityless_report_is_used_when_nothing_matches(self, tmp_path):
        write_report(tmp_path, "release", "legacy", {})
        data = load_report_data_json(tmp_path, MODEL_ID, MODEL_REPO)
        assert data["marker"] == "legacy"

    def test_mismatched_report_is_surfaced_not_dropped(self, tmp_path):
        """Regression: a report claiming an unrecognised identity was neither
        matched nor collected, so the bundle disappeared behind a warning."""
        write_report(
            tmp_path, "release", "unknown-spelling", {"model_repo": "some-other-name"}
        )
        data = load_report_data_json(tmp_path, MODEL_ID, MODEL_REPO)
        assert data is not None
        assert data["marker"] == "unknown-spelling"

    def test_identityless_wins_over_mismatched(self, tmp_path):
        write_report(tmp_path, "release", "mismatch", {"model_repo": "other/model"})
        write_report(tmp_path, "benchmarks", "legacy", {})
        data = load_report_data_json(tmp_path, MODEL_ID, MODEL_REPO)
        assert data["marker"] == "legacy"

    def test_newest_mismatch_wins_across_workflow_dirs(self, tmp_path):
        older = write_report(
            tmp_path, "benchmarks", "older", {"model_repo": "other/model"}
        )
        newer = write_report(
            tmp_path, "release", "newer", {"model_repo": "other/model"}
        )
        stale, fresh = time.time() - 500, time.time()
        os.utime(older, (stale, stale))
        os.utime(newer, (fresh, fresh))
        data = load_report_data_json(tmp_path, MODEL_ID, MODEL_REPO)
        assert data["marker"] == "newer"

    def test_missing_reports_root_returns_none(self, tmp_path):
        assert load_report_data_json(tmp_path / "absent", MODEL_ID, MODEL_REPO) is None

    def test_empty_reports_root_returns_none(self, tmp_path):
        assert load_report_data_json(tmp_path, MODEL_ID, MODEL_REPO) is None
