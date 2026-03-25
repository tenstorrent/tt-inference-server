# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC


from workflows.workflow_venvs import ensure_librispeech_yaml_tasks, ensure_whisper_tt_model


class TestEnsureLibrispeechYamlTasks:
    def test_creates_yaml_when_missing(self, tmp_path):
        """YAML file is written to the correct location when absent."""
        ensure_librispeech_yaml_tasks(tmp_path)
        yaml_file = (
            tmp_path
            / "lmms_eval"
            / "tasks"
            / "librispeech"
            / "librispeech_test_other.yaml"
        )
        assert yaml_file.exists(), f"Expected YAML file at {yaml_file}"
        content = yaml_file.read_text()
        assert 'task: "librispeech_test_other"' in content
        assert "librispeech_wer" in content

    def test_does_not_overwrite_existing(self, tmp_path):
        """Existing YAML files are left unchanged (idempotent)."""
        yaml_dir = tmp_path / "lmms_eval" / "tasks" / "librispeech"
        yaml_dir.mkdir(parents=True)
        yaml_file = yaml_dir / "librispeech_test_other.yaml"
        yaml_file.write_text("existing content")
        ensure_librispeech_yaml_tasks(tmp_path)
        assert yaml_file.read_text() == "existing content"

    def test_creates_intermediate_directories(self, tmp_path):
        """Parent directories are created if they don't exist."""
        site_packages = tmp_path / "lib" / "python3.10" / "site-packages"
        ensure_librispeech_yaml_tasks(site_packages)
        yaml_file = (
            site_packages
            / "lmms_eval"
            / "tasks"
            / "librispeech"
            / "librispeech_test_other.yaml"
        )
        assert yaml_file.exists()


class TestEnsureWhisperTtModel:
    def test_copies_when_missing(self, tmp_path):
        """whisper_tt.py is copied to models/simple/ when absent."""
        src_dir = tmp_path / "lmms_eval" / "models"
        src_dir.mkdir(parents=True)
        src = src_dir / "whisper_tt.py"
        src.write_text("# whisper_tt model")
        ensure_whisper_tt_model(tmp_path)
        dst = src_dir / "simple" / "whisper_tt.py"
        assert dst.exists()
        assert dst.read_text() == "# whisper_tt model"

    def test_does_not_overwrite_existing(self, tmp_path):
        """Existing file in simple/ is left unchanged (idempotent)."""
        src_dir = tmp_path / "lmms_eval" / "models"
        src_dir.mkdir(parents=True)
        (src_dir / "whisper_tt.py").write_text("source content")
        dst_dir = src_dir / "simple"
        dst_dir.mkdir(parents=True)
        dst = dst_dir / "whisper_tt.py"
        dst.write_text("existing content")
        ensure_whisper_tt_model(tmp_path)
        assert dst.read_text() == "existing content"

    def test_skips_when_source_missing(self, tmp_path):
        """No error when source whisper_tt.py doesn't exist."""
        ensure_whisper_tt_model(tmp_path)
        dst = tmp_path / "lmms_eval" / "models" / "simple" / "whisper_tt.py"
        assert not dst.exists()
