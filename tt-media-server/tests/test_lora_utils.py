# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

from unittest.mock import MagicMock, patch

import pytest
from utils.lora_utils import (
    _find_safetensors_filename,
    _get_triggers_from_model_card,
    _get_triggers_from_readme,
    _get_triggers_from_safetensors,
    get_lora_trigger_words,
    prepare_prompt_with_lora,
    resolve_lora_path,
)

REPO_ID = "artificialguybr/ColoringBookRedmond-V2"
LORA_FILENAME = "ColoringBookRedmond-ColoringBook-ColoringBookAF.safetensors"


class TestFindSafetensorsFilename:
    def test_single_safetensors_file(self):
        with patch(
            "utils.lora_utils.list_repo_files",
            return_value=["README.md", LORA_FILENAME],
        ):
            assert _find_safetensors_filename(REPO_ID) == LORA_FILENAME

    def test_no_safetensors_file_raises(self):
        with patch(
            "utils.lora_utils.list_repo_files",
            return_value=["README.md", "config.json"],
        ):
            with pytest.raises(FileNotFoundError, match="No .safetensors file"):
                _find_safetensors_filename(REPO_ID)

    def test_repository_not_found_raises(self):
        from huggingface_hub.utils import RepositoryNotFoundError

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.headers = {}
        with patch(
            "utils.lora_utils.list_repo_files",
            side_effect=RepositoryNotFoundError("not found", response=mock_response),
        ):
            with pytest.raises(FileNotFoundError, match="Repository not found"):
                _find_safetensors_filename("bad/repo")

    def test_multiple_files_prefers_root(self):
        with patch(
            "utils.lora_utils.list_repo_files",
            return_value=[
                "subfolder/other.safetensors",
                "root_model.safetensors",
                "README.md",
            ],
        ):
            assert _find_safetensors_filename(REPO_ID) == "root_model.safetensors"

    def test_multiple_root_files_returns_first(self):
        with patch(
            "utils.lora_utils.list_repo_files",
            return_value=["b_model.safetensors", "a_model.safetensors"],
        ):
            assert _find_safetensors_filename(REPO_ID) == "b_model.safetensors"

    def test_all_nested_returns_first(self):
        with patch(
            "utils.lora_utils.list_repo_files",
            return_value=[
                "v1/model.safetensors",
                "v2/model.safetensors",
            ],
        ):
            assert _find_safetensors_filename(REPO_ID) == "v1/model.safetensors"


class TestResolveLoraPath:
    def test_local_file_returns_resolved_path(self, tmp_path):
        lora_file = tmp_path / "adapter.safetensors"
        lora_file.write_bytes(b"fake weights")

        result = resolve_lora_path(str(lora_file))
        assert result == str(lora_file.resolve())

    def test_hf_repo_delegates_to_hf_hub_download(self, tmp_path):
        hf_cached = str(tmp_path / "hub" / "snapshots" / "abc123" / LORA_FILENAME)

        with patch(
            "utils.lora_utils.list_repo_files",
            return_value=[LORA_FILENAME],
        ):
            with patch(
                "utils.lora_utils.hf_hub_download",
                return_value=hf_cached,
            ) as mock_download:
                result = resolve_lora_path(REPO_ID)

        assert result == hf_cached
        mock_download.assert_called_once_with(
            repo_id=REPO_ID,
            filename=LORA_FILENAME,
        )

    def test_nonexistent_local_path_treated_as_hf_repo(self, tmp_path):
        hf_cached = str(tmp_path / "hub" / "snapshots" / "abc123" / "model.safetensors")

        with patch(
            "utils.lora_utils.list_repo_files",
            return_value=["model.safetensors"],
        ):
            with patch(
                "utils.lora_utils.hf_hub_download",
                return_value=hf_cached,
            ):
                result = resolve_lora_path("/nonexistent/path.safetensors")

        assert result == hf_cached

    def test_download_failure_wraps_in_lora_not_found(self):
        with patch(
            "utils.lora_utils.list_repo_files",
            return_value=["model.safetensors"],
        ):
            with patch(
                "utils.lora_utils.hf_hub_download",
                side_effect=OSError("Network error"),
            ):
                with pytest.raises(FileNotFoundError, match="Failed to download LoRA"):
                    resolve_lora_path("bad/repo")

    def test_list_files_failure_propagates(self):
        with patch(
            "utils.lora_utils.list_repo_files",
            side_effect=OSError("Network error"),
        ):
            with pytest.raises(OSError, match="Network error"):
                resolve_lora_path("bad/repo")


class TestGetTriggersFromModelCard:
    def test_returns_instance_prompt(self):
        mock_card = MagicMock()
        mock_card.data.instance_prompt = "ColoringBookAF, Coloring Book"

        with patch("utils.lora_utils.ModelCard") as mc:
            mc.load.return_value = mock_card
            result = _get_triggers_from_model_card("some/repo")

        assert result == ["ColoringBookAF", "Coloring Book"]

    def test_returns_single_trigger(self):
        mock_card = MagicMock()
        mock_card.data.instance_prompt = "pixel art"

        with patch("utils.lora_utils.ModelCard") as mc:
            mc.load.return_value = mock_card
            assert _get_triggers_from_model_card("some/repo") == ["pixel art"]

    def test_returns_none_when_empty(self):
        mock_card = MagicMock()
        mock_card.data.instance_prompt = ""

        with patch("utils.lora_utils.ModelCard") as mc:
            mc.load.return_value = mock_card
            assert _get_triggers_from_model_card("some/repo") is None

    def test_returns_none_when_missing(self):
        mock_card = MagicMock(spec=[])
        mock_card.data = MagicMock(spec=[])

        with patch("utils.lora_utils.ModelCard") as mc:
            mc.load.return_value = mock_card
            assert _get_triggers_from_model_card("some/repo") is None

    def test_returns_none_on_network_error(self):
        with patch("utils.lora_utils.ModelCard") as mc:
            mc.load.side_effect = OSError("Network error")
            assert _get_triggers_from_model_card("some/repo") is None


class TestGetTriggersFromSafetensors:
    def test_reads_trigger_word_key(self):
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.metadata.return_value = {"trigger_word": "pixel art, style"}

        with patch("safetensors.safe_open", return_value=mock_file):
            result = _get_triggers_from_safetensors("/fake/path.safetensors")

        assert result == ["pixel art", "style"]

    def test_reads_ss_trigger_words_key(self):
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.metadata.return_value = {"ss_trigger_words": "my_style"}

        with patch("safetensors.safe_open", return_value=mock_file):
            result = _get_triggers_from_safetensors("/fake/path")

        assert result == ["my_style"]

    def test_returns_none_when_no_metadata(self):
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.metadata.return_value = None

        with patch("safetensors.safe_open", return_value=mock_file):
            assert _get_triggers_from_safetensors("/fake/path") is None

    def test_returns_none_on_import_error(self):
        import sys

        saved = sys.modules.pop("safetensors", None)
        try:
            with patch.dict(sys.modules, {"safetensors": None}):
                assert _get_triggers_from_safetensors("/fake/path") is None
        finally:
            if saved is not None:
                sys.modules["safetensors"] = saved


class TestGetTriggersFromReadme:
    def test_parses_trigger_word_line(self, tmp_path):
        readme = tmp_path / "README.md"
        readme.write_text("# Model\n\nTrigger word: my_token\n\nSome description.")

        with patch("utils.lora_utils.hf_hub_download", return_value=str(readme)):
            result = _get_triggers_from_readme("some/repo")

        assert result == ["my_token"]

    def test_parses_bold_trigger_words(self, tmp_path):
        readme = tmp_path / "README.md"
        readme.write_text("**Trigger Words:** `style_A, style_B`")

        with patch("utils.lora_utils.hf_hub_download", return_value=str(readme)):
            result = _get_triggers_from_readme("some/repo")

        assert result == ["style_A", "style_B"]

    def test_parses_activation_word(self, tmp_path):
        readme = tmp_path / "README.md"
        readme.write_text("Activation word: TOK")

        with patch("utils.lora_utils.hf_hub_download", return_value=str(readme)):
            result = _get_triggers_from_readme("some/repo")

        assert result == ["TOK"]

    def test_parses_use_in_prompt_pattern(self, tmp_path):
        readme = tmp_path / "README.md"
        readme.write_text("Use `sks_style` in your prompt to activate the LoRA.")

        with patch("utils.lora_utils.hf_hub_download", return_value=str(readme)):
            result = _get_triggers_from_readme("some/repo")

        assert result == ["sks_style"]

    def test_returns_none_when_no_match(self, tmp_path):
        readme = tmp_path / "README.md"
        readme.write_text("# A nice model\nNo trigger info here.")

        with patch("utils.lora_utils.hf_hub_download", return_value=str(readme)):
            assert _get_triggers_from_readme("some/repo") is None

    def test_returns_none_on_download_error(self):
        with patch(
            "utils.lora_utils.hf_hub_download", side_effect=OSError("not found")
        ):
            assert _get_triggers_from_readme("some/repo") is None


class TestGetLoraTriggerWords:
    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        get_lora_trigger_words.cache_clear()
        yield
        get_lora_trigger_words.cache_clear()

    def test_local_file_uses_safetensors(self, tmp_path):
        lora_file = tmp_path / "model.safetensors"
        lora_file.write_bytes(b"fake")

        with patch(
            "utils.lora_utils._get_triggers_from_safetensors",
            return_value=["my_token"],
        ):
            result = get_lora_trigger_words(str(lora_file))

        assert result == ("my_token",)

    def test_hf_repo_prefers_model_card(self):
        with patch(
            "utils.lora_utils._get_triggers_from_model_card",
            return_value=["card_token"],
        ):
            result = get_lora_trigger_words("org/repo")

        assert result == ("card_token",)

    def test_hf_repo_falls_back_to_safetensors_repo(self):
        mock_card = patch(
            "utils.lora_utils._get_triggers_from_model_card", return_value=None
        )
        mock_st = patch(
            "utils.lora_utils._get_triggers_from_safetensors_repo",
            return_value=["st_token"],
        )
        with mock_card, mock_st:
            result = get_lora_trigger_words("org/repo")

        assert result == ("st_token",)

    def test_hf_repo_falls_back_to_readme(self):
        mock_card = patch(
            "utils.lora_utils._get_triggers_from_model_card", return_value=None
        )
        mock_st = patch(
            "utils.lora_utils._get_triggers_from_safetensors_repo",
            return_value=None,
        )
        mock_readme = patch(
            "utils.lora_utils._get_triggers_from_readme",
            return_value=["readme_token"],
        )
        with mock_card, mock_st, mock_readme:
            result = get_lora_trigger_words("org/repo")

        assert result == ("readme_token",)

    def test_returns_none_when_all_sources_fail(self):
        mock_card = patch(
            "utils.lora_utils._get_triggers_from_model_card", return_value=None
        )
        mock_st = patch(
            "utils.lora_utils._get_triggers_from_safetensors_repo",
            return_value=None,
        )
        mock_readme = patch(
            "utils.lora_utils._get_triggers_from_readme", return_value=None
        )
        with mock_card, mock_st, mock_readme:
            assert get_lora_trigger_words("org/repo") is None

    def test_result_is_cached(self):
        with patch(
            "utils.lora_utils._get_triggers_from_model_card",
            return_value=["cached_token"],
        ) as mock_card:
            get_lora_trigger_words("org/repo")
            get_lora_trigger_words("org/repo")

        mock_card.assert_called_once()


class TestPreparePromptWithLora:
    def test_injects_trigger(self):
        with patch(
            "utils.lora_utils.get_lora_trigger_words",
            return_value=("pixel art",),
        ):
            result = prepare_prompt_with_lora("a cute cat", "nerijs/pixel-art-xl")

        assert result == "a cute cat, pixel art"

    def test_skips_when_trigger_already_present(self):
        with patch(
            "utils.lora_utils.get_lora_trigger_words",
            return_value=("pixel art",),
        ):
            result = prepare_prompt_with_lora(
                "a cute cat in pixel art style", "nerijs/pixel-art-xl"
            )

        assert result == "a cute cat in pixel art style"

    def test_case_insensitive_dedup(self):
        with patch(
            "utils.lora_utils.get_lora_trigger_words",
            return_value=("Pixel Art",),
        ):
            result = prepare_prompt_with_lora(
                "a cute pixel art cat", "nerijs/pixel-art-xl"
            )

        assert result == "a cute pixel art cat"

    def test_no_lora_path_returns_original(self):
        assert prepare_prompt_with_lora("a cute cat", None) == "a cute cat"

    def test_empty_prompt_returns_unchanged(self):
        assert prepare_prompt_with_lora("", "some/repo") == ""

    def test_no_triggers_found_returns_original(self):
        with patch("utils.lora_utils.get_lora_trigger_words", return_value=None):
            result = prepare_prompt_with_lora("a cute cat", "some/repo")

        assert result == "a cute cat"

    def test_auto_inject_false_skips(self):
        with patch(
            "utils.lora_utils.get_lora_trigger_words",
            return_value=("pixel art",),
        ):
            result = prepare_prompt_with_lora(
                "a cute cat", "nerijs/pixel-art-xl", auto_inject=False
            )

        assert result == "a cute cat"
