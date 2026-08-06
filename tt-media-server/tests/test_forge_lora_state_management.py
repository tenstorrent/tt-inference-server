# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""LoRA state management for the Forge (tt-xla) SDXL path.

The Forge runner fuses the adapter into the UNet *before* compile, so unlike
the trace runner it cannot swap weights on a live graph — every LoRA change is
a rebuild + recompile. These tests pin the state machine that keeps those
recompiles to the minimum, and pin the two details that made the adapter
silently no-op before: the ``prefix="unet"`` argument and the fuse-then-unload
order.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# conftest mocks utils.logger but only provides TTLogger.
sys.modules["utils.logger"].log_exception_chain = MagicMock()

# conftest registers "tt_model_runners.forge_runners" as a bare stub module with
# no __path__, which makes real submodule imports fail with "is not a package".
# Restore __path__ so the real sdxl_forge_runner loads, while leaving the rest
# of conftest's mocks (torch, diffusers, ...) in place.
_forge_pkg = sys.modules.get("tt_model_runners.forge_runners")
if _forge_pkg is not None and not hasattr(_forge_pkg, "__path__"):
    _forge_pkg.__path__ = [
        str(Path(__file__).resolve().parents[1] / "tt_model_runners" / "forge_runners")
    ]

from domain.image_generate_request import ImageGenerateRequest  # noqa: E402
from tt_model_runners.forge_runners.sdxl_forge_runner import (  # noqa: E402
    SDXLForgeRunner,
)


def _make_runner():
    """Build a runner without running __init__ (which touches env + devices)."""
    runner = SDXLForgeRunner.__new__(SDXLForgeRunner)
    runner.logger = MagicMock()
    runner.device_id = "0"
    runner.device = "cpu"
    runner.unet = MagicMock(name="initial_unet")
    runner._current_lora_path = None
    runner._current_lora_scale = None
    runner._model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    runner._variant = "fp16"
    return runner


def _make_request(lora_path=None, lora_scale=0.5, **kwargs):
    defaults = dict(
        prompt="a cat",
        negative_prompt="",
        guidance_scale=5.0,
        num_inference_steps=20,
        seed=42,
    )
    defaults.update(kwargs)
    return ImageGenerateRequest.model_construct(
        lora_path=lora_path, lora_scale=lora_scale, **defaults
    )


class TestEnsureLoraState:
    """State transitions — each rebuild is minutes of recompile, so no-ops matter."""

    def test_no_lora_requested_on_clean_runner_is_noop(self):
        runner = _make_runner()
        with patch.object(runner, "_build_unet") as build:
            runner._ensure_lora_state(_make_request())
        build.assert_not_called()

    def test_first_lora_request_rebuilds(self):
        runner = _make_runner()
        with patch.object(runner, "_build_unet") as build:
            runner._ensure_lora_state(
                _make_request(lora_path="adapter-A", lora_scale=0.5)
            )
        build.assert_called_once_with("adapter-A", 0.5)
        assert runner._current_lora_path == "adapter-A"
        assert runner._current_lora_scale == 0.5

    def test_same_path_and_scale_does_not_recompile(self):
        runner = _make_runner()
        runner._current_lora_path = "adapter-A"
        runner._current_lora_scale = 0.5
        with patch.object(runner, "_build_unet") as build:
            runner._ensure_lora_state(
                _make_request(lora_path="adapter-A", lora_scale=0.5)
            )
        build.assert_not_called()

    def test_same_path_new_scale_recompiles(self):
        runner = _make_runner()
        runner._current_lora_path = "adapter-A"
        runner._current_lora_scale = 0.5
        with patch.object(runner, "_build_unet") as build:
            runner._ensure_lora_state(
                _make_request(lora_path="adapter-A", lora_scale=0.8)
            )
        build.assert_called_once_with("adapter-A", 0.8)
        assert runner._current_lora_scale == 0.8

    def test_switching_adapter_recompiles(self):
        runner = _make_runner()
        runner._current_lora_path = "adapter-A"
        runner._current_lora_scale = 0.5
        with patch.object(runner, "_build_unet") as build:
            runner._ensure_lora_state(
                _make_request(lora_path="adapter-B", lora_scale=1.0)
            )
        build.assert_called_once_with("adapter-B", 1.0)
        assert runner._current_lora_path == "adapter-B"

    def test_dropping_lora_rebuilds_plain_unet(self):
        """Rollback to baseline must rebuild without an adapter, not keep the fused one."""
        runner = _make_runner()
        runner._current_lora_path = "adapter-A"
        runner._current_lora_scale = 0.5
        with patch.object(runner, "_build_unet") as build:
            runner._ensure_lora_state(_make_request(lora_path=None))
        build.assert_called_once_with(None, 0.5)
        assert runner._current_lora_path is None
        assert runner._current_lora_scale is None

    def test_scale_change_ignored_when_no_lora_requested(self):
        """scale is meaningless without a path; must not trigger a recompile."""
        runner = _make_runner()
        with patch.object(runner, "_build_unet") as build:
            runner._ensure_lora_state(_make_request(lora_path=None, lora_scale=0.9))
        build.assert_not_called()

    def test_build_failure_clears_state_and_raises(self):
        runner = _make_runner()
        with patch.object(runner, "_build_unet", side_effect=OSError("no such repo")):
            with pytest.raises(RuntimeError, match="Failed to apply LoRA adapter"):
                runner._ensure_lora_state(_make_request(lora_path="bad-adapter"))
        assert runner._current_lora_path is None
        assert runner._current_lora_scale is None

    def test_state_cleared_after_failure_so_next_request_retries(self):
        runner = _make_runner()
        with patch.object(runner, "_build_unet", side_effect=OSError("transient")):
            with pytest.raises(RuntimeError):
                runner._ensure_lora_state(_make_request(lora_path="adapter-A"))
        with patch.object(runner, "_build_unet") as build:
            runner._ensure_lora_state(_make_request(lora_path="adapter-A"))
        build.assert_called_once()

    def test_old_unet_released_before_rebuild(self):
        """Holding two UNets on device risks DRAM OOM on N150."""
        runner = _make_runner()
        seen = {}

        def _capture(path, scale):
            seen["unet_during_build"] = runner.unet
            return MagicMock(name="new_unet")

        with patch.object(runner, "_build_unet", side_effect=_capture):
            runner._ensure_lora_state(_make_request(lora_path="adapter-A"))
        assert seen["unet_during_build"] is None


class TestBuildUnet:
    """The two details that made the adapter silently no-op before."""

    @pytest.fixture
    def _unet_cls(self):
        with patch("diffusers.UNet2DConditionModel") as cls, patch(
            "tt_model_runners.forge_runners.sdxl_forge_runner.resolve_lora_path",
            side_effect=lambda p: f"/local/{p}",
        ):
            cls.from_pretrained.return_value = MagicMock(name="unet")
            yield cls

    def test_lora_loaded_with_unet_prefix(self, _unet_cls):
        """diffusers defaults to prefix='transformer', which matches no SDXL UNet key."""
        runner = _make_runner()
        runner._build_unet("adapter-A", 0.7)
        unet = _unet_cls.from_pretrained.return_value
        unet.load_lora_adapter.assert_called_once_with(
            "/local/adapter-A", prefix="unet"
        )

    def test_fused_then_unloaded_so_graph_stays_static(self, _unet_cls):
        runner = _make_runner()
        runner._build_unet("adapter-A", 0.7)
        unet = _unet_cls.from_pretrained.return_value
        unet.fuse_lora.assert_called_once_with(lora_scale=0.7)
        # Order matters: unloading before fusing would discard the weights.
        assert unet.method_calls.index(
            call.fuse_lora(lora_scale=0.7)
        ) < unet.method_calls.index(call.unload_lora())

    def test_none_scale_defaults_to_one(self, _unet_cls):
        runner = _make_runner()
        runner._build_unet("adapter-A", None)
        _unet_cls.from_pretrained.return_value.fuse_lora.assert_called_once_with(
            lora_scale=1.0
        )

    def test_no_lora_touches_no_lora_api(self, _unet_cls):
        runner = _make_runner()
        runner._build_unet()
        unet = _unet_cls.from_pretrained.return_value
        unet.load_lora_adapter.assert_not_called()
        unet.fuse_lora.assert_not_called()


class TestRunWiring:
    """Regression guard for the original bug: run() dropped lora_path entirely."""

    def test_run_applies_lora_before_generating(self):
        runner = _make_runner()
        order = []

        def _generate(**kwargs):
            order.append("generate")
            return MagicMock(name="image_tensor")

        with patch.object(
            runner, "_ensure_lora_state", side_effect=lambda r: order.append("lora")
        ) as ensure, patch.object(runner, "_generate", side_effect=_generate), patch(
            "tt_model_runners.forge_runners.sdxl_forge_runner.prepare_prompt_with_lora",
            side_effect=lambda prompt, lora_path: prompt,
        ), patch(
            # torch is a conftest MagicMock, so the tensor->PIL tail cannot run.
            "tt_model_runners.forge_runners.sdxl_forge_runner.Image"
        ):
            runner.run([_make_request(lora_path="adapter-A")])

        ensure.assert_called_once()
        assert order == ["lora", "generate"], (
            "LoRA must be applied before the graph runs"
        )

    def test_run_injects_lora_trigger_words(self):
        """The coloring-book adapter needs its trigger word in the prompt to show up."""
        runner = _make_runner()

        with patch.object(runner, "_ensure_lora_state"), patch.object(
            runner, "_generate"
        ) as generate, patch(
            "tt_model_runners.forge_runners.sdxl_forge_runner.prepare_prompt_with_lora",
            side_effect=lambda prompt, lora_path: f"{prompt}, ColoringBookAF",
        ), patch("tt_model_runners.forge_runners.sdxl_forge_runner.Image"):
            runner.run([_make_request(lora_path="adapter-A", prompt="a cat")])

        assert generate.call_args.kwargs["prompt"] == "a cat, ColoringBookAF"
