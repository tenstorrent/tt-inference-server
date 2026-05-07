#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for the tt_symbiote-aware ModelSpecTemplate validator and the
tt_symbiote env-var defaults injection in ModelSpec.__post_init__.

See workflows/model_spec.py and docs/tt_symbiote_integration_pipeline.md §4.
"""

from typing import Dict

import pytest

from workflows.model_spec import (
    DeviceModelSpec,
    ModelSpec,
    ModelSpecTemplate,
    _TT_SYMBIOTE_DEFAULT_ENV_VARS,
    tt_symbiote_impl,
    tt_transformers_impl,
)
from workflows.workflow_types import DeviceTypes, InferenceEngine, ModelStatusTypes


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _good_symbiote_device_spec(**overrides) -> DeviceModelSpec:
    """A DeviceModelSpec that satisfies every tt_symbiote validator rule.

    The override_tt_config / env_vars dicts can be patched via overrides for
    individual negative-path tests.
    """
    override_tt_config: Dict[str, str] = {
        "enable_model_warmup": True,
        "trace_mode": "none",
        "trace_region_size": 200000000,
        "fabric_config": "FABRIC_1D_RING",
    }
    env_vars: Dict[str, str] = {
        "TT_SYMBIOTE_DISPATCHER": "CPU",
        "MESH_DEVICE": "T3K",
        "DISABLE_METAL_OP_TIMEOUT": "1",
    }
    if "override_tt_config" in overrides:
        override_tt_config.update(overrides.pop("override_tt_config"))
    if "env_vars" in overrides:
        env_vars.update(overrides.pop("env_vars"))
    base = dict(
        device=DeviceTypes.T3K,
        max_concurrency=1,
        max_context=2048,
        default_impl=True,
        vllm_args={"max_model_len": "2048", "max_num_seqs": "1", "block_size": "64"},
        override_tt_config=override_tt_config,
        env_vars=env_vars,
    )
    base.update(overrides)
    return DeviceModelSpec(**base)


def _good_symbiote_template(**overrides) -> ModelSpecTemplate:
    """A ModelSpecTemplate that satisfies every tt_symbiote validator rule."""
    base = dict(
        weights=["fixture/symbiote-test"],
        impl=tt_symbiote_impl,
        tt_metal_commit="abc1234",
        vllm_commit="def5678",
        inference_engine=InferenceEngine.VLLM.value,
        device_model_specs=[_good_symbiote_device_spec()],
        status=ModelStatusTypes.EXPERIMENTAL,
        has_builtin_warmup=True,
    )
    if "device_model_specs" in overrides:
        base["device_model_specs"] = overrides.pop("device_model_specs")
    base.update(overrides)
    return ModelSpecTemplate(**base)


# ---------------------------------------------------------------------------
# Positive path: a fully-correct template passes
# ---------------------------------------------------------------------------


class TestSymbioteValidatorAccepts:
    def test_canonical_symbiote_template_validates(self):
        # Should not raise.
        template = _good_symbiote_template()
        assert template.has_builtin_warmup is True
        assert template.impl is tt_symbiote_impl

    def test_non_symbiote_template_skips_validator(self):
        # tt_transformers templates must not trigger the symbiote validator
        # even if they would otherwise fail it (e.g. has_builtin_warmup=False).
        ModelSpecTemplate(
            weights=["fixture/tt-transformers-test"],
            impl=tt_transformers_impl,
            tt_metal_commit="abc1234",
            vllm_commit="def5678",
            inference_engine=InferenceEngine.VLLM.value,
            device_model_specs=[
                DeviceModelSpec(
                    device=DeviceTypes.T3K,
                    max_concurrency=32,
                    max_context=4096,
                    default_impl=True,
                ),
            ],
            status=ModelStatusTypes.EXPERIMENTAL,
            has_builtin_warmup=False,  # would fail the symbiote validator
        )

    def test_fabric_config_unset_logs_warning_but_does_not_assert(self, caplog):
        """Multi-chip devices without fabric_config should warn but not fail.

        Gemma-4 currently relies on the tt-metal multi-chip default; we want
        the validator to nudge the operator without breaking that working
        config. Only Ling-mini-2.0 needs the explicit FABRIC_1D_RING.
        """
        ds = _good_symbiote_device_spec(
            override_tt_config={"fabric_config": None},  # noop in dict.update
        )
        # Force fabric_config out of the dict by rebuilding.
        ds_dict = {
            "device": DeviceTypes.T3K,
            "max_concurrency": 1,
            "max_context": 2048,
            "default_impl": True,
            "override_tt_config": {
                "enable_model_warmup": True,
                "trace_mode": "none",
                "trace_region_size": 200000000,
            },
            "env_vars": {"TT_SYMBIOTE_DISPATCHER": "CPU", "MESH_DEVICE": "T3K"},
        }
        ds_no_fabric = DeviceModelSpec(**ds_dict)

        with caplog.at_level("WARNING", logger="workflows.model_spec.tt_symbiote"):
            _good_symbiote_template(device_model_specs=[ds_no_fabric])

        warnings = [r for r in caplog.records if "fabric_config" in r.getMessage()]
        assert warnings, "expected fabric_config warning when unset on multi-chip"


# ---------------------------------------------------------------------------
# Negative paths: misconfigurations raise AssertionError
# ---------------------------------------------------------------------------


class TestSymbioteValidatorRejects:
    def test_has_builtin_warmup_false_rejected(self):
        with pytest.raises(AssertionError, match=r"has_builtin_warmup=True"):
            _good_symbiote_template(has_builtin_warmup=False)

    def test_enable_model_warmup_false_rejected(self):
        ds = _good_symbiote_device_spec(
            override_tt_config={"enable_model_warmup": False},
        )
        with pytest.raises(AssertionError, match=r"enable_model_warmup"):
            _good_symbiote_template(device_model_specs=[ds])

    def test_trace_mode_traced_rejected(self):
        ds = _good_symbiote_device_spec(override_tt_config={"trace_mode": "traced"})
        with pytest.raises(AssertionError, match=r"trace_mode"):
            _good_symbiote_template(device_model_specs=[ds])

    def test_unknown_dispatcher_rejected(self):
        ds = _good_symbiote_device_spec(
            env_vars={"TT_SYMBIOTE_DISPATCHER": "BOGUS_DISPATCHER"},
        )
        with pytest.raises(AssertionError, match=r"TT_SYMBIOTE_DISPATCHER"):
            _good_symbiote_template(device_model_specs=[ds])

    def test_mesh_device_mismatch_rejected(self):
        ds = _good_symbiote_device_spec(env_vars={"MESH_DEVICE": "N300"})
        with pytest.raises(AssertionError, match=r"MESH_DEVICE"):
            _good_symbiote_template(device_model_specs=[ds])

    def test_dispatcher_unset_is_fine(self):
        # Dispatcher is auto-injected by ModelSpec.__post_init__ when missing,
        # so the template-level validator should not require it.
        ds_dict = {
            "device": DeviceTypes.T3K,
            "max_concurrency": 1,
            "max_context": 2048,
            "default_impl": True,
            "override_tt_config": {
                "enable_model_warmup": True,
                "trace_mode": "none",
                "trace_region_size": 200000000,
                "fabric_config": "FABRIC_1D_RING",
            },
            "env_vars": {"MESH_DEVICE": "T3K"},  # no TT_SYMBIOTE_DISPATCHER
        }
        _good_symbiote_template(device_model_specs=[DeviceModelSpec(**ds_dict)])


# ---------------------------------------------------------------------------
# Env-var defaults injection in ModelSpec.__post_init__
# ---------------------------------------------------------------------------


class TestSymbioteEnvVarInjection:
    def _build_minimal_symbiote_model_spec(self, env_vars=None) -> ModelSpec:
        """Build a ModelSpec directly (skipping ModelSpecTemplate.expand_to_specs)
        so we can isolate the env-var injection behaviour."""
        ds = _good_symbiote_device_spec(env_vars=env_vars or {})
        return ModelSpec(
            model_id="id_tt-symbiote_fixture_t3k",
            impl=tt_symbiote_impl,
            hf_model_repo="fixture/symbiote-test",
            model_name="symbiote-test",
            inference_engine=InferenceEngine.VLLM.value,
            device_type=DeviceTypes.T3K,
            tt_metal_commit="abc1234",
            device_model_spec=ds,
            vllm_commit="def5678",
            has_builtin_warmup=True,
        )

    def test_defaults_injected_for_symbiote_specs(self):
        spec = self._build_minimal_symbiote_model_spec()
        for key, default_val in _TT_SYMBIOTE_DEFAULT_ENV_VARS.items():
            assert key in spec.env_vars, (
                f"expected tt_symbiote default env var {key!r} on a tt_symbiote ModelSpec"
            )
            # When the device-spec didn't override, the default should win.
            if key not in {"TT_SYMBIOTE_DISPATCHER", "DISABLE_METAL_OP_TIMEOUT"}:
                # _good_symbiote_device_spec sets DISPATCHER and METAL_OP_TIMEOUT
                # explicitly; assert they survive (matching the default in this
                # fixture). For the rest the default value should pass through.
                assert spec.env_vars[key] == default_val

    def test_per_spec_override_wins_over_default(self):
        """Mirrors Ling-mini-2.0 in the real spec: override watchdog to 180s."""
        spec = self._build_minimal_symbiote_model_spec(
            env_vars={"TT_SYMBIOTE_PREFILL_WATCHDOG_SEC": "180"},
        )
        assert spec.env_vars["TT_SYMBIOTE_PREFILL_WATCHDOG_SEC"] == "180"
        # Other defaults still applied.
        assert spec.env_vars["TT_SYMBIOTE_DECODE_WATCHDOG_SEC"] == "30"

    def test_no_injection_for_non_symbiote_specs(self):
        ds = DeviceModelSpec(
            device=DeviceTypes.T3K,
            max_concurrency=32,
            max_context=4096,
            default_impl=True,
        )
        spec = ModelSpec(
            model_id="id_tt-transformers_fixture_t3k",
            impl=tt_transformers_impl,
            hf_model_repo="fixture/tt-transformers-test",
            model_name="tt-transformers-test",
            inference_engine=InferenceEngine.VLLM.value,
            device_type=DeviceTypes.T3K,
            tt_metal_commit="abc1234",
            device_model_spec=ds,
            vllm_commit="def5678",
        )
        for key in _TT_SYMBIOTE_DEFAULT_ENV_VARS:
            assert key not in spec.env_vars, (
                f"tt_transformers ModelSpec should NOT have tt_symbiote default {key!r} injected"
            )
