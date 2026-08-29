import copy
import json
from pathlib import Path

import pytest

from scripts.prepare_gemma_quetzal_enrollment import (
    EnrollmentError,
    HF_REVISION,
    INIT_SHA256,
    MODEL,
    PATCHSET,
    QUETZAL_SOURCE,
    RUNNER,
    SCHEMA,
    SHIELD_SOURCE,
    TTIS_SOURCE,
    TT_METAL,
    render_fragments,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ID = "sha256-" + "1" * 64 + "-" + "2" * 64


def evidence():
    return {
        "schema_version": SCHEMA, "decision": "approved", "administrator_owned": True,
        "read_only": True, "no_writable_aliases": True, "revocation_status": "active",
        "identity": {"model_id": MODEL, "hf_revision": HF_REVISION,
                     "quetzal_source_revision": QUETZAL_SOURCE, "ttis_revision": TTIS_SOURCE,
                     "shield_revision": SHIELD_SOURCE, "tt_metal_revision": TT_METAL,
                     "tt_metal_patchset_sha256": PATCHSET, "patchset_applied_manifest_matches": True,
                     "initialization_milestones_sha256": INIT_SHA256},
        "package_id": PACKAGE_ID, "package_manifest_sha256": "3" * 64,
        "host_package_root": f"/mnt/models/quetzal/immutable/v1/{PACKAGE_ID}",
        "container_package_root": f"/home/container_app_user/quetzal/packages/{PACKAGE_ID}",
        "profile": {"batch_size": 1, "concurrency": 1, "prefill_capacity": 1024,
                    "decode_capacity": 2048, "precision": "BFP8"},
        "topology": {"chip_count": 4, "mesh_shape": [2, 2], "collective": "Ring",
                     "links": 2, "runner_label": RUNNER},
        "roles": {"compiled_weights": "compiled_weights/gemma/weights.pt",
                  "generated_prefill": "compiled/gemma/prefill/generated.py",
                  "generated_decode": "compiled/gemma/decode/generated.py",
                  "qualification_manifest": "qualification_manifest.yaml"},
        "qualification": {"pcc": 0.991, "fresh": True, "exact_package_identity": PACKAGE_ID,
                          "endpoint_isl": 1024, "endpoint_osl": 512, "http_200": True,
                          "clean_unload": True, "zero_device_holders_after": True,
                          "initialization_terminal": {"event": "engine_ready", "state": "complete"}},
    }


def test_exact_evidence_renders_schema_valid_non_dispatching_fragments():
    rendered = render_fragments(evidence(), ROOT)
    row = rendered["implementation"]
    assert set(row["ci"]) == {"nightly", "release"}
    assert rendered["catalogue"]["templates"][0]["device_model_specs"][0]["max_context"] == 2048
    env = rendered["catalogue"]["templates"][0]["device_model_specs"][0]["env_vars"]
    assert env["TTQ_ROW_ALL_REDUCE_TOPOLOGY"] == "Ring"
    assert env["TTQ_TUNED_ROW_ALL_REDUCE_LINKS"] == "2"
    assert env["VLLM_PLUGINS"] == "quetzal_model_registry,tt"
    assert env["TT_VLLM_BUILTIN_MODELS"] == "0"
    assert rendered["handoff"]["quetzal_source_revision"] == QUETZAL_SOURCE
    assert rendered["handoff"]["fallback_allowed"] is False


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda x: x.update(decision="pending"), "decision"),
        (lambda x: x["identity"].update(quetzal_source_revision="0" * 40), "quetzal_source_revision"),
        (lambda x: x["profile"].update(decode_capacity=1024), "profile"),
        (lambda x: x["topology"].update(runner_label="p300x2"), "runner_label"),
        (lambda x: x["qualification"].update(pcc=0.989), "pcc"),
        (lambda x: x["qualification"].update(endpoint_osl=511), "endpoint_osl"),
        (lambda x: x["qualification"].update(initialization_terminal=None), "initialization_terminal"),
        (lambda x: x["roles"].update(generated_decode="../generated.py"), "contained relative"),
    ],
)
def test_dispatch_critical_mismatch_fails_closed(mutation, match):
    bad = copy.deepcopy(evidence())
    mutation(bad)
    with pytest.raises(EnrollmentError, match=match):
        render_fragments(bad, ROOT)


def test_active_config_intentionally_has_no_gemma_quetzal_lane():
    config = json.loads((ROOT / ".github/workflows/models-ci-config.json").read_text())
    rows = config["models"]["gemma-4-31B-it"]["implementations"]
    assert all(row.get("impl") != "quetzal" for row in rows)
    blocker = json.loads((ROOT / "productization/gemma4_31b_models_ci_enrollment.blocked.json").read_text())
    assert blocker["status"] == "blocked_not_dispatchable"
