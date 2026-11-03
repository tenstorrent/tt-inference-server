# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
"""Explicit coordinator preflight/go evidence shared by scoped goal12 arms."""
import json
import os
from pathlib import Path


def verify_receipts(intent, eval_name, source_manifest_sha256, controls, environ=None):
    if intent == 'isolated_smoke':
        return {'intent':intent, 'scored':False, 'preflight':None, 'go':None}
    if intent not in ('cost_sample', 'paired_confirmation', 'full_suite'):
        raise ValueError('Unknown evaluation intent')
    env = os.environ if environ is None else environ
    if env.get("TT_QWEN38_LOCAL_SERVER_JSON"):
        raise ValueError("Local proof is not supported by this release route")
    def receipt(variable, field):
        path = env.get(variable) or controls.get(field)
        if not path:
            raise ValueError('Missing ' + variable + ' receipt')
        data = json.loads(Path(path).read_text())
        if data.get('model') != 'Qwen/Qwen3.8-27B':
            raise ValueError('Receipt model mismatch')
        if data.get('source_manifest_sha256') != source_manifest_sha256:
            raise ValueError('Receipt source manifest mismatch')
        if data.get("scope") == "local_measurement":
            raise ValueError("Local receipt cannot prove the release route")
        if controls.get("cohort") is not None:
            from reference_config.evals.qwen38_goal12 import compute_cohort_binding
            binding = compute_cohort_binding(controls)
            if (controls["cohort"] != binding or binding.get("scope") != "release"
                    or data.get("cohort") != binding):
                raise ValueError("Receipt explicit cohort binding mismatch")
        return data, str(path)

    preflight, preflight_path = receipt('TT_QWEN38_PREFLIGHT_JSON', 'preflight_json')
    for check in ('release_values', 'registry', 'disk', 'release_route', 'isolation_smoke'):
        if preflight.get(check, {}).get('ok') is not True:
            raise ValueError('Preflight not passed: ' + check)
    identity = preflight.get('runtime_sources', {})
    original = controls.get('selected_protocol') == 'original-stage11-cutoff-repair'
    if original:
        observed=preflight.get('observed_runtime_identity',{})
        marker=observed.get('generator_full64_marker',{})
        if (not identity.get('host_tis_sha') or not identity.get('image_digest')
                or observed.get('kind')!='owned_image_full64_policy_checkpoint'
                or observed.get('image_id')!=identity['image_digest']
                or marker.get('ok') is not True or marker.get('layer_count')!=64
                or not marker.get('policy_sha256') or not observed.get('checkpoint_receipt_sha256')
                or not observed.get('checkpoint_manifest_sha256') or not observed.get('serving_processes')):
            raise ValueError('Original repair needs observed image/full64/policy/checkpoint identity')
        for key in ('metal_sha','plugin_sha'):
            if identity.get(key) is None and not observed.get('missing_reasons',{}).get(key):
                raise ValueError('Unknown runtime SHA needs concrete missing reason')
    elif not all(identity.get(k) for k in ('host_tis_sha', 'metal_sha', 'plugin_sha', 'image_digest')):
        raise ValueError('Preflight must record host TIS, metal, plugin and image identity')
    if identity.get('reused_image') and not identity.get('equivalence_evidence'):
        raise ValueError('Reused image requires source equivalence evidence')
    go_path = None
    if intent == 'full_suite':
        go, go_path = receipt('TT_QWEN38_GO_JSON', 'go_json')
        decision = go.get('evals', {}).get(eval_name, {})
        if decision.get('decision') != 'go' or not decision.get('evidence'):
            raise ValueError('Full suite requires explicit go with evidence')
    return {'intent':intent, 'preflight':preflight_path, 'go':go_path,
            'runtime_sources':identity, 'source_manifest_sha256':source_manifest_sha256}
