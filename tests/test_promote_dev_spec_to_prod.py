# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

import json
import textwrap

from scripts.release.promote_dev_spec_to_prod import (
    DEFAULT_CI_CONFIG,
    DEFAULT_DEV_DIR,
    DEFAULT_PROD_DIR,
    ReleaseCombo,
    collect_release_combos,
    find_matches,
    iter_implementations,
    main,
    model_name_from_weight,
    promote,
    split_into_blocks,
    template_identity,
    template_matches,
    upsert_block,
)
from workflows.workflow_types import DeviceTypes, InferenceEngine


def test_model_name_from_weight_strips_org_prefix():
    assert model_name_from_weight("meta-llama/Llama-3.1-8B-Instruct") == (
        "Llama-3.1-8B-Instruct"
    )
    assert model_name_from_weight("openai/gpt-oss-20b") == "gpt-oss-20b"


def test_iter_implementations_flat_shape():
    entry = {"inference_engine": "FORGE", "ci": {"nightly": {"devices": ["P150"]}}}
    assert list(iter_implementations(entry)) == [entry]


def test_iter_implementations_array_shape():
    impl_a = {"inference_engine": "vLLM", "ci": {}}
    impl_b = {"inference_engine": "FORGE", "ci": {}}
    entry = {"implementations": [impl_a, impl_b]}
    assert list(iter_implementations(entry)) == [impl_a, impl_b]


def test_collect_release_combos_array_and_flat_shapes():
    ci_config = {
        "models": {
            "Llama-3.1-8B-Instruct": {
                "implementations": [
                    {
                        "inference_engine": "vLLM",
                        "ci": {
                            "nightly": {"devices": ["N150"]},
                            "release": {"devices": ["GALAXY", "P300X2"]},
                        },
                    },
                    {
                        "inference_engine": "FORGE",
                        "ci": {"nightly": {"devices": ["P150"]}},
                    },
                ]
            },
            "whisper-large-v3": {
                "inference_engine": "MEDIA",
                "ci": {"release": {"devices": ["P150"]}},
            },
            "Falcon3-7B-Instruct": {
                "inference_engine": "FORGE",
                "ci": {"nightly": {"devices": ["P150"]}},
            },
        }
    }
    combos = collect_release_combos(ci_config)
    assert combos == {
        ReleaseCombo("Llama-3.1-8B-Instruct", InferenceEngine.VLLM, DeviceTypes.GALAXY),
        ReleaseCombo("Llama-3.1-8B-Instruct", InferenceEngine.VLLM, DeviceTypes.P300X2),
        ReleaseCombo("whisper-large-v3", InferenceEngine.MEDIA, DeviceTypes.P150),
    }


def test_collect_release_combos_ignores_nightly_and_weekly():
    ci_config = {
        "models": {
            "m": {
                "inference_engine": "vLLM",
                "ci": {
                    "nightly": {"devices": ["GALAXY"]},
                    "weekly": {"devices": ["GALAXY"]},
                },
            }
        }
    }
    assert collect_release_combos(ci_config) == set()


def _llama_template():
    return {
        "weights": ["meta-llama/Llama-3.1-8B-Instruct"],
        "impl": "tt_transformers",
        "inference_engine": "VLLM",
        "device_model_specs": [
            {"device": "GALAXY", "max_concurrency": 32},
            {"device": "N150", "max_concurrency": 1},
            {"device": "P300X2", "max_concurrency": 8},
        ],
    }


def test_template_matches_on_basename_engine_and_device():
    combo = ReleaseCombo(
        "Llama-3.1-8B-Instruct", InferenceEngine.VLLM, DeviceTypes.GALAXY
    )
    assert template_matches(_llama_template(), combo) is True


def test_template_does_not_match_wrong_device():
    combo = ReleaseCombo("Llama-3.1-8B-Instruct", InferenceEngine.VLLM, DeviceTypes.T3K)
    assert template_matches(_llama_template(), combo) is False


def test_template_does_not_match_wrong_engine():
    combo = ReleaseCombo(
        "Llama-3.1-8B-Instruct", InferenceEngine.FORGE, DeviceTypes.GALAXY
    )
    assert template_matches(_llama_template(), combo) is False


def test_template_identity_is_impl_engine_weights_and_devices():
    # devices are part of the identity: the catalogue holds multiple blocks per
    # (impl, engine, weights), one per device group, so they must not collide.
    assert template_identity(_llama_template()) == (
        "tt_transformers",
        InferenceEngine.VLLM,
        frozenset({"meta-llama/Llama-3.1-8B-Instruct"}),
        frozenset({DeviceTypes.GALAXY, DeviceTypes.N150, DeviceTypes.P300X2}),
    )


def test_template_identity_distinguishes_blocks_with_different_devices():
    # Same model/impl/engine, different device groups → different identities.
    base = {
        "weights": ["meta-llama/Llama-3.1-8B-Instruct"],
        "impl": "tt_transformers",
        "inference_engine": "VLLM",
    }
    galaxy = {**base, "device_model_specs": [{"device": "GALAXY"}]}
    p300 = {**base, "device_model_specs": [{"device": "P300X2"}]}
    assert template_identity(galaxy) != template_identity(p300)


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text))


def test_find_matches_picks_whole_template_and_reports_unmatched(tmp_path):
    dev = tmp_path / "dev"
    _write(
        dev / "llm.yaml",
        """
        templates:
        - weights:
            - meta-llama/Llama-3.1-8B-Instruct
          impl: tt_transformers
          inference_engine: VLLM
          device_model_specs:
            - device: GALAXY
              max_concurrency: 32
            - device: N150
              max_concurrency: 1
        """,
    )
    combos = {
        ReleaseCombo("Llama-3.1-8B-Instruct", InferenceEngine.VLLM, DeviceTypes.GALAXY),
        ReleaseCombo("nonexistent", InferenceEngine.VLLM, DeviceTypes.GALAXY),
    }
    matches_by_file, unmatched = find_matches(dev, combos)

    assert list(matches_by_file.keys()) == ["llm.yaml"]
    picked = matches_by_file["llm.yaml"]
    assert len(picked) == 1
    # whole template: the non-release N150 device is still present
    devices = [d["device"] for d in picked[0].template["device_model_specs"]]
    assert devices == ["GALAXY", "N150"]
    # the raw source lines are carried for verbatim splicing
    assert "".join(picked[0].lines).startswith("- weights:")
    assert unmatched == {
        ReleaseCombo("nonexistent", InferenceEngine.VLLM, DeviceTypes.GALAXY)
    }


def test_find_matches_dedups_template_matched_by_two_combos(tmp_path):
    dev = tmp_path / "dev"
    _write(
        dev / "llm.yaml",
        """
        templates:
        - weights:
            - meta-llama/Llama-3.1-8B-Instruct
          impl: tt_transformers
          inference_engine: VLLM
          device_model_specs:
            - device: GALAXY
            - device: P300X2
        """,
    )
    combos = {
        ReleaseCombo("Llama-3.1-8B-Instruct", InferenceEngine.VLLM, DeviceTypes.GALAXY),
        ReleaseCombo("Llama-3.1-8B-Instruct", InferenceEngine.VLLM, DeviceTypes.P300X2),
    }
    matches_by_file, unmatched = find_matches(dev, combos)
    assert len(matches_by_file["llm.yaml"]) == 1
    assert unmatched == set()


def test_split_into_blocks_roundtrips_and_separates_filler():
    text = textwrap.dedent(
        """\
        templates:
        # banner
        - weights:
            - openai/gpt-oss-20b
          impl: gpt_oss
          inference_engine: VLLM

        - weights:
            - meta-llama/Llama-3.1-8B-Instruct
          impl: tt_transformers
          inference_engine: VLLM
        """
    )
    segments = split_into_blocks(text)
    # joining all segment lines reproduces the input byte-for-byte
    assert "".join(line for _, lines in segments for line in lines) == text
    kinds = [kind for kind, _ in segments]
    # header + banner are filler; two template blocks; blank line between is filler
    assert kinds.count("block") == 2
    blocks = ["".join(lines) for kind, lines in segments if kind == "block"]
    assert blocks[0].startswith("- weights:\n    - openai/gpt-oss-20b")
    assert blocks[1].startswith("- weights:\n    - meta-llama/Llama-3.1-8B-Instruct")


def test_split_into_blocks_keeps_blank_line_inside_a_body():
    # A blank line followed by more indented body is interior to the block, not a
    # separator — otherwise the block (and its parsed identity) would be truncated.
    text = textwrap.dedent(
        """\
        templates:
        - weights:
            - openai/gpt-oss-20b
          impl: gpt_oss
          inference_engine: VLLM

          device_model_specs:
            - device: T3K
        """
    )
    segments = split_into_blocks(text)
    assert "".join(line for _, lines in segments for line in lines) == text
    blocks = [lines for kind, lines in segments if kind == "block"]
    assert len(blocks) == 1
    from scripts.release.promote_dev_spec_to_prod import parse_block

    parsed = parse_block(blocks[0])
    # the device survived the interior blank line
    assert parsed["device_model_specs"] == [{"device": "T3K"}]


def _block_lines(text):
    text = textwrap.dedent(text)
    (kind, lines) = next(seg for seg in split_into_blocks(text) if seg[0] == "block")
    assert kind == "block"
    return lines


def test_upsert_block_appends_new_template():
    segments = split_into_blocks("templates:\n")
    lines = _block_lines(
        """
        - weights:
            - meta-llama/Llama-3.1-8B-Instruct
          impl: tt_transformers
          inference_engine: VLLM
          device_model_specs:
            - device: GALAXY
        """
    )
    identity = ("tt_transformers", InferenceEngine.VLLM, frozenset({"x"}))
    action = upsert_block(segments, identity, lines)
    assert action == "appended"
    assert sum(1 for kind, _ in segments if kind == "block") == 1


def test_upsert_block_replaces_same_identity_and_leaves_others_byte_identical():
    other = """\
- weights:
    - openai/gpt-oss-20b
  impl: gpt_oss
  inference_engine: VLLM
  version: "9.9.9"
  device_model_specs:
    - device: T3K
"""
    old = """\
- weights:
    - meta-llama/Llama-3.1-8B-Instruct
  impl: tt_transformers
  inference_engine: VLLM
  version: "0.1.0"
  device_model_specs:
    - device: GALAXY
"""
    segments = split_into_blocks("templates:\n" + other + old)

    new_lines = _block_lines(old.replace("0.1.0", "0.2.0"))
    from scripts.release.promote_dev_spec_to_prod import parse_block

    identity = template_identity(parse_block(_block_lines(old)))
    action = upsert_block(segments, identity, new_lines)

    assert action == "updated"
    rendered = "".join(line for _, lines in segments for line in lines)
    # the other template is untouched byte-for-byte; only the target version changed
    assert other in rendered
    assert "0.2.0" in rendered
    assert "0.1.0" not in rendered


def _build_tree(tmp_path):
    """dev has a Llama template with an inline comment; prod has an older copy."""
    dev = tmp_path / "dev"
    prod = tmp_path / "prod"
    _write(
        dev / "llm.yaml",
        """
        templates:
        - weights:
            - meta-llama/Llama-3.1-8B-Instruct
          impl: tt_transformers
          inference_engine: VLLM
          version: "0.2.0"
          device_model_specs:
            - device: GALAXY
              max_context: 16384  # 16 * 1024
            - device: N150
              max_context: 4096
        """,
    )
    _write(
        prod / "llm.yaml",
        """
        templates:
        - weights:
            - meta-llama/Llama-3.1-8B-Instruct
          impl: tt_transformers
          inference_engine: VLLM
          version: "0.1.0"
          device_model_specs:
            - device: GALAXY
              max_context: 16384
            - device: N150
              max_context: 4096
        """,
    )
    ci = tmp_path / "ci.json"
    ci.write_text(
        json.dumps(
            {
                "models": {
                    "Llama-3.1-8B-Instruct": {
                        "inference_engine": "vLLM",
                        "ci": {"release": {"devices": ["GALAXY"]}},
                    }
                }
            }
        )
    )
    return ci, dev, prod


def test_promote_updates_prod_preserving_comment_and_whole_template(tmp_path):
    ci, dev, prod = _build_tree(tmp_path)
    report = promote(ci, dev, prod, dry_run=False)

    text = (prod / "llm.yaml").read_text()
    assert "0.2.0" in text  # version bumped from dev
    assert "0.1.0" not in text  # old block replaced in place, not duplicated
    assert "# 16 * 1024" in text  # inline comment preserved
    assert "device: N150" in text  # whole template copied (non-release device)
    assert report["unmatched"] == set()
    assert "llm.yaml" in report["changed_files"]


def test_promote_appends_new_release_template_and_stays_valid_yaml(tmp_path):
    import yaml as _yaml

    dev = tmp_path / "dev"
    prod = tmp_path / "prod"
    # dev has a release template on P300X2 that prod does not have yet.
    _write(
        dev / "llm.yaml",
        """
        templates:
        - weights:
            - meta-llama/Llama-3.1-8B-Instruct
          impl: tt_transformers
          inference_engine: VLLM
          device_model_specs:
            - device: P300X2
              max_context: 4096
        """,
    )
    # prod has only an unrelated template; it must survive byte-for-byte. It also
    # deliberately ends WITHOUT a trailing newline to exercise the append seam.
    existing = (
        "templates:\n"
        "- weights:\n"
        "    - openai/gpt-oss-20b\n"
        "  impl: gpt_oss\n"
        "  inference_engine: VLLM\n"
        "  device_model_specs:\n"
        "    - device: T3K"  # no trailing newline
    )
    (prod).mkdir(parents=True, exist_ok=True)
    (prod / "llm.yaml").write_text(existing)
    ci = tmp_path / "ci.json"
    ci.write_text(
        json.dumps(
            {
                "models": {
                    "Llama-3.1-8B-Instruct": {
                        "inference_engine": "vLLM",
                        "ci": {"release": {"devices": ["P300X2"]}},
                    }
                }
            }
        )
    )

    report = promote(ci, dev, prod, dry_run=False)

    assert report["actions"]["llm.yaml"] == [
        (
            template_identity(
                {
                    "impl": "tt_transformers",
                    "inference_engine": "VLLM",
                    "weights": ["meta-llama/Llama-3.1-8B-Instruct"],
                    "device_model_specs": [{"device": "P300X2"}],
                }
            ),
            "appended",
        )
    ]
    text = (prod / "llm.yaml").read_text()
    # the pre-existing block is intact (not fused with the appended one)
    assert "    - device: T3K\n" in text
    # the result is still valid YAML with both templates
    parsed = _yaml.safe_load(text)
    weights = {tuple(t["weights"]) for t in parsed["templates"]}
    assert ("openai/gpt-oss-20b",) in weights
    assert ("meta-llama/Llama-3.1-8B-Instruct",) in weights


def test_promote_dry_run_writes_nothing(tmp_path):
    ci, dev, prod = _build_tree(tmp_path)
    before = (prod / "llm.yaml").read_text()
    report = promote(ci, dev, prod, dry_run=True)
    # Disk is untouched, but the intended change is still reported.
    assert (prod / "llm.yaml").read_text() == before
    assert "llm.yaml" in report["changed_files"]


def test_promote_is_idempotent(tmp_path):
    ci, dev, prod = _build_tree(tmp_path)
    promote(ci, dev, prod, dry_run=False)
    after_first = (prod / "llm.yaml").read_text()
    report = promote(ci, dev, prod, dry_run=False)
    assert (prod / "llm.yaml").read_text() == after_first
    assert report["changed_files"] == []


def test_promote_reports_unmatched_combo(tmp_path):
    ci, dev, prod = _build_tree(tmp_path)
    ci.write_text(
        json.dumps(
            {
                "models": {
                    "ghost-model": {
                        "inference_engine": "vLLM",
                        "ci": {"release": {"devices": ["GALAXY"]}},
                    }
                }
            }
        )
    )
    report = promote(ci, dev, prod, dry_run=False)
    assert (
        ReleaseCombo("ghost-model", InferenceEngine.VLLM, DeviceTypes.GALAXY)
        in report["unmatched"]
    )


def test_main_returns_zero_and_writes(tmp_path, capsys):
    ci, dev, prod = _build_tree(tmp_path)
    rc = main(["--ci-config", str(ci), "--dev-dir", str(dev), "--prod-dir", str(prod)])
    assert rc == 0
    assert "0.2.0" in (prod / "llm.yaml").read_text()


def test_main_dry_run_writes_nothing(tmp_path):
    ci, dev, prod = _build_tree(tmp_path)
    before = (prod / "llm.yaml").read_text()
    rc = main(
        [
            "--ci-config",
            str(ci),
            "--dev-dir",
            str(dev),
            "--prod-dir",
            str(prod),
            "--dry-run",
        ]
    )
    assert rc == 0
    assert (prod / "llm.yaml").read_text() == before


def test_main_returns_nonzero_on_unmatched(tmp_path, capsys):
    ci, dev, prod = _build_tree(tmp_path)
    ci.write_text(
        json.dumps(
            {
                "models": {
                    "ghost-model": {
                        "inference_engine": "vLLM",
                        "ci": {"release": {"devices": ["GALAXY"]}},
                    }
                }
            }
        )
    )
    rc = main(["--ci-config", str(ci), "--dev-dir", str(dev), "--prod-dir", str(prod)])
    assert rc == 1
    assert "ghost-model" in capsys.readouterr().out


def test_real_repo_release_combos_all_match_dev():
    """Every release-marked combo in the real ci-config exists in the dev catalogue."""
    ci_config = json.loads(DEFAULT_CI_CONFIG.read_text())
    combos = collect_release_combos(ci_config)
    assert combos, "expected at least one release combo in the real ci-config"
    _, unmatched = find_matches(DEFAULT_DEV_DIR, combos)
    assert unmatched == set(), f"release combos missing from dev: {unmatched}"


def test_real_repo_dry_run_against_prod_succeeds(tmp_path):
    """Dry-run against the real catalogues runs cleanly and writes nothing to repo."""
    report = promote(DEFAULT_CI_CONFIG, DEFAULT_DEV_DIR, DEFAULT_PROD_DIR, dry_run=True)
    assert report["unmatched"] == set()


def test_real_repo_promote_is_noop_when_dev_equals_prod(tmp_path):
    """Promoting must not reformat untouched templates.

    dev and prod are currently byte-identical, so every release template already
    matches its prod copy exactly. A real write must therefore change nothing and
    leave each prod file byte-for-byte identical.
    """
    import shutil

    prod_copy = tmp_path / "prod"
    shutil.copytree(DEFAULT_PROD_DIR, prod_copy)
    before = {p.name: p.read_text() for p in prod_copy.glob("*.yaml")}

    report = promote(DEFAULT_CI_CONFIG, DEFAULT_DEV_DIR, prod_copy, dry_run=False)

    assert report["changed_files"] == [], (
        f"promote reformatted untouched templates: {report['changed_files']}"
    )
    after = {p.name: p.read_text() for p in prod_copy.glob("*.yaml")}
    assert after == before
