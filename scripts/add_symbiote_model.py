#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Scaffolder for new tt_symbiote model integrations.

Emits the five artefacts a new tt_symbiote model needs (adapter subclass,
SYMBIOTE_MODEL_REGISTRY entry, ModelSpecTemplate block, EvalConfig stub,
model_performance_reference.json stub) ready for paste-in. Read-only by
default: prints to stdout with clear FILE markers indicating where each
block belongs.

Usage:
    python3 scripts/add_symbiote_model.py \\
        --hf-arch BailingMoeV2ForCausalLM \\
        --weights inclusionAI/Ling-mini-2.0 \\
        --short-name Ling-mini-2.0 \\
        --device t3k \\
        --max-context 2048 \\
        --max-concurrency 1 \\
        --tt-metal-commit 489d8b0 \\
        --vllm-commit 6f6d817

See docs/add_support_for_new_symbiote_model.md for the surrounding workflow.
"""

from __future__ import annotations

import argparse
import re
import sys
import textwrap
from typing import List


# ---------------------------------------------------------------------------
# Derived-name helpers
# ---------------------------------------------------------------------------


def _to_snake(name: str) -> str:
    """Convert any reasonable identifier into snake_case ASCII.

    Examples:
        Ling-mini-2.0 -> ling_mini_2_0
        Qwen3-32B     -> qwen3_32b
        Gemma-4-31B   -> gemma_4_31b
    """
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    return re.sub(r"_+", "_", cleaned)


def _to_upper_key(name: str) -> str:
    """Convert short-name into a MODEL_KEY suitable for env-var TT_SYMBIOTE_<KEY>_*.

    Strips trailing version suffixes ('-2.0', '-2-0') so e.g. 'Ling-mini-2.0' -> 'LING'
    rather than 'LING_MINI_2_0'. The user can override via --model-key.
    """
    base = re.split(r"[-_]\d", name, maxsplit=1)[0]
    return _to_snake(base).upper()


def _adapter_classname(hf_arch: str) -> str:
    """SymbioteFooForCausalLM derived from FooForCausalLM."""
    base = hf_arch
    for suffix in ("ForCausalLM", "ForConditionalGeneration", "Model"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            return f"Symbiote{base}{suffix}"
    return f"Symbiote{base}"


# ---------------------------------------------------------------------------
# Block emitters
# ---------------------------------------------------------------------------


def _emit_adapter(args: argparse.Namespace) -> str:
    classname = _adapter_classname(args.hf_arch)
    snake = _to_snake(args.short_name)
    model_key = args.model_key or _to_upper_key(args.short_name)
    target_path = (
        f"~/tt-metal/models/experimental/tt_symbiote/vllm/generator_vllm_{snake}.py"
    )
    return textwrap.dedent(
        f'''\
        # ===== FILE: {target_path} =====
        # SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
        # SPDX-License-Identifier: Apache-2.0

        """vLLM adapter for {args.short_name} ({args.hf_arch}) via tt_symbiote.

        Bridges vLLM's serving interface to the HF model whose modules have been
        replaced with TTNN equivalents. The boilerplate (DIAG, watchdog,
        _to_host_tensor, prefill/decode forward, warmup loops, KV cache
        hand-off) lives in
        models.experimental.tt_symbiote.vllm.symbiote_adapter_base.SymbioteAdapterBase;
        this module only carries the model-specific load + replacement +
        KV cache build.
        """

        import logging
        from typing import Optional

        import torch
        from tqdm import tqdm

        from models.experimental.tt_symbiote.vllm.symbiote_adapter_base import (
            SymbioteAdapterBase,
        )

        logger = logging.getLogger(__name__)


        class {classname}(SymbioteAdapterBase):
            """vLLM-compatible adapter for {args.short_name} on TT hardware.

            All four contract methods are inherited from SymbioteAdapterBase. This
            subclass overrides only _build_model_and_kv_cache (the model-specific
            HF load + module replacement + KV cache allocation).
            """

            MODEL_KEY = "{model_key}"
            # TODO: replace with the model's TTNN attention kernel module name
            # so the [WATCHDOG] log lines guide the operator to the right file.
            WATCHDOG_PREFILL_KERNEL_HINT = "{snake} attention"

            # TODO: cover every ISL the benchmark sweep exercises against
            # max_context={args.max_context}. Values <= max_position_embeddings are
            # filtered at runtime in warmup_model_prefill.
            WARMUP_PREFILL_SEQ_LENS = (128, 1024)

            @classmethod
            def _build_model_and_kv_cache(
                cls,
                hf_config,
                mesh_device,
                max_batch_size,
                max_seq_len,
                tt_data_parallel,
                optimizations: Optional[str] = None,
                **kwargs,
            ):
                """Load HF {args.short_name}, run module replacement, allocate KV cache.

                Returns (model, kv_cache, model_device); the base then runs
                model.eval(), grad disable, and the final device-property patch.
                """
                from transformers import AutoModelForCausalLM
                # TODO: import the TTNN modules used by your model below
                # from models.experimental.tt_symbiote.modules.<your_modules> import (
                #     <your TTNN classes>,
                # )
                from models.experimental.tt_symbiote.utils.device_management import set_device
                from models.experimental.tt_symbiote.utils.module_replacement import (
                    register_module_replacement_dict,
                )

                model_name = hf_config._name_or_path
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                )

                # Capture model_device BEFORE module replacement -- after replacement
                # TTNN modules lack _parameters so model.parameters() would fail.
                model_device = next(model.parameters()).device

                # TODO: build the TTNN replacement dict(s) for your model. See
                # generator_vllm.py (Gemma-4) for the 2-pass pattern, or
                # generator_vllm_ling.py (Ling) for the 3-pass pattern.
                # nn_to_ttnn = {{...}}
                # modules = register_module_replacement_dict(model, nn_to_ttnn, model_config=None)

                set_device(model, mesh_device)

                # logger.info(f"Preprocessing {{len(modules)}} TTNN modules weights...")
                # for name, mod in tqdm(modules.items(), desc="Preprocessing & moving weights"):
                #     mod.preprocess_weights()
                #     mod.move_weights_to_device()

                # TODO: allocate the paged KV cache appropriate to your model.
                # kv_cache = ...
                kv_cache = None  # replace before serving

                return model, kv_cache, model_device
        '''
    )


def _emit_registry_entry(args: argparse.Namespace) -> str:
    classname = _adapter_classname(args.hf_arch)
    snake = _to_snake(args.short_name)
    target_path = (
        "~/tt-metal/models/experimental/tt_symbiote/vllm/__init__.py"
        " (add the new entry to SYMBIOTE_MODEL_REGISTRY)"
    )
    return textwrap.dedent(
        f'''\
        # ===== FILE: {target_path} =====
            "TT{args.hf_arch}": (
                "models.experimental.tt_symbiote.vllm.generator_vllm_{snake}:{classname}"
            ),
        '''
    )


def _emit_model_spec(args: argparse.Namespace) -> str:
    device_const = args.device.upper()
    mesh_device = args.device.upper()
    fabric_line = (
        '                    "fabric_config": "FABRIC_1D_RING",'
        if args.device.lower() != "n150"
        else "                    # fabric_config not required for single-chip device"
    )
    return textwrap.dedent(
        f'''\
        # ===== FILE: tt-inference-server/workflows/model_spec.py =====
        # ===== Add this ModelSpecTemplate near the existing tt_symbiote entries
        # ===== (search for "impl=tt_symbiote_impl") =====
            ModelSpecTemplate(
                weights=["{args.weights}"],
                impl=tt_symbiote_impl,
                tt_metal_commit="{args.tt_metal_commit}",
                vllm_commit="{args.vllm_commit or "TODO"}",
                inference_engine=InferenceEngine.VLLM.value,
                device_model_specs=[
                    DeviceModelSpec(
                        device=DeviceTypes.{device_const},
                        max_concurrency={args.max_concurrency},
                        max_context={args.max_context},
                        default_impl=True,
                        vllm_args={{
                            "max_model_len": "{args.max_context}",
                            "max_num_seqs": "{args.max_concurrency}",
                            "block_size": "64",
                            "trust-remote-code": True,
                        }},
                        override_tt_config={{
                            "enable_model_warmup": True,
                            "trace_mode": "none",
                            "trace_region_size": 200000000,
        {fabric_line}
                        }},
                        env_vars={{
                            "TT_SYMBIOTE_DISPATCHER": "CPU",
                            "MESH_DEVICE": "{mesh_device}",
                            # ModelSpec.__post_init__ injects sensible defaults
                            # for TT_SYMBIOTE_DIAG / WATCHDOG / SYNC env vars; override
                            # below only when this model needs different values
                            # (e.g. raise PREFILL_WATCHDOG_SEC if cold-boot JIT > 60s).
                        }},
                    ),
                ],
                status=ModelStatusTypes.EXPERIMENTAL,
                has_builtin_warmup=True,
            ),
        '''
    )


def _emit_eval_config(args: argparse.Namespace) -> str:
    return textwrap.dedent(
        f'''\
        # ===== FILE: tt-inference-server/evals/eval_config.py =====
        # ===== Add this EvalConfig to EVAL_CONFIGS =====
        EvalConfig(
            hf_model_repo="{args.weights}",
            tasks=[
                # TODO: pick at least 2 tasks per model; see existing entries
                # (e.g. ifeval, mmlu_pro for Ling-mini-2.0). Set published_score
                # to None on first integration; populate once you have a
                # GPU-reference run.
                EvalTask(
                    task_name="ifeval",
                    score=EvalTaskScore(
                        published_score=None,
                        published_score_ref=None,
                        score_func=score_task_single_key,
                        score_func_kwargs={{
                            "result_keys": ["prompt_level_strict_acc,none"],
                            "unit": "percent",
                        }},
                    ),
                    workflow_venv_type=WorkflowVenvType.EVALS_COMMON,
                    include_path="work_dir",
                    apply_chat_template=True,
                    model_kwargs={{
                        "model": "{args.weights}",
                        "base_url": "http://127.0.0.1:8000/v1/completions",
                        "tokenizer_backend": "huggingface",
                        "max_length": {args.max_context},
                    }},
                    gen_kwargs={{"stream": "false", "max_gen_toks": "256"}},
                    seed=42,
                    num_fewshot=0,
                    log_samples=True,
                    limit_samples_map={{
                        EvalLimitMode.CI_NIGHTLY: 0.5,
                        EvalLimitMode.SMOKE_TEST: 0.01,
                    }},
                ),
            ],
        ),
        '''
    )


def _emit_perf_ref_stub(args: argparse.Namespace) -> str:
    return textwrap.dedent(
        f'''\
        # ===== FILE: tt-inference-server/benchmarking/benchmark_targets/model_performance_reference.json =====
        # ===== Add this entry. Theoretical numbers are placeholders; replace =====
        # ===== with measured baseline performance from your first benchmark run. =====
            "{args.short_name}": {{
                "{args.device.lower()}": [
                    {{
                        "isl": 128,
                        "osl": 128,
                        "max_concurrency": {args.max_concurrency},
                        "num_prompts": 8,
                        "task_type": "text",
                        "image_height": null,
                        "image_width": null,
                        "images_per_prompt": null,
                        "targets": {{
                            "theoretical": {{
                                "ttft_ms": null,
                                "tput_user": null
                            }}
                        }}
                    }},
                    {{
                        "isl": 1024,
                        "osl": 128,
                        "max_concurrency": {args.max_concurrency},
                        "num_prompts": 4,
                        "task_type": "text",
                        "targets": {{
                            "theoretical": {{
                                "ttft_ms": null,
                                "tput_user": null
                            }}
                        }}
                    }}
                ]
            }},
        '''
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--hf-arch",
        required=True,
        help="HuggingFace architecture name (architectures[0] in config.json), "
        'e.g. "BailingMoeV2ForCausalLM".',
    )
    parser.add_argument(
        "--weights",
        required=True,
        help='HuggingFace repo path, e.g. "inclusionAI/Ling-mini-2.0".',
    )
    parser.add_argument(
        "--short-name",
        required=True,
        help='Short name used in MODEL_SPECS keys and registry, e.g. "Ling-mini-2.0".',
    )
    parser.add_argument(
        "--device",
        required=True,
        choices=["n150", "n300", "t3k", "galaxy", "galaxy_t3k", "n150x4"],
        help="Target device. Multi-chip devices (everything except n150) need "
        "fabric_config in override_tt_config.",
    )
    parser.add_argument(
        "--max-context",
        required=True,
        type=int,
        help="Max context length in tokens (filters benchmark sweep rows by isl+osl).",
    )
    parser.add_argument(
        "--max-concurrency",
        required=True,
        type=int,
        help="Max concurrent sequences. Set to 1 unless model code supports batching.",
    )
    parser.add_argument(
        "--tt-metal-commit",
        required=True,
        help='Pinned tt-metal commit SHA (short or full), e.g. "489d8b0".',
    )
    parser.add_argument(
        "--vllm-commit",
        default=None,
        help="Pinned vllm fork commit SHA. Optional but recommended.",
    )
    parser.add_argument(
        "--model-key",
        default=None,
        help="Override the auto-derived MODEL_KEY (used in TT_SYMBIOTE_<KEY>_PREFILL_SYNC env "
        'var name and watchdog log strings). Default: derived from --short-name.',
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Print all five blocks with clear separators. The operator pastes each
    # block at the marked location; nothing is mutated in-place.
    blocks = [
        ("1/5 Adapter subclass (NEW FILE)", _emit_adapter(args)),
        ("2/5 SYMBIOTE_MODEL_REGISTRY entry", _emit_registry_entry(args)),
        ("3/5 ModelSpecTemplate block", _emit_model_spec(args)),
        ("4/5 EvalConfig block", _emit_eval_config(args)),
        ("5/5 model_performance_reference.json stub", _emit_perf_ref_stub(args)),
    ]

    sep = "=" * 76
    print(sep)
    print(
        f" tt_symbiote scaffolder for {args.short_name} ({args.hf_arch}) on {args.device}"
    )
    print(sep)
    print()
    print(
        "Five blocks follow. Paste each at its marked file path; nothing is\n"
        "mutated in place. After pasting:\n"
        "  - Fill in the TODOs in the new adapter file (TTNN module imports,\n"
        "    replacement dicts, KV cache build).\n"
        "  - Run the validator unit tests:\n"
        "      pytest tests/workflows/test_tt_symbiote_validator.py\n"
        "  - Bring up the local server with the canonical command (CLAUDE.md §4.1)\n"
        "    and confirm HTTP 200 on /v1/models.\n"
    )

    for title, body in blocks:
        print(sep)
        print(f" {title}")
        print(sep)
        print(body)
        print()

    print(sep)
    print(
        " Done. See docs/add_support_for_new_symbiote_model.md for the full\n"
        " bring-up checklist."
    )
    print(sep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
