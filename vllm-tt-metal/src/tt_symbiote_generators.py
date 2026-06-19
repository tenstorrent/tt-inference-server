# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Generic, tiered vLLM Generator adapter for tt_symbiote-ported models.

`tt_symbiote` is deliberately HuggingFace-shaped (``from_pretrained`` ->
``set_device`` -> ``generate``) and ships **no** vLLM/server code. This module is
the serving glue — it lives entirely in tt-inference-server and wraps any
tt_symbiote model in the tt-metal :class:`Generator` contract that the
Tenstorrent vLLM plugin drives.

Scalability: ONE generic adapter serves every model. A per-model *serving tier*
(declared model-authoritatively in ``tt_symbiote`` ``RUNTIME_PINS[arch]
["serving_tier"]``) selects the prefill/decode strategy. Adding model #101 is a
single registry row — no new adapter class.

Tiers (see docs/development/tt_inference_server_integration.md §9 in tt_symbiote):
  * S0_GREEDY_ENGINE  - model emits tokens (on-device argmax); bridged to vLLM
                        via the one-hot-logits trick. Greedy, model-managed KV,
                        max_num_seqs=1. (e.g. dots.ocr ``TTNNDotsOCRPipeline``)
  * S1_LOGITS_UNPAGED - model.forward returns logits, KV not vLLM-paged; one
                        request at a time, vLLM samples. (fallback)
  * S2_PAGED          - model.forward over a vLLM-page-table-aware paged KV
                        cache; full continuous batching. (target for text LLMs;
                        needs the shared set_vllm_page_table hook in tt_symbiote)

Contract reference: ``$TT_METAL_HOME/models/tt_transformers/tt/generator.py``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _ensure_transformers_vision2seq_alias() -> None:
    """Bridge a transformers 4.x→5.x rename for tt-metal's import chain.

    tt_symbiote requires transformers 5.9.0, which removed ``AutoModelForVision2Seq``
    (renamed to ``AutoModelForImageTextToText`` — same auto class). tt-metal's vLLM
    ``Generator`` base transitively imports ``models/common/llama_models.py``, which
    still does ``from transformers import AutoModelForVision2Seq`` (true at both the
    dots.ocr pin c09f09c3 and later commits). Pure tt_symbiote HF usage never imports
    that module, so this incompatibility only appears on the tt-inference-server
    integration path — making the integration layer the right place to bridge it.
    Aliasing the old name is semantically exact (it is the same class, renamed).
    """
    import transformers

    if not hasattr(transformers, "AutoModelForVision2Seq") and hasattr(
        transformers, "AutoModelForImageTextToText"
    ):
        transformers.AutoModelForVision2Seq = transformers.AutoModelForImageTextToText


_ensure_transformers_vision2seq_alias()

# tt-metal's Generator base. Importable in the image because TT_METAL_HOME is on
# PYTHONPATH; importing this module does not import ttnn/torch eagerly.
from models.tt_transformers.tt.generator import Generator  # noqa: E402

# Serving tier constants (mirror tt_symbiote._runtime_pins.SERVING_TIERS).
S0_GREEDY_ENGINE = "S0_GREEDY_ENGINE"
S1_LOGITS_UNPAGED = "S1_LOGITS_UNPAGED"
S2_PAGED = "S2_PAGED"

# Large logit so vLLM's greedy sampler argmax-selects exactly the pipeline's
# token (one-hot-logits bridge for S0). Any strictly-dominant value works.
_ONEHOT_LOGIT = 30.0


@dataclass(frozen=True)
class ServingRecipe:
    """How a tt_symbiote HF architecture is served under vLLM."""

    hf_arch: str  # HF config.architectures[0], e.g. "DotsOCRForCausalLM"
    tier: str
    multimodal: bool = False


# Local fallback, used only if tt_symbiote metadata cannot be imported at
# registration time. The authoritative source is tt_symbiote RUNTIME_PINS.
_FALLBACK_RECIPES: Dict[str, ServingRecipe] = {
    "DotsOCRForCausalLM": ServingRecipe(
        "DotsOCRForCausalLM", S0_GREEDY_ENGINE, multimodal=True
    ),
}

_MULTIMODAL_EXTRAS = ("vision", "video", "audio")


def get_serving_recipes() -> Dict[str, ServingRecipe]:
    """Map HF architecture -> ServingRecipe.

    Source of truth: tt_symbiote ``RUNTIME_PINS`` (model-authoritative — the tier
    lives next to the tt-metal commit pin). Falls back to a local table if
    tt_symbiote is not importable (keeps registration robust).
    """
    recipes: Dict[str, ServingRecipe] = dict(_FALLBACK_RECIPES)
    try:
        from tt_symbiote.models._runtime_pins import RUNTIME_PINS, serving_tier_for

        for arch, pin in RUNTIME_PINS.items():
            recipes[arch] = ServingRecipe(
                hf_arch=arch,
                tier=serving_tier_for(arch),
                multimodal=any(e in _MULTIMODAL_EXTRAS for e in pin.get("extras", [])),
            )
    except Exception as e:  # pragma: no cover - import-environment dependent
        logger.warning(
            "tt_symbiote serving recipes: falling back to local table (%s)", e
        )

    # Opt-in serving-tier override (non-destructive S2 migration switch).
    # `TT_SYMBIOTE_SERVING_TIER` is a comma list of `Arch=Tier` pairs, e.g.
    # `DotsOCRForCausalLM=S2_PAGED`. Leaving it unset keeps the model-authoritative
    # pin (dots.ocr stays S0_GREEDY_ENGINE), so the validated demo is never broken;
    # set it to evaluate the S2 paged-logits path without editing _runtime_pins.py.
    override = os.environ.get("TT_SYMBIOTE_SERVING_TIER", "").strip()
    if override:
        valid = {S0_GREEDY_ENGINE, S1_LOGITS_UNPAGED, S2_PAGED}
        for pair in override.split(","):
            if "=" not in pair:
                continue
            arch, tier = (s.strip() for s in pair.split("=", 1))
            if tier not in valid:
                logger.warning("ignoring TT_SYMBIOTE_SERVING_TIER %r: unknown tier %r", pair, tier)
                continue
            base = recipes.get(arch)
            recipes[arch] = ServingRecipe(
                hf_arch=arch,
                tier=tier,
                multimodal=base.multimodal if base is not None else False,
            )
            logger.info("tt_symbiote serving tier override: %s -> %s", arch, tier)
    return recipes


class _TTSymbioteGenerator(Generator):
    """Generic tiered adapter; tier subclasses set SERVING_TIER + capabilities.

    We intentionally do NOT call ``Generator.__init__`` — it assumes
    tt_transformers ``ModelArgs`` / trace bookkeeping that tt_symbiote models do
    not use. The TTModelRunner-called hooks we don't need (warmup, kv spec) are
    overridden as no-ops for model-managed-KV tiers.
    """

    SERVING_TIER: str = S1_LOGITS_UNPAGED
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
    }

    def __init__(self, hf_model: Any, mesh_device: Any, recipe: ServingRecipe):
        self.hf_model = hf_model
        self.mesh_device = mesh_device
        self.recipe = recipe
        # tt_symbiote attaches the optimized TTNN pipeline (S0 models) at
        # set_device time via recipe.make_kv_cache.
        self.pipeline = getattr(hf_model, "_tt_pipeline", None)
        # S2 models attach a TTNNPagedAttentionKVCache to ``_tt_kv_cache`` at
        # set_device time. vLLM re-sizes it to its own block budget in
        # allocate_kv_cache; until then this is a sensible default.
        self._paged_cache = getattr(hf_model, "_tt_kv_cache", None)
        self.data_parallel = 1
        self.mode = None
        self._warmed_up = False
        # TS-7: prefill-token budget (vLLM's max_num_batched_tokens), installed by
        # the runner via set_prefill_chunk_size. ``None`` keeps the validated
        # single-shot (traced) prefill; when a prompt exceeds the budget the S2
        # path loops the eager chunked prefill over the paged KV.
        self._prefill_chunk_size: Optional[int] = None

    # ------------------------------------------------------------------
    # Construction (vLLM factory entry point, called by TTModelLoader)
    # ------------------------------------------------------------------
    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config: Any,
        mesh_device: Any,
        max_batch_size: int,
        max_seq_len: Optional[int] = None,
        tt_data_parallel: int = 1,
        optimizations: Any = None,
        **kwargs: Any,
    ) -> "_TTSymbioteGenerator":
        import torch
        from tt_symbiote import AutoModelForCausalLM, set_device

        arch = (getattr(hf_config, "architectures", None) or [None])[0]
        recipe = get_serving_recipes().get(arch) or ServingRecipe(
            arch or "?", cls.SERVING_TIER
        )
        model_path = cls._resolve_model_path(hf_config)
        logger.info(
            "tt_symbiote: loading %s (arch=%s, tier=%s, mm=%s, max_seq_len=%s)",
            model_path,
            arch,
            cls.SERVING_TIER,
            recipe.multimodal,
            max_seq_len,
        )
        # vLLM's TT platform opens the mesh from MESH_DEVICE, which a model spec
        # may set to a literal shape tuple (e.g. "(8, 1)" so dots.ocr's required
        # T3K data-parallel mesh is created). But tt_symbiote resolves the device
        # *arch* from MESH_DEVICE as an arch NAME ("T3K") in several build-time
        # places (e.g. the dots.ocr attention-class selection,
        # _select_attention_class) and forward-time @run_on_devices guards. Now
        # that the worker has already opened the mesh, normalize MESH_DEVICE to
        # the arch name implied by the *live* device so every tt_symbiote arch
        # lookup resolves. determine_device_name disambiguates Wormhole/Blackhole
        # via ttnn.get_arch_name() + device count.
        try:
            from tt_symbiote.core.arch import determine_device_name

            arch_name = determine_device_name(mesh_device)
            if arch_name and os.environ.get("MESH_DEVICE") != arch_name:
                logger.info(
                    "tt_symbiote: normalizing MESH_DEVICE %r -> %r (from live mesh)",
                    os.environ.get("MESH_DEVICE"),
                    arch_name,
                )
                os.environ["MESH_DEVICE"] = arch_name
        except Exception as e:
            logger.warning(
                "tt_symbiote: could not normalize MESH_DEVICE from live device: %s", e
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        # Binds the worker's (externally opened) mesh; builds TTNN pipeline/cache.
        set_device(model, mesh_device)
        return cls(model, mesh_device, recipe)

    @staticmethod
    def _resolve_model_path(hf_config: Any) -> str:
        hf_model_env = os.environ.get("HF_MODEL")
        if hf_model_env:
            return hf_model_env
        name_or_path = getattr(hf_config, "_name_or_path", None)
        if name_or_path:
            return name_or_path
        raise RuntimeError(
            "tt_symbiote adapter: cannot resolve model path; set HF_MODEL or "
            "provide hf_config._name_or_path."
        )

    # ------------------------------------------------------------------
    # TTModelRunner hooks we bypass for model-managed-KV tiers (S0/S1)
    # ------------------------------------------------------------------
    def warmup_model_prefill(self, *args: Any, **kwargs: Any) -> None:
        # S0 pipelines warm themselves on first prefill; nothing to do here.
        return None

    def warmup_model_decode(self, *args: Any, **kwargs: Any) -> None:
        return None

    def set_prefill_chunk_size(self, max_num_batched_tokens: Any) -> None:
        """TS-7: record vLLM's per-step prefill-token budget for chunked prefill.

        vLLM's own chunked-prefill scheduler stays disabled for TT (the plugin
        asserts ``chunked_prefill_enabled is False`` and feeds whole prompts), so
        a prompt longer than this budget is chunked *inside* ``_prefill_s2`` via
        the pipeline's eager chunked path (fill-then-attend over the paged KV).
        Snap to a tile-friendly, block-aligned multiple of 256 so each
        ``chunk_start_idx`` keeps a large power-of-two divisor (the chunked-SDPA
        ``q_chunk_size`` constraint). ``None``/non-positive disables chunking.
        """
        try:
            budget = int(max_num_batched_tokens)
        except (TypeError, ValueError):
            self._prefill_chunk_size = None
            return
        if budget <= 0:
            self._prefill_chunk_size = None
            return
        self._prefill_chunk_size = max(256, (budget // 256) * 256)

    # ------------------------------------------------------------------
    # KV cache
    # ------------------------------------------------------------------
    def _validate_pipeline_kv_geometry(self, kv_cache_shape: Any) -> None:
        """T4: check vLLM's proposed KV geometry against the pipeline cache.

        vLLM passes ``kv_cache_shape = (num_blocks, num_kv_heads, block_size,
        head_size)``. The pipeline owns the real device cache, so we do not build
        from this tuple, but a mismatch on ``head_size`` / ``block_size`` means
        the vLLM block manager and the device kernels disagree about KV layout --
        that corrupts attention. Raise on those. ``num_blocks`` is allowed to
        differ (the pipeline sizes its own vision-aware block budget), and
        ``num_kv_heads`` may differ by an integer mesh head-shard factor (the
        plugin shards heads TP-style while a DP pipeline replicates the full
        cache per device); both are surfaced as info only.
        """
        cache = self._paged_cache
        cfg = getattr(cache, "config", None)
        if cfg is None:
            return
        pipe_geom = {
            "num_kv_heads": int(getattr(cache, "num_kv_heads", -1)),
            "head_size": int(getattr(cache, "head_dim", -1)),
            "block_size": int(getattr(cfg, "block_size", -1)),
            "num_blocks": int(getattr(cfg, "max_num_blocks", -1)),
        }
        if kv_cache_shape is None or len(kv_cache_shape) != 4:
            logger.warning(
                "tt_symbiote S2/T4: vLLM kv_cache_shape=%r is not a 4-tuple; "
                "skipping geometry validation (pipeline cache geometry=%s)",
                kv_cache_shape,
                pipe_geom,
            )
            return
        v_num_blocks, v_num_kv_heads, v_block_size, v_head_size = (int(x) for x in kv_cache_shape)
        mismatches = []
        # ``num_kv_heads``: the vLLM plugin shards KV heads across the mesh under a
        # tensor-parallel assumption (``spec.num_kv_heads // min(num_devices,
        # num_kv_heads)``). dots.ocr instead runs *data parallel*: the paged cache
        # is REPLICATED across the mesh, so every device holds the full
        # ``pipe num_kv_heads``. The vLLM block manager only uses its (sharded)
        # head count for byte/num_blocks accounting -- which the pipeline overrides
        # with its own block budget -- and block *indexing* (page table -> physical
        # block id) is head-count-agnostic. So an integer head-shard factor
        # (``pipeline % vLLM == 0``) is a benign mesh-accounting artifact, not KV
        # corruption. Only a non-divisor head count is a genuine layout conflict.
        pipe_kv_heads = pipe_geom["num_kv_heads"]
        if v_num_kv_heads != pipe_kv_heads:
            if v_num_kv_heads >= 1 and pipe_kv_heads % v_num_kv_heads == 0:
                logger.info(
                    "tt_symbiote S2/T4: vLLM num_kv_heads=%s differs from pipeline "
                    "cache num_kv_heads=%s by an integer mesh head-shard factor "
                    "(%sx); the pipeline owns a replicated (DP) paged cache and "
                    "block indexing is head-agnostic, so this is expected.",
                    v_num_kv_heads,
                    pipe_kv_heads,
                    pipe_kv_heads // v_num_kv_heads,
                )
            else:
                mismatches.append(
                    f"num_kv_heads vLLM={v_num_kv_heads} pipeline={pipe_kv_heads} "
                    "(not an integer mesh head-shard factor)"
                )
        if v_head_size != pipe_geom["head_size"]:
            mismatches.append(f"head_size vLLM={v_head_size} pipeline={pipe_geom['head_size']}")
        if v_block_size != pipe_geom["block_size"]:
            mismatches.append(f"block_size vLLM={v_block_size} pipeline={pipe_geom['block_size']}")
        if mismatches:
            raise ValueError(
                "tt_symbiote S2/T4: vLLM KV geometry is incompatible with the pipeline "
                "paged cache; the block manager and device kernels would disagree on KV "
                "layout (KV corruption). Mismatches: " + "; ".join(mismatches) + ". "
                "Align the vLLM model config (num_kv_heads / head_size / block_size) with "
                f"the pipeline cache geometry {pipe_geom}."
            )
        if v_num_blocks != pipe_geom["num_blocks"]:
            logger.info(
                "tt_symbiote S2/T4: vLLM num_blocks=%s differs from pipeline cache "
                "num_blocks=%s; using the pipeline's own (vision-aware) block budget.",
                v_num_blocks,
                pipe_geom["num_blocks"],
            )

    def allocate_kv_cache(
        self,
        kv_cache_shape: Any = None,
        dtype: Any = None,
        num_layers: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Build the paged KV cache vLLM will drive (S2 only).

        The vLLM TTModelRunner calls this with
        ``kv_cache_shape = (num_blocks, num_kv_heads, block_size, head_size)``
        — exactly tt_symbiote's :class:`TTNNPagedAttentionKVCache` cache shape —
        and uses the return value as the ``kv_cache`` kwarg to prefill/decode.

        For S0/S1 the model owns its KV cache and vLLM does not page it, so we
        return ``None``.
        """
        if self.SERVING_TIER != S2_PAGED:
            return None

        # Pipeline-backed models (e.g. dots.ocr) own a device paged KV cache built
        # by their recipe with the model's own DP batch-shard mapper and
        # vision-sized blocks. Reuse it directly rather than building a fresh
        # `modules` cache: the S2 forward path drives `pipeline.forward_logits_*`,
        # which reads/writes `pipeline.paged_cache`. vLLM's block manager still
        # owns block assignment via `set_vllm_page_table` (installed per request in
        # _prefill_s2/_decode_s2).
        if self.pipeline is not None and getattr(self.pipeline, "paged_cache", None) is not None:
            self._paged_cache = self.pipeline.paged_cache
            # T4: the pipeline sizes its own (vision-aware) paged cache, so we
            # reuse it rather than the geometry vLLM proposes. But a vLLM block
            # manager configured with a KV geometry the pipeline cache cannot
            # honor would silently corrupt KV (wrong head/dim/block size) -- so
            # validate the proposed 4-tuple against the pipeline cache and fail
            # loudly on any correctness-affecting mismatch.
            self._validate_pipeline_kv_geometry(kv_cache_shape)
            logger.info(
                "tt_symbiote S2: reusing pipeline paged KV cache (DP-mapper) for %s",
                type(self.pipeline).__name__,
            )
            return self._paged_cache

        import torch
        from tt_symbiote.modules.ttnn_attention import (
            PagedAttentionConfig,
            TTNNPagedAttentionKVCache,
        )

        if kv_cache_shape is None or len(kv_cache_shape) != 4:
            raise ValueError(
                f"S2 allocate_kv_cache expects a 4-tuple "
                f"(num_blocks, num_kv_heads, block_size, head_size); got {kv_cache_shape!r}"
            )
        num_blocks, num_kv_heads, block_size, head_size = kv_cache_shape
        layers = int(num_layers) if num_layers is not None else self.hf_model.config.num_hidden_layers

        config = PagedAttentionConfig(
            block_size=int(block_size),
            max_num_blocks=int(num_blocks),
            batch_size=1,
        )
        cache = TTNNPagedAttentionKVCache(
            num_layers=layers,
            num_kv_heads=int(num_kv_heads),
            head_dim=int(head_size),
            config=config,
            device=None,
            dtype=torch.bfloat16,
        ).to_device(self.mesh_device)
        # Make the model forward use this exact cache as its HF past_key_values.
        self._paged_cache = cache
        self.hf_model._tt_kv_cache = cache
        logger.info(
            "tt_symbiote S2: allocated paged KV cache shape=%s layers=%s",
            tuple(kv_cache_shape),
            layers,
        )
        return cache

    # ------------------------------------------------------------------
    # prefill / decode dispatch by tier
    # ------------------------------------------------------------------
    def prefill_forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.SERVING_TIER == S0_GREEDY_ENGINE:
            return self._prefill_s0(*args, **kwargs)
        if self.SERVING_TIER == S2_PAGED:
            return self._prefill_s2(*args, **kwargs)
        raise NotImplementedError(
            f"prefill_forward for tier {self.SERVING_TIER} not implemented; "
            "see design §9.2 (S1: per-request logits loop)."
        )

    def decode_forward(self, *args: Any, **kwargs: Any) -> Any:
        if self.SERVING_TIER == S0_GREEDY_ENGINE:
            return self._decode_s0(*args, **kwargs)
        if self.SERVING_TIER == S2_PAGED:
            return self._decode_s2(*args, **kwargs)
        raise NotImplementedError(
            f"decode_forward for tier {self.SERVING_TIER} not implemented; "
            "see design §9.2."
        )

    def process_decode_output_host(self, tt_out: Any, is_tokens: bool = False) -> Any:
        """Pass-through host post-processing for the plugin's async-decode path.

        ``async_decode.finalize_decode`` calls this whenever the model defines it.
        The tt-metal ``Generator`` base implementation converts on-device ttnn
        outputs to torch using ``self.model_args``/per-DP model objects — neither
        of which a tt_symbiote adapter has. Every tt_symbiote tier's
        ``decode_forward`` already returns host torch logits (S0's one-hot bridge,
        S1/S2 real logits), which is exactly what ``finalize_decode`` falls back
        to consuming when ``process_decode_output_host`` is absent. Overriding it
        as a pass-through keeps that contract without the on-device-token machinery.
        """
        return tt_out

    # ------------------------------------------------------------------
    # S0: one-hot-logits bridge over a token-emitting pipeline
    # ------------------------------------------------------------------
    def _prefill_s0(
        self,
        tokens: Any,
        page_table: Any = None,
        kv_cache: Any = None,
        prompt_lens: Any = None,
        empty_slots: Any = None,
        **kwargs: Any,
    ) -> Any:
        """Run pipeline prefill; return one-hot logits for the first token.

        vLLM owns scheduling/sampling; ``page_table``/``kv_cache`` are ignored
        (model-managed KV). Multimodal image tensors arrive via kwargs from the
        vLLM multimodal pipeline (``pixel_values`` / ``image_grid_thw``).
        """
        if self.pipeline is None:
            raise RuntimeError(
                "tt_symbiote S0 adapter: model has no _tt_pipeline; set_device "
                "must build it (e.g. dots.ocr recipe.make_kv_cache)."
            )
        import torch

        input_ids = tokens if torch.is_tensor(tokens) else torch.as_tensor(tokens)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        # vLLM's TT model_runner delivers multimodal kwargs as nested lists
        # (per request -> per image, with None for text-only requests). Flatten
        # to the single concatenated pixel tensor + stacked grid the HF-shaped
        # pipeline expects. Mirrors tt-metal's Qwen2.5-VL generator_vllm
        # prefill_forward normalization.
        def _flatten_mm(nested: Any) -> list:
            items: list = []
            if nested is None:
                return items
            if not isinstance(nested, (list, tuple)):
                return [nested]
            for per_user in nested:
                if per_user is None:
                    continue
                if isinstance(per_user, (list, tuple)):
                    items.extend([x for x in per_user if x is not None])
                else:
                    items.append(per_user)
            return items

        pixel_values = kwargs.get("pixel_values")
        image_grid_thw = kwargs.get("image_grid_thw")
        if isinstance(pixel_values, (list, tuple)):
            pv_items = _flatten_mm(pixel_values)
            pixel_values = torch.concat(pv_items, dim=0) if pv_items else None
        if isinstance(image_grid_thw, (list, tuple)):
            grid_items = _flatten_mm(image_grid_thw)
            image_grid_thw = torch.stack(grid_items, dim=0) if grid_items else None

        if pixel_values is not None and pixel_values.dtype != torch.bfloat16:
            pixel_values = pixel_values.to(torch.bfloat16)

        if not self._warmed_up:
            self.pipeline.warmup(
                input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw
            )
            self._warmed_up = True

        first = self.pipeline.prefill(
            input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw
        )
        token_ids = self._as_token_list(first)
        return self._onehot_logits(token_ids)

    def _decode_s0(
        self,
        tokens: Any,
        start_pos: Any = None,
        page_table: Any = None,
        kv_cache: Any = None,
        **kwargs: Any,
    ) -> Any:
        """One decode step: feed vLLM's last token to the pipeline, return
        one-hot logits for the pipeline's next (argmax) token."""
        import torch

        last = tokens if torch.is_tensor(tokens) else torch.as_tensor(tokens)
        prev_ids = self._as_token_list(last)
        prev = prev_ids[0] if len(prev_ids) == 1 else prev_ids
        nxt = self.pipeline.decode_step(prev)
        token_ids = self._as_token_list(nxt)
        return self._onehot_logits(token_ids)

    # ------------------------------------------------------------------
    # S2: paged forward over a vLLM-page-table-aware KV cache
    #
    # The model stays pure HF: we install vLLM's block table on the paged cache
    # via the shared `set_vllm_page_table` hook (validated on T3K), then call
    # `hf_model(...)` with that cache as `past_key_values` and return the logits
    # vLLM samples from. No model code is vLLM-aware.
    #
    # NOTE: hardware-validated at the hook level (paged_fill + paged_sdpa_decode
    # honor the external table); full multi-user continuous-batching e2e is
    # pending a registered S2 model (see design §9.2 / M2 follow-up).
    # ------------------------------------------------------------------
    def _resolve_paged_cache(self, kv_cache: Any):
        # Pipeline-backed models own the cache the logits forward reads/writes.
        if self.pipeline is not None and getattr(self.pipeline, "paged_cache", None) is not None:
            return self.pipeline.paged_cache

        from tt_symbiote.modules.ttnn_attention import TTNNPagedAttentionKVCache

        if isinstance(kv_cache, TTNNPagedAttentionKVCache):
            return kv_cache
        if isinstance(self._paged_cache, TTNNPagedAttentionKVCache):
            return self._paged_cache
        raise RuntimeError(
            "tt_symbiote S2 adapter: no TTNNPagedAttentionKVCache available; "
            "allocate_kv_cache must run first."
        )

    @staticmethod
    def _extract_mm_inputs(kwargs: dict) -> tuple:
        """Flatten vLLM's nested multimodal kwargs into (pixel_values, grid).

        Mirrors the normalization in ``_prefill_s0`` so the S2 prefill path feeds
        the pipeline the single concatenated pixel tensor + stacked grid it
        expects.
        """
        import torch

        def _flatten(nested: Any) -> list:
            items: list = []
            if nested is None:
                return items
            if not isinstance(nested, (list, tuple)):
                return [nested]
            for per_user in nested:
                if per_user is None:
                    continue
                if isinstance(per_user, (list, tuple)):
                    items.extend([x for x in per_user if x is not None])
                else:
                    items.append(per_user)
            return items

        pixel_values = kwargs.get("pixel_values")
        image_grid_thw = kwargs.get("image_grid_thw")
        if isinstance(pixel_values, (list, tuple)):
            pv_items = _flatten(pixel_values)
            pixel_values = torch.concat(pv_items, dim=0) if pv_items else None
        if isinstance(image_grid_thw, (list, tuple)):
            grid_items = _flatten(image_grid_thw)
            image_grid_thw = torch.stack(grid_items, dim=0) if grid_items else None
        if pixel_values is not None and pixel_values.dtype != torch.bfloat16:
            pixel_values = pixel_values.to(torch.bfloat16)
        return pixel_values, image_grid_thw

    @staticmethod
    def _install_page_table(cache, page_table: Any, broadcast: bool = False) -> None:
        """Map vLLM's per-request block table onto the pipeline's DP page table.

        vLLM hands ``block_tables`` shaped ``[num_active_seqs, n_blocks]`` (physical
        block ids per scheduled sequence). The dots.ocr paged cache requires a full
        ``[batch_size, blocks_per_sequence]`` table -- one row per DP stream (mesh
        device): the KV buffer is *replicated* per device and the page table is
        *sharded* (row ``d`` -> device ``d``), so each stream indexes its own
        private 512-block buffer with no cross-device collision.

        Two install modes back correct continuous batching:

        * ``broadcast=True`` (single-request prefill): tile the one request's
          blocks across *every* stream row, so the SIMD prefill writes that
          request's KV to its blocks on *all* mesh devices. Because every device
          then holds the request's KV at its true block ids, a later decode step
          can serve the request from *any* row -- which removes the need for a
          stable vLLM-slot -> DP-stream mapping (vLLM compacts/reorders the active
          batch as requests finish).
        * positional (decode, or a multi-request prefill): active sequence ``i``
          occupies row ``i`` and reads its own blocks on device ``i`` (which holds
          its KV from the broadcast prefill). Inactive rows keep the cache's
          existing (valid) mapping. Column padding past ``n_blocks`` is handled by
          ``set_vllm_page_table`` (those columns are never read).
        """
        if page_table is None:
            return
        import torch

        pt = page_table if torch.is_tensor(page_table) else torch.as_tensor(page_table)
        if pt.dim() == 1:
            pt = pt.unsqueeze(0)
        pt = pt.to(torch.int32)

        cfg = getattr(cache, "config", None)
        bs = int(getattr(cfg, "batch_size", pt.shape[0]))
        bps = int(getattr(cfg, "blocks_per_sequence", pt.shape[1]))

        if broadcast and int(pt.shape[0]) == 1 and bs > 1:
            # Replicate the single request's blocks to every DP stream so all mesh
            # devices write/hold this request's KV (replicated-buffer cache).
            pt = pt.expand(bs, pt.shape[1]).contiguous()

        if int(pt.shape[0]) == bs and int(pt.shape[1]) <= bps:
            # Full DP-width (broadcast tiled, or already batch_size rows): install.
            cache.set_vllm_page_table(pt)
            return

        # Positional expand [num_active, n_blocks] -> [bs, bps], seeding inactive
        # rows from the cache's current (valid) table.
        base = cache.page_table
        full = (base.clone() if torch.is_tensor(base) else torch.as_tensor(base)).to(torch.int32)
        rows = min(int(pt.shape[0]), bs)
        cols = min(int(pt.shape[1]), bps)
        full[:rows, :cols] = pt[:rows, :cols]
        cache.set_vllm_page_table(full)

    @staticmethod
    def _pad_prefill_to_dp_batch(
        input_ids: Any,
        pixel_values: Any,
        image_grid_thw: Any,
        dp_batch: int,
    ) -> tuple:
        """Pad an ``[N, S]`` prefill (N<=dp_batch active seqs) to the DP width.

        The dots.ocr pipeline prefills SIMD-style over exactly ``config.batch_size``
        (== mesh devices) streams and requires ``input_ids.shape[0] == batch_size``.
        vLLM, with TT scheduler-level chunked prefill off, prefills a request (or a
        co-scheduled group) with ``N <= dp_batch`` rows. We replicate the first
        active stream's prompt (and its image) into the inactive rows so the SIMD
        batch is full and every row shares the uniform seq_len / image grid the
        batched-vision path requires. Only the N active rows' logits/KV are used
        (the caller slices ``[:N]``); padding rows write to their own (per-device,
        replicated) KV blocks and are discarded.
        """
        import torch

        ids = input_ids if torch.is_tensor(input_ids) else torch.as_tensor(input_ids)
        n = int(ids.shape[0])
        if n >= dp_batch:
            return input_ids, pixel_values, image_grid_thw
        pad = dp_batch - n
        ids_full = torch.cat([ids, ids[:1].expand(pad, ids.shape[1])], dim=0).contiguous()

        if pixel_values is None or image_grid_thw is None:
            return ids_full, pixel_values, image_grid_thw

        grids = image_grid_thw if torch.is_tensor(image_grid_thw) else torch.as_tensor(image_grid_thw)
        if grids.dim() == 1:
            grids = grids.unsqueeze(0)
        pv = pixel_values if torch.is_tensor(pixel_values) else torch.as_tensor(pixel_values)
        # First image's patch block (rows 0..count0) is replicated for padding.
        count0 = int(grids[0][0]) * int(grids[0][1]) * int(grids[0][2])
        first_block = pv[:count0]
        pv_full = torch.cat([pv] + [first_block] * pad, dim=0).contiguous()
        grids_full = torch.cat(
            [grids, grids[:1].expand(pad, grids.shape[1])], dim=0
        ).contiguous()
        return ids_full, pv_full, grids_full

    @staticmethod
    def _slice_request_mm(pixel_values: Any, image_grid_thw: Any, r: int) -> tuple:
        """Extract request ``r``'s image patches from a concatenated MM batch.

        vLLM concatenates every scheduled request's vision patches into a single
        ``pixel_values`` tensor (rows stacked) with one ``image_grid_thw`` row per
        request. Per-request broadcast prefill processes one request at a time, so
        we slice out request ``r``'s patch block ``[offset_r : offset_r + count_r]``
        (``count_r = t*h*w`` from its grid row) and its single grid row. Returns
        ``(None, None)`` for a text-only step.
        """
        import torch

        if pixel_values is None or image_grid_thw is None:
            return None, None
        grids = image_grid_thw if torch.is_tensor(image_grid_thw) else torch.as_tensor(image_grid_thw)
        if grids.dim() == 1:
            grids = grids.unsqueeze(0)
        pv = pixel_values if torch.is_tensor(pixel_values) else torch.as_tensor(pixel_values)
        # Single image shared by the whole step (already one grid row): no slice.
        if int(grids.shape[0]) <= 1:
            return pv, grids
        counts = [int(grids[i][0]) * int(grids[i][1]) * int(grids[i][2]) for i in range(int(grids.shape[0]))]
        offset = sum(counts[:r])
        count_r = counts[r]
        pv_r = pv[offset:offset + count_r].contiguous()
        grid_r = grids[r:r + 1].contiguous()
        return pv_r, grid_r

    @staticmethod
    def _uniform_prefix_len(num_computed_tokens: Any, batch: int, seq_len: int) -> int:
        """TS-8: shared cached-prefix length across active streams, else 0.

        vLLM hands a block-aligned cached-prefix length per request. The pipeline
        skips one *shared* prefix for the whole batch (the M1 shared-system-prompt
        win), so we only honor a prefix when every active stream agrees on a
        positive value strictly inside the prompt; mixed per-stream prefixes fall
        back to a full prefill (correct, just no skip).
        """
        if num_computed_tokens is None:
            return 0
        try:
            vals = [int(x) for x in list(num_computed_tokens)[:batch]]
        except (TypeError, ValueError):
            return 0
        if not vals or any(v != vals[0] for v in vals):
            return 0
        p = vals[0]
        # Need a non-empty suffix and a genuinely cached prefix.
        if p <= 0 or p >= seq_len:
            return 0
        return p

    def _prefill_s2(
        self,
        tokens: Any,
        page_table: Any = None,
        kv_cache: Any = None,
        prompt_lens: Any = None,
        empty_slots: Any = None,
        num_computed_tokens: Any = None,
        **kwargs: Any,
    ) -> Any:
        import torch

        cache = self._resolve_paged_cache(kv_cache)

        input_ids = tokens if torch.is_tensor(tokens) else torch.as_tensor(tokens)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        pt_full = None
        if page_table is not None:
            pt_full = page_table if torch.is_tensor(page_table) else torch.as_tensor(page_table)
            if pt_full.dim() == 1:
                pt_full = pt_full.unsqueeze(0)

        # Pipeline-backed (dots.ocr): run the device prefill graph in logits mode.
        # The pipeline owns embedding/vision/scatter/decoder/lm_head on T3K; we get
        # last-position logits [B, vocab] back and present them as [B, 1, vocab] so
        # vLLM's `[:, -1, :]` slice yields the per-request next-token logits.
        if self.pipeline is not None:
            pixel_values, image_grid_thw = self._extract_mm_inputs(kwargs)
            seq_len = int(input_ids.shape[-1])
            batch = int(input_ids.shape[0])
            dp_batch = int(getattr(self.pipeline.config, "batch_size", 1))

            # CONTINUOUS-BATCHING CORRECTNESS: prefill each request ON ITS OWN,
            # broadcasting its KV to EVERY DP stream (mesh device). vLLM may admit
            # requests across multiple prefill waves and then decode them together
            # in a single DP step; a request's row in that combined decode is not
            # the row it prefilled on. Because each device owns a *replicated* KV
            # buffer, broadcasting a request's prefill to all rows writes its KV to
            # every device, so the later decode can place it on ANY row and still
            # read the right KV. (Per-request prefill is not DP-parallel across
            # distinct requests, but OCR is decode-bound and the DP decode step is
            # still fully batched -- the throughput win is preserved.)
            logits_rows = []
            for r in range(batch):
                ids_r = input_ids[r:r + 1]
                pv_r, grid_r = self._slice_request_mm(pixel_values, image_grid_thw, r)
                pt_r = pt_full[r:r + 1] if pt_full is not None else None
                # Tile this request's blocks across every DP stream so the SIMD
                # prefill writes its KV to all mesh devices.
                self._install_page_table(cache, pt_r, broadcast=True)

                seq_len_r = int(ids_r.shape[-1])
                # TS-8 per-request prefix: skip an already-cached prefix.
                prefix_len = 0
                if num_computed_tokens is not None:
                    try:
                        p = int(list(num_computed_tokens)[r])
                        if 0 < p < seq_len_r:
                            prefix_len = p
                    except (TypeError, ValueError, IndexError):
                        prefix_len = 0
                # TS-7 chunked prefill for long prompts (or to carry a prefix).
                chunk_size = self._prefill_chunk_size
                use_chunk = (
                    chunk_size is not None and chunk_size > 0 and seq_len_r > chunk_size
                ) or prefix_len > 0
                if use_chunk and (chunk_size is None or chunk_size <= 0):
                    chunk_size = 256
                if use_chunk:
                    logger.info(
                        "tt_symbiote S2: chunked prefill req=%d/%d seq_len=%d "
                        "chunk_size=%d prefix_len=%d (%d suffix chunks)",
                        r,
                        batch,
                        seq_len_r,
                        chunk_size,
                        prefix_len,
                        (seq_len_r - prefix_len + chunk_size - 1) // chunk_size,
                    )

                # The pipeline prefills exactly ``batch_size`` SIMD streams; pad
                # the single request up to the DP width (its prompt + image are
                # replicated into every row, matching the broadcast page table).
                pf_ids, pf_pv, pf_grid = ids_r, pv_r, grid_r
                if dp_batch > 1:
                    pf_ids, pf_pv, pf_grid = self._pad_prefill_to_dp_batch(
                        ids_r, pv_r, grid_r, dp_batch
                    )

                if not use_chunk and not self._warmed_up:
                    # Warm the S2 logits graphs once (chunked path is eager).
                    self.pipeline.warmup(
                        pf_ids,
                        pixel_values=pf_pv,
                        image_grid_thw=pf_grid,
                        return_logits=True,
                    )
                    self._warmed_up = True
                forward_kwargs: Dict[str, Any] = {}
                if use_chunk:
                    forward_kwargs["chunk_size"] = chunk_size
                    if prefix_len > 0:
                        forward_kwargs["prefix_len"] = prefix_len
                logits_r = self.pipeline.forward_logits_prefill(
                    pf_ids,
                    pixel_values=pf_pv,
                    image_grid_thw=pf_grid,
                    **forward_kwargs,
                )
                if not torch.is_tensor(logits_r):
                    logits_r = torch.as_tensor(logits_r)
                logits_rows.append(logits_r[:1])  # active (broadcast) row

            logits = torch.cat(logits_rows, dim=0)  # [batch, vocab]
            return logits.unsqueeze(1)  # [B, vocab] -> [B, 1, vocab]

        cache_position = torch.arange(input_ids.shape[-1])
        out = self.hf_model(
            input_ids=input_ids,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
        )
        # vLLM expects logits [B, seq, vocab]; it slices [:, -1, :] itself.
        return out.logits

    def _decode_s2(
        self,
        tokens: Any,
        start_pos: Any = None,
        page_table: Any = None,
        kv_cache: Any = None,
        **kwargs: Any,
    ) -> Any:
        import torch

        cache = self._resolve_paged_cache(kv_cache)

        last = tokens if torch.is_tensor(tokens) else torch.as_tensor(tokens)

        # Pipeline-backed (dots.ocr): one device decode step in logits mode, with
        # vLLM's per-sequence start_pos threaded through as the decode positions.
        if self.pipeline is not None:
            # vLLM's TT model_runner pads the decode batch to ``max_num_reqs``
            # (token rows -> 0, position rows -> -1) and appends padding AFTER the
            # ``unpadded_batch_size`` real rows (see model_runner.py:922-929). The
            # active sequences are therefore the contiguous front rows whose
            # ``start_pos >= 0``. Feeding the padding rows straight into the
            # pipeline builds a ``[max_num_reqs, ...]`` decode activation and trips
            # the sharded-RMSNorm physical-height assert; slice to the active rows.
            tok_t = last.reshape(-1)
            pos_t = None
            if start_pos is not None:
                pos_t = (start_pos if torch.is_tensor(start_pos) else torch.as_tensor(start_pos)).reshape(-1)
                n_active = int((pos_t >= 0).sum().item())
                if n_active <= 0:
                    n_active = pos_t.numel()
                tok_t = tok_t[:n_active]
                pos_t = pos_t[:n_active].to(torch.int32)
            n_active = int(tok_t.numel())

            pipe_batch = int(getattr(self.pipeline.config, "batch_size", 1))
            if n_active > pipe_batch:
                raise NotImplementedError(
                    f"S2 served decode received {n_active} concurrent sequences but the "
                    f"pipeline was built for batch_size={pipe_batch}. Enable the DP batch "
                    f"pipeline (DOTS_OCR_PARALLELISM=DP, batch_size==num_devices) for "
                    f"multi-sequence continuous batching (design W4/M2)."
                )

            # CONTINUOUS-BATCHING CORRECTNESS: the pipeline decodes SIMD over
            # exactly ``pipe_batch`` (== mesh devices) rows. When fewer requests
            # are active, the inactive rows MUST NOT write KV through stale page
            # table rows (left pointing at other requests' blocks) -- that slowly
            # corrupts resident KV across request cycles and breaks single-stream
            # serving after a few requests. Instead we pad by REPLICATING active
            # row 0 into every inactive row: padding rows re-decode a real request
            # (same token, position AND blocks), so their KV writes are idempotent
            # duplicates on those rows' devices (which already hold that request's
            # KV from the broadcast prefill) and corrupt nothing. The active rows'
            # logits are sliced back out below.
            pt_full = None
            if page_table is not None:
                pt_full = page_table if torch.is_tensor(page_table) else torch.as_tensor(page_table)
                if pt_full.dim() == 1:
                    pt_full = pt_full.unsqueeze(0)
                pt_full = pt_full.to(torch.int32)[:n_active]

            tok_list = [int(x) for x in tok_t.tolist()]
            pos_list = [int(x) for x in pos_t.tolist()] if pos_t is not None else None
            if pipe_batch > 1 and n_active < pipe_batch:
                pad = pipe_batch - n_active
                tok_list = tok_list + [tok_list[0]] * pad
                if pos_list is not None:
                    pos_list = pos_list + [pos_list[0]] * pad
                if pt_full is not None:
                    pt_full = torch.cat([pt_full, pt_full[:1].expand(pad, pt_full.shape[1])], dim=0).contiguous()

            # Install the full-width (replicated) page table so every SIMD row maps
            # to a real request's blocks (no stale rows).
            self._install_page_table(cache, pt_full)

            prev = tok_list[0] if len(tok_list) == 1 else tok_list
            cache_position = torch.as_tensor(pos_list, dtype=torch.int32) if pos_list is not None else None
            logits = self.pipeline.forward_logits_decode(prev, cache_position=cache_position)
            if not torch.is_tensor(logits):
                logits = torch.as_tensor(logits)
            if int(logits.shape[0]) > n_active:
                logits = logits[:n_active]  # drop replicated padding rows
            # vLLM's host-sampling path indexes tt_out[start:start+sz, -1, :], so
            # return [B, 1, vocab] (same rank as prefill), not [B, vocab].
            return logits.unsqueeze(1)

        input_ids = last
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(-1)  # [B] -> [B, 1]
        if start_pos is not None:
            pos = start_pos if torch.is_tensor(start_pos) else torch.as_tensor(start_pos)
            cache_position = pos.reshape(-1)[:1].to(torch.long)
        else:
            cache_position = torch.as_tensor([cache.get_seq_length(0)], dtype=torch.long)

        out = self.hf_model(
            input_ids=input_ids,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
        )
        logits = out.logits
        # vLLM decode expects [B, vocab]; collapse the length-1 sequence axis.
        return logits[:, -1, :] if logits.dim() == 3 else logits

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _as_token_list(value: Any) -> List[int]:
        import torch

        if torch.is_tensor(value):
            return [int(x) for x in value.flatten().tolist()]
        if isinstance(value, (list, tuple)):
            flat: List[int] = []
            for v in value:
                flat.extend(_TTSymbioteGenerator._as_token_list(v))
            return flat
        return [int(value)]

    def _onehot_logits(self, token_ids: List[int]):
        """Build ``[B, 1, vocab]`` logits that argmax to ``token_ids`` so vLLM's
        greedy sampler reproduces the pipeline's deterministic choice."""
        import torch

        vocab = int(getattr(self.hf_model.config, "vocab_size"))
        b = len(token_ids)
        logits = torch.zeros(b, 1, vocab, dtype=torch.float32)
        for i, tid in enumerate(token_ids):
            logits[i, 0, int(tid)] = _ONEHOT_LOGIT
        return logits


class TTSymbioteGeneratorS0(_TTSymbioteGenerator):
    """Greedy generate-engine tier (dots.ocr today)."""

    SERVING_TIER = S0_GREEDY_ENGINE
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
    }


class TTSymbioteGeneratorS1(_TTSymbioteGenerator):
    """Logits, model-managed (unpaged) KV; one request at a time."""

    SERVING_TIER = S1_LOGITS_UNPAGED
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
    }


class TTSymbioteGeneratorS2(_TTSymbioteGenerator):
    """Paged, logits, continuous batching (target for text LLMs)."""

    SERVING_TIER = S2_PAGED
    model_capabilities = {
        "supports_prefix_caching": True,
        "supports_async_decode": False,
    }


# Tier -> concrete class (used by run_vllm_api_server.register_tt_models).
TIER_TO_CLASS: Dict[str, type] = {
    S0_GREEDY_ENGINE: TTSymbioteGeneratorS0,
    S1_LOGITS_UNPAGED: TTSymbioteGeneratorS1,
    S2_PAGED: TTSymbioteGeneratorS2,
}

_MODULE = "tt_symbiote_generators"


def _copy_native_multimodal_registration(arch: str, subclass: type) -> bool:
    """Attach vLLM's native per-arch multimodal processor onto ``subclass``.

    vLLM ships built-in multimodal model classes (e.g.
    ``vllm.model_executor.models.dots_ocr.DotsOCRForCausalLM``) already decorated
    with ``@MULTIMODAL_REGISTRY.register_processor(...)``, which sets a
    ``_processor_factory`` (info + dummy-inputs + processor). vLLM's v1
    ``InputProcessor`` requires the *registered* model class to carry that factory
    (``multimodal/registry.py``). The tt_symbiote serving adapter reuses the
    native processor verbatim (no reimplemented image preprocessing), so a VLM is
    still a single RUNTIME_PINS row. Returns True if a native registration was
    found and copied.
    """
    try:
        from vllm import ModelRegistry as _VllmRegistry

        native_cls = _VllmRegistry._try_load_model_cls(arch)
    except Exception as e:  # pragma: no cover - import-environment dependent
        logger.debug("tt_symbiote MM: _try_load_model_cls(%s) raised: %r", arch, e)
        return False
    if native_cls is None:
        logger.debug("tt_symbiote MM: no native vLLM model registered for %s", arch)
        return False
    factory = getattr(native_cls, "_processor_factory", None)
    if factory is None:
        logger.debug(
            "tt_symbiote MM: native %s (%s) has no _processor_factory", arch, native_cls
        )
        return False
    logger.info("tt_symbiote MM: wired %s multimodal processor from native %s", arch, native_cls)
    subclass._processor_factory = factory
    # Mirror the capability flags vLLM queries on the model *class* so the
    # multimodal pipeline treats the adapter like the native VLM.
    subclass.supports_multimodal = True
    for flag in (
        "supports_multimodal_raw_input_only",
        "supports_multimodal_pruning",
        "supports_pp",
    ):
        if hasattr(native_cls, flag):
            setattr(subclass, flag, getattr(native_cls, flag))
    # vLLM's chat templating calls ``get_placeholder_str`` on the model *class*
    # to insert the per-modality placeholder tokens (e.g. the dots.ocr
    # <|imgpad|> span) into the prompt. It is a pure classmethod (modality -> str,
    # no instance state), so rebind the native implementation onto the subclass.
    native_gph = getattr(native_cls, "get_placeholder_str", None)
    if native_gph is not None:
        func = getattr(native_gph, "__func__", native_gph)
        subclass.get_placeholder_str = classmethod(func)
    return True


def _dots_ocr_max_tokens_all_users(
    model_name: Any = None,
    num_devices: int = 8,
    tt_data_parallel: int = 1,
    **_: Any,
) -> int:
    """Token budget sizing get_num_available_blocks_tt to the dots.ocr buffer.

    The pipeline paged cache (see pipeline._create_paged_kv_cache) is built with
    ``block_size=64``, ``blocks_per_sequence=64`` and
    ``max_num_blocks = max(256, batch_size*64)`` where ``batch_size==num_devices``
    under DP. The worker turns the returned token budget into
    ``num_tt_blocks = ceil(tokens/block_size) + block_size_headroom``; we subtract
    one DP batch's worth of headroom so the final block count lands at (or just
    below) ``max_num_blocks`` and every vLLM-assigned block ID stays in range.
    """
    block_size = 64
    blocks_per_sequence = 64
    batch_size = max(1, int(num_devices))
    max_num_blocks = max(256, batch_size * blocks_per_sequence)
    # Reserve a full batch of blocks for the worker's worst-case +block_size*max_batch
    # headroom; clamp so we never return a non-positive budget.
    usable_blocks = max(blocks_per_sequence, max_num_blocks - batch_size)
    return usable_blocks * block_size


def _build_registered_classes() -> Dict[str, type]:
    """Resolve, per arch, the concrete class to register under ``TT<Arch>``.

    Text-only archs register the shared tier class. Multimodal archs get a
    per-arch subclass (so the per-class ``_processor_factory`` does not leak
    across archs that share a tier) with vLLM's native processor copied in. The
    subclasses are injected as module globals so the lazily-imported registry
    path ``"tt_symbiote_generators:<name>"`` resolves in any process (including a
    vLLM engine subprocess that re-imports this module).
    """
    out: Dict[str, type] = {}
    for arch, recipe in get_serving_recipes().items():
        base = TIER_TO_CLASS.get(recipe.tier, TTSymbioteGeneratorS1)
        if not recipe.multimodal:
            out[arch] = base
            continue
        name = f"{base.__name__}_{arch}"
        cls = globals().get(name)
        if cls is None:
            cls = type(name, (base,), {"__module__": __name__})
            _copy_native_multimodal_registration(arch, cls)
            # dots.ocr's paged cache resets its (per-layer, global) sequence
            # bookkeeping on every prefill, so it cannot honor a vLLM prefix-cache
            # hit (the skipped prefix would not be accounted for) -- disable prefix
            # caching for this arch so continuous batching stays correct (every
            # request gets a full prefill broadcast to all DP streams).
            if arch == "DotsOCRForCausalLM":
                cls.model_capabilities = dict(
                    getattr(base, "model_capabilities", {}),
                    supports_prefix_caching=False,
                )
                # Cap vLLM's KV block budget to the pipeline's paged-cache buffer.
                # dots.ocr's DP cache is a *replicated* 512-block buffer per mesh
                # device (max(256, batch_size*64), batch_size==num_devices). With
                # broadcast prefill every request's blocks live on every device, so
                # vLLM must allocate block IDs strictly inside [0, 512). Without this
                # cap, get_num_available_blocks_tt falls back to 131072 tokens (2056
                # blocks); once vLLM hands out an ID >= 512 (after enough requests or
                # long generations), the paged kernels index past the buffer and
                # silently corrupt KV. Sizing the budget to the buffer keeps every
                # block ID in range.
                cls.get_max_tokens_all_users = staticmethod(_dots_ocr_max_tokens_all_users)
            globals()[name] = cls
        out[arch] = cls
    return out


# Built at import so synthesized multimodal subclasses exist as importable
# module globals everywhere this module is loaded. Guarded so a missing
# tt_symbiote/vLLM at import time never breaks registration outright.
try:
    _REGISTERED_CLASSES: Dict[str, type] = _build_registered_classes()
except Exception as e:  # pragma: no cover - import-environment dependent
    logger.warning("tt_symbiote registered-class build failed: %s", e)
    _REGISTERED_CLASSES = {}


def registration_entries() -> Dict[str, str]:
    """Return ``{"TT<Arch>": "tt_symbiote_generators:<Class>"}`` for every
    tt_symbiote serving recipe. Drives ModelRegistry registration so adding a
    model is a single RUNTIME_PINS row, no code here.
    """
    classes = _REGISTERED_CLASSES or _build_registered_classes()
    return {f"TT{arch}": f"{_MODULE}:{cls.__name__}" for arch, cls in classes.items()}
