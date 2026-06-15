# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""KernelDispatcher — shape-keyed kernel instance cache with model matrix support.

At init:
  1. Loads model_matrix.toml and validates schema_version.
  2. Acquires a file lock at ~/.dispatch.lock to prevent concurrent device use.
  3. Validates every model entry (GQA constraint).
  4. Warms the tt-metal kernel cache from pre-compiled binaries in kernels/manifest.json
     (if present); falls back to JIT with a warning when binaries are missing.

At inference:
  - dispatch_* methods cache compiled kernel instances keyed by shape.
  - resolve(model_name_or_path) returns the ModelMatrixEntry for a model.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import pathlib
import tarfile
import tomllib
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .compat import (
    resolve_attn_config,
    resolve_norm_config,
    resolve_swiglu_config,
    to_tiles,
)

# Must match schema_version in model_matrix.toml.  Bump both together.
DISPATCHER_SCHEMA_VERSION = 3

_DEFAULT_MATRIX_PATH = pathlib.Path(__file__).parent / "model_matrix.toml"
_DEVICE_LOCK_PATH = pathlib.Path.home() / ".dispatch.lock"
_KERNELS_DIR = pathlib.Path(__file__).parent.parent / "kernels"
_MANIFEST_PATH = _KERNELS_DIR / "manifest.json"
_TT_METAL_CACHE = pathlib.Path.home() / ".cache" / "tt-metal-cache"


@dataclass
class Capabilities:
    """Derived (not stored) decode-path capabilities for a resolved model.

    Computed from a ModelMatrixEntry's behavioral fields + dims (+ HardwareConfig).
    These replace the runner's inline eligibility computations:
      fast_path        <- _ondevice_attn_eligible + _setup_paged_decode ok_shape
      lm_head_ondevice <- _setup_ondevice_lmhead eligibility
    """
    fast_path: bool
    lm_head_ondevice: bool
    attn_backend: str


@dataclass
class ResolvedConfig:
    """Model dims view exposing the ModelConfig attribute names the runner reads.

    Built from a matrix entry so the matrix is authoritative for dims (#3 Phase D),
    replacing the per-arch dim modules / hardcoded CONFIGS dicts in registry.py for
    listed models. Novel models keep detect_model_family()'s HF-config introspection.
    """
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    activation: str
    norm_type: str


def _compute_capabilities(
    *,
    norm_type: str,
    rotary_pct: float,
    attn_bias: bool,
    head_norm: bool,
    head_dim: int,
    n_heads: int,
    n_kv_heads: int,
    attn_backend: str = "ttnn",
) -> Capabilities:
    """Shared capability formula used by both the matrix entry and the unlisted
    auto-derive path, so listed and novel models compute eligibility identically.

    fast_path (traced-ttnn decode) is eligible iff the model is a plain RMSNorm
    full-RoPE attention with no bias / head-norm, a tile-aligned head_dim, and a GQA
    group the decode SDPA's pad-to-32 can represent (group | 32, nh <= 32). This merges
    the runner's `_ondevice_attn_eligible` flag with the `_setup_paged_decode` ok_shape
    gate (and the nh<=32 / group|32 check).

    lm_head_ondevice (on-device final-norm + matmul + argmax) needs an RMSNorm-family
    final norm — rmsnorm or gemma_rms, the latter being rmsnorm with a (1+w) weight baked
    in at upload. LayerNorm (mean-sub + bias) is not yet supported on-device. Unaligned
    vocab is handled by the runner's pad-mask, so no vocab alignment constraint applies.
    This matches the runner's current `not self._uses_layernorm` gate.
    """
    group = (n_heads // n_kv_heads) if n_kv_heads else 0
    fast_path = (
        norm_type == "rmsnorm"
        and rotary_pct == 1.0
        and not attn_bias
        and not head_norm
        and head_dim % 32 == 0
        and group >= 1
        and 32 % group == 0
        and n_heads <= 32
    )
    lm_head_ondevice = (norm_type != "layernorm")
    return Capabilities(
        fast_path=fast_path,
        lm_head_ondevice=lm_head_ondevice,
        attn_backend=attn_backend,
    )


def derive_capabilities(hf_config) -> Capabilities:
    """Auto-derive decode-path capabilities for an UNLISTED (novel) model from its
    HuggingFace config alone — the universal floor of the two-tier resolution (#3).

    This is intentionally config-only and conservative; the matrix overrides it for
    listed models. A novel model still runs (flagged community/unverified); the output
    validator catches breakage. NOTE: some signals the runner sniffs from loaded module
    structure (e.g. Qwen2's always-on qkv bias) are not visible here — those are exactly
    the cases the matrix is meant to record an override for.
    """
    text_cfg = getattr(hf_config, "text_config", hf_config)
    archs = getattr(hf_config, "architectures", None) or []
    arch = archs[0] if archs else ""
    model_type = str(getattr(text_cfg, "model_type", "")).lower()

    n_heads = getattr(text_cfg, "num_attention_heads", 32)
    n_kv = getattr(text_cfg, "num_key_value_heads", n_heads) or n_heads
    hidden = getattr(text_cfg, "hidden_size", 4096)
    head_dim = getattr(text_cfg, "head_dim", None) or (hidden // n_heads)

    if "gemma" in model_type or arch.startswith("Gemma"):
        norm_type = "gemma_rms"
    elif hasattr(text_cfg, "rms_norm_eps"):
        norm_type = "rmsnorm"
    elif hasattr(text_cfg, "layer_norm_eps") or hasattr(text_cfg, "layer_norm_epsilon"):
        norm_type = "layernorm"
    else:
        norm_type = "rmsnorm"

    rotary_pct = float(getattr(text_cfg, "partial_rotary_factor",
                               getattr(text_cfg, "rotary_pct", 1.0)) or 1.0)
    # LayerNorm-family archs (GPTNeoX/Pythia) carry qkv bias; otherwise read the flag.
    attn_bias = bool(getattr(text_cfg, "attention_bias", False)) or (norm_type == "layernorm")
    head_norm = arch.startswith("Qwen3") or "gemma3" in model_type
    attn_backend = "ttnn" if (n_kv and n_heads % n_kv == 0 and 32 % (n_heads // n_kv) == 0) else "custom"

    return _compute_capabilities(
        norm_type=norm_type,
        rotary_pct=rotary_pct,
        attn_bias=attn_bias,
        head_norm=head_norm,
        head_dim=head_dim,
        n_heads=n_heads,
        n_kv_heads=n_kv,
        attn_backend=attn_backend,
    )


@dataclass
class ModelMatrixEntry:
    name: str
    arch: str
    local_path: str
    hidden_size: int
    intermediate_size: int
    n_layers: int
    vocab_size: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    max_seq: int
    rope_theta: float
    swa_window: int
    tie_embeddings: bool
    kernels: List[str]
    min_hw: str
    status: str
    # --- behavioral fields (schema v2, issue #3). Defaults reproduce today's
    #     HF-derived behavior so an entry that omits them is a no-op. ---
    norm_type: str = "rmsnorm"          # rmsnorm | layernorm | gemma_rms
    activation: str = "silu"            # silu | gelu | gelu_tanh | relu2
    rotary_pct: float = 1.0             # 1.0 full; 0.25 partial (GPTNeoX/StableLM)
    attn_bias: bool = False             # qkv + o-proj bias present
    head_norm: bool = False             # per-head q/k RMSNorm (Qwen3/Gemma3)
    parallel_residual: bool = False     # attn + MLP branch the same pre-norm (GPTNeoX)
    embed_scale: str = "none"           # none | sqrt_hidden (Gemma scales embeds)
    attn_backend: str = "ttnn"          # ttnn | custom | cpu (#34 backend-per-op)
    # Granite (#43) architectural scaling factors. None = fall through to the HF config
    # value (so an omitted field is a no-op); a set value OVERRIDES it, same as the other
    # behavioral fields. Read by the runner's _scaling_factor().
    embedding_multiplier: Optional[float] = None   # scales input embeddings
    residual_multiplier: Optional[float] = None    # scales each sublayer output pre-residual-add
    attention_multiplier: Optional[float] = None   # overrides the 1/sqrt(head_dim) softmax scale
    logits_scaling: Optional[float] = None          # divides final logits (no-op for greedy argmax)
    issue: Optional[int] = None
    notes: str = ""

    def model_config(self) -> ResolvedConfig:
        """Return the matrix-authoritative dims view consumed by the runner as self._cfg."""
        return ResolvedConfig(
            hidden_size=self.hidden_size,
            num_heads=self.n_heads,
            num_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            intermediate_size=self.intermediate_size,
            activation=self.activation,
            norm_type=self.norm_type,
        )

    def capabilities(self, hw_config=None) -> Capabilities:
        """Compute the DERIVED decode-path gates from fields + dims.

        See _compute_capabilities for the eligibility rules.
        """
        return _compute_capabilities(
            norm_type=self.norm_type,
            rotary_pct=self.rotary_pct,
            attn_bias=self.attn_bias,
            head_norm=self.head_norm,
            head_dim=self.head_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            attn_backend=self.attn_backend,
        )


class SchemaVersionError(RuntimeError):
    """Raised when model_matrix.toml schema_version doesn't match dispatcher."""


class ModelNotFoundError(ValueError):
    """Raised when a requested model is not in the matrix."""


class UnsafeModelError(ValueError):
    """Raised when a 'community' model is requested without --unsafe."""


class DeviceLockError(RuntimeError):
    """Raised when the device lock is already held by another process."""


class KernelDispatcher:
    """Dispatches tensor shapes to cached kernel instances.

    Parameters
    ----------
    device:
        An open ttnn device.
    model_config:
        Optional pre-loaded ModelMatrixEntry.
    matrix_path:
        Path to model_matrix.toml.  Defaults to the file in this package.
    unsafe:
        If True, allow loading 'community' status models with a warning.
    """

    def __init__(
        self,
        device,
        model_config=None,
        matrix_path: Optional[pathlib.Path] = None,
        unsafe: bool = False,
        hw_config=None,
    ):
        self._device = device
        self._model_config = model_config
        self._hw_config = hw_config
        self._cache: Dict[Tuple, Any] = {}
        self._unsafe = unsafe
        self._lock_fh = None

        self._acquire_device_lock()
        _warm_kernel_cache(device)
        _install_kernel_patch_overlay()

        matrix_path = matrix_path or _DEFAULT_MATRIX_PATH
        self._matrix, self._index = _load_and_validate_matrix(matrix_path)

    # ------------------------------------------------------------------
    # Device lock
    # ------------------------------------------------------------------

    def _acquire_device_lock(self) -> None:
        """Acquire an exclusive file lock so only one dispatcher runs at a time."""
        try:
            fh = open(_DEVICE_LOCK_PATH, "w")
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fh.write(str(os.getpid()))
            fh.flush()
            self._lock_fh = fh
        except BlockingIOError:
            raise DeviceLockError(
                f"Device lock {_DEVICE_LOCK_PATH} is held by another process. "
                "Stop the other bench/dispatch process before starting a new one."
            )

    def release(self) -> None:
        """Release the device lock. Best-effort and shutdown-safe: __del__ can run during
        interpreter finalization, when module globals (fcntl) may already be None — so
        re-import locally and swallow any error rather than raise from a destructor."""
        fh = self._lock_fh
        if fh is None:
            return
        self._lock_fh = None
        try:
            import fcntl as _fcntl
            _fcntl.flock(fh, _fcntl.LOCK_UN)
            fh.close()
        except Exception:
            pass

    def __del__(self):
        self.release()

    # ------------------------------------------------------------------
    # Model matrix
    # ------------------------------------------------------------------

    def resolve(self, model_name_or_path: str) -> ModelMatrixEntry:
        """Return the ModelMatrixEntry for a model name or local path.

        Raises ModelNotFoundError if not found.
        Raises UnsafeModelError if status='community' and unsafe=False.
        """
        entry = self._index.get(model_name_or_path)

        # Also match by local_path basename (e.g. "llama3-8b")
        if entry is None:
            basename = pathlib.Path(model_name_or_path).name
            for e in self._matrix:
                if pathlib.Path(e.local_path).name == basename:
                    entry = e
                    break

        if entry is None:
            known = ", ".join(sorted(self._index.keys()))
            raise ModelNotFoundError(
                f"Model '{model_name_or_path}' not found in model matrix. "
                f"Known models: {known}. "
                "To attempt an unsupported model, pass unsafe=True to load_model()."
            )

        if entry.status == "community" and not self._unsafe:
            raise UnsafeModelError(
                f"Model '{entry.name}' has status='community' (unverified). "
                "Pass unsafe=True to load_model() to proceed with a visible warning."
            )

        if entry.status == "community":
            warnings.warn(
                f"Loading community model '{entry.name}': correctness unverified on this hardware.",
                UserWarning,
                stacklevel=2,
            )

        return entry

    def lookup(self, model_name_or_path: str) -> Optional[ModelMatrixEntry]:
        """Return the ModelMatrixEntry for a name/path, or None if unlisted.

        Unlike resolve(), this applies no trust (unsafe) gate and never raises — it
        answers only "is there a matrix entry?". The runner uses it to prefer the matrix
        for listed models and fall back to derive_capabilities() for novel ones. The
        --unsafe / community gating is enforced at the load_model() layer (#25) via
        resolve().
        """
        entry = self._index.get(model_name_or_path)
        if entry is None:
            basename = pathlib.Path(model_name_or_path).name
            for e in self._matrix:
                if pathlib.Path(e.local_path).name == basename:
                    entry = e
                    break
        return entry

    def models(self) -> List[ModelMatrixEntry]:
        """Return all entries in the model matrix."""
        return list(self._matrix)

    # ------------------------------------------------------------------
    # Kernel dispatch (shape-keyed cache)
    # ------------------------------------------------------------------

    def swiglu(self, gate, w_gate, bias_gate, up, w_up, bias_up, out, activation="silu"):
        from tt_inference_server.kernels.swiglu import make_swiglu_kernel
        M, K, N = gate.shape[0], gate.shape[1], w_gate.shape[1]
        key = ("swiglu", M, K, N, activation)
        if key not in self._cache:
            cfg = resolve_swiglu_config(M, K, N, activation, hw_config=self._hw_config)
            self._cache[key] = make_swiglu_kernel(**cfg)
        return self._cache[key](gate, w_gate, bias_gate, up, w_up, bias_up, out)

    def flash_attn(self, Q, K, V, scale, neg_inf, zero, zero_head, ones, mask, out,
                   N_heads: int = None, N_kv_heads: int = None):
        head_dim = Q.shape[-1]
        n_heads = N_heads or Q.shape[0]
        n_kv = N_kv_heads or K.shape[0]
        kv_seq = K.shape[0] // n_kv if n_kv else K.shape[0]
        key = ("flash_attn", n_heads, n_kv, head_dim)
        if key not in self._cache:
            from tt_inference_server.kernels.flash_attn import make_flash_attn_kernel
            cfg = resolve_attn_config(n_heads, n_kv, head_dim, kv_seq, hw_config=self._hw_config)
            self._cache[key] = make_flash_attn_kernel(**cfg)
        return self._cache[key](Q, K, V, scale, neg_inf, zero, zero_head, ones, mask, out)

    def kv_decode(self, Q, K_cache, V_cache, scale, neg_inf, zero, zero_head, out, ones=None):
        from tt_inference_server.kernels.kv_decode import make_kv_decode_kernel
        N_kv_heads, head_dim = Q.shape[0], Q.shape[1]
        max_seq = K_cache.shape[0] // N_kv_heads
        key = ("kv_decode", N_kv_heads, head_dim, max_seq)
        if key not in self._cache:
            self._cache[key] = make_kv_decode_kernel(
                N_kv_heads=N_kv_heads,
                head_dim_tiles=to_tiles(head_dim),
                max_seq_tiles=to_tiles(max_seq),
            )
        return self._cache[key](Q, K_cache, V_cache, scale, neg_inf, zero, zero_head, out)

    def rmsnorm(self, x, weight, scaler, out):
        from tt_inference_server.kernels.rmsnorm import make_rmsnorm_kernel
        seq, hidden = x.shape[0], x.shape[1]
        key = ("rmsnorm", seq, hidden)
        if key not in self._cache:
            cfg = resolve_norm_config(seq, hidden)
            self._cache[key] = make_rmsnorm_kernel(**cfg)
        return self._cache[key](x, weight, scaler, out)

    def layernorm(self, x, weight, bias, scaler, out):
        from tt_inference_server.kernels.layernorm import make_layernorm_kernel
        seq, hidden = x.shape[0], x.shape[1]
        key = ("layernorm", seq, hidden)
        if key not in self._cache:
            cfg = resolve_norm_config(seq, hidden)
            self._cache[key] = make_layernorm_kernel(**cfg)
        return self._cache[key](x, weight, bias, scaler, out)

    def moe_route(self, hidden, w_router, probs_out, N_experts: int, top_k: int, ones=None):
        """Run router projection + softmax on device, then top-k on host."""
        import torch
        import ttnn
        hidden_dim = hidden.shape[1]
        expert_tiles = max(1, (N_experts + 31) // 32)
        key = ("moe_router", hidden_dim, N_experts)
        if key not in self._cache:
            from tt_inference_server.kernels.moe_router import make_moe_router_kernel
            self._cache[key] = make_moe_router_kernel(
                hidden_tiles=to_tiles(hidden_dim),
                expert_tiles=expert_tiles,
            )
        self._cache[key](hidden, w_router, probs_out)
        probs = ttnn.to_torch(probs_out).float()[:, :N_experts]
        weights, indices = torch.topk(probs, k=top_k, dim=-1)
        return indices, weights

    def cache_size(self) -> int:
        return len(self._cache)


# ------------------------------------------------------------------
# Pre-compiled kernel cache warming
# ------------------------------------------------------------------

def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _device_build_key(device) -> Optional[str]:
    """Return the tt-metal build key dir name for the current device.

    The build key is the numeric directory under ~/.cache/tt-metal-cache/ that
    tt-metal creates for this hardware/toolchain combination.  We identify it as
    the directory that contains a 'kernels/' subdirectory (vs. 'firmware/').
    Returns None if nothing is found (cache empty or not yet populated).
    """
    if not _TT_METAL_CACHE.is_dir():
        return None
    candidates = [
        d for d in _TT_METAL_CACHE.iterdir()
        if d.is_dir() and (d / "kernels").is_dir()
    ]
    if not candidates:
        return None
    # Pick the most recently modified — matches the current running device.
    return max(candidates, key=lambda p: p.stat().st_mtime).name


def _warm_kernel_cache(device) -> None:
    """Extract pre-compiled kernel ELFs into the tt-metal disk cache.

    Reads manifest.json from the kernels/ package directory.  For each variant
    whose bundle (.tar.gz) is present and whose sha256 matches, extracts the
    bundle into ~/.cache/tt-metal-cache/<build_key>/.  This makes the tt-metal
    JIT cache warm so the first kernel call skips the 30–120 s Clang compilation.

    Falls back silently (with a one-time warning) when:
      - manifest.json is absent (no pre-compiled binaries shipped)
      - a bundle file is missing
      - sha256 mismatch (corrupted download)
    """
    if not _MANIFEST_PATH.is_file():
        return

    try:
        with open(_MANIFEST_PATH) as f:
            manifest = json.load(f)
    except Exception as exc:
        warnings.warn(f"Could not read kernel manifest ({exc}); using JIT compilation.", UserWarning)
        return

    build_key = _device_build_key(device)
    if build_key is None:
        # Cache dir not yet populated — tt-metal hasn't compiled anything yet.
        # We can't extract to the right location, so let JIT run on first call.
        return

    cache_dest = _TT_METAL_CACHE / build_key
    warmed = 0
    missing = 0

    for variant_key, entry in manifest.get("variants", {}).items():
        bundle_path = _KERNELS_DIR / entry["path"]
        if not bundle_path.is_file():
            missing += 1
            continue

        expected_sha = entry.get("sha256", "")
        if expected_sha and _sha256(bundle_path) != expected_sha:
            warnings.warn(
                f"Kernel bundle sha256 mismatch for '{variant_key}' — skipping pre-compiled binary.",
                UserWarning,
            )
            missing += 1
            continue

        # Extract only entries whose paths don't already exist (avoid re-extracting on every init).
        with tarfile.open(bundle_path, "r:gz") as tar:
            for member in tar.getmembers():
                dest = cache_dest / member.name
                if not dest.exists():
                    tar.extract(member, path=cache_dest, set_attrs=False)
        warmed += 1

    if missing > 0 and warmed == 0:
        warnings.warn(
            f"{missing} kernel variant(s) have no pre-compiled binary in {_KERNELS_DIR}. "
            "First inference will trigger JIT compilation (30–120 s per variant). "
            "Run `make release-kernels` in tt-lang to build the binaries.",
            UserWarning,
        )


# ------------------------------------------------------------------
# JIT kernel-patch overlay (#50)
# ------------------------------------------------------------------

_KERNEL_PATCH_ENV = "DISPATCH_KERNEL_PATCH_DIR"


def _install_kernel_patch_overlay() -> None:
    """A/B kernel-variant dev loop without a tt-metal/tt-lang rebuild (issue #50).

    When DISPATCH_KERNEL_PATCH_DIR is set, overlay patched kernel .cpp files over the
    tt-lang-generated source at the JIT boundary. Unset => no monkeypatch, exact current
    behavior (this is a pure no-op then).

    How it works: tt-lang generates each kernel's C++ via ttkernel_to_cpp_by_name() and
    writes it through ttl.ttl_api._write_kernel_to_tmp(name, source) before tt-metal
    JIT-compiles that file. We wrap that writer: if <patch_dir>/<name>.cpp exists, its
    contents REPLACE the generated source for kernel `name`. The file is then content-
    hashed and compiled exactly as usual — so a no-op patch (a verbatim copy of the
    generated .cpp) reproduces the baseline bit-for-bit, and an edited patch triggers a
    fresh compile of the variant. The .riscv pre-compiled artifact system (#1/#26) stays
    the release path; this is the iterate-fast dev path that feeds kernel-variant A/B arms
    (tests/experiments/matrix.toml).

    Dump a kernel's current source to copy into a patch dir: tt-lang already prints each
    generated kernel and writes it to /tmp/$USER/ttlang_kernel_<name>_<hash>.cpp.
    """
    patch_dir = os.environ.get(_KERNEL_PATCH_ENV)
    if not patch_dir:
        return
    patch_path = pathlib.Path(patch_dir).expanduser().resolve()
    if not patch_path.is_dir():
        warnings.warn(
            f"{_KERNEL_PATCH_ENV}={patch_dir} is not a directory; no kernel patches applied.",
            UserWarning)
        return
    try:
        from ttl import ttl_api
    except Exception as exc:  # ttl not importable (e.g. no-device CI) — nothing to patch
        warnings.warn(
            f"Could not import ttl.ttl_api for kernel patching ({exc}); no patches applied.",
            UserWarning)
        return
    if getattr(ttl_api, "_dispatch_patch_installed", False):
        return  # idempotent — only wrap once per process

    _orig_write = ttl_api._write_kernel_to_tmp

    def _patched_write(name, source):
        cand = patch_path / f"{name}.cpp"
        if cand.is_file():
            print(f"  [kernel-patch] overlaying '{name}' <- {cand}")
            return _orig_write(name, cand.read_text())
        return _orig_write(name, source)

    ttl_api._write_kernel_to_tmp = _patched_write
    ttl_api._dispatch_patch_installed = True
    available = sorted(p.stem for p in patch_path.glob("*.cpp"))
    print(f"  Kernel-patch overlay ACTIVE ({_KERNEL_PATCH_ENV}={patch_path}); "
          f"patches: {available or '(none found)'}")


# ------------------------------------------------------------------
# Matrix loading helpers
# ------------------------------------------------------------------

def _opt_float(v):
    """Coerce an optional numeric matrix field to float, preserving None (= unset)."""
    return None if v is None else float(v)


def _load_and_validate_matrix(
    path: pathlib.Path,
) -> Tuple[List[ModelMatrixEntry], Dict[str, ModelMatrixEntry]]:
    """Load model_matrix.toml, validate schema, return (list, name→entry index)."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    version = raw.get("schema_version")
    if version != DISPATCHER_SCHEMA_VERSION:
        raise SchemaVersionError(
            f"model_matrix.toml schema_version={version} but dispatcher expects "
            f"{DISPATCHER_SCHEMA_VERSION}. Redeploy model_matrix.toml or update "
            "DISPATCHER_SCHEMA_VERSION in dispatcher.py."
        )

    entries: List[ModelMatrixEntry] = []
    index: Dict[str, ModelMatrixEntry] = {}

    for raw_entry in raw.get("model", []):
        entry = ModelMatrixEntry(
            name=raw_entry["name"],
            arch=raw_entry["arch"],
            local_path=raw_entry["local_path"],
            hidden_size=raw_entry["hidden_size"],
            intermediate_size=raw_entry["intermediate_size"],
            n_layers=raw_entry["n_layers"],
            vocab_size=raw_entry["vocab_size"],
            n_heads=raw_entry["n_heads"],
            n_kv_heads=raw_entry["n_kv_heads"],
            head_dim=raw_entry["head_dim"],
            max_seq=raw_entry["max_seq"],
            rope_theta=float(raw_entry["rope_theta"]),
            swa_window=raw_entry["swa_window"],
            tie_embeddings=raw_entry["tie_embeddings"],
            kernels=list(raw_entry["kernels"]),
            min_hw=raw_entry["min_hw"],
            status=raw_entry["status"],
            norm_type=raw_entry.get("norm_type", "rmsnorm"),
            activation=raw_entry.get("activation", "silu"),
            rotary_pct=float(raw_entry.get("rotary_pct", 1.0)),
            attn_bias=raw_entry.get("attn_bias", False),
            head_norm=raw_entry.get("head_norm", False),
            parallel_residual=raw_entry.get("parallel_residual", False),
            embed_scale=raw_entry.get("embed_scale", "none"),
            attn_backend=raw_entry.get("attn_backend", "ttnn"),
            embedding_multiplier=_opt_float(raw_entry.get("embedding_multiplier")),
            residual_multiplier=_opt_float(raw_entry.get("residual_multiplier")),
            attention_multiplier=_opt_float(raw_entry.get("attention_multiplier")),
            logits_scaling=_opt_float(raw_entry.get("logits_scaling")),
            issue=raw_entry.get("issue"),
            notes=raw_entry.get("notes", ""),
        )

        if entry.n_heads % entry.n_kv_heads != 0:
            raise ValueError(
                f"Model '{entry.name}': n_heads={entry.n_heads} not divisible by "
                f"n_kv_heads={entry.n_kv_heads}. Fix model_matrix.toml."
            )

        if entry.n_heads < 1 or entry.n_kv_heads < 1:
            raise ValueError(
                f"Model '{entry.name}': n_heads and n_kv_heads must be >= 1."
            )

        entries.append(entry)
        index[entry.name] = entry

    return entries, index
