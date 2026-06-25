# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Runner contract + selection seam for the dispatch serving runtime.

`load_model()` resolves *which* runner serves a model. The generic
``TTModelRunner`` (a standard decoder-only transformer) is the fallback. A
**custom runner** — for a model whose tt-nn graph is not a standard transformer
(MoE, Gated-DeltaNet/Mamba-state, multimodal) — is selected, in order of
precedence (most-explicit wins):

  1. Explicit override   — ``runner="module:Class"`` arg / ``DISPATCH_RUNNER`` env.
  2. Entry-point match   — an installed package declaring the
     ``tt_inference_server.runners`` entry-point group whose claim matches the
     model's HF config.
  3. HF self-declaration — ``config.json`` ``tt_runner`` field or a
     ``tt_dispatch.json`` sidecar in the model repo. Trust-gated (``--unsafe``).
  4. Fallback            — the generic ``TTModelRunner`` (unchanged behavior).

Third-party runners live in their OWN package and satisfy `BaseRunner`; no model
code is vendored here, and dispatch hard-codes no model names — it matches on the
HF config alone.
"""

from __future__ import annotations

import importlib
import json
import os
import pathlib
from typing import Iterator, Optional, Protocol, Union, runtime_checkable

# The entry-point group external runner packages register under, e.g. in their
# pyproject.toml:
#   [project.entry-points."tt_inference_server.runners"]
#   qwen3_5_moe = "some_pkg.dispatch_runner:Qwen36Runner"
ENTRY_POINT_GROUP = "tt_inference_server.runners"

# Env overrides (parallel to the CLI flags) — useful for `python -m ... serve`
# wrappers and tests.
ENV_RUNNER = "DISPATCH_RUNNER"          # "module:Class" explicit override
ENV_ALLOW = "DISPATCH_RUNNER_ALLOW"     # comma-separated allowlist for self-declared runners


@runtime_checkable
class BaseRunner(Protocol):
    """The complete contract a runner must satisfy to back a ``ModelHandle``.

    `ModelHandle` only ever calls three methods and reads three attributes — this
    Protocol is exactly that surface. Custom runners implement it directly; they
    do NOT need to subclass anything (it is structural / duck-typed).

    Optional class-level discovery hooks (any subset; checked in this priority):
        ``claims(cls, hf_config) -> bool``     classmethod, most flexible matcher
        ``supported_architectures: set[str]``  matched against config.architectures[0]
        ``supported_model_types: set[str]``    matched against config.model_type

    Optional lifecycle hook:
        ``MANAGES_OWN_DEVICE: bool = False``   if True, `load_model` does NOT open a
            ttnn device; the runner opens/owns/closes its own (e.g. a 1x1 mesh) and
            registers its own cleanup. The runner is then constructed with
            ``device=None``.
            OBLIGATION: such a runner MUST close its device cleanly on shutdown
            (e.g. ``atexit``-registered ``ttnn.close_mesh_device`` / ``close_device``).
            This is required for reliable model hot-swap: a serve process exiting must
            leave the card pristine for the next model. An ungraceful teardown can leave
            a locked device mutex that wedges the card until a manual ``tt-smi -r``.

    Constructor contract `load_model` calls (keyword args filtered to what the
    constructor actually accepts — a runner may take ``**kwargs`` and ignore the rest):
        ``__init__(self, model_path, device, *, max_seq, unsafe,
                   force_novel=False, trace_region_size=..., **kwargs)``
    """

    _tokenizer: object
    _listed: bool
    _community: bool

    def generate(self, prompt: str, max_new_tokens: int = 50,
                 temperature: float = 1.0, chat: bool = True) -> str: ...

    def generate_stream(self, prompt: str, max_new_tokens: int = 50,
                        temperature: float = 1.0,
                        chat: bool = True) -> Iterator[Union[str, dict]]: ...

    def benchmark(self, prompt: str, n_tokens: int = 50) -> "tuple[float, str]": ...


_REQUIRED_METHODS = ("generate", "generate_stream", "benchmark")
_REQUIRED_ATTRS = ("_tokenizer", "_listed", "_community")


def validate_runner(obj) -> None:
    """Fail fast with a clear message if ``obj`` does not satisfy `BaseRunner`.

    Called on a constructed runner instance, so attribute checks see instance
    attributes set in ``__init__`` (a Protocol ``isinstance`` check alone would
    not, since the attrs are annotations, not class defaults).
    """
    missing_methods = [m for m in _REQUIRED_METHODS if not callable(getattr(obj, m, None))]
    missing_attrs = [a for a in _REQUIRED_ATTRS if not hasattr(obj, a)]
    if missing_methods or missing_attrs:
        cls = type(obj).__name__
        parts = []
        if missing_methods:
            parts.append(f"missing method(s): {', '.join(missing_methods)}")
        if missing_attrs:
            parts.append(f"missing attribute(s): {', '.join(missing_attrs)} "
                         "(set them in __init__)")
        raise TypeError(
            f"Runner {cls!r} does not satisfy the BaseRunner contract: "
            + "; ".join(parts)
        )


def _load_dotted(spec: str):
    """Import and return the object named by ``"module:Class"`` or ``"module.Class"``.

    The colon form is preferred (unambiguous when the attribute is nested). Raises
    a clear error rather than a bare ImportError/AttributeError.
    """
    spec = spec.strip()
    if ":" in spec:
        module_name, _, attr = spec.partition(":")
    elif "." in spec:
        module_name, _, attr = spec.rpartition(".")
    else:
        raise ValueError(
            f"Invalid runner spec {spec!r}: expected 'module:Class' (or 'module.Class')."
        )
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 — surface the original cause clearly
        raise ImportError(
            f"Could not import module {module_name!r} for runner {spec!r}: "
            f"{type(exc).__name__}: {exc}. Is the runner package installed in the "
            "serving environment?"
        ) from exc
    try:
        obj = getattr(module, attr)
    except AttributeError as exc:
        raise ImportError(
            f"Module {module_name!r} has no attribute {attr!r} for runner {spec!r}."
        ) from exc
    return obj


# --------------------------------------------------------------------------- #
# HF config loading (for discovery / matching — does NOT execute remote code)
# --------------------------------------------------------------------------- #

def load_hf_config(model_path: str):
    """Best-effort load of a model's HF config for runner discovery.

    Returns a transformers config object, or ``None`` if it cannot be loaded
    without ``trust_remote_code`` (discovery then proceeds on whatever raw
    config.json fields it can read). Never executes model repo code.
    """
    try:
        from transformers import AutoConfig
        return AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    except Exception:  # noqa: BLE001 — custom arch may need trust_remote_code; fall back
        return None


def _raw_config_json(model_path: str) -> dict:
    """Read the raw config.json as a dict (local path or hub download). {} on failure.

    Used to read non-standard fields (``model_type``, ``tt_runner``) without
    needing transformers to fully parse a possibly-custom config.
    """
    p = pathlib.Path(model_path)
    if p.is_dir():
        cfg = p / "config.json"
        if cfg.is_file():
            try:
                return json.loads(cfg.read_text())
            except Exception:  # noqa: BLE001
                return {}
        return {}
    # Hub id: download just config.json (no model code execution).
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(model_path, "config.json")
        return json.loads(pathlib.Path(path).read_text())
    except Exception:  # noqa: BLE001
        return {}


def _config_arch(hf_config, raw: Optional[dict] = None) -> str:
    archs = getattr(hf_config, "architectures", None) if hf_config is not None else None
    if not archs and raw:
        archs = raw.get("architectures")
    return (archs[0] if archs else "") or ""


def _config_model_type(hf_config, raw: Optional[dict] = None) -> str:
    mt = getattr(hf_config, "model_type", "") if hf_config is not None else ""
    if not mt and raw:
        mt = raw.get("model_type", "")
    return str(mt or "")


# --------------------------------------------------------------------------- #
# Entry-point discovery (precedence step 2)
# --------------------------------------------------------------------------- #

def iter_entry_point_runners():
    """Yield (name, EntryPoint) for every registered ``tt_inference_server.runners``.

    Tolerant of the importlib.metadata API differences across Python versions.
    """
    from importlib import metadata
    try:
        eps = metadata.entry_points(group=ENTRY_POINT_GROUP)  # py3.10+
    except TypeError:  # pragma: no cover — py3.9 returns a dict
        eps = metadata.entry_points().get(ENTRY_POINT_GROUP, [])
    for ep in eps:
        yield ep.name, ep


# Specificity ranks for tie-breaking: a more-specific claim wins.
_RANK_CLAIMS = 3
_RANK_ARCH = 2
_RANK_MODEL_TYPE = 1


def _match_rank(runner_cls, hf_config, arch: str, model_type: str) -> int:
    """Return how specifically ``runner_cls`` claims this model (0 = no match)."""
    claims = getattr(runner_cls, "claims", None)
    if callable(claims):
        try:
            if claims(hf_config):
                return _RANK_CLAIMS
        except Exception:  # noqa: BLE001 — a broken claim() must not break discovery
            pass
    archs = getattr(runner_cls, "supported_architectures", None)
    if archs and arch and arch in archs:
        return _RANK_ARCH
    mtypes = getattr(runner_cls, "supported_model_types", None)
    if mtypes and model_type and model_type in mtypes:
        return _RANK_MODEL_TYPE
    return 0


def match_runner(hf_config, raw: Optional[dict] = None):
    """Select the entry-point runner class claiming this model, or None.

    Precedence within entry points: ``claims()`` > ``supported_architectures`` >
    ``supported_model_types``. Most-specific match wins; a tie at the top rank is
    ambiguous and raises (caller should ask for an explicit ``--runner``).
    """
    arch = _config_arch(hf_config, raw)
    model_type = _config_model_type(hf_config, raw)

    candidates = []  # (rank, name, runner_cls)
    for name, ep in iter_entry_point_runners():
        try:
            runner_cls = ep.load()
        except Exception as exc:  # noqa: BLE001 — skip a broken package, keep going
            import sys
            sys.stderr.write(
                f"[dispatch] WARNING: runner entry point {name!r} failed to load: "
                f"{type(exc).__name__}: {exc}\n"
            )
            continue
        rank = _match_rank(runner_cls, hf_config, arch, model_type)
        if rank > 0:
            candidates.append((rank, name, runner_cls))

    if not candidates:
        return None

    top = max(c[0] for c in candidates)
    winners = [c for c in candidates if c[0] == top]
    if len(winners) > 1:
        names = ", ".join(sorted(n for _, n, _ in winners))
        raise RuntimeError(
            f"Ambiguous runner selection: entry points [{names}] all claim this model "
            f"(arch={arch!r}, model_type={model_type!r}) at the same specificity. "
            "Disambiguate with an explicit --runner module:Class."
        )
    return winners[0][2]


# --------------------------------------------------------------------------- #
# HF repo self-declaration (precedence step 3 — trust-gated)
# --------------------------------------------------------------------------- #

def parse_self_declared_runner(model_path: str, hf_config=None,
                               raw: Optional[dict] = None) -> Optional[str]:
    """Return a ``"module:Class"`` runner spec self-declared by the model repo, or None.

    Sources, in order:
      1. ``tt_dispatch.json`` sidecar in a local model dir: ``{"runner": "module:Class"}``.
      2. ``tt_runner`` field in the model's ``config.json``.

    This points at code that will be IMPORTED AND EXECUTED — treat exactly like
    ``trust_remote_code``. The caller gates honoring it behind ``--unsafe`` and an
    optional allowlist (see ``runner_allowed``).
    """
    p = pathlib.Path(model_path)
    if p.is_dir():
        sidecar = p / "tt_dispatch.json"
        if sidecar.is_file():
            try:
                data = json.loads(sidecar.read_text())
                spec = data.get("runner")
                if spec:
                    return str(spec)
            except Exception:  # noqa: BLE001
                pass
    if raw is None:
        raw = _raw_config_json(model_path)
    spec = raw.get("tt_runner") if raw else None
    return str(spec) if spec else None


def runner_allowed(spec: str) -> bool:
    """Whether ``spec`` passes the optional ``DISPATCH_RUNNER_ALLOW`` allowlist.

    If the env var is unset/empty, all specs are allowed (the ``--unsafe`` gate is
    the only barrier). If set, only listed ``module:Class`` (or module-prefix)
    entries are honored.
    """
    allow = os.environ.get(ENV_ALLOW, "").strip()
    if not allow:
        return True
    entries = {e.strip() for e in allow.split(",") if e.strip()}
    if spec in entries:
        return True
    # Allow a bare module prefix to whitelist a whole package.
    module = spec.partition(":")[0].partition(".")[0]
    return module in entries or spec.partition(":")[0] in entries


# --------------------------------------------------------------------------- #
# Top-level resolver (used by load_model)
# --------------------------------------------------------------------------- #

def resolve_runner_class(model_path: str, explicit: Optional[str], unsafe: bool):
    """Resolve the runner class for ``model_path`` by precedence.

    Returns ``(runner_cls_or_None, source)``. ``None`` means "use the generic
    TTModelRunner fallback". ``source`` is a short human-readable tag for logging.
    """
    # 1. Explicit override (arg or env). User-supplied -> trusted.
    explicit = explicit or os.environ.get(ENV_RUNNER) or None
    if explicit:
        return _load_dotted(explicit), f"explicit:{explicit}"

    # Lazily load config only if there is any chance of a non-generic match.
    eps = list(iter_entry_point_runners())
    hf_config = None
    raw = None
    if eps:
        hf_config = load_hf_config(model_path)
        raw = _raw_config_json(model_path)
        matched = match_runner(hf_config, raw)
        if matched is not None:
            return matched, f"entry_point:{matched.__module__}:{matched.__name__}"

    # 3. HF repo self-declaration — trust-gated behind --unsafe (+ allowlist).
    if unsafe:
        if raw is None:
            raw = _raw_config_json(model_path)
        spec = parse_self_declared_runner(model_path, hf_config, raw)
        if spec:
            if not runner_allowed(spec):
                raise RuntimeError(
                    f"Self-declared runner {spec!r} is not in DISPATCH_RUNNER_ALLOW; "
                    "refusing to load it. Add it to the allowlist or pass --runner."
                )
            import sys
            sys.stderr.write(
                f"[dispatch] WARNING: honoring self-declared runner {spec!r} from the "
                "model repo (trust_remote_code-class: this imports and executes "
                "repo-referenced code).\n"
            )
            return _load_dotted(spec), f"self_declared:{spec}"

    # 4. Generic fallback.
    return None, "generic"


__all__ = [
    "BaseRunner",
    "ENTRY_POINT_GROUP",
    "validate_runner",
    "load_hf_config",
    "iter_entry_point_runners",
    "match_runner",
    "parse_self_declared_runner",
    "runner_allowed",
    "resolve_runner_class",
]
