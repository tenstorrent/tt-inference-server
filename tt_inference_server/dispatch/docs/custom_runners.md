# Custom runners for the dispatch serving runtime

The dispatch runtime serves standard decoder-only transformers with a generic
runner (`TTModelRunner`) that introspects a HuggingFace module for "the
attention," "the MLP," embeddings, and per-layer norms. Models whose tt-nn graph
is **not** a standard transformer — MoE, Gated-DeltaNet/Mamba-state, multimodal —
cannot be expressed that way.

A **custom runner** lets you serve such a model through the same
`load_model()` / `serve` entry points. Your runner lives in **your own package**
— no model code goes into this repo, and no dispatch code goes into yours. The
only shared surface is the contract below.

> Custom runners run arbitrary code you point dispatch at. The runtime carries no
> correctness or SLA guarantee for them — that is what `--unsafe` acknowledges.

## 1. The contract — `BaseRunner`

A runner must provide three methods and three attributes. That is the entire
surface `ModelHandle` uses. You do **not** need to subclass anything; the contract
is structural (duck-typed), checked by `validate_runner()` right after construction.

```python
class MyRunner:
    # --- required attributes (set in __init__) ---
    _tokenizer: object   # a transformers-like tokenizer (ModelHandle.tokenizer reads this)
    _listed: bool        # True if this is a known/validated model; else False
    _community: bool     # True if unverified/community (drives the serve "community" tag)

    # --- required methods ---
    def generate(self, prompt: str, max_new_tokens: int = 50,
                 temperature: float = 1.0, chat: bool = True) -> str: ...

    def generate_stream(self, prompt: str, max_new_tokens: int = 50,
                        temperature: float = 1.0, chat: bool = True):
        # yield decoded text deltas (str), one per decode step;
        # the FINAL yielded item MUST be a dict:
        #   {"finish_reason": str, "prompt_tokens": int, "completion_tokens": int}
        ...

    def benchmark(self, prompt: str, n_tokens: int = 50) -> tuple[float, str]:
        # returns (tokens_per_second, output_text)
        ...
```

`generate_stream` is the one the OpenAI server actually drives (both streaming and
non-streaming responses), so make sure the final-dict usage record is correct.

### Constructor

`load_model()` constructs your runner as:

```python
Runner(model_path, device, max_seq=..., unsafe=..., force_novel=...,
       trace_region_size=..., device_ids=...)
```

It passes only the keyword arguments your `__init__` actually declares (accept
`**kwargs` to future-proof, or declare just the ones you use). `device` is an open
ttnn device unless you manage your own (see §4).

## 2. Selecting a runner

In precedence order (most-explicit wins):

| # | Mechanism | Trust |
|---|---|---|
| 1 | Explicit: `serve --runner module:Class` or env `DISPATCH_RUNNER=module:Class` | user-supplied |
| 2 | Entry-point match (installed package, see §3) | installed package |
| 3 | HF-repo self-declaration (`tt_runner` / `tt_dispatch.json`) | **`--unsafe` only** |
| 4 | Generic `TTModelRunner` fallback | default |

```bash
# Explicit (works for a source checkout — no install needed):
python -m tt_inference_server.dispatch.serve serve --unsafe \
    --runner my_pkg.dispatch_runner:Qwen36Runner /path/to/checkpoint

# Auto-discovery (your package is installed and claims the model):
python -m tt_inference_server.dispatch.serve serve --unsafe /path/to/checkpoint
```

If two installed runners claim the same model at the same specificity, dispatch
errors and asks for an explicit `--runner`.

## 3. Entry-point auto-discovery

Declare the `tt_inference_server.runners` entry-point group in **your** package's
`pyproject.toml`, then `pip install` it into the serving environment:

```toml
[project.entry-points."tt_inference_server.runners"]
qwen3_5_moe = "my_pkg.dispatch_runner:Qwen36Runner"
```

Dispatch scans this group and matches against the model's HF config, by priority:

1. `@classmethod claims(cls, hf_config) -> bool` — most flexible; overrides the sets.
2. `supported_architectures: set[str]` — matched against `config.architectures[0]`.
3. `supported_model_types: set[str]` — matched against `config.model_type`.

```python
class Qwen36Runner:
    supported_model_types = {"qwen3_5_moe"}
    # or, for finer control:
    # @classmethod
    # def claims(cls, hf_config): return hf_config.model_type == "qwen3_5_moe"
```

> Entry points only resolve for **installed** packages (`pip install` /
> `pip install -e .`). For a plain source checkout that isn't installed, use the
> `--runner module:Class` path instead.

## 4. Device / mesh ownership

By default `load_model()` opens a single ttnn device and passes it to the runner,
and closes it at exit. If your runner needs a different device topology — e.g. a
1×1 **mesh** via `open_mesh_device` — set:

```python
class Qwen36Runner:
    MANAGES_OWN_DEVICE = True
    def __init__(self, model_path, device, **kwargs):
        # device is None here. Open, own, and register cleanup for your own device:
        import ttnn
        self.mesh = ttnn.open_mesh_device(...)
        import atexit; atexit.register(ttnn.close_mesh_device, self.mesh)
```

When `MANAGES_OWN_DEVICE = True`, dispatch does **not** open a device and passes
`device=None`. You are responsible for opening, owning, and closing it (and for
reserving any trace region you need — `trace_region_size` is passed to your
constructor if you accept it). This avoids the double-open/double-close that would
otherwise conflict with a mesh runner.

**Clean teardown is an obligation, not an option.** A `MANAGES_OWN_DEVICE` runner
**must** close its device on shutdown (register `ttnn.close_mesh_device` / `close_device`
via `atexit`, or close it in your own lifecycle). This is what makes Ollama-style model
hot-swap reliable: serving runs one model per process, and when that process exits the
next model must find a pristine card. An ungraceful teardown can leave a locked device
mutex (`/dev/shm/tt_device_*`) that wedges the card until a manual `tt-smi -r` — so the
clean-close path must be the common one.

## 5. HF-repo self-declaration (trust-gated)

A model repo can point at its runner so plain `serve --unsafe <repo>` finds it
without an installed entry point. Either:

- a `tt_runner` field in the model's `config.json`:
  ```json
  { "model_type": "qwen3_5_moe", "tt_runner": "my_pkg.dispatch_runner:Qwen36Runner" }
  ```
- or a `tt_dispatch.json` sidecar in the model directory:
  ```json
  { "runner": "my_pkg.dispatch_runner:Qwen36Runner", "max_seq": 4096 }
  ```

This is honored **only under `--unsafe`** because it imports and executes
repo-referenced code — treat it exactly like `trust_remote_code`. Optionally
constrain it with an allowlist:

```bash
export DISPATCH_RUNNER_ALLOW="my_pkg"          # module-prefix, or full module:Class
```

The referenced module must still be importable in the serving environment.

## 6. Environment coupling

A custom runner pins to whatever tt-metal / ttnn build its model code was written
against; the serving environment must satisfy the runner package's imports
(declare them as the package's dependencies). Firmware is environment-wide, not
per-runner — match it to what the model needs before serving.

## 7. Warm starts (optional)

Cold JIT-compiling kernels on first serve can be slow for large models. The
orthogonal `tt-kernel-manager` tool can pull a precompiled tt-metal kernel cache
matching the serving environment before `serve`, so the first request is a cache
hit:

```bash
tt-kernel pull <bundle-matching-your-env>   # then serve as usual
```

This is independent of the runner seam — it only removes compile latency.
