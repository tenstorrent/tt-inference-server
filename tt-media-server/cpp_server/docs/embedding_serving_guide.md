# How cpp_server is configured to serve embedding models

This guide explains, from first principles, how the C++ media server gets its
configuration, which environment variables it reads and at which point of its
lifecycle, why the current setup requires a hand-built environment, and what
the v1 cleanup changes. It is written in layers: each section assumes only the
ones before it.

Everything here was verified empirically on an n150 with BGE-large-en-v1.5
(tt-metal pinned at `65718bb`), where the unmodified server produced
bit-identical embeddings to the Python reference once the environment was
right.

---

## Level 0 — the one-paragraph version

`tt_media_server_cpp` is one operating-system process that contains two worlds:
a C++ web server (Drogon) that accepts HTTP requests, and an embedded Python
interpreter that actually runs the model on the Tenstorrent device. Both worlds
read their configuration from the same place — the process environment — but
they read *different variables*, at *different moments*, and nobody translates
between them. The server works today only if you manually set every variable
each world expects before launching. The v1 cleanup makes the C++ side the
single owner of configuration and has it explicitly hand the Python side what
it needs.

---

## Level 1 — first principles

### What is an environment variable?

Every process on Linux carries a private list of `NAME=value` string pairs
called its *environment*. Two rules explain almost everything in this guide:

1. **A child process gets a copy of its parent's environment at the moment it
   is started.** Your shell is a process; when you run `./server`, the server
   is a child of the shell and inherits whatever the shell had *at that
   instant*.
2. **The copy is a snapshot.** If you change a variable in the shell after the
   server started, the running server never sees the change. Likewise, a
   program can change its *own* environment (`setenv` in C, `os.environ` in
   Python) and its future children inherit that, but its parent is unaffected.

### `VAR=x ./cmd` vs `export VAR=x` vs `env VAR= ./cmd`

- `VAR=x ./cmd` (prefix form) sets `VAR` **only for that one command**. Your
  shell keeps its old value. This is the safe, precise way to configure a
  single launch.
- `export VAR=x` sets `VAR` **for the whole shell session** — this command and
  every command you run afterwards inherits it. Convenient, but it is how
  variables "leak": you export something for one experiment and three hours
  later a different program silently picks it up. (That exact thing happened
  to us — see the PYTHONPATH story below.)
- `env VAR= ./cmd` runs the command with `VAR` set to the **empty string**,
  overriding whatever the shell had. We used `env PYTHONPATH= ...` to *scrub*
  an inherited value for one launch without touching the shell. (`env -u VAR`
  would remove the variable entirely instead of emptying it; for `PYTHONPATH`
  the two are equivalent in effect.)

### What is a virtual environment (venv), really?

A venv is just a folder (here
`/localdev/jzivanovic/tt-metal-65718bb/python_env/`) containing:

- `bin/python3` — a Python executable (or symlink),
- `lib/python3.10/site-packages/` — its own private collection of installed
  packages (this is where `ttnn` lives, installed from the pinned tt-metal),
- `pyvenv.cfg` — a marker file that tells Python "you are inside a venv, use
  the packages next to me".

"Activating" a venv (`source .../bin/activate`) does something very mundane:
it puts the venv's `bin/` directory **first on your `PATH`** and sets
`VIRTUAL_ENV`. That's it. Any process that then looks for `python3` finds the
venv's one, and that Python resolves its packages from the venv's
`site-packages`.

**Why does a C++ server care about a venv?** Because it *embeds* a Python
interpreter (next section). When that embedded interpreter boots, it has to
figure out where the Python standard library and installed packages live. Part
of that search involves locating a `python3` on `PATH`. With the venv
activated, it locates the venv's `python3`, sees the `pyvenv.cfg` next to it,
and therefore uses the venv's `site-packages` — which is the only place the
correctly pinned `ttnn` is installed. Launch the server without the venv on
`PATH` and the embedded interpreter resolves a different, wrong set of
packages (or none).

### What is PYTHONPATH?

When Python executes `import ttnn`, it walks an ordered list of directories
(`sys.path`) and takes the **first** match. `PYTHONPATH` is an environment
variable whose directories get **prepended** to that list. It is powerful and
dangerous for the same reason: whatever is in it wins, and it is invisible —
nothing in your command line shows it is set.

Extra subtlety that bit us: the tt-metal `models/` directory is a *namespace
package* (it has no `__init__.py`). For namespace packages Python does not
stop at the first match — it **merges every `models/` directory found across
all of `sys.path`** into one logical package, and for each submodule the
earliest directory wins. Our shell had `PYTHONPATH=/localdev/jzivanovic/tt-metal`
(the HEAD checkout) exported from some earlier setup. The server inherited it,
so `models.demos...bge_large_en` silently loaded from HEAD while `ttnn` came
from the pinned venv — a Frankenstein stack that happened to produce identical
numbers, but only by luck. That is why the working recipe launches with
`env PYTHONPATH=`.

### What is an embedded Python interpreter?

Normally Python is a program (`python3`) that runs your script. But Python is
also a library (`libpython3.10.so`), and any C++ program can link it and call
`Py_Initialize()` to boot an interpreter **inside its own process**. That is
what `embedding_runner.cpp` does. Important consequences:

- The interpreter shares the process, so it shares the process **environment**.
  Every rule from above applies to it.
- It shares the process's `PATH`-based discovery, which is why venv activation
  matters at *server launch time*, not at build time only.
- The Python code it imports (`tt_model_runners/embedding_runner.py`) is
  ordinary `tt-media-server` code — the same code the pure-Python server would
  run. The C++ server is a different front door to the same model runners.

---

## Level 2 — the two configuration systems

There is no single config file. There are **two independent configuration
systems staring at the same process environment**:

### System 1: C++ `Settings` (`cpp_server/src/config/settings.cpp`)

Reads env vars on demand (each accessor calls `getenv`, most cache the result
on first use). It decides *server-level* things:

| Question | Variable | Default |
|---|---|---|
| Which modality does this process serve? | `MODEL_SERVICE` | `llm` |
| Which devices / how many workers? | `DEVICE_IDS` | `(0)` |
| Where is the tt-media-server Python code? | `TT_PYTHON_PATH` | `..` |
| How many requests may be batched together? | `MAX_IN_FLIGHT_COUNT` | 32 |
| How long to wait to fill a batch? | `MAX_BATCH_DELAY_TIME_MS` | 5 |
| What API key must clients send? | `OPENAI_API_KEY` (read in `main.cpp`) | `your-secret-key` |

### System 2: Python `Settings` (`tt-media-server/config/settings.py`)

A pydantic singleton **created once, at import time**, inside the embedded
interpreter of each worker. It reads:

- `MODEL` — must be a value of the internal `ModelNames` enum, e.g.
  `bge-large-en-v1.5`. **Not** the HuggingFace id `BAAI/bge-large-en-v1.5`;
  passing the HF id raises `ValueError: ... is not a valid ModelNames` and the
  worker dies during import.
- `DEVICE` — e.g. `n150`.

From `(MODEL, DEVICE)` it looks up `config/constants.py` (`ModelConfigs`) and
derives everything else: `model_runner`, `max_batch_size` (8 for BGE on n150),
`device_mesh_shape`, `is_galaxy`, `model_weights_path`, ...

If `MODEL` is missing it logs `Skipping config overrides` and keeps the class
defaults — which are SDXL-flavoured (`model_runner='tt-sdxl-trace'`,
`max_batch_size=1`, SDXL weights path). Crucially it does **not** crash.

### The gap between them

The C++ side never sets `MODEL` or `DEVICE` for the Python side. Both systems
just read the ambient environment you happened to launch with. If you set the
C++ variables but forget the Python ones, you get a server that *boots and even
answers single requests correctly* but is silently misconfigured — the
nastiest failure mode there is. This gap is the main thing the cleanup closes
(the plan's `p4-python-env` step: `workerProcessMain` will explicitly export
`MODEL`/`DEVICE`/`MODEL_RUNNER` for the forked worker before Python starts).

---

## Level 3 — the lifecycle: which variable is read at which moment

```
you run ./build/tt_media_server_cpp -p 8000
│   the process snapshots the environment NOW; later shell changes are invisible
│
├─ main()
│    reads MODEL_SERVICE        → picks the modality: "embedding" | "image" | "tts" | "llm"
│                                  registers ONLY that modality's HTTP routes
│                                  (other paths will 404: "route_not_found")
│    reads OPENAI_API_KEY       → arms the Bearer-token auth filter
│
├─ EmbeddingService::start()
│    reads DEVICE_IDS           → one worker process per device group, e.g. "(0)" = 1 worker
│    reads MAX_IN_FLIGHT_COUNT  → maxBatchSize of the dispatch loop  (note: NOT MAX_BATCH_SIZE)
│    reads MAX_BATCH_DELAY_TIME_MS → how long the batcher waits before flushing a partial batch
│    fork() each worker         → child inherits the environment
│    setenv TT_VISIBLE_DEVICES  → set BY the service FOR each worker (which chips it may open)
│
├─ worker process
│    Py_Initialize()            → embedded interpreter boots
│                                  finds python3 via PATH → venv → venv site-packages (ttnn)
│                                  PYTHONPATH (if any) is prepended to sys.path — leak risk
│    sys.path += TT_PYTHON_PATH → so `import tt_model_runners...` resolves
│    import tt_model_runners.embedding_runner
│         └─ imports config.settings
│              └─ Python Settings singleton constructed  ← MODEL and DEVICE read HERE,
│                 exactly once; the entire Python-side config is frozen at this instant
│    BGELargeENRunner()         → the class is HARDCODED in embedding_runner.cpp;
│                                  MODEL does not choose the class today, it only
│                                  fixes max_batch_size / mesh / weights in Settings
│    set_device(), warmup()     → tt-metal PCC gate must pass (≥ 0.90)
│    "[Worker 0] Ready"
│
└─ Drogon listens on the port
     per request:
       auth filter              → Authorization: Bearer <OPENAI_API_KEY value>
       EmbeddingController      → request.model empty? default "BAAI/bge-large-en-v1.5"
                                  (cosmetic: echoed in the response, routes nothing)
       queue → dispatch thread  → drains up to maxBatchSize requests into one batch
       pipe (binary codec)      → worker → Python run() → embeddings back over the pipe
```

Key lifecycle insight: **almost everything is read once, early, and frozen.**
`MODEL_SERVICE` at `main()`, worker count at service start, the whole Python
config at first import. Nothing is re-read per request. Configuration is a
launch-time contract, not a runtime knob.

---

## Level 4 — modality and model selection

**Modality** (embedding vs image vs TTS vs LLM): chosen once per process by
`MODEL_SERVICE` at startup. One process serves exactly one modality; requests
to other modalities' routes get 404.

**Model within the modality**: also one per process, and today it is doubly
locked in:

1. `embedding_runner.cpp` hardcodes the Python class `BGELargeENRunner`.
2. Python `Settings` is a frozen singleton, so even the Python side cannot
   switch models after import.

The `"model"` field in the request body does **not** select anything. You can
send `"model": "banana"` and get BGE embeddings back with `"model": "banana"`
echoed. Serving two embedding models simultaneously means running two server
processes, on different ports, with disjoint `DEVICE_IDS`.

After the cleanup this becomes: the C++ config carries the model name, a
factory picks the right runner class from it (instead of the hardcoded class),
and the request field can at least be *validated* against the loaded model.
Still one model per process — the model is loaded onto the device at warmup,
so that is the natural granularity, same as vLLM and friends.

---

## Level 5 — the working launch recipe, justified line by line

> **Superseded as of the v1 cleanup.** Skip to *Level 7* for the current
> recipe. The version below is the *pre-cleanup* one; it is kept because every
> line of it documents a real failure, and because the problems it works around
> are what Level 6 and 7 explain.

```bash
# 1) venv on PATH → embedded interpreter finds the pinned ttnn
source /localdev/jzivanovic/tt-metal-65718bb/python_env/bin/activate

env \
  PYTHONPATH= \                # 2) scrub ambient PYTHONPATH; ours had the HEAD
                               #    tt-metal checkout, which silently overrides
                               #    the pinned model code (namespace-package merge)
  TT_METAL_HOME=/localdev/jzivanovic/tt-metal-65718bb \
                               # 3) tt-metal runtime (kernels, firmware) from the
                               #    pinned worktree, matching the venv's ttnn
  MODEL='bge-large-en-v1.5' \  # 4) Python Settings: internal enum value, NOT the
                               #    HF id → gives max_batch_size=8, right mesh, weights
  DEVICE=n150 \                # 5) Python Settings: with MODEL, selects the row
                               #    in ModelConfigs for this hardware
  MODEL_SERVICE=embedding \    # 6) C++: this process serves the embedding modality
  DEVICE_IDS='(0)' \           # 7) C++: one worker, on chip 0
  MAX_IN_FLIGHT_COUNT=8 \      # 8) C++ batch cap. MUST NOT exceed the model's
                               #    max_batch_size (8 for BGE on n150). The default
                               #    is 32: at >8 simultaneous requests the dispatcher
                               #    forms an oversized batch, Python asserts
                               #    "Batch size 15 exceeds max 8", and every request
                               #    in that batch gets HTTP 500 (verified empirically)
  TT_PYTHON_PATH=/localdev/jzivanovic/tt-inference-server/tt-media-server \
                               # 9) C++ → embedded Python: where tt_model_runners lives
  ./build/tt_media_server_cpp -p 8000

# requests must carry the key from OPENAI_API_KEY (default shown):
curl -H "Authorization: Bearer your-secret-key" \
     -H "Content-Type: application/json" \
     -d '{"input": "hello world", "model": "BAAI/bge-large-en-v1.5"}' \
     http://localhost:8000/v1/embeddings
```

Every line above was discovered through a real failure. Omit (4)–(5) and the
server *still boots* but crashes with HTTP 500 under concurrent load
(`Batch size 7 exceeds max 1`). Omit (2) and you may be running a different
tt-metal's model code than you think. Use the HF id in (4) and the worker dies
at import. Omit (8) and everything works up to 8 simultaneous requests, then
collapses (a 16-request burst produced a batch of 15 → 15 × HTTP 500). Omit
the auth header and you get 401.

Note the asymmetry in (8): the *model's* batch size is derived automatically
from `MODEL`+`DEVICE` on the Python side, but the *C++ dispatcher's* cap is an
unrelated variable with an unrelated default, and nothing checks they are
consistent. Two knobs, one invariant (`C++ cap ≤ model max`), zero enforcement
— a textbook example of the two-config-systems problem.

One more operational note: the warmup PCC gate compares against random inputs
with no fixed seed, so it is noisy. We observed pass values 0.9099–0.9348 and
one spurious *failure* at 0.8822 on the exact same pinned code that passed
before and after. A single warmup failure is not proof of a regression — retry
before investigating.

---

## Level 6 — what is wrong with this, precisely

The baseline proved the data path is correct (bit-exact vs the Python
reference, batching and concurrency included). The problems are all in how
configuration reaches the code:

1. **Silent misconfiguration instead of failure.** With `MODEL` unset the
   server boots, warms up, and answers single requests — on n150 only, because
   the pydantic *defaults* happen to describe an n150 (`device='n150'`, mesh
   `(1,1)`). The wrong `model_runner` default is masked by the hardcoded C++
   class. The wrong `max_batch_size=1` is a landmine that only detonates under
   concurrency. A config error should kill the process at startup with a clear
   message, not degrade it invisibly.
2. **Two config systems, no bridge.** C++ knows the modality and devices;
   Python knows the model; each trusts the launcher to have set the other's
   variables. The fix: C++ parses and validates everything once, then
   explicitly exports what Python needs into each worker before the
   interpreter starts.
3. **Ambient environment as config channel.** `PYTHONPATH` and PATH-based venv
   discovery mean the *shell you happen to launch from* silently changes which
   code runs. The pybind11 rewrite should set `sys.path` deliberately from
   validated config instead of inheriting ambience.
4. **Hardcoded model.** One C++ file names one Python class. Adding the next
   three embedding models this way means three more hardcoded branches. The
   cleanup replaces this with config-driven selection (a real
   `EmbeddingConfig`, embedding `ModelRunnerType` values, a factory).
5. **Misnamed knobs and dead code.** Batches are capped by
   `MAX_IN_FLIGHT_COUNT` (a queue-depth concept) instead of a `MAX_BATCH_SIZE`,
   and nothing enforces `cap ≤ model max` — confirmed empirically (batch of 15
   against a model max of 8 → mass HTTP 500). There is a stray
   `setenv("MAX_BATCH_SIZE")` in `stop()` that does nothing useful; the
   `ModelRunnerType::MOCK` registration creates the *real* runner.
6. **The enum-vs-HF-id footgun.** The external API speaks HF ids
   (`BAAI/bge-large-en-v1.5`), the config speaks enum values
   (`bge-large-en-v1.5`), and nothing maps between them at the boundary.

"Parse the environment from config files instead of hand-setting it per run"
is the right instinct, with one refinement: the target is not necessarily a
config *file*, but a **single validated entry point**. Whether values arrive
via a file, env vars, or CLI flags, they should be parsed once by the C++
`Settings`, validated (fail fast on nonsense), and then *pushed* to every
consumer — including the embedded Python — rather than each consumer pulling
from the raw environment independently.

---

## Level 7 — the current recipe, after the v1 cleanup

Model selection is now configuration. `MODEL_RUNNER_TYPE` picks a row in the
`embeddingModels()` table in `settings.cpp`, and that row carries only what the
parent process must know before any interpreter exists: the HF id (the
controller's default when a client omits `"model"`), the internal `MODEL` enum
value, and the per-device `max_batch_size` (the parent forms the batches). The
forked worker exports `MODEL` / `DEVICE` / `MODEL_RUNNER` into its own
environment before Python is imported, so those are no longer yours to set.

Which Python class implements a model is deliberately *not* in that table. The
worker calls `tt_model_runners.runner_fabric.get_device_runner()`, which maps
`settings.model_runner` — set from the `MODEL_RUNNER` we exported — to a class.
That registry already exists in Python and is where new models get added
anyway, so C++ asks for the runner by name rather than keeping a second copy of
the mapping that could drift.

```bash
source /localdev/jzivanovic/tt-metal-65718bb/python_env/bin/activate

env PYTHONPATH= \
    TT_METAL_HOME=/localdev/jzivanovic/tt-metal-65718bb \
    MODEL_SERVICE=embedding \
    MODEL_RUNNER_TYPE=tt_bge_large_en \
    DEVICE=n150 \
    DEVICE_IDS='(0)' \
    TT_PYTHON_PATH=/localdev/jzivanovic/tt-inference-server/tt-media-server \
    ./build/tt_media_server_cpp -p 8000
```

Three variables from the old recipe are gone. `MODEL` is derived from
`MODEL_RUNNER_TYPE` — and setting it yourself is now an error rather than an
override, because that is exactly how the wrong model's config used to get
paired with the runner. `MAX_IN_FLIGHT_COUNT` is gone because batches are
capped by the model's own `max_batch_size`; the invariant that nothing used to
enforce is now structural. The enum-vs-HF-id footgun is gone with it: the table
holds both spellings, so the controller's default model id comes from config
instead of a string literal.

`DEVICE` stays, and cannot be derived: it describes the machine, not the model,
and Python's `ModelConfigs` is keyed on it (BGE is batch 8 on n150, 16 on t3k).
Set it wrong and you get a startup error listing the supported values.

Because the C++ table mirrors numbers that Python owns, the worker re-reads
`settings.max_batch_size` after import and refuses to start if the two
disagree, logging `max_batch_size=8 (agrees with Python)` when they match. A
drifted table is a hard error at boot rather than an oversized batch hours
later.

### Running with no hardware

`MODEL_RUNNER_TYPE=embedding_mock` selects a runner that imports no Python and
opens no device. It answers with a deterministic unit-length 1024-dim vector
derived from a hash of the input text, so identical inputs
give identical vectors and different inputs give different ones. It also
reproduces the real runner's two failure modes — unknown model name, oversized
batch — so the HTTP, batching, and codec layers can be regression-tested in CI:

```bash
MODEL_SERVICE=embedding MODEL_RUNNER_TYPE=embedding_mock DEVICE_IDS='(0)' \
    ./build/tt_media_server_cpp -p 8000
```

No venv, no `TT_METAL_HOME`, no `DEVICE` needed, and it is ready in
milliseconds. The vectors are meaningless as embeddings — this is a plumbing
stand-in, not an approximation of the model.

To add another embedding model: one enumerator in `ModelRunnerType`, its two
string cases (`toString`, and `toClientRunnerName` which must match Python's
`ModelRunners` value), and one row in `embeddingModels()`. The runner class
comes from Python's fabric, so nothing else in C++ changes.

---

## Appendix A — PYTHONPATH vs TT_PYTHON_PATH vs venv, side by side

All three influence the same thing — the ordered folder list (`sys.path`) that
Python walks when it executes an `import` — but they answer three different
questions:

| Mechanism | Question it answers | Who defines it | What it contributes |
|---|---|---|---|
| venv (via `PATH`) | "Which Python installation am I?" | Python itself | stdlib + `site-packages` (ttnn, torch, ...) — the *installed* packages |
| `PYTHONPATH` | "What extra folders should *every* Python search?" | Python itself | ambient, invisible, inherited prepends — we want it **empty** |
| `TT_PYTHON_PATH` | "Where is the tt-media-server source code?" | **this project** (Python ignores it) | one folder, inserted explicitly by `embedding_runner.cpp` so `import tt_model_runners` works |

The embedded interpreter has no `python3` binary of its own, so at boot it
searches `PATH` for one to decide which installation it belongs to — that is
what venv activation feeds it. `TT_PYTHON_PATH` is the controlled, on-purpose
version of what `PYTHONPATH` does ambiently: the repo's runner code is plain
source, never pip-installed, so the C++ adds its folder to `sys.path` by hand.

## Appendix B — building cpp_server (CMake, build.sh, version-locking)

The server's C++ calls functions like `Py_Initialize()`. Compiling that needs
two things from *some* Python installation:

- **headers** (`Python.h`) at compile time — the declarations, and
- **libpython3.10.so** at link time — the implementation.

**CMake** is the tool that finds them. It reads `CMakeLists.txt` (the build
recipe: source files, dependencies) and locates every dependency on the
machine — Drogon, JsonCpp, and Python. For Python it takes the first `python3`
on `PATH` and asks it where its headers and library live. **Build with the
venv active** so that CMake finds the venv's Python 3.10 and bakes that
installation in. **build.sh** is only a convenience wrapper that invokes CMake
with the right flags and then runs the compilation (use it without `--blaze`;
the LLM engine is irrelevant for embeddings).

The finished binary is stamped with "I need `libpython3.10.so`"; at launch the
dynamic linker loads exactly that library. So the interpreter inside the
server is **version-locked at build time** — switching Python versions means
rebuilding, not re-configuring. The venv therefore matters twice, for two
different reasons: at *build* time it decides which Python gets compiled in;
at *run* time it decides where that interpreter finds its packages.

Practical notes: `install_dependencies.sh` covers Drogon, JsonCpp, etc.;
`tokenizers-cpp` additionally needs a Rust toolchain (`rustup`), which that
script does not install. A clean build takes ≈ 4 minutes on this machine once
dependencies exist and produces `build/tt_media_server_cpp`.
