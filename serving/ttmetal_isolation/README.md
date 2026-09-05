# Per-model tt-metal build isolation (Exabox QB2)

Prevents the failure where one model's tt-metal build corrupts another model's live serve.

## The failure this fixes

On Exabox, bare-metal Quetzal/ttis serves point `TT_METAL_HOME` (== `TT_METAL_RUNTIME_ROOT`)
at a tt-metal tree. That **one tree is used for two things at once**:

1. the **built lib** the process loaded (`build/lib/libtt_metal.so`, `build*/lib/_ttnn.so`), and
2. the **JIT kernel-source root** that TTNN compiles device kernels from *at serve time*.

So the tree's git **source HEAD must equal the commit its built lib was compiled at**. If a
later build runs `git checkout <other-commit>` inside that same tree, the source moves out
from under the running serve and its next JIT compile fails, e.g.:

```
error: 'NUM_WORKER_CORES' was not declared in this scope
```

That is exactly what happened: a gpt-oss "sink" build advanced the shared
`/home/nkapre/tt-metal` source HEAD to `2a8253ad…` while the Llama-1B serve's built lib +
runtime-identity were `1c2aff50…`, breaking Llama until the source was reverted.

`/home` is **node-local NVMe** on Exabox (each compute node has its own tree at the same
path). The corruption is therefore a *single-node, single-shared-tree* problem: multiple
(model) builds and serves sharing one `/home/nkapre/tt-metal`.

## The layout

Give every `(model, tt-metal commit)` its **own immutable, node-local tree**:

```
$HOME/.cache/tt-metal/<40-hex-commit>/     # canonical per-commit tree
    build/ build_Release/ python_env/ ...  # built lib
    .ttq-runtime-identity.json             # {"base_revision":"<commit>", ...}  == HEAD
```

Rules:

- A tree is **immutable**: once built and stamped, its `HEAD` == its directory name ==
  `.ttq-runtime-identity.json:base_revision`. A build **never** `git checkout`s a different
  commit into an existing tree — a different commit gets a different directory.
- Serves set `TT_METAL_HOME`/`TT_METAL_RUNTIME_ROOT` to their pinned per-commit tree.
- Never build or write tt-metal into `/data` (shared NFS). Node-local `$HOME` only;
  build scratch in `/tmp/$USER`.
- The generic `$HOME/tt-metal` name is deprecated as a *serve target*; it is the classic
  shared tree that caused the incident.

`.ttq-runtime-identity.json` is the same file `vllm-tt-metal/src/run_vllm_api_server.py`
fail-closes on (`base_revision != actual_runtime`). Stamping the per-commit tree makes the
serve preflight authoritative for bare-metal too.

## The tools

| Script | Role |
| --- | --- |
| `ttmetal_lib.sh` | Shared helpers: canonical path, HEAD, identity read/write, verdict, live-serve enumeration. |
| `ttmetal_guard.sh` | The guard (see below). |
| `ttmetal_build.sh` | Build-entrypoint wrapper: resolves the per-commit tree, runs the guard, reuses if already coherent, else clones/checks-out/builds/creates-venv/**stamps**. Never targets the shared tree. |
| `resolve_commit.py` | Resolves the required 40-hex tt-metal commit from a model spec / tree identity / env. |

### The guard — `ttmetal_guard.sh`

Run it **on the node** where a build/checkout would happen (serves are node-local).

```
ttmetal_guard.sh check <tree>                  # coherence verdict for a tree
ttmetal_guard.sh check-pair <headsha> <idsha>  # pure verdict on explicit shas (tests/CI)
ttmetal_guard.sh live-serves                   # live serves on this node + tree each uses
ttmetal_guard.sh guard-checkout <tree> <commit># is it safe to checkout/build here?
ttmetal_guard.sh stamp <tree> <commit>         # write the runtime-identity stamp
```

Verdicts (`check`) and exit codes:

| Verdict | Exit | Meaning |
| --- | --- | --- |
| `COHERENT` | 0 | identity present, `base_revision == HEAD`, build present |
| `CORRUPT` | 3 | identity present but `base_revision != HEAD` — **the 2a8253ad-vs-1c2aff50 class** |
| `UNSTAMPED` | 4 | built tree with no identity (built-lib commit cannot be attested) |
| `NOBUILD` | 5 | git tree, no built lib |
| `NOTREE` | 2 | not a git worktree |

`guard-checkout <tree> <commit>` refuses to let a build clobber a serve:

- **exit 10 REFUSE** — a live serve on this node depends on this exact tree and `<commit>`
  is not what the tree coherently provides (checkout/build here would break its JIT).
- **exit 11 REFUSE** — the tree is an immutable per-commit tree already coherent at a
  *different* commit; relocate to the per-commit path for `<commit>`.
- **exit 0 ALLOW** — no live serve depends on the tree (or it already coherently provides
  `<commit>`).

### The build wrapper — `ttmetal_build.sh`

```
ttmetal_build.sh <commit40hex> [--reference <existing-tt-metal>] [--jobs N] [--dry-run]
ttmetal_build.sh --from-tree <tree>          # resolve the commit from a tree's identity
ttmetal_build.sh --from-spec <spec.json|id>  # resolve via resolve_commit.py
```

It resolves `TREE=$HOME/.cache/tt-metal/<commit>`, guards it, reuses if already `COHERENT`,
else builds and stamps, and prints the `TT_METAL_HOME`/`TT_METAL_RUNTIME_ROOT` a serve
should use. Because the target is always the per-commit path, a build **cannot** land in the
shared tree or another model's tree.

## Proof (measured against the live incident SHAs)

`gpt = 2a8253ad20a9270102f12431f26639288330fe4b`,
`llama = 1c2aff5064f19936d70127c80fe9333d687cc427`.

- Exact mismatch class: `check-pair $gpt $llama` -> `CORRUPT` (exit 3); `check-pair $llama
  $llama` -> `COHERENT` (exit 0).
- Live `check /home/nkapre/tt-metal`: on p01t06 (Llama serve) -> `COHERENT
  head=1c2aff50 identity=1c2aff50`; on p06t01 (shared tree left at the sink commit) ->
  `UNSTAMPED head=2a8253ad`.
- Live `guard-checkout /home/nkapre/tt-metal $gpt` on p01t06 (Llama live) -> **REFUSE
  exit 10**, naming the serve pid and pointing the build at
  `$HOME/.cache/tt-metal/2a8253ad…` — i.e. it would have prevented the incident.
- `guard-checkout $HOME/.cache/tt-metal/$gpt $gpt` -> ALLOW; re-checkout of Llama's own
  commit into its live tree -> ALLOW (no-op, no false positive).

## Adopting it

- New model / commit: `ttmetal_build.sh <commit>` (add `--reference /home/$USER/tt-metal`
  to clone fast from an existing local tree), then point the serve's
  `TT_METAL_HOME`/`TT_METAL_RUNTIME_ROOT` at the printed per-commit path.
- Migrate the still-shared Llama serve: build `1c2aff50…` into
  `$HOME/.cache/tt-metal/1c2aff50…` and repoint on the next serve restart.
- Backfill stamps on the already-per-commit trees that lack them (gpt `tt-metal-b534-gdn`,
  qwen `tt-metal-f9377427`): `ttmetal_guard.sh stamp <tree> <HEAD>`.
- CI/pre-build hook: run `ttmetal_guard.sh guard-checkout` before any `git checkout`/
  `build_metal.sh` in an existing tree.
