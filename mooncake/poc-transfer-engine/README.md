# poc-transfer-engine — Tenstorrent custom backend for the Mooncake Transfer Engine

The C++ proof-of-concept for issue
[#3890](https://github.com/tenstorrent/tt-inference-server/issues/3890),
*"Custom backend for DRAM access via UMD"*: move a tensor from one galaxy's
**device DRAM** to another's, through the
[Mooncake Transfer Engine](https://github.com/kvcache-ai/Mooncake), staged through
a UMD-backed device-DRAM storage backend.

> **Where the code lives.** Unlike poc1–poc3 (self-contained Python that runs against
> a Mooncake master), this PoC is **C++ that builds *inside* the cpp_server build** —
> it depends on cpp_server's CMake, tt-metal, and the Mooncake `transfer_engine`
> target, which is built directly from the vendored Mooncake submodule at
> [`tt-media-server/cpp_server/third_party/Mooncake`](../../tt-media-server/cpp_server/third_party/Mooncake).
> The source stays in [`tt-media-server/cpp_server/`](../../tt-media-server/cpp_server) where it
> compiles; **this folder is the design hub** (rationale, diagrams, and the map below).

---

## What this PoC answers

The Transfer Engine has two mechanisms — **Storage** (how bytes move between a
backing store and a host staging buffer) and **Transport** (how the staging buffer
moves host→host between galaxies). Mooncake registers/DMAs **host** virtual
addresses only and cannot touch TT device DRAM, so the custom backend stages device
DRAM into a registered host **bounce buffer** while the transport stays generic:

```
   device DRAM ──readInto (UMD)──► host staging buf ──submitAndWait (TCP/RDMA)──►
                                                       peer host buf ──writeFrom (UMD)──► device DRAM
```

A `MooncakeMigrationWorker` drives the three #3890 scope items 1:1 —
**write** a known tensor on the sender, **transfer** it via the engine + custom
backend, **verify** it byte-for-byte on the receiver.

See [`adr-mooncake-backend.md`](adr-mooncake-backend.md) for the full design
rationale, the storage/transport interface split, the verified Mooncake API
surface, and how this attaches to the existing tt-llm-engine migration worker.

## Where it fits among the PoCs

| PoC | Language | Scope |
|-----|----------|-------|
| [poc1](../poc1) | Python | Raw `MooncakeDistributedStore` put/get smoke test |
| [poc2](../poc2) | Python | Control-plane orchestration loop (mocked migration worker) |
| [poc3](../poc3) | Python | Mooncake Store multi-tier storage (DRAM / SSD / remote) |
| **poc-transfer-engine** | **C++** | **The custom UMD device-DRAM backend + Mooncake transport, in cpp_server** |

## Source map (in `tt-media-server/cpp_server/`)

| Path | Role |
|------|------|
| [`include/transport/`](../../tt-media-server/cpp_server/include/transport) | Interfaces + placeholder types (`IStorageBackend`, `ITransferEngine`, `transfer_types.hpp`, …) |
| [`src/transport/`](../../tt-media-server/cpp_server/src/transport) | Implementations (host/device DRAM backends, UMD access, Mooncake engine, migration worker) |
| [`src/transport/README.md`](../../tt-media-server/cpp_server/src/transport/README.md) | Code-level orientation (file-by-file) |
| [`tests/unit/transport/transport_test.cpp`](../../tt-media-server/cpp_server/tests/unit/transport/transport_test.cpp) | Smoke tests over every interface, all build configs |
| [`tests/e2e/transport_migration_e2e.cpp`](../../tt-media-server/cpp_server/tests/e2e/transport_migration_e2e.cpp) + [`run_transport_migration_e2e.sh`](../../tt-media-server/cpp_server/tests/e2e/scripts/run_transport_migration_e2e.sh) | Two-process acceptance harness (`--mooncake` builds only) |

## Dependencies & first-time setup (for `--mooncake` builds)

The Mooncake `transfer_engine` is built **from source** out of the vendored submodule
(`third_party/Mooncake`), so a fresh clone needs two one-time setup steps before
`./build.sh --mooncake` will configure. (Plain `./build.sh` / `--blaze` builds don't
need any of this — Mooncake is only pulled in by `--mooncake`.)

### 1. Check out the Mooncake submodule — recursively

It has its own nested submodules (`extern/yalantinglibs`, `extern/pybind11`) that
**must** be present, so `--recursive` is required:

```bash
# from the repo root
git submodule update --init --recursive \
  tt-media-server/cpp_server/third_party/Mooncake
```

> ⚠️ **Common gotcha — do NOT run the init under `sudo`.** The repo is owned by your
> user; running `git submodule update` (or `dependencies.sh`, which calls it) as root
> trips git's *"detected dubious ownership"* guard, and the nested submodules silently
> stay **empty**. The symptom shows up only later as a CMake error:
> `Could not find a package configuration file provided by "yalantinglibs"`.
> Always init submodules as your **normal user**.

### 2. Install Mooncake's system dependencies

The standalone Transfer Engine build expects these to already be installed — it uses
`find_package(... REQUIRED)`, so a missing one is a hard configure error:

`libibverbs-dev`, `libnuma-dev`, `libgoogle-glog-dev`, `libgflags-dev`,
`libjsoncpp-dev`, `libyaml-cpp-dev`, plus an **installed** `yalantinglibs`
(resolved via `find_package(yalantinglibs CONFIG)`).

Mooncake ships a helper that installs the apt packages **and** builds + installs
yalantinglibs from its submodule:

```bash
# Init submodules (step 1) FIRST, as your user, THEN run this — it needs sudo for apt.
sudo third_party/Mooncake/dependencies.sh
```

If `dependencies.sh` ran *before* the submodules were checked out (the sudo gotcha
above), yalantinglibs was never installed. Install it manually:

```bash
cd third_party/Mooncake/extern/yalantinglibs
cmake -S . -B build -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF -DBUILD_UNIT_TESTS=OFF
cmake --build build -j"$(nproc)"
sudo cmake --install build      # installs yalantinglibsConfig.cmake under /usr/local
```

> **RDMA is always compiled in.** The standalone Transfer Engine has no library-level
> toggle to disable RDMA (`libibverbs` / `rdma_transport` are linked unconditionally),
> so `--mooncake` already includes it — there is no separate `--mooncake-rdma` flag.
>
> **Storage is not built.** Only `mooncake-transfer-engine` is compiled (the single
> target cpp_server links). Mooncake's **store** and its Python/Rust bindings are
> intentionally left out — they would pull in Python3-dev, xxhash/zstd/liburing and
> Rust for artifacts cpp_server doesn't use. Enabling the store later is a one-block
> change in `cpp_server/CMakeLists.txt` (`add_subdirectory(third_party/Mooncake)` with
> `WITH_STORE=ON`); the `transfer_engine` target name is unchanged, so this PoC's
> wiring would not need to change.

## Build & run

All commands run from **`tt-media-server/cpp_server/`** (the e2e harness expects the
build output under that root — **not** from inside `build/`):

```bash
./build.sh                    # both guards OFF — transport_lib/transport_test still build (no-op fallbacks)
./build.sh --blaze            # real UMD device-DRAM backend (USE_METAL_CPP_LIB)
./build.sh --mooncake         # real Mooncake transport (TCP+RDMA) → also builds transport_migration_e2e
./build.sh --blaze --mooncake # both real backends — required for the device-DRAM (real HW) e2e path

cd build && ctest --output-on-failure        # runs transport_test in any configuration

# Two-process acceptance harness (stays in cpp_server; needs a --mooncake build):
tests/e2e/scripts/run_transport_migration_e2e.sh                 # transport-only loopback, no HW
STORAGE=device SRC_DEVICE_ID=0 DST_DEVICE_ID=1 \
  tests/e2e/scripts/run_transport_migration_e2e.sh               # real device DRAM, two boards (build with --blaze --mooncake)
```

Build guards: `USE_METAL_CPP_LIB` (real UMD I/O, via `--blaze`) and
`TT_TRANSPORT_WITH_MOONCAKE` (real Mooncake transport, via `--mooncake`). Each real
backend sits behind a guard with a no-op fallback, so the library and unit test build
in **every** configuration.

## Status

| Step | Status |
|------|--------|
| Interfaces + host-DRAM round-trip + worker staging (`transport_test`, any build) | impl |
| Device-DRAM backend single-galaxy round-trip (UMD, `--blaze`) | impl |
| Mooncake transport loopback TCP (host backend, `--mooncake`) | impl |
| Two-galaxy acceptance, both backends enabled | pending a two-process HW run |

## Contents of this folder

```
README.md                          this hub
adr-mooncake-backend.md            design record (rationale, interfaces, integration plan)
diagrams/3890-implemented.excalidraw   editable source of the architecture diagram
diagrams/architecture.png              rendered architecture diagram
```
