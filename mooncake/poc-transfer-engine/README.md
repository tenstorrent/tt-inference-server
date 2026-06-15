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
> target vendored in `tt-llm-engine`. So the source stays in
> [`tt-media-server/cpp_server/`](../../tt-media-server/cpp_server) where it compiles;
> **this folder is the design hub** (rationale, diagrams, and the map below).

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
| [`tests/transport_test.cpp`](../../tt-media-server/cpp_server/tests/transport_test.cpp) | Smoke tests over every interface, all build configs |
| [`tests/integration/transport_migration_e2e.cpp`](../../tt-media-server/cpp_server/tests/integration/transport_migration_e2e.cpp) + [`run_transport_migration_e2e.sh`](../../tt-media-server/cpp_server/tests/integration/run_transport_migration_e2e.sh) | Two-process acceptance harness (`--mooncake` builds only) |

## Build & run

All commands run from **`tt-media-server/cpp_server/`** (the e2e harness expects the
build output under that root):

```bash
./build.sh                  # both guards OFF — transport_lib/transport_test still build (no-op fallbacks)
./build.sh --blaze          # real UMD device-DRAM backend (USE_METAL_CPP_LIB)
./build.sh --mooncake       # real Mooncake transport (implies --blaze) → also builds transport_migration_e2e

cd build && ctest --output-on-failure        # runs transport_test in any configuration

# Two-process acceptance harness (stays in cpp_server; needs a --mooncake build):
tests/integration/run_transport_migration_e2e.sh                  # transport-only loopback, no HW
STORAGE=device SRC_DEVICE_ID=0 DST_DEVICE_ID=1 \
  tests/integration/run_transport_migration_e2e.sh                # real device DRAM, two boards

# #4209 worker discovery via the Mooncake Metadata Service (host RAM only, two hosts):
tests/integration/run_mooncake_metadata_server.sh                 # start the metadata service
tests/integration/run_migration_worker_discovery.sh               # single-host smoke (auto-starts service)
```

Build guards: `USE_METAL_CPP_LIB` (real UMD I/O, via `--blaze`) and
`TT_TRANSPORT_WITH_MOONCAKE` (real Mooncake transport, via `--mooncake`). Each real
backend sits behind a guard with a no-op fallback, so the library and unit test build
in **every** configuration. RDMA: `--mooncake-rdma`.

## Status

| Step | Status |
|------|--------|
| Interfaces + host-DRAM round-trip + worker staging (`transport_test`, any build) | impl |
| Device-DRAM backend single-galaxy round-trip (UMD, `--blaze`) | impl |
| Mooncake transport loopback TCP (host backend, `--mooncake`) | impl |
| Two-galaxy acceptance, both backends enabled | pending a two-process HW run |
| Metadata-service worker discovery, two hosts, host RAM (#4209) | impl |

## Contents of this folder

```
README.md                          this hub
adr-mooncake-backend.md            design record (rationale, interfaces, integration plan)
diagrams/3890-implemented.excalidraw   editable source of the architecture diagram
diagrams/architecture.png              rendered architecture diagram
```
