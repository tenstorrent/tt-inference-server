<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->

# `tt::transport` — Mooncake KV-migration PoC (issue [#3890](https://github.com/tenstorrent/tt-inference-server/issues/3890))

A proof-of-concept for a **Tenstorrent custom backend for the
[Mooncake Transfer Engine](https://github.com/kvcache-ai/Mooncake)**: move a tensor
from one galaxy's *device DRAM* to another's, through Mooncake transport, with a
UMD-backed device-DRAM storage backend.

This PoC is cataloged alongside the other Mooncake explorations in the repo's
Mooncake hub, [`mooncake/poc-transfer-engine/`](../../../../mooncake/poc-transfer-engine/)
— start there for the design rationale ([`adr-mooncake-backend.md`](../../../../mooncake/poc-transfer-engine/adr-mooncake-backend.md))
and diagrams. This file is the code-level orientation for `src/transport/` +
`include/transport/`.

## Core idea: storage / transport split

Mooncake registers and DMAs **host virtual addresses only** — it cannot touch TT
device DRAM. So #3890 splits a "Transfer Engine" into two mechanisms and stages
device DRAM through a registered host buffer (a **bounce buffer**, two extra copies;
zero-copy/RDMA-direct is follow-up):

```
                     ┌──────────── sender galaxy ────────────┐
                     │                                       │
   device DRAM  ──readInto (UMD)──►  host staging buffer  ───┤
   (NocAddr)                          (registered)           │
                                                             │ submitAndWait
                                                             │  (Mooncake TCP/RDMA)
                     ┌──────────── receiver galaxy ──────────┤
                     │                                       ▼
   device DRAM  ◄──writeFrom (UMD)──  host staging buffer  ◄─┘
   (NocAddr)                           (registered)
        │
        └──readInto──►  byte-compare against expected pattern  →  PASS / FAIL
```

- **Storage** = how bytes move between a backing store and the host staging buffer
  (host DRAM memcpy, or device DRAM via UMD).
- **Transport** = how the staging buffer moves host→host between galaxies
  (Mooncake TCP / RDMA).

## Components

```
                MooncakeMigrationWorker          ← the #3890 process:
                 write / transfer / verify          maps 1:1 to the 3 scope items
                          │
                          ▼
        ┌──────── ITransferEngine ────────┐      transport interface
        │   MooncakeTransferEngine        │ ─────pimpl───► mooncake::TransferEngine
        │   (init/register/openSegment/   │               (TT_TRANSPORT_WITH_MOONCAKE)
        │    submitAndWait)               │
        └──────────────┬──────────────────┘
                       │ holds a
                       ▼
        ┌──────── IStorageBackend ────────┐      storage interface
        │  HostDramStorageBackend         │      baseline (plain memcpy)
        │  DeviceDramStorageBackend  ◄────┼── THE CUSTOM BACKEND (#3890)
        └──────────────┬──────────────────┘
                       │ delegates to
                       ▼
                 UmdDeviceAccess  ─────pimpl───►  tt-metal device DRAM I/O
                                                  (USE_METAL_CPP_LIB)
```

| File | Role |
|------|------|
| `transfer_types.hpp` | `NocAddr` (`channel<<32 \| local_addr`), `TransferRequest/Status`, `SegmentHandle`, `EngineConfig`, `StorageMedium`, `TransportProtocol` — placeholder types that mirror Mooncake's surface. |
| `i_storage_backend.hpp` | Storage interface: `readInto` / `writeFrom` between a backing store and the host staging buffer. |
| `host_dram_storage_backend.*` | Backing store is host memory; `addr` is a host VA, staging is a `memcpy`. Transport-only baseline. |
| `device_dram_storage_backend.*` | **The custom backend.** Backing store is device DRAM; `addr` is a `NocAddr`; delegates to `UmdDeviceAccess`. |
| `umd_device_access.*` | Stages bytes host ↔ device DRAM via tt-metal `Read/WriteFromDeviceDRAMChannel`. Real under `USE_METAL_CPP_LIB`, no-op fallback otherwise. |
| `i_transfer_engine.hpp` | Transport interface: `init → registerLocalMemory → openSegment → submitAndWait`. Mirrors the Mooncake engine surface. |
| `mooncake_transfer_engine.*` | Wraps `mooncake::TransferEngine` + composes an `IStorageBackend`. Real under `TT_TRANSPORT_WITH_MOONCAKE`, no-op fallback otherwise. |
| `mooncake_migration_worker.*` | Drives the three #3890 scope items: `writeTensorOnSender`, `transferToReceiver`, `verifyTensorOnReceiver`. |

## Build guards

Each real backend sits **behind a build guard with a no-op fallback**, so
`transport_lib` and `transport_test` build in *every* configuration with no hard
Mooncake/tt-metal dependency. When a guard is off, methods log and report failure
rather than crashing. tt-metal / Mooncake are hidden behind pimpls so headers stay
dependency-free.

| Guard | Enables | Build flag |
|-------|---------|------------|
| `USE_METAL_CPP_LIB` | Real UMD device-DRAM I/O | `--blaze` / `-DTT_METAL_HOME=…` |
| `TT_TRANSPORT_WITH_MOONCAKE` | Real Mooncake transport | `--mooncake` / `-DENABLE_MOONCAKE=ON` (implies `--blaze`) |

## How to build

From `cpp_server/`:

```bash
./build.sh                  # default: both guards OFF — transport_lib/test still build (no-op fallbacks)
./build.sh --blaze          # real UMD device-DRAM backend (USE_METAL_CPP_LIB)
./build.sh --mooncake       # real Mooncake transport (implies --blaze) → builds transport_migration_e2e
```

Mooncake is built inside the `tt-llm-engine` sub-build (`DS_ENABLE_MOONCAKE`), which
is why `--mooncake` implies `--blaze`. RDMA: add `-DENABLE_MOONCAKE_RDMA=ON`.

## How to run tests

**Unit / smoke (`transport_test`)** — runs in *any* build configuration; exercises
every interface, the host-DRAM memcpy round-trip, and the worker's storage-staging
path. Guarded transport/UMD paths report failure without crashing.

```bash
cd build && ctest --output-on-failure        # all tests
./build/transport_test                        # just this binary
```

**Two-galaxy acceptance (`transport_migration_e2e`)** — only built with `--mooncake`.
A manual two-process sender/receiver driver (not a ctest case). Both sides derive the
same deterministic byte pattern, so the receiver verifies without out-of-band payload.

```bash
# Transport-only smoke test over real Mooncake TCP, no hardware (host stands in for device DRAM):
tests/integration/run_transport_migration_e2e.sh

# Two boards on one host, real device DRAM (needs --blaze+--mooncake build, TT_METAL_HOME, real HW):
STORAGE=device SRC_DEVICE_ID=0 DST_DEVICE_ID=1 BYTES=1048576 \
  tests/integration/run_transport_migration_e2e.sh
```

The harness exchanges the receiver's *actual* segment name via a rendezvous file
(under `P2PHANDSHAKE` the transport binds a random port). See the script header and
`tests/integration/transport_migration_e2e.cpp` for all env overrides
(`STORAGE`, `BYTES`, `SRC_ADDR`/`DST_ADDR`, `TIMEOUT_SEC`, …).

## Worker discovery via the Mooncake Metadata Service (#4209)

OS-assigned ports change on every start, so two independent hosts can't hard-code each
other's address. The metadata service is a shared registry that fixes this: the receiver
registers under a **predefined logical name** mapped to its **auto-detected IP + dynamic
port**, and the sender looks up that name to resolve the live address. No rendezvous
file, no fixed port. **Host RAM only.**

```
   receiver: init(metadata, name="kv-receiver-0") + registerLocalMemory
                 └─► metadata service: kv-receiver-0 → {IP, dynamic port}
   sender:   openSegment("kv-receiver-0")  ──► service resolves the address
                 └─► submitTransfer (TCP, one tensor)
```

Driver: `migration_worker_discovery` (`--mooncake` only). Start the metadata service on a
host both peers can reach, then run a receiver and a sender:

```bash
# Single-host smoke test (auto-starts the service, runs both workers):
tests/e2e/scripts/run_migration_worker_discovery.sh

# Two-host run:
#  metadata host (META_HOST): serves http://0.0.0.0:18080/metadata
tests/integration/run_mooncake_metadata_server.sh
#  receiver host:
MC_TCP_BIND_ADDRESS=<receiver-ip> build/migration_worker_discovery --role receiver \
  --metadata http://META_HOST:18080/metadata --name kv-receiver-0 --bytes 1048576
#  sender host:
build/migration_worker_discovery --role sender \
  --metadata http://META_HOST:18080/metadata --name kv-sender-0 \
  --peer kv-receiver-0 --bytes 1048576
```

Gotchas:
- **HTTP metadata plugin must be compiled in** (`USE_HTTP ON` in
  `tt-llm-engine/cmake/mooncake.cmake`, with `libcurl` on the include/library path), else
  the client aborts with `Unable to find metadata storage plugin http`.
- **Use the wheel's `http_metadata_server.py`** (what `run_mooncake_metadata_server.sh`
  starts), not `mooncake_master` — the latter serves a different route and 404s every
  PUT/GET.
- **Multi-NIC hosts:** set `MC_TCP_BIND_ADDRESS=<this host's IP>` so the engine advertises
  a reachable interface (auto-detect may pick `docker0`/`flannel.1`).
- The `404 metadata not found` lines at startup are **expected** — the engine probes for
  its own name before registering it.

## Mooncake Migration Worker discovery

`bringup_mooncake_worker` is the worker's entry point / composition root (one process per
worker). `PeerDiscoveryService` owns *how* peers are resolved (the resolve-with-retry loop +
timeout); `MooncakeMigrationWorker` owns the ordered lifecycle — allocate host-DRAM pool → init
engine → register/publish (makes us discoverable) → **delegate** peer discovery → hold until
SIGTERM → teardown in reverse. **Register-before-discover** is the invariant the worker owns.
Workers are symmetric peers: each takes its own `--name` and its peers as `--peer`; success
is `CONNECTED to N peers` then `READY`. Logic is launcher-agnostic — a bash loop, MPI, or an
orchestrator all just spawn one process per worker.

**MPI e2e test** (`tests/e2e/scripts/run_migration_workers_mpi.sh`, ctest
`MooncakeMpiDiscovery`): starts the metadata service, then `mpirun -np 20` launches 4 prefill +
16 decode workers on one host. `migration_worker_rank_launch.sh` maps each rank to a
disaggregated topology — `prefill-p` peers `decode-(4p..4p+3)`, each `decode-d` peers back to
`prefill-(d/4)` — and the test passes once all 20 log `CONNECTED` within the timeout.

```bash
# all 20 workers, self-contained (auto-starts metadata service):
WORKER_BIN=./build/bringup_mooncake_worker \
  tests/e2e/scripts/run_migration_workers_mpi.sh
```

## Validation status

| Step | Status |
|------|--------|
| Interfaces + host-DRAM round-trip + worker staging (`transport_test`, any build) | impl |
| Device-DRAM backend single-galaxy round-trip (UMD, `--blaze`) | impl |
| Mooncake transport loopback TCP (host backend, `--mooncake`) | impl |
| Two-galaxy acceptance, both backends enabled | pending a two-process HW run |
| Metadata-service worker discovery, two hosts, host RAM (#4209, `migration_worker_discovery`) | **validated** (two hosts, 1 MiB tensor, byte-verified MATCH) |
| Productionized discovery worker (#4294, `bringup_mooncake_worker`) | **validated locally** (single host, MPI `-np 20` = 4 prefill + 16 decode, all `CONNECTED`→`READY`; run manually, not yet wired into CI) |

Note: the unit/smoke `transport_test` runs in any CI build; the MPI discovery
e2e (`MooncakeMpiDiscovery`) is currently a manual/local check (it needs the
Mooncake build + a metadata service) and is not yet in a GitHub workflow.

## Future work — wiring into the tt-llm-engine migration worker

The existing KV-migration worker lives in
[`../../tt-llm-engine/disaggregation/migration/`](../../tt-llm-engine/disaggregation/migration/)
and already moves KV cache galaxy-to-galaxy over **MPI/DCN** with ULFM fault
tolerance. It routes all transport through an abstract `SenderBackend` /
`ReceiverBackend` interface (today only `DcnSenderBackend`). Mooncake fills
*capability* gaps (multi-NIC RDMA bandwidth, dynamic membership, a pooled global
KV-cache store with cross-request prefix reuse), not a correctness hole.

**Integration goal:** add a `MooncakeSenderBackend` implementing that same interface,
delegating the host→host hop to this branch's `ITransferEngine`. `MigrationFrontend`,
`MigrationClient`, and the worker stay unchanged; Mooncake becomes a runtime-selectable
transport alongside DCN/MPI.

```
   MigrationFrontend  (tt-llm-engine, unchanged)
            │
   ┌──── SenderBackend / ReceiverBackend interface (existing) ────┐
   │   DcnSenderBackend (existing)      MooncakeSenderBackend (TODO)│
   └────────┬───────────────────────────────────┬─────────────────┘
            │                                    │ delegates host→host hop
   MultiDeviceReader/Writer                ITransferEngine  (this branch)
   (shared device DRAM I/O)                       │
            │                                     │ TCP / RDMA
        ══ MPI/DCN ══►  peer worker  ◄══════ Mooncake ══
```

**Open design decision — where Mooncake attaches:**

- **Option A (recommended): attach at `SenderBackend`.** Reuse `MigrationFrontend`,
  `MultiDeviceReader`, address tables, and the worker's threading/ULFM model; use
  Mooncake only for the wire. This branch's `DeviceDramStorageBackend` / `UmdDeviceAccess`
  then **dissolve** (the existing `MultiDeviceReader` already does device I/O); only
  `MooncakeTransferEngine` survives.
- **Option B: attach at the byte interface.** Keep this branch's full storage+transport
  stack and bypass `MultiDeviceReader` — more PoC code survives, but re-implements
  addressing, zero-copy, and completion that already exist.

**Tensions to resolve under either** (see ADR-002 for detail): the existing interface
*fuses* device-read + transport with a zero-copy `acquire/publish` while this branch
*splits* them (two copies); two UMD device-access layers now exist (pick one);
routing/metadata differ (MPI rank + `KvChunkAddressTable` vs `openSegment`/offset +
`P2PHANDSHAKE`); receiver-write and ACK/fault semantics differ (DCN
receiver-drains-then-writes vs Mooncake one-sided write); Mooncake currently builds
inside tt-llm-engine while `transport_lib` is dependency-free.

**Other follow-ups:** custom zero-copy / RDMA-direct `Transport` subclass (register the
UMD DRAM mapping instead of bouncing); multi-tensor / batched transfers; concurrency
and failure/retry semantics matching the incumbent's ULFM behaviour.

**Discovery lifecycle (post-merge):** discovery resolves peers once at bring-up and the
worker then holds the handles until SIGTERM. Steady-state membership changes are not yet
handled — if a peer crashes and restarts on a new dynamic port, its cached `SegmentHandle`
goes stale. Production needs periodic peer health checks and re-discovery/reconnection on
peer restart (plus metrics: peer count, time-to-ready, reconnection events). Discovery is
already cancellable (a SIGTERM during bring-up aborts the poll loop promptly); wiring the
MPI e2e into CI is also pending.
