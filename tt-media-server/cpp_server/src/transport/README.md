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

## Worker discovery via the Mooncake Metadata Service (issue [#4209](https://github.com/tenstorrent/tt-inference-server/issues/4209))

The `transport_migration_e2e` harness above uses `metadata_uri = "P2PHANDSHAKE"`,
where each engine binds a **random OS-assigned port** and there is no shared
registry — so the receiver's *actual* `host:randomPort` name has to be smuggled to
the sender through a **rendezvous file** on a shared path. That does not work across
two independent hosts.

#4209 removes that hack by validating the **Mooncake Metadata Service** as the
discovery mechanism. With a real metadata service (the HTTP metadata server, etcd, or
redis), the receiver advertises under a **predefined logical name**;
Mooncake's "new RPC mapping" path
([`transfer_engine_impl.cpp`](../../tt-llm-engine/third_party/mooncake/mooncake-transfer-engine/src/transfer_engine_impl.cpp))
registers `<name> → {auto-detected IP, OS-assigned dynamic port}` in that service.
The sender opens the predefined name and the service resolves the dynamic address —
no rendezvous file, no predefined port. **Host RAM only** (no device memory).

```
   receiver: init(metadata, name="kv-receiver-0") + registerLocalMemory
                 └─► metadata service: kv-receiver-0 → {IP, dynamic port}
   sender:   openSegment("kv-receiver-0")  ──► service resolves the dynamic port
                 └─► submitTransfer (TCP, one tensor) across hosts
```

Driver: `migration_worker_discovery` (built next to `transport_migration_e2e`,
`--mooncake` only). Start the metadata service once on a host both peers can reach,
then run a receiver and a sender:

```bash
# Single-host smoke test (auto-starts the metadata service, runs both workers):
tests/integration/run_migration_worker_discovery.sh

# Two-host run (the real PoC):
#  metadata host (META_HOST):
tests/integration/run_mooncake_metadata_server.sh           # serves http://0.0.0.0:18080/metadata
#  receiver host (advertise this host's own LAN IP on multi-NIC boxes):
MC_TCP_BIND_ADDRESS=<receiver-ip> build/migration_worker_discovery --role receiver \
  --metadata http://META_HOST:18080/metadata --name kv-receiver-0 --bytes 1048576
#  sender host:
build/migration_worker_discovery --role sender \
  --metadata http://META_HOST:18080/metadata --name kv-sender-0 \
  --peer kv-receiver-0 --bytes 1048576
```

### Build + runtime requirements (learned the hard way)

- **HTTP metadata plugin must be compiled in.** `tt-llm-engine/cmake/mooncake.cmake`
  forces `USE_HTTP OFF` by default; flip it to `ON` (and have `libcurl` headers/libs on
  the include/library path) or the C++ client aborts with
  `Unable to find metadata storage plugin http`.
- **Use the wheel's `http_metadata_server.py`, not `mooncake_master`.** The
  `mooncake_master` binary's embedded HTTP server answers a different route than the
  vendored C++ client expects (you get `http=404 metadata not found` on every PUT/GET).
  `run_mooncake_metadata_server.sh` launches the Python server, which serves the
  `GET/PUT/DELETE /metadata?key=...` API the client actually calls.
- **Multi-NIC hosts:** set `MC_TCP_BIND_ADDRESS=<this host's IP>` so the engine
  advertises the interface the peer can reach (auto-detection may pick `docker0`/
  `flannel.1`). Each host advertises *its own* IP.

The `http=404 ... metadata not found` lines you see at startup are **expected** — that's
the engine probing for a pre-existing descriptor for its own name before it registers.

## Validation status

| Step | Status |
|------|--------|
| Interfaces + host-DRAM round-trip + worker staging (`transport_test`, any build) | impl |
| Device-DRAM backend single-galaxy round-trip (UMD, `--blaze`) | impl |
| Mooncake transport loopback TCP (host backend, `--mooncake`) | impl |
| Two-galaxy acceptance, both backends enabled | pending a two-process HW run |
| Metadata-service worker discovery, two hosts, host RAM (#4209, `migration_worker_discovery`) | **validated** (two hosts, 1 MiB tensor, byte-verified MATCH) |

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
