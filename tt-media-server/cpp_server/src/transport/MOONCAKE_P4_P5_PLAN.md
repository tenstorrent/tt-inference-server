# Mooncake KV-Cache Migration — Phase 4 + Phase 5 Implementation Plan

**Status:** plan / scoping
**Scope:** Phase 4 (k8s + metadata-service discovery) and Phase 5 (Kafka trigger), as **integration** of already-merged scaffolding with the P0–P3 data plane.
**Deferred:** Phase 6 (RDMA-direct — drop mirror + drain). Not due.
**Branch context:** `mooncake_merge` / `visnjakrsmanovicTT/4173-mooncake-tcp-kv-cache-block-transfer`.

---

## 1. Where we are

`#4173` (commit `fb74d916a`) is a **squash-merge** that folded all of Phase 0–3 into one commit, sitting on top of **#4406 "Combine Kafka and Mooncake discovery"**. So the branch tree contains **two parallel streams** that were developed independently and must now converge (hence `mooncake_merge`):

| Stream | What it is | State |
|---|---|---|
| **P0–P3 data plane** (`src/transport/`) | `KvMigrationMultiHostSender`, `KvMigrationSender/Receiver` orchestrators, `MooncakeKvSender/Receiver`, `KvCacheMirror`/`KvCacheLayout`/`RemoteRegion`, real `KvChunkAddressTable` adapter (gated by `ENABLE_KV_TABLE`), coexistence UMD device IO | Done; unit + host-mode e2e green |
| **P4/P5 scaffolding** (#4406) | `PeerDiscoveryService`, `MooncakeMigrationWorker::bringUp`, `bringup_mooncake_worker` binary, `src/messaging/*` (Kafka), `IMigrationExecutor` seam + `StubMigrationExecutor`, dev-kafka tooling, MPI discovery harness | Done, but **wired to stubs** |

### 1.1 The two seams the original plan reserved — both now exist as designed

- **Resolution interface** → `PeerDiscoveryService::discover(engine, peerNames) → map<name, SegmentHandle>`, backed by a **real Mooncake HTTP metadata service** (`EngineConfig::metadata_uri`), not a static list.
- **`migrate()` trigger entrypoint** → `IMigrationExecutor::execute(migrationId, MigrationRequest, onDone)`, with `StubMigrationExecutor` explicitly labelled a placeholder *"until the real Mooncake-backed executor lands."*

**Verdict:** the Phase 0–6 plan still stands. The only reframe is that **P4 and P5 are now integration work, not greenfield** — the scaffolding is merged; it just isn't connected to the real bytes.

### 1.2 The keystone gap

**No binary drives the real data plane.**

- `bringup_mooncake_worker`'s Kafka loop only **logs + acks** — it never calls `transferToReceiver` / `migrate`.
- `tt_kv_migration_consumer` uses `StubMigrationExecutor` (log + `onDone(SUCCESSFUL)`).
- `KvMigrationMultiHostSender::migrate(uuid, req)` exists and is tested, but nothing calls it from a triggered path.
- **Receiver-side orchestration is entirely unwired in production**: `bringup_mooncake_worker` registers a generic host-DRAM pool, **not** a `KvCacheMirror` segment, and never runs `KvMigrationReceiver::run()`.

Everything below closes that gap.

---

## 2. Relevant interfaces (verbatim, for reference)

**Trigger seam** (`include/runtime/worker/migration_executor.hpp`):
```cpp
class IMigrationExecutor {
 public:
  using DoneCallback = std::function<void(tt::services::MigrationStatus status)>;
  virtual ~IMigrationExecutor() = default;
  // Must return immediately; onDone invoked exactly once with terminal status.
  virtual void execute(uint64_t migrationId,
                       const tt::services::MigrationRequest& request,
                       DoneCallback onDone) = 0;
};
```

**Data-plane entrypoint** (`include/transport/kv_migration_multi_host_sender.hpp`):
```cpp
KvMigrationMultiHostSender(
    std::shared_ptr<ITransferEngine> engine,
    IDeviceIo& device,
    std::shared_ptr<const IKvTable> prefillTable,
    std::shared_ptr<const IKvTable> decodeTable,
    std::string prefillHost,
    std::unordered_map<std::string, KvControlChannel*> channels);

bool migrate(uint64_t uuid, const MigrationRequest& request);  // true iff ALL touched hosts succeed
```

**Discovery seam** (`include/transport/peer_discovery_service.hpp`):
```cpp
std::optional<std::map<std::string, SegmentHandle>> discover(
    ITransferEngine& engine,
    const std::vector<std::string>& peerNames,
    const std::atomic<bool>* cancelToken = nullptr);
```

**Request-shape mismatch to resolve:**
- Wire / `tt::services::MigrationRequest` — **single-layer, symmetric**: `{src_slot, dst_slot, layer_id, position_start, position_end}`.
- Transport `tt::transport::MigrationRequest` — **multi-layer, asymmetric**: `{src_slot, dst_slot, layer_begin, layer_end, src_position_begin/end, dst_position_begin/end}` with `srcSlice()` / `dstSlice()`.

---

## 3. Step-by-step plan (integration-first critical path)

Order: **5a → 4a → 5b → 4b → 4c**. 5a/4a/5b are standalone-testable on one box (no HW, no 16-galaxy).

> **Ordering correction (2026-06-30):** an earlier draft listed `5a → 5b → 4a`,
> but 5b ("Kafka → executor → real migration → ack") cannot run until the
> executor has a live `KvMigrationMultiHostSender` — which needs 4a's per-host
> `KvControlChannel`s (the connector) and the decode-side `KvMigrationReceiver`
> loop. So **4a is a hard prerequisite of 5b** and now precedes it. Step
> contents are unchanged; only the order moved.

### Phase 5a — The real executor (keystone; do first) — **DONE (2026-06-30)**

Smallest change that flips stubs → real migration.

1. **DONE — `MooncakeMigrationExecutor : tt::worker::IMigrationExecutor`**, in
   `include/src transport/mooncake_migration_executor.{hpp,cpp}` (compiled into
   `transport_lib`, which already exposes the worker/services headers — no
   `llm_runner_lib` coupling). Wraps `KvMigrationMultiHostSender`. `execute()`
   enqueues and returns; one background thread runs migrations serially (the
   per-host senders/channels aren't built for concurrent `migrate()`), then
   fires `onDone` **exactly once**: `migrate()==true → SUCCESSFUL`, else (false /
   threw / shutting down) `FAILED`. The dtor drops queued-but-unstarted jobs
   without firing their callbacks (they capture the dying owner; scheduler sees
   a timeout — the safe degraded path) and joins the in-flight one. A `MigrateFn`
   ctor allows DI for unit tests.
2. **DONE — wire/contract reshape (no adapter).** `tt::services::MigrationRequest`
   and `tt::messaging::MigrationRequestMessage` now mirror
   `tt::transport::MigrationRequest` field-for-field
   (`src_slot, dst_slot, layer_begin, layer_end, src/dst_position_begin/end`), so
   the only mapping is a by-name struct copy. JSON keys + the Python producers
   (`migration_cli.py`, `migration_e2e/acks.py`) updated to match.
3. **DEFERRED to 5b — swap the stub in `kv_migration_consumer_main.cpp`.**
   Constructing the real executor there needs a live `KvMigrationMultiHostSender`
   (engine + discovery + per-host control channels + tables + device), which is
   the 4a/5b control-plane work. The executor is ready and tested; only the
   composition root is pending.
4. **DONE — standalone test.** `mooncake_migration_executor_test` (8 tests:
   async dispatch, status mapping, field passthrough, exception→FAILED,
   sequential, dtor-drain) + `KvMigrationMultiHost.DrivenThroughMooncakeExecutor`
   in `kv_migration_orchestrator_test`: a Kafka-shaped `services::MigrationRequest`
   driven through the executor fans out to a real 2-host data plane and
   byte-verifies both decode hosts. No HW, no Kafka, no k8s. All transport tests
   green; Kafka-gated TUs syntax-checked (KAFKA_ENABLED=OFF in this env).

### Phase 4a — Receiver-side orchestration + mirror-as-segment (prerequisite of 5b)

Genuinely-missing production piece, not just wiring. Steps 5 & 6 are
`transport_lib` components, **DONE + unit-tested here** over a loopback fake
transport (no Kafka / HW); step 7 needs the worker binary and lands with 5b.

5. **DONE — decode worker runs `KvMigrationReceiver`** via new
   `KvMigrationReceiverServer` (`transport/kv_migration_endpoints.{hpp,cpp}`):
   injected server-transport factory → `KvControlChannel` → runs
   `KvMigrationReceiver::run()` on a background thread; `stop()` tears the
   transport down (unblocks the loop) and joins. The `MooncakeKvReceiver` it
   drives **already registers its full-table `KvCacheMirror` as the one Mooncake
   segment** at construction — so "mirror-as-segment" needed no new code, just
   this server wiring.
6. **DONE — control-channel connector** `KvControlChannelConnector` (same file):
   given a `host → Endpoint{ip,port}` map + an injected `TransportFactory`, opens
   a client `KvControlChannel` per decode host and exposes the
   `unordered_map<host, KvControlChannel*>` `KvMigrationMultiHostSender` takes.
   The transport factory is **injected** (not a direct `src/sockets` dep) so
   `transport_lib` stays socket-decoupled and the test runs over a loopback fake;
   production supplies a `TcpSocketTransport` factory. `connect()` returns false
   if any endpoint's factory failed but keeps the rest (same comprehensive-report
   contract as the multi-host sender). **Still open (5b/4-7):** endpoint
   resolution source — piggyback host:port on the metadata service **or** a
   parallel registry keyed by the discovery host names.
   **Tests (5 in `kv_migration_endpoints_test`, all green):** connector builds
   one channel/endpoint; skips a failed host but keeps the rest; server
   start-fails on null factory; and **end-to-end single-host + 2-host fan-out**
   drive the real data plane through connector + receiver-server and byte-verify
   each decode host. (Shared `Pipe`/`BlockingFakeTransport` lifted into
   `transport_test_fakes.hpp`; `stop()` now closes the inbound pipe so a server
   stops cleanly.)
7. **Table provisioning — production source is the ENGINE, not disk.** The
   `KvChunkAddressTable` is engine-authored: NoC addresses come from the engine's
   on-device KV-cache tensor allocation, and the **device map** (`FabricNode →
   ASIC unique id`) can only be resolved against live chips by a
   CreateDevice-capable process (the engine). Disaggregation ships both to the
   co-located worker over its shmem `MigrationLayerClient`
   (`send_kv_chunk_table` + `send_device_map`); the `.pb` files in the repo are
   only test fixtures.

   **DONE (approach 2 — our own engine→worker handoff, decoupled from the shmem
   client):** new `transport/device_map.{hpp}` (`DeviceMap`: FabricNode → u64 UMD
   ASIC id, keyed by encodeDevice) + `transport/engine_table_handoff.{hpp,cpp}`:
   a self-contained framed contract carrying `{ serialized table blob, device
   map }` over any `ISocketTransport` (a local engine↔worker socket in prod, a
   loopback fake in tests). `serializeEngineHandoff`/`parseEngineHandoff`,
   `sendEngineHandoff` (engine/producer), `receiveEngineHandoff` +
   `IEngineTableSource`/`SocketEngineTableSource` (worker/consumer; reuses
   `deserializeKvTable`). No dependency on the disaggregation shmem client, MPI,
   or a worker-side `.pb`. The engine team implements the producer against real
   device-resolved data (decision #4); a shmem source can replace
   `SocketEngineTableSource` behind the same seam later. **Tests
   (`engine_table_handoff_test`, all green):** serialize round-trip (incl. empty
   devmap), truncation rejection, loopback send/receive, and a real-table case
   that hands the actual decode `.pb` bytes + a device map through
   `SocketEngineTableSource` and verifies the worker's table resolves identically
   + the device map survives (SKIPs without `ENABLE_KV_TABLE`/the `.pb`).
   **Consuming the `DeviceMap` in device IO** (FabricNode→chip open) is 4b step 12.

   **DONE (approach B — peer load + exchange, kept as the standalone/test path).**
   `transport/kv_table_provisioning.{hpp,cpp}`: each worker loads **only its own**
   `.pb` (`loadKvTableFile` → parsed `IKvTable` + the raw blob), then the two
   sides **swap tables over the control channel** (`exchangeTableBlob`, role-based
   send/recv ordering so one channel never deadlocks) and the peer blob is
   reconstructed with `KvChunkAddressTableAdapter::fromProtobuf`
   (`provisionPeerTable`). So a prefill host never needs the decode `.pb` on disk —
   it gets the decode table over the wire before building its sender. The two
   orchestrator `exchangeTables` methods now delegate to the shared
   `exchangeTableBlob` (one wire implementation). **Tests
   (`kv_table_provisioning_test`, all green):** blob swap between roles, nullopt
   on closed channel, missing-file + empty-blob guards, and a real-table case
   that loads `prefill_kv_chunk_table.pb` + `decoder_kv_table.pb`, exchanges the
   ~52 MB decode table over a loopback control channel, and verifies the
   wire-obtained table resolves byte-identically to the directly-loaded one
   (SKIPs when `ENABLE_KV_TABLE` is off or the `.pb`s are absent). **5b wiring:**
   the prefill worker calls `loadKvTableFile(prefill.pb)` then
   `provisionPeerTable(channel, Sender, blob)` per decode host before constructing
   `KvMigrationMultiHostSender`; the decode worker loads its `.pb` and the
   `KvMigrationReceiverServer` answers the exchange.

### Phase 5b — Wire the trigger end to end (Kafka → executor → ack)

8. **DONE — unified worker binary.** New `src/mooncake_kv_migration_worker_main.cpp`
   → target `mooncake_kv_migration_worker` (gated `transfer_engine` + `KAFKA_ENABLED`,
   next to bringup). One `--role prefill|decode` process composing the tested
   building blocks:
   - **prefill:** init Mooncake engine → `loadKvTableFile` (prefill + decode) →
     `MultiDeviceUmd` → `KvControlChannelConnector` (static `--peer-control
     NAME=host:port`, `TcpSocketTransport` client factory) → `KvMigrationMultiHostSender`
     → `MooncakeMigrationExecutor` behind `KvMigrationWorker` (the factored Kafka
     consume→executor→ack loop).
   - **decode:** init engine → `loadKvTableFile` → `MultiDeviceUmd` →
     `MooncakeKvReceiver` (mirror registered as the segment) →
     `KvMigrationReceiverServer` (`TcpSocketTransport` server factory) → idle to SIGTERM.

   Notably **drops `MooncakeMigrationWorker` + `PeerDiscoveryService`** from the data
   path: the decode worker advertises its segment name over the control channel
   (`MirrorReady`), so the prefill only needs the engine init'd to `openSegment`.
   Supersedes `bringup_mooncake_worker` (log-only loop) and `tt_kv_migration_consumer`
   (stub) — both left in place for now; retire after HW validation. **Verified
   here:** clang `-fsyntax-only` clean against real headers + CMake configures (the
   target itself builds on a `KAFKA_ENABLED` + `--blaze --mooncake --kv-table` build,
   not in this `KAFKA_ENABLED=OFF` env). Device chip resolution still uses the
   `device & 0xFFFF` placeholder → real `umd_chip_id` via the device map is 4b-12.
9. **DONE (inherited) — migration_id / ack semantics:** `KvMigrationWorker` already
   threads `migration_id` → `MooncakeMigrationExecutor` runs it as the orchestrator
   `uuid` and acks the terminal status; the orchestrator only returns `SUCCESSFUL`
   after the receiver's drain `Ack` (not after the WRITE). Idempotent retry of the
   same request is the higher layer's job (drain is idempotent by design).
10. **Standalone test:** 1 prefill + N receiver processes on one box (each its own `HostDeviceIo` + mirror segment + control channel), `dev-kafka.sh up` + `migration_cli.py produce` → real executor → real loopback migration → `tail acks` shows `SUCCESSFUL`; byte-verify across all N decode hosts. This is the Phase-3 n→m harness over the real discovery + Kafka path. No 16-galaxy needed.

### Phase 4b — Device IO + real-cluster mapping (needs HW)

11. **Coexistence device IO in the worker:** wire Phase-1 `MultiDeviceUmd` (raw `UmdDevice`, no flock) so the worker reads/writes real device DRAM beside a live engine (today it uses `HostDramStorageBackend`).
12. **`FabricNodeId → UMD chip` mapping** for the real cluster (currently `chip = chip_id`).
13. **HW test:** extend the MPI discovery harness (4 prefill + 16 decode, validated single-host for discovery) to drive a real migration through the executor on hardware. Byte-verify 1→1, then 1→16, then 4→16.

### Phase 4c — k8s deployment (the actual "Phase 4")

14. **Replace MPI launch** (the harness uses `mpirun` only as a launcher) with **k8s manifests**: one worker pod per device-host, co-located with the engine (sidecar or daemonset — open decision), Mooncake HTTP metadata service as Deployment+Service, Kafka as managed/Deployment.
15. **Liveness/readiness + TCP heartbeat** (heartbeat loop already in the worker) replacing ULFM.
16. **Metadata-service choice:** keep the Mooncake-wheel HTTP metadata server for the first k8s cut; revisit etcd later.

---

## 4. Decisions to lock before coding

1. **Wire message shape** — extend to multi-layer + asymmetric positions (recommended) vs. single-layer minimal. *Blocks 5a.*
2. **One worker binary with `--role`** (recommended) vs. keep two. *Blocks 5b.*
3. **Control-channel endpoint resolution** — piggyback on metadata service vs. parallel registry. *Blocks 4a.*
4. **Coordinate with dmadic's live branches** — `prefill-mooncake-1` ("Mooncake integration", "Functional standalone prefill") and `offload-mooncake-interface` build the prefill-engine side that **produces the trigger**. The real "which slot to migrate" likely arrives via that offload interface, not just `migration_cli.py`; the executor's `MigrationRequest` must match what they emit.

---

## 5. Standalone testing matrix

| Step | Standalone validation | HW |
|---|---|---|
| 5a | `transport_kv_migration_e2e` host mode driven through `MooncakeMigrationExecutor`; byte-verify | no |
| 4a | connector + receiver wiring unit-tested over loopback sockets (gtest in `transport_lib`) | no |
| 5b | 1 prefill + N receiver procs on one box; `dev-kafka.sh` + `migration_cli.py produce` → executor → loopback → ack `SUCCESSFUL`; byte-verify n→m | no |
| 4b | device DRAM read/write beside live engine; 1→1 then 1→16 then 4→16 | yes |
| 4c | k8s pods + metadata service + Kafka; liveness/heartbeat | yes |
