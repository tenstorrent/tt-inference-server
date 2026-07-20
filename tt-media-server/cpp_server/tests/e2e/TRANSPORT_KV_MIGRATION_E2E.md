<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. -->

# Transport KV-migration e2e test — run guide

End-to-end test of the Mooncake KV-cache migration data plane: seed a known
blob at the **source** addresses, migrate it (real table addressing + Mooncake
one-sided transfer + drain), and **byte-verify** it at the **destination**, with
per-side timing. Runs with no hardware (host mode), on 1 galaxy, or across
N galaxies, and can be triggered directly (CLI) or through the full unified
worker path (Kafka).

- Binary: `build/transport_kv_migration_e2e`
- Scripts: `tests/e2e/scripts/run_transport_kv_migration_e2e.sh` (1→1),
  `tests/e2e/scripts/run_transport_kv_migration_e2e_multihost.sh` (1→N)
- In-process gtest: `TransportKvMigrationE2E`
- Production worker (`mooncake_kv_migration_worker`) + byte-checker
  (`kv_seed_verify`): **§13** — the deployable Kafka→migration path, distinct
  from this harness.

---

## 0. TL;DR — the fast paths

| Goal | One command |
|---|---|
| Smoke (no HW) | `ctest --test-dir build -R TransportKvMigrationE2E -V` |
| 1→1, host, no HW | `bash tests/e2e/scripts/run_transport_kv_migration_e2e.sh` |
| 1→2, host, no HW | `bash tests/e2e/scripts/run_transport_kv_migration_e2e_multihost.sh` |
| 1→2, host, via Kafka | `TRIGGER=kafka bash tests/e2e/scripts/run_transport_kv_migration_e2e_multihost.sh` (broker up) |
| 1→1, 1 galaxy (HW) | `MODE=device bash tests/e2e/scripts/run_transport_kv_migration_e2e.sh` |
| 2 galaxies, 1→1 (HW) | `ROLE`-split single-host script — see §7.5 |
| N galaxies, 1→N (HW) | `ROLE=receiver` per decode host + `ROLE=sender` — see §7.6 |
| 1→1 or 1→N over **RDMA** (HW) | add `PROTOCOL=rdma` to the §7.5 / §7.6 split — see §7.8 |
| Production worker, 2 galaxies, Kafka | `mooncake_kv_migration_worker` + `kv_seed_verify` — see §13 |

The **Mooncake wire transport defaults to TCP**; `PROTOCOL=rdma` (harness) /
`--protocol rdma` (worker) switches the per-window data WRITEs to one-sided RDMA.
RDMA needs a real RNIC and is a cross-host path (§7.8) — the no-HW host-mode
rows above stay TCP.

Every run ends with `RESULT: PASS` (or `FAIL`) and prints `[migration] …`
timing lines. `verification: PASS` means the bytes arrived correct.

---

## 1. Building

Pick the build for the scenario. From `tt-media-server/cpp_server/`:

```bash
# host mode only (no hardware, builtin tables):
./build.sh --mooncake

# real .pb tables (adds the real KvChunkAddressTable adapter) — needs tt-metal:
export TT_METAL_HOME="$PWD/tt-llm-engine/tt-metal"     # absolute path
./build.sh --mooncake --kv-table

# device mode (real device DRAM via DRISC NOC-DMA) — needs tt-metal + HW:
export TT_METAL_HOME="$PWD/tt-llm-engine/tt-metal"
./build.sh --blaze --mooncake --kv-table
# device mode also needs the DRISC service-kernel ELF at run time:
export MIGRATION_DRISC_SERVICE_ELF=/path/to/drisc_service.elf

# add the Kafka trigger (only needed for --trigger kafka):
./build.sh --blaze --mooncake --kv-table --kafka

# if CMake complains about a generator mismatch (Makefiles vs Ninja), add --fresh once:
./build.sh --mooncake --fresh
```

Notes:
- `--kv-table` requires `TT_METAL_HOME` set to a **built** tt-metal (has
  `build_Release/`), else configure fails with a `TT_METAL_API_FOUND` error.
- Produces `build/transport_kv_migration_e2e`. If it's missing, the build didn't
  include `--mooncake`.

---

## 2. Modes, tables, transport, triggers (what the knobs mean)

**`--mode`**
- `host` — device I/O is an in-process host-memory store. **No hardware.** The
  bytes still flow over real Mooncake TCP between processes, so it exercises
  addressing + transport + orchestration + byte-verify, but the numbers are
  loopback memcpy, **not** a real transfer-time.
- `device` — device I/O is DRISC NOC-DMA over the coexistence UMD path (opens
  the chip via the raw UMD Cluster, no `start_device`, no flock; the transfer
  itself is the on-device DRISC service kernel). **Needs hardware**, a `--blaze`
  build, and `MIGRATION_DRISC_SERVICE_ELF` set. This is the only mode that gives
  a real transfer-time.

**`--table`**
- `builtin` — tiny reduced single-host table (2 layers, 64 B chunks, 512 B
  total). For the 1→1 script / gtest. Auto seed+verify.
- `builtin2` — 2-host split builtin (layer 0 → `decode-0`, layer 1 → `decode-1`),
  still 64 B chunks. For the 1→N host-mode script. **Host mode only** — in device
  mode both decode procs would target the same physical chips and clobber each
  other; use real tables for device multi-host.
- `<prefill.pb>` (+ `--decode-table <decode.pb>`) — the real
  `KvChunkAddressTable`s. Real addresses, real 19584 B chunks. Needs
  `--kv-table`.

**`--transport`** — the KV data plane. `bounce` is the only path: a small
double-pinned bounce buffer on the decode side, filled window-by-window by the
sender and drained to device (the "RDMA-over-host" architecture, so named
whether the wire is TCP or RDMA). There is no full-table mirror; each window's
`WindowReady` carries the section descriptors telling the receiver where to
drain. Scripts default `TRANSPORT=bounce`.

**`--protocol`** — the Mooncake **wire** underneath the bounce data plane.
- `tcp` (**default**) — per-window WRITEs go over Mooncake TCP. Works with no
  RNIC, including single-box loopback; this is what every host-mode path uses.
- `rdma` — per-window WRITEs are one-sided RDMA. The double-pinned staging /
  bounce buffers become the `ibv_reg_mr`'d MRs the NIC targets (still NOC-mapped
  for device DMA). **Needs a real RNIC + libibverbs**, and is effectively a
  **cross-host** path (loopback RDMA generally won't work). RDMA is always
  compiled into a `--mooncake` build — there is no separate build flag. See §7.8.
  On a host with no RNIC it fails fast with `RdmaTransport: No available RNIC →
  installTransport(rdma) failed`. The bounce data plane code is identical either
  way — only the installed transport differs. Scripts: `PROTOCOL=tcp|rdma`.

**`--trigger`** (sender)
- `cli` (default) — the sender calls `migrate()` directly. `migrate_total` =
  pure data-plane time. Use this to measure transfer time.
- `kafka` — the sender drives the migration through the unified worker path
  (`KvMigrationWorker` + `MooncakeMigrationExecutor`), self-produces the request,
  and waits for the `SUCCESSFUL` ack. `migrate_total` includes the Kafka
  round-trip. Needs a `--kafka` build + a running broker. Requires
  `--peer-control` (uses the multi-host path).

**`--seed-verify`** — seed a deterministic dummy blob at the source and
byte-verify the destination. Auto for `builtin`/`builtin2`; the scripts pass it
by default (`SEED_VERIFY=1`). Set `SEED_VERIFY=0` for a real table against a
**live model** (don't overwrite real KV).

**`--device-map FILE`** — device mode only. Lines `mesh chip umd_chip_id`. Maps
each table `FabricNode` to the chip to open, by 64-bit ASIC unique_id (resolved
to a local device index at open). Omit it to use the placeholder
`chip = table_chip_id`; if the placeholder is wrong on your cluster the
byte-verify FAILS — that's your signal to supply the map. See §9.

---

## 3. Prerequisites checklist

| Scenario | Needs |
|---|---|
| host, builtin | `--mooncake` build |
| host, real tables | `--mooncake --kv-table` build + `TT_METAL_HOME` + the `.pb` files |
| device (any) | `--blaze --mooncake --kv-table` build + `TT_METAL_HOME` + a HW reservation |
| Kafka trigger | above + `--kafka` build + a broker with topics (see §6) |
| multi-galaxy | routable IPs between hosts + open control & Mooncake ports (see §5) |

---

## 4. Quick smoke — the in-process gtest (no HW)

One terminal:
```bash
cd tt-media-server/cpp_server
ctest --test-dir build -R TransportKvMigrationE2E -V
# or directly:
./build/transport_kv_migration_e2e
```
Expect:
```
[migration] sender: chunks=8 bytes=512 | migrate_total=… ms (… GB/s)
[  PASSED  ] 2 tests.
```
This forks a sender + runs the receiver in-process, host mode, builtin table,
and asserts byte content. Fastest confidence check.

---

## 5. Concepts you must get right for multi-process / multi-galaxy

Each worker process needs two endpoints, and they must be reachable from the
sender:

1. **Control channel** — a TCP port the *receiver* listens on and the *sender*
   connects to (`--control-port` on the receiver; `--control-host` +
   `--control-port` on the sender, or `--peer-control HOST=ip:port` per decode
   host). Carries Begin/BounceReady/[WindowReady/WindowAck]*/Done/Ack.
2. **Mooncake segment name** — `--mooncake-name ip:port`. The harness defaults
   to `P2PHANDSHAKE` (`METADATA=P2PHANDSHAKE`), so the sender opens the
   receiver's segment by **connecting to that `ip:port`** — it must be a
   **routable** address and an **open port**, distinct per process. To resolve
   segments through a metadata service instead, pass `METADATA=<uri>` (e.g.
   `http://HOST:PORT/metadata`) to every process; the receiver registers its
   segment there and the sender resolves it by name. This only changes
   data-plane segment discovery — the control channel is still the direct
   `--control-host`/`--peer-control` dial above.

So per decode host you need: a control port and a Mooncake port, both reachable
from the prefill host. On loopback (one box) use `127.0.0.1` and distinct ports.
Across galaxies use the host's real IP.

**Finding a host's IP** (run on each host):
```bash
hostname -I | awk '{print $1}'      # first IPv4
# or pick the interface on the inter-host fabric:
ip -4 addr show | grep -w inet
```
Use the IP that is **routable between the galaxies** (same cluster
fabric/subnet). Confirm reachability + ports before running:
```bash
# from the prefill host, after the receiver is up on DECODE_IP:
nc -vz "$DECODE_IP" 18650      # control port
nc -vz "$DECODE_IP" 17777      # Mooncake port
```

**The request must match on all processes** — same `--slot --layer-begin
--layer-end --pos-begin --pos-end` (the scripts pass one set to every process).

---

## 6. Kafka broker setup (only for `--trigger kafka`)

One terminal (leave running):
```bash
cd tt-media-server/cpp_server
scripts/dev-kafka.sh up
python scripts/migration_cli.py setup     # creates kv-migration-{requests,acks}
scripts/dev-kafka.sh status                # confirm broker + topics
```
Default broker `localhost:9092` (override with `KAFKA_BROKERS`). Tear down with
`scripts/dev-kafka.sh down`.

---

## 7. Scenario walk-throughs

### 7.1 One box, host mode, 1→1 (no HW)

Terminal 1:
```bash
cd tt-media-server/cpp_server
bash tests/e2e/scripts/run_transport_kv_migration_e2e.sh
```
It launches the receiver (background) + sender (foreground) on `127.0.0.1`.
Expect:
```
[migration] sender: chunks=8 bytes=512 | migrate_total=… ms (… GB/s)
[migration] receiver[decode]: chunks=8 bytes=1024 | serve_total=… ms (… GB/s)
[receiver] verification: PASS
RESULT: PASS
```

### 7.2 One box, host mode, 1→2 (no HW)

Terminal 1:
```bash
bash tests/e2e/scripts/run_transport_kv_migration_e2e_multihost.sh
```
Launches `decode-0` + `decode-1` receivers + 1 sender (fan-out to both). Expect
two `receiver[decode-N]: … verification: PASS` lines and `RESULT: PASS`.

### 7.3 One box, host mode, 1→2, via Kafka (full unified-worker e2e)

Terminal 1 — broker (see §6, leave running).
Terminal 2:
```bash
TRIGGER=kafka bash tests/e2e/scripts/run_transport_kv_migration_e2e_multihost.sh
```
Same as 7.2, but the sender produces the request to Kafka and a
`KvMigrationWorker`/`MooncakeMigrationExecutor` consumes it and drives the
migration. Extra line: `[sender] produced migration request …` and
`[sender] ack … status=2` (2 = SUCCESSFUL). Needs a `--kafka` build.

### 7.4 One box = one galaxy, device mode, 1→1 (HW)

Reserve the box / galaxy first. Terminal 1:
```bash
export TT_METAL_HOME="$PWD/tt-llm-engine/tt-metal"
MODE=device bash tests/e2e/scripts/run_transport_kv_migration_e2e.sh
```
Uses real device DRAM (builtin addresses) on loopback. `verification: PASS`
means the bytes made a real DRAM→Mooncake→DRAM round trip. Timing is a real
single-host transfer number (loopback Mooncake, real DRAM).

For real-table device single-host (real addresses, real chunk sizes):
```bash
MODE=device TABLE=/path/prefill.pb DECODE_TABLE=/path/decoder.pb \
  bash tests/e2e/scripts/run_transport_kv_migration_e2e.sh
```
(Needs `--kv-table` in the build. `SEED_VERIFY=1` is on by default, so it writes
a dummy blob to those addresses and verifies — do **not** run this against a
live model on the same chips; set `SEED_VERIFY=0` if you must.)

### 7.5 Two galaxies, 1→1 (HW) — split launch

Prefill on galaxy A (`PREFILL_IP`), decode on galaxy B (`DECODE_IP`). Find the
IPs with `hostname -I` on each host (see §5).

For real tables you must pass the **host tags** that match the tables'
`fabric_node_host` (`PREFILL_HOST`, `DECODE_HOST`) — else the request resolves to
no chunks. Get them from whoever produced the `.pb` (or dump with the adapter
test). Pick a `LAYER_*` range whose chunks live on that one decode host.

**Terminal 1 — on the DECODE host (galaxy B), start FIRST** (binds the control
port + advertises its Mooncake segment):
```bash
cd tt-media-server/cpp_server
export TT_METAL_HOME="$PWD/tt-llm-engine/tt-metal"
ROLE=receiver MODE=device \
  TABLE=/path/decoder.pb DECODE_HOST=<decode-tag> \
  RECV_NAME="${DECODE_IP}:17777" CONTROL_PORT=18650 \
  LAYER_BEGIN=0 LAYER_END=1 POS_BEGIN=0 POS_END=3424 \
  bash tests/e2e/scripts/run_transport_kv_migration_e2e.sh
```

**Terminal 2 — on the PREFILL host (galaxy A):**
```bash
cd tt-media-server/cpp_server
export TT_METAL_HOME="$PWD/tt-llm-engine/tt-metal"
ROLE=sender MODE=device \
  TABLE=/path/prefill.pb DECODE_TABLE=/path/decoder.pb \
  PREFILL_HOST=<prefill-tag> DECODE_HOST=<decode-tag> \
  SEND_NAME="${PREFILL_IP}:17778" \
  CONTROL_HOST="${DECODE_IP}" CONTROL_PORT=18650 \
  LAYER_BEGIN=0 LAYER_END=1 POS_BEGIN=0 POS_END=3424 \
  bash tests/e2e/scripts/run_transport_kv_migration_e2e.sh
```
No Kafka involved — the single-host script uses the direct (CLI) trigger. The
decode host prints `verification: PASS` (+ `serve_total`); the prefill host
prints `[migration] sender: … migrate_total=…`. Both must say `RESULT: PASS`.
The `LAYER_*`/`POS_*`/`SLOT` values MUST be identical on both commands.

### 7.6 N galaxies, 1→N (HW) — turnkey `ROLE` split

The multi-host script has a `ROLE` mode so each galaxy runs one clean command
(no ssh, no raw binary flags). Its `both` default is one-box; `receiver` /
`sender` are for physically distinct hosts.

Prep (see §5): get each host's routable IP (`hostname -I`), pick each decode
host's **tag** from the real decode table (its `fabric_node_host`), and make sure
the control port + Mooncake port are open between hosts.

**On EACH decode host `k`** (IP `DECODE_IP_k`, tag `TAG_k`) — start these
FIRST:
```bash
cd tt-media-server/cpp_server
export TT_METAL_HOME="$PWD/tt-llm-engine/tt-metal"
ROLE=receiver MODE=device \
  DECODE_HOST="TAG_k" CONTROL_PORT=18650 MOONCAKE_NAME="${DECODE_IP_k}:17777" \
  TABLE=/path/decoder.pb \
  LAYER_BEGIN=0 LAYER_END=61 POS_BEGIN=0 POS_END=3424 \
  [DEVICE_MAP=/path/devmap_k.txt] \
  bash tests/e2e/scripts/run_transport_kv_migration_e2e_multihost.sh
```

**On the prefill host** (IP `PREFILL_IP`) — one `tag=ip:cport` per decode host in
`PEERS`:
```bash
cd tt-media-server/cpp_server
export TT_METAL_HOME="$PWD/tt-llm-engine/tt-metal"
ROLE=sender MODE=device \
  SEND_NAME="${PREFILL_IP}:17778" \
  TABLE=/path/prefill.pb DECODE_TABLE=/path/decoder.pb \
  PEERS="TAG_0=${DECODE_IP_0}:18650 TAG_1=${DECODE_IP_1}:18650 ..." \
  LAYER_BEGIN=0 LAYER_END=61 POS_BEGIN=0 POS_END=3424 \
  [TRIGGER=kafka KAFKA_BROKERS=<BROKERS>] \
  bash tests/e2e/scripts/run_transport_kv_migration_e2e_multihost.sh
```
Each decode host prints its `verification: PASS` + `serve_total`; the prefill
prints `[migration] sender: … migrate_total=…` and the overall `RESULT`. The
request (`SLOT`/`LAYER_*`/`POS_*`) must be identical on every command. Add
`TRIGGER=kafka` (+ a reachable broker, §6) for the full unified-worker path.

> The tags and which hosts a request touches come from the decode table
> (`hostsForRequest`). If you don't know the tags, dump the table with the
> `kv_chunk_address_table_adapter_test` or ask the engine/tooling that produced
> the `.pb`.

### 7.7 Real-table, targeted volume (any of the above)

Set the request range to select how much data moves (see §10). Example ~2 MiB on
one layer:
```
--layer-begin 0 --layer-end 1 --pos-begin 0 --pos-end 3424
```

### 7.8 Running over RDMA (`PROTOCOL=rdma`)

The bounce data plane is transport-agnostic; `rdma` only changes the Mooncake
wire (one-sided RDMA instead of TCP for the per-window WRITEs). It is an
**overlay on the §7.5 / §7.6 cross-host runs** — same commands, plus
`PROTOCOL=rdma`.

**Prerequisites**
- **Build:** any `--mooncake` build (RDMA is always compiled in — no extra flag).
  `libibverbs` must be present at run time (it enumerates the NIC).
- **NIC:** a real RDMA NIC (RNIC, e.g. `mlx5_0`) visible to libibverbs on each
  host. No RNIC ⇒ `RdmaTransport: No available RNIC` and the engine init fails.
- **Fabric:** RDMA reachability **between the two hosts** (same RoCE/IB fabric).
  RDMA is a cross-host path — single-box loopback generally won't work, so keep
  the no-HW / one-box smoke runs on TCP.
- The Mooncake handshake/metadata bootstrap still goes over TCP even under RDMA,
  so `MC_TCP_BIND_ADDRESS` / the routable `--mooncake-name ip:port` still apply
  (§5); only the data WRITEs move to RDMA.

**Harness (2 galaxies, 1→1)** — the §7.5 commands with `PROTOCOL=rdma` on both
sides. The harness auto-discovers the single present NIC (it has no
NIC-selection flag):
```bash
# decode host, FIRST:
ROLE=receiver MODE=device PROTOCOL=rdma \
  TABLE=/path/decoder.pb DECODE_HOST=<decode-tag> \
  RECV_NAME="${DECODE_IP}:17777" CONTROL_PORT=18650 \
  LAYER_BEGIN=0 LAYER_END=1 POS_BEGIN=0 POS_END=3424 \
  bash tests/e2e/scripts/run_transport_kv_migration_e2e.sh
# prefill host:
ROLE=sender MODE=device PROTOCOL=rdma \
  TABLE=/path/prefill.pb DECODE_TABLE=/path/decoder.pb \
  PREFILL_HOST=<prefill-tag> DECODE_HOST=<decode-tag> \
  SEND_NAME="${PREFILL_IP}:17778" \
  CONTROL_HOST="${DECODE_IP}" CONTROL_PORT=18650 \
  LAYER_BEGIN=0 LAYER_END=1 POS_BEGIN=0 POS_END=3424 \
  bash tests/e2e/scripts/run_transport_kv_migration_e2e.sh
```
**Harness (N galaxies, 1→N)** — the §7.6 `ROLE=receiver`/`ROLE=sender` commands
with `PROTOCOL=rdma` added to every command.

**Production worker** — pass `--protocol rdma` (or `MIGRATION_MOONCAKE_PROTOCOL=rdma`)
to every worker (§13.6). On a multi-NIC host, pin the NIC(s) with repeatable
`--rdma-nic mlx5_0` (or `MIGRATION_RDMA_NICS=mlx5_0,mlx5_1`); the default (no
filter) auto-discovers the single present NIC. The NIC filter is a **worker-only**
knob — the e2e harness always auto-discovers.

---

## 8. Reading the timing / performance

Each run prints, per side:
```
[migration] sender: chunks=<N> bytes=<B> | migrate_total=<T> ms (<G> GB/s)
[migration] receiver[<tag>]: chunks=<N> bytes=<B> | serve_total=<T> ms (<G> GB/s)
```
- `bytes` = logical KV volume (one replica), derived from the plan.
- `migrate_total` (sender) = the whole migration:
  - `--trigger cli`: read + Mooncake write + drain + control round-trips → the
    **data-plane transfer time**. Use this for "how long to move X".
  - `--trigger kafka`: also includes Kafka produce/consume/ack → the **deployed
    worker latency**.
- `serve_total` (receiver) = bounce-buffer register + windowed drain (its share).
- `GB/s` = `bytes / time`. Real tables use **2-replica** device groups, so the
  wire actually carries ~2× `bytes` (each chunk written to both replicas);
  interpret throughput accordingly.

To isolate Kafka overhead: run the same size with `--trigger cli` and
`--trigger kafka`; the difference ≈ Kafka + trigger cost.

Only **`--mode device` on real hardware** gives a representative transfer-time;
host mode is loopback memcpy. The data plane is currently **serial per chunk**
(one staging buffer, synchronous writes, hosts driven serially), so the number
reflects that (not a pipelined ideal).

---

## 9. Device map: do you need it?

Each worker only opens its **own host's local chips** (cross-host movement is
Mooncake, not device fabric). The placeholder opens `chip = table_chip_id`
directly as a device index. That is correct **iff** each worker host owns chips
from **one mesh** and the local device index equals the table's chip id.

**Detector:** run `--mode device --seed-verify`. If `verification: PASS`, you do
**not** need a device map. If it byte-mismatches while host mode passes, supply
one: a file with `mesh chip umd_chip_id` per line, via `DEVICE_MAP=<file>` (or
`--device-map`). Resolving `umd_chip_id` (ASIC unique id) is a
CreateDevice-capable step; get it from the engine/tooling for your cluster. At
open, the worker enumerates the visible chips once and maps each `umd_chip_id`
to its local device index; a value that is a plain local index (not the ASIC
unique id) or names a chip not visible on the host fails to open rather than
silently addressing the wrong chip.

**Discovery is NOT required** — the static `--peer-control` list is the control
endpoint interface; discovery is only its production replacement.

---

## 10. Sizing the blob (hit a target MB)

```
migrated_bytes = num_layers × (pos_end − pos_begin)/chunk_n_tokens × chunk_size_bytes
```
Real `decoder_kv_table.pb`: `chunk_size_bytes = 19584`, `chunk_n_tokens = 32`.
With `pos_begin = 0`, **one layer**: `bytes = pos_end × 612`.

| Target (1 layer) | `--pos-end` | chunks | bytes |
|---|---|---|---|
| ~2 MiB | `3424` | 107 | 2,095,488 |
| ~2 MB (dec) | `3264` | 102 | 1,997,568 |
| ~5 MiB | `8576` | 268 | 5,248,512 |

For 2 MiB **across 2 hosts**, use 2 layers ~1 MiB each: `--layer-begin 0
--layer-end 2 --pos-end 1728`. Easiest: just bump `--pos-end` and read the
printed `bytes=`. (Verify your table's `chunk_size_bytes`/`chunk_n_tokens` — the
`kv_chunk_address_table_adapter_test` prints them.)

---

## 11. Script env-var reference

**`run_transport_kv_migration_e2e.sh`** (1→1):
`ROLE` (both|receiver|sender), `MODE`, `PROTOCOL` (tcp|rdma, default tcp),
`TRANSPORT` (bounce, the only path), `METADATA` (default P2PHANDSHAKE; a service
URI resolves segments via that service), `TABLE`, `DECODE_TABLE`, `CONTROL_HOST`,
`CONTROL_PORT` (18650), `RECV_NAME`/`SEND_NAME` (127.0.0.1:17777/17778),
`SLOT`, `LAYER_BEGIN`/`LAYER_END`, `POS_BEGIN`/`POS_END`, `TIMEOUT_SEC`,
`SEED_VERIFY` (1), `DEVICE_MAP`.

**`run_transport_kv_migration_e2e_multihost.sh`** (1→N):
`ROLE` (both|receiver|sender), `MODE`, `PROTOCOL` (tcp|rdma, default tcp),
`TRANSPORT` (bounce, the only path), `METADATA` (default P2PHANDSHAKE; a service
URI resolves segments via that service), `SEED_VERIFY` (1), `DEVICE_MAP`,
`SLOT`/`LAYER_*`/`POS_*`, `TIMEOUT_SEC`, `TRIGGER` (cli|kafka),
`KAFKA_BROKERS` (localhost:9092). Then per role:
- **both** (one box): `TABLE` (builtin2|prefill.pb), `DECODE_TABLE`,
  `PEERS="tag:cport:mport …"`, `HOST_IP` (127.0.0.1), `SEND_NAME`
  (127.0.0.1:17780).
- **receiver** (each decode host): `DECODE_HOST` (tag), `CONTROL_PORT` (18650),
  `MOONCAKE_NAME` (thisIP:port), `TABLE` (decode.pb|builtin2).
- **sender** (prefill host): `SEND_NAME` (thisIP:port), `TABLE`
  (prefill.pb|builtin2), `DECODE_TABLE`, `PEERS="tag=ip:cport …"`.

Note `PEERS` differs by role: `both` uses `tag:cport:mport` (shared `HOST_IP`);
`sender` uses `tag=ip:cport` (routable per-host endpoints).

---

## 12. Troubleshooting

- **`… not found/executable`** → build with `--mooncake` (+ `--kv-table` /
  `--blaze` / `--kafka` as needed).
- **`TT_METAL_API_FOUND is OFF`** at configure → `export TT_METAL_HOME=<built
  tt-metal>` before `--kv-table`/`--blaze`.
- **CMake generator mismatch** → `./build.sh … --fresh` once.
- **`real-table mode needs … -DENABLE_KV_TABLE=ON`** → build lacked `--kv-table`.
- **`no prefill/decode chunks for request on host`** → the request range doesn't
  land on that host's tag; fix `--decode-host`/`--prefill-host` or the layer/pos
  range against the table.
- **`--trigger kafka needs a KAFKA_ENABLED build`** → add `--kafka`.
- **No ack / hangs in kafka mode** → broker not up / topics missing (§6), or the
  ack consumer missed the ack (offset reset) — check `migration_cli.py tail
  acks`.
- **`control connect failed` / hang across galaxies** → wrong IP or blocked
  port; verify with `nc -vz` (§5). Start the receiver **before** the sender.
- **`verification: FAIL` in device mode but host mode passes** → chip mapping is
  wrong; supply `DEVICE_MAP` (§9).
- **`Local segment descriptor not found`** (Mooncake) → benign P2PHANDSHAKE
  bootstrap log, not a failure; check the final `RESULT:` line.
- **`RdmaTransport: No available RNIC` / `installTransport(rdma) failed`** (only
  under `PROTOCOL=rdma` / `--protocol rdma`) → no RDMA NIC visible to libibverbs
  on this host. Use a host with an RNIC (and a cross-host run — loopback RDMA
  won't work), or drop back to TCP (`PROTOCOL=tcp`, the default). On a multi-NIC
  host where discovery picks the wrong NIC, pin it with the worker's
  `--rdma-nic` / `MIGRATION_RDMA_NICS` (§13.6). See §7.8.

---

## 13. Production worker (`mooncake_kv_migration_worker`) + `kv_seed_verify`

Everything above drives the **e2e harness** (`transport_kv_migration_e2e`), which
self-contains the sender, receiver, and byte-verify in one process. This section
is the **production-shaped path**: two long-lived `mooncake_kv_migration_worker`
processes (one `--role prefill`, one `--role decode`) triggered by **real
external Kafka messages** (`migration_cli.py produce`), and byte-verified
out-of-band with the standalone **`kv_seed_verify`** helper.

Difference from §7.3 / §7.6: there the harness *self-produces* the request and
verifies internally. Here the worker is the actual deployable binary — it moves
whatever KV the table addresses and does **no** seed/verify of its own, so
correctness is checked by bracketing the migration with `kv_seed_verify`.

### 13.1 Build (both galaxies)

```bash
./install_dependencies.sh --kafka          # librdkafka for KAFKA_ENABLED (both boxes)
export TT_METAL_HOME="$PWD/tt-llm-engine/tt-metal"
./build.sh --blaze --mooncake --kv-table --kafka
```
Produces `build/mooncake_kv_migration_worker` and `build/kv_seed_verify`.

### 13.2 What `kv_seed_verify` does (and why)

The worker moves real KV and never writes/reads a test pattern. To byte-check it
without a live model, bracket the migration:
- `--mode seed` on **prefill**, *before* the trigger — writes a deterministic
  pattern into the source DRAM at the prefill table's addresses.
- `--mode verify` on **decode**, *after* the ack — reads the dest DRAM at the
  decode table's addresses and compares to the same pattern.

It reuses the real tables, `buildHostPlan`, the device map, and the harness's
pattern, so a `VERIFY: PASS` is the same byte guarantee the harness gives.
Symmetric request only (src slot/positions == dst).

### 13.3 Kafka broker — cross-galaxy (host networking)

`scripts/dev-kafka.sh` (§6) targets a single `tt_net` box: it advertises
`kafka:9092`, which a `--network host` container **cannot resolve**. For two
galaxies, run the broker on the **prefill host** advertising `localhost:9092`
(the prefill worker and CLI share the host's localhost under `--network host`):

```bash
# on the prefill HOST shell (docker runs here, not inside the tt container):
sed 's#advertised.listeners=.*#advertised.listeners=PLAINTEXT://localhost:9092#' \
    scripts/kafka-server.properties > ~/kafka-hostnet.properties
docker rm -f kafka 2>/dev/null || true
docker run -d --name kafka --network host \
  -v "$HOME/kafka-hostnet.properties:/mnt/shared/config/server.properties:ro" \
  apache/kafka:4.0.0
sleep 8 && docker logs kafka 2>&1 | tail -5     # expect broker started, NOT "Is a directory"
```
The config file **must exist before `docker run`** — a missing `-v` source makes
Docker create an empty directory and mount it (→ `server.properties: Is a
directory`). Then create topics from inside the prefill container:
```bash
export KAFKA_BROKERS=localhost:9092
python3 scripts/migration_cli.py status && python3 scripts/migration_cli.py setup
```

### 13.4 Two-galaxy run

Prefill `PREFILL_IP`, decode `DECODE_IP` (routable LAN IPs, both containers
`--network host` — see §5). Every device binary needs `TT_METAL_HOME` +
`TT_METAL_RUNTIME_ROOT` + `MIGRATION_DRISC_SERVICE_ELF` (the DRISC service
kernel); the CLI and prefill worker need `KAFKA_BROKERS`. Keep these in a
`~/env.sh` and `source` it in each shell.

**Decode worker (`DECODE_IP`) — start first:**
```bash
./build/mooncake_kv_migration_worker --role decode \
  --metadata P2PHANDSHAKE --name ${DECODE_IP}:17777 --host <decode-tag> \
  --table /path/decoder.pb --control-port 18650 \
  --device-map /path/decode.devmap
# wait for: [worker] decode ... READY: segment=... control_port=18650
```
**Prefill worker (`PREFILL_IP`):**
```bash
KAFKA_BROKERS=localhost:9092 ./build/mooncake_kv_migration_worker --role prefill \
  --metadata P2PHANDSHAKE --name ${PREFILL_IP}:17778 --host <prefill-tag> \
  --prefill-table /path/prefill.pb --decode-table /path/decoder.pb \
  --peer-control <decode-tag>=${DECODE_IP}:18650
# wait for: [worker] prefill ... READY: 1 decode peers ...
```
**Seed → trigger → verify** (same `slot`/`layer`/`pos` in all three; the layer
range must live on the decode host — see §5/§9, KV is sharded by layer across
hosts):
```bash
# prefill: seed BEFORE producing (single-mesh prefill needs no device map)
./build/kv_seed_verify --mode seed --table /path/prefill.pb --host <prefill-tag> \
  --slot 5 --layer-begin 32 --layer-end 36 --pos-begin 0 --pos-end 61408   # -> SEED: OK

# prefill: trigger (use a fresh --migration-id each run)
python3 scripts/migration_cli.py produce --migration-id 702 \
  --src-slot 5 --dst-slot 5 --layer-begin 32 --layer-end 36 \
  --src-pos-begin 0 --src-pos-end 61408 --dst-pos-begin 0 --dst-pos-end 61408
#   decode worker: bounce buffer ready -> windowed drain -> OK
#   prefill worker: migrate(...) complete + SUCCESSFUL ack

# decode: verify AFTER the ack
./build/kv_seed_verify --mode verify --table /path/decoder.pb --host <decode-tag> \
  --device-map /path/decode.devmap \
  --slot 5 --layer-begin 32 --layer-end 36 --pos-begin 0 --pos-end 61408   # -> VERIFY: PASS
```

### 13.5 Device map (decode worker + verify)

The worker's `--device-map` uses the same `mesh chip umd_chip_id` file as §9. It
is **required on the decode side when that host's table spans multiple meshes**:
the placeholder `chip = chip_id` aliases meshes onto the same physical chips, so
a multi-layer migration silently overwrites itself. The worker's channel-range
self-check catches a table/device-map mismatch only when a KV location's NoC
channel exceeds the opened chip's DRAM-channel count; it does **not** catch mesh
aliasing (only `kv_seed_verify` would). Prefill is typically single-mesh → no
map. Pass the **same** map to the decode worker and to `kv_seed_verify --mode
verify`. Build it by listing the host's `(mesh, chip)` nodes from the table and
giving each its chip's 64-bit ASIC `umd_chip_id` (the same value as §9 /
`print_local_device_map`), which the worker resolves to the local device index
at open — do **not** hand-assign a small local index.

### 13.6 Worker CLI reference

`mooncake_kv_migration_worker --role prefill|decode --metadata URI --name NAME
--host TAG [--device-map FILE] [--protocol tcp|rdma] [--rdma-nic NAME]…`
- **prefill:** `--prefill-table P.pb --decode-table D.pb --peer-control
  NAME=host:port` (repeatable). Kafka via env: `KAFKA_BROKERS`,
  `KAFKA_MIGRATION_REQUEST_TOPIC`, `KAFKA_MIGRATION_ACK_TOPIC`, `KAFKA_GROUP_ID`.
- **decode:** `--table D.pb --control-port N [--segment NAME]`. Leave `--segment`
  unset under P2PHANDSHAKE — it defaults to the engine's live server name (the
  auto-assigned RPC endpoint the sender must open).
- **both roles (transport):** `--protocol tcp|rdma` (default `tcp`; env fallback
  `MIGRATION_MOONCAKE_PROTOCOL`, flag wins). Under `rdma`, pin the NIC(s) on a
  multi-NIC host with repeatable `--rdma-nic mlx5_0` (env fallback
  `MIGRATION_RDMA_NICS=mlx5_0,mlx5_1`); default (no filter) auto-discovers the
  single present NIC. Pass the **same** `--protocol` to every worker. See §7.8.

### 13.7 Worker gotchas

- **Start the decode worker before the prefill worker** — the prefill opens its
  control channel to the decode at startup.
- `TT_METAL_HOME` on every device binary, or UMD can't find
  `blackhole_140_arch.yaml` (it falls back to `cwd/tt_metal` and all device
  opens fail).
- `librdkafka-dev` on **both** boxes: the one binary links Kafka even for
  `--role decode`.
- `migration_cli.py --brokers` is a **global** flag (before the subcommand);
  simplest is `export KAFKA_BROKERS=localhost:9092`.
- Throughput is latency-bound (serial per-chunk one-sided writes). The decode
  bounce buffer is a small fixed-size registration (tens of MiB, independent of
  the host KV size) done once at startup, separate from per-migration transfer
  time.
- Decode replicas are 2-way, so the wire carries ~2× the logical bytes (§8).
