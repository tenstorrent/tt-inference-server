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

# device mode (real device DRAM via coexistence UMD) — needs tt-metal + HW:
export TT_METAL_HOME="$PWD/tt-llm-engine/tt-metal"
./build.sh --blaze --mooncake --kv-table

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

## 2. Modes, tables, triggers (what the knobs mean)

**`--mode`**
- `host` — device I/O is an in-process host-memory store. **No hardware.** The
  bytes still flow over real Mooncake TCP between processes, so it exercises
  addressing + transport + orchestration + byte-verify, but the numbers are
  loopback memcpy, **not** a real transfer-time.
- `device` — device I/O is the coexistence UMD (`read_dram`/`write_dram` on real
  chip DRAM, no `start_device`, no flock). **Needs hardware** and a `--blaze`
  build. This is the only mode that gives a real transfer-time.

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
each table `FabricNode` to the UMD chip to open. Omit it to use the placeholder
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
   host). Carries Begin/MirrorReady/Done/Ack.
2. **Mooncake segment name** — `--mooncake-name ip:port`. The harness uses
   `P2PHANDSHAKE` (no metadata server), so the sender opens the receiver's
   segment by **connecting to that `ip:port`**. It must be a **routable** address
   and an **open port**, distinct per process.

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

**Terminal 1 — on the DECODE host (galaxy B), start FIRST** (it binds the
control port + advertises its Mooncake segment):
```bash
cd tt-media-server/cpp_server
export TT_METAL_HOME="$PWD/tt-llm-engine/tt-metal"
ROLE=receiver MODE=device \
  TABLE=/path/decoder.pb \
  RECV_NAME="${DECODE_IP}:17777" CONTROL_PORT=18650 \
  bash tests/e2e/scripts/run_transport_kv_migration_e2e.sh
```

**Terminal 2 — on the PREFILL host (galaxy A):**
```bash
cd tt-media-server/cpp_server
export TT_METAL_HOME="$PWD/tt-llm-engine/tt-metal"
ROLE=sender MODE=device \
  TABLE=/path/prefill.pb DECODE_TABLE=/path/decoder.pb \
  SEND_NAME="${PREFILL_IP}:17778" \
  CONTROL_HOST="${DECODE_IP}" CONTROL_PORT=18650 \
  bash tests/e2e/scripts/run_transport_kv_migration_e2e.sh
```
The decode host prints `verification: PASS` (+ `serve_total`); the prefill host
prints `[migration] sender: … migrate_total=…`. Both must say `RESULT: PASS`.
Pick a request range so the chunks land on the one decode host (see §7.7 / §10).

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
- `serve_total` (receiver) = prepareMirror + drain (its share).
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

## 9. Device map (item 1): do you need it?

Each worker only opens its **own host's local chips** (cross-host movement is
Mooncake, not device fabric). The placeholder passes `chip = table_chip_id` to
the UMD. That is correct **iff** each worker host owns chips from **one mesh**
and the UMD local index equals the table's chip id.

**Detector:** run `--mode device --seed-verify`. If `verification: PASS`, you do
**not** need a device map. If it byte-mismatches while host mode passes, supply
one: a file with `mesh chip umd_chip_id` per line, via `DEVICE_MAP=<file>` (or
`--device-map`). Resolving `umd_chip_id` (ASIC unique id) is a
CreateDevice-capable step; get it from the engine/tooling for your cluster.

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
`ROLE` (both|receiver|sender), `MODE`, `TABLE`, `DECODE_TABLE`, `CONTROL_HOST`,
`CONTROL_PORT` (18650), `RECV_NAME`/`SEND_NAME` (127.0.0.1:17777/17778),
`SLOT`, `LAYER_BEGIN`/`LAYER_END`, `POS_BEGIN`/`POS_END`, `TIMEOUT_SEC`,
`SEED_VERIFY` (1), `DEVICE_MAP`.

**`run_transport_kv_migration_e2e_multihost.sh`** (1→N):
`ROLE` (both|receiver|sender), `MODE`, `SEED_VERIFY` (1), `DEVICE_MAP`,
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
