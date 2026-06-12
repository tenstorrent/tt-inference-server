# Mooncake PoCs

Exploration of [Mooncake](https://github.com/kvcache-ai/Mooncake) — its
KV-cache Store and Transfer Engine — for Tenstorrent KV-cache migration and
tiering. Each subfolder is an independent proof-of-concept with its own README;
together they move from "does Mooncake work at all" toward "a custom TT backend
that moves device-DRAM KV between galaxies."

| PoC | Language | What it proves |
|-----|----------|----------------|
| [poc1](poc1) | Python | Raw `MooncakeDistributedStore` `put`/`get` smoke test — the baseline. |
| [poc2](poc2) | Python | The control-plane orchestration loop (scheduler looks local→remote, pulls via a **mocked** migration worker). Targets #4017. |
| [poc3](poc3) | Python | Mooncake Store's multi-tier storage (local DRAM / remote DRAM / SSD) in isolation, with an authoritative read-tier classifier. |
| [poc-transfer-engine](poc-transfer-engine) | C++ | The **custom UMD device-DRAM backend** for the Transfer Engine + Mooncake transport, driving a sender→transfer→verify migration worker. Targets #3890. |

## Python PoCs (poc1–poc3)

Self-contained scripts that run against a Mooncake master. Install the wheel into
your tt-metal venv and start a master with the bundled `master_startup.sh`:

```bash
pip install mooncake-transfer-engine==0.3.6.post1
./pocN/master_startup.sh    # in a separate terminal
```

See each PoC's README for its specific run/test commands.

## C++ PoC (poc-transfer-engine)

The transfer-engine PoC is **C++ that builds inside the cpp_server build**, so its
source lives in [`tt-media-server/cpp_server/`](../tt-media-server/cpp_server)
(`src/transport/`, `include/transport/`, tests) while
[`poc-transfer-engine/`](poc-transfer-engine) holds its design record and diagrams.
Build and run from the cpp_server root — see
[`poc-transfer-engine/README.md`](poc-transfer-engine/README.md).
