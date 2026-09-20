<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
-->

# Co-located training + inference on one Galaxy

The **launcher** for co-locating a trainer and a vLLM inference server on a
single Galaxy (one `tt-run` MPI world, 16/16 device split, device-to-device
weight transfer) lives in **`tt-training-service`**, since that service owns
job launching (SLURM / `tt-run`):

> `tt-training-service/grpo/colocate/` — MGD, rank bindings, the per-rank
> dispatcher, and a README. It is wired through `job_manager`'s
> `run_command_override` (`training_types._build_colocated_grpo_run_command`),
> enabled per-request via `inference_server.transport: device_socket`.

## What stays in this repo

The **inference-server-side change** that lets vLLM join an externally-launched
`tt-run` world:

`tt-vllm-plugin/.../v1/worker/tt_worker.py` → `open_mesh_device()`, when
`TT_COLOCATED_INFERENCE=1`:

1. joins the launcher's shared ttnn distributed context, so the server opens
   **only its bound submesh** (`MESH_DEVICE=(8,2)`) instead of all 32 chips, and
2. sets **`FABRIC_2D`** (required for inter-mesh device sockets).

This is gated entirely on the env var, so the normal `docker run` path is
unchanged. The receiver endpoints for weight hot-swap are
`vllm-tt-metal/src/weight_update_api.py` (`POST /v1/internal/weights/update`,
payload `{sender_rank, hf_rope}`) → `TTWorker.update_weights`
(`WeightBridge` `role="ttt"`).
