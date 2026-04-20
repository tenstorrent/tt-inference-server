#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Prefill runner with multi-rank support for disaggregated prefill/decode.

Thin wrapper around TtDeepSeekPrefillPipeline (tt-metal) that:
  - Opens the global 32x4 mesh device via TTNN's distributed context
  - Builds the TtDeepSeekPrefillPipeline (model + KV cache + optional migration layer)
  - Bridges the C++ inference server via shared memory (SHM) on rank 0
  - Broadcasts requests to worker ranks via MPI
  - Calls pipeline.prefill() collectively on all ranks

Rank 0 (coordinator):
  - Reads prefill requests from SHM
  - Broadcasts token_ids + metadata to workers via MPI
  - Calls pipeline.prefill() and writes the first token back to SHM

Ranks 1-3 (workers):
  - Receive request via MPI broadcast
  - Call pipeline.prefill() collectively (TTNN handles cross-host CCL)

KV cache migration happens inside MLA (inside pipeline.prefill()) after each
layer's fill_cache_for_user_(). The pipeline wires the migration callback
automatically when migration_layer is provided.

Environment variables required:
    TT_IPC_SHM_C2P: Name of input  SHM segment (rank 0 only)
    TT_IPC_SHM_P2C: Name of output SHM segment (rank 0 only)
    OMPI_COMM_WORLD_RANK or RANK: Rank ID (0-3)

Usage:
    tt-run --rank-binding 32x4_rank_bindings.yaml \
           --mpi-args "--host $HOSTS --rankfile <rankfile> --bind-to none --tag-output" \
           python prefill_runner.py
"""

import os
import signal
import sys

from mpi4py import MPI

import ttnn
from models.demos.deepseek_v3_d_p.tt.tt_deepseek_prefill_pipeline import (
    TtDeepSeekPrefillPipeline,
    TtPrefillPipelineConfig,
)

from shared_memory import PREFILL_MAX_TOKEN_IDS, SharedMemory

_shutdown = False

# -- Config constants --
# TODO: load real HF config + state_dict from disk. For now these are placeholders
# that match DeepSeek V3 671B with a 32x4 global mesh.
GLOBAL_MESH_SHAPE = (32, 4)
NUM_LAYERS = 61
MAX_SEQ_LEN = 102400


def _handle_sigterm(signum, frame):
    global _shutdown
    _shutdown = True


def _is_shutdown() -> bool:
    return _shutdown


# ================================================================
# Rank 0: coordinator
# ================================================================


def run_rank0_coordinator(pipeline: TtDeepSeekPrefillPipeline, comm) -> None:
    """Rank 0: Read from SHM, broadcast request, run prefill, return token to SHM."""
    c2p_name = os.environ.get("TT_IPC_SHM_C2P")
    p2c_name = os.environ.get("TT_IPC_SHM_P2C")

    if not (c2p_name and p2c_name):
        print("Error: TT_IPC_SHM_C2P or TT_IPC_SHM_P2C not set", file=sys.stderr)
        sys.exit(1)

    print(f"Rank 0: Opening shared memory C2P={c2p_name}, P2C={p2c_name}", file=sys.stderr)

    try:
        with SharedMemory(
            c2p_name, max_token_ids=PREFILL_MAX_TOKEN_IDS, is_shutdown=_is_shutdown
        ) as c2p, SharedMemory(
            p2c_name, max_token_ids=1, is_shutdown=_is_shutdown,
        ) as p2c:
            print("Rank 0: SHM bridge started, waiting for prefill requests...", file=sys.stderr)

            while not _shutdown:
                msg = c2p.read()
                if msg is None:
                    # Shutdown — broadcast None to workers so they exit too
                    comm.bcast(None, root=0)
                    break

                task_id = msg.task_id  # uint32_t per PR #3033
                print(
                    f"Rank 0: Received prefill request task_id={task_id}, "
                    f"num_tokens={len(msg.token_ids)}, "
                    f"tokens={msg.token_ids[:5]}{'...' if len(msg.token_ids) > 5 else ''}",
                    file=sys.stderr,
                )

                # Broadcast request to all ranks
                request = {
                    "token_ids": msg.token_ids,
                    "slot_id": 0,  # TODO: extract from inference server protocol
                }
                comm.bcast(request, root=0)

                # All ranks run prefill collectively
                first_token = pipeline.prefill(
                    token_ids=request["token_ids"],
                    slot_id=request["slot_id"],
                )

                p2c.write_token(task_id, first_token)
                print(f"Rank 0: Sent prefill token {first_token} for task {task_id}", file=sys.stderr)

    except KeyboardInterrupt:
        print("Rank 0: Interrupted by user", file=sys.stderr)

    print("Rank 0: Shutdown complete", file=sys.stderr)


# ================================================================
# Ranks 1-3: workers
# ================================================================


def run_worker_rank(rank: int, pipeline: TtDeepSeekPrefillPipeline, comm) -> None:
    """Ranks 1-3: Receive request via MPI broadcast, participate in collective prefill."""
    print(f"Rank {rank}: Waiting for MPI broadcasts from rank 0...", file=sys.stderr)

    try:
        while not _shutdown:
            request = comm.bcast(None, root=0)
            if request is None:
                break  # shutdown

            print(
                f"Rank {rank}: Processing prefill seq_len={len(request['token_ids'])} "
                f"slot_id={request['slot_id']}",
                file=sys.stderr,
            )

            pipeline.prefill(
                token_ids=request["token_ids"],
                slot_id=request["slot_id"],
            )

    except KeyboardInterrupt:
        print(f"Rank {rank}: Interrupted by user", file=sys.stderr)

    print(f"Rank {rank}: Shutdown complete", file=sys.stderr)


# ================================================================
# Main
# ================================================================


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    # TODO: multi-subcontext support (prefill + decode in one MPI job).
    # When tt-metal PR #42000 (sub-context API) lands and the dual-rank-bindings
    # launcher feature is implemented, we'll need to:
    #   1. Detect our sub-context:  subcontext_id = int(os.environ["TT_RUN_SUBCONTEXT_ID"])
    #   2. If we're not the prefill sub-context, early-exit (decode has its own runner).
    #   3. Split MPI_COMM_WORLD by subcontext_id so bcast()/scatter() stay within prefill:
    #        comm = MPI.COMM_WORLD.Split(color=subcontext_id, key=rank)
    #   4. Dispatch fabric config based on sub-context instead of hardcoding.
    # See models/demos/deepseek_v3_b1/docs/example_dual_rankbindings_one_psd.md in tt-metal.
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    print(f"Rank {rank}/{world_size}: Starting prefill runner", file=sys.stderr)

    # -- Open the global 32x4 mesh (all ranks) --
    ttnn.init_distributed_context()
    # TODO: fabric config, router config from real config (see sub-context TODO above)
    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*GLOBAL_MESH_SHAPE))

    # -- Build the pipeline (all ranks) --
    # TODO: load real HF config, state_dict, migration_layer
    # hf_config = load_hf_config(...)
    # state_dict = load_state_dict(...)
    # migration_layer = setup_migration_layer(...)
    hf_config = None
    state_dict = {}
    migration_layer = None

    pipeline_config = TtPrefillPipelineConfig(
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
        mesh_shape=GLOBAL_MESH_SHAPE,
        is_balanced=True,
    )

    pipeline = TtDeepSeekPrefillPipeline(
        mesh_device=mesh_device,
        hf_config=hf_config,
        state_dict=state_dict,
        config=pipeline_config,
        migration_layer=migration_layer,
    )
    pipeline.compile()

    comm.Barrier()  # all ranks ready
    print(f"Rank {rank}: Setup complete, entering request loop", file=sys.stderr)

    # -- Run --
    if rank == 0:
        run_rank0_coordinator(pipeline, comm)
    elif rank in [1, 2, 3]:
        run_worker_rank(rank, pipeline, comm)
    else:
        print(f"Invalid rank: {rank}. Must be 0-3", file=sys.stderr)
        sys.exit(1)

    # -- Cleanup --
    del pipeline  # releases KV cache + model refs via __del__
    ttnn.close_mesh_device(mesh_device)
    print(f"Rank {rank}: Shutdown complete", file=sys.stderr)


if __name__ == "__main__":
    main()
