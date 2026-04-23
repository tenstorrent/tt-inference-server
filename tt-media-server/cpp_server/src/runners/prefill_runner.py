#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Prefill runner with multi-rank support for disaggregated prefill/decode.

Thin wrapper around TtDeepSeekPrefillPipeline (tt-metal) that:
  - Opens the mesh device via TTNN's distributed context
  - Builds the TtDeepSeekPrefillPipeline (model + KV cache + optional migration layer)
  - Bridges the C++ inference server via shared memory (SHM) on rank 0
  - Broadcasts requests to worker ranks via MPI
  - Calls pipeline.prefill() collectively on all ranks

Rank 0 (coordinator):
  - Reads prefill requests from SHM
  - Broadcasts token_ids + metadata to workers via MPI (skipped for single rank)
  - Calls pipeline.prefill() and writes the first token back to SHM

Ranks 1+ (workers):
  - Receive request via MPI broadcast
  - Call pipeline.prefill() collectively (TTNN handles cross-host CCL)

KV cache migration happens inside MLA (inside pipeline.prefill()) after each
layer's fill_cache_for_user_(). The pipeline wires the migration callback
automatically when migration_layer is provided.

Environment variables:
    TT_IPC_SHM_C2P:              Input SHM segment name (rank 0 only)
    TT_IPC_SHM_P2C:              Output SHM segment name (rank 0 only)
    OMPI_COMM_WORLD_RANK or RANK: Rank ID

    DEEPSEEK_V3_HF_MODEL:        Path to HuggingFace model dir (for loading config)
    TT_DS_PREFILL_TTNN_CACHE:    Path to TTNN weight cache dir
    TT_DS_PREFILL_HOST_REF_CACHE: Path to host reference cache (for PCC checks)

    PREFILL_SP:             SP factor, default 32 (use 8 for single Galaxy)
    PREFILL_TP:             TP factor, default 4
    PREFILL_NUM_LAYERS:     Number of transformer layers, default 61
    PREFILL_MAX_SEQ_LEN:    Max sequence length, default 3200 * PREFILL_SP
    PREFILL_RANDOM_WEIGHTS: Set to "1" to use random weights (no cache needed)
    PREFILL_INPUT_FILE:     Optional JSON file of pre-tokenized token_ids. When set,
                            SHM is used only as a "run now" trigger and the token_ids
                            in the SHM message are ignored. Accepts a plain list
                            [t0, t1, ...] or {"token_ids": [...]}.
    PREFILL_INPUT_PROMPT_FILE: Optional path to a prompt file (plain text, or JSON
                            with a "prompt" key — e.g. cached InfiniteBench
                            subsets). When set and PREFILL_INPUT_FILE is not, the
                            runner loads the HF tokenizer from DEEPSEEK_V3_HF_MODEL
                            and tokenizes to PREFILL_ISL tokens.
    PREFILL_ISL:            Target input-sequence length when tokenizing a prompt
                            file. Must be divisible by PREFILL_SP. Default: PREFILL_MAX_SEQ_LEN.

Usage:
    # Single Galaxy (no tt-run needed):
    export PREFILL_SP=8
    export DEEPSEEK_V3_HF_MODEL=/mnt/MLPerf/.../DeepSeek-R1-0528
    export TT_DS_PREFILL_TTNN_CACHE=/mnt/MLPerf/.../DeepSeek-R1-0528-Cache-prefill
    export TT_DS_PREFILL_HOST_REF_CACHE=/mnt/MLPerf/.../DeepSeek-R1-0528-Reference-prefill
    export TT_IPC_SHM_C2P=tt_ipc_c2p_12345
    export TT_IPC_SHM_P2C=tt_ipc_p2c_12345
    python prefill_runner.py

    # 4-Galaxy ring:
    tt-run --rank-binding 32x4_rank_bindings.yaml \
           --mpi-args "--host $HOSTS --rankfile <rankfile> --bind-to none --tag-output" \
           python prefill_runner.py
"""

import gc
import json
import os
import signal
import sys
from pathlib import Path

import ttnn  # must be imported before mpi4py to avoid MPI library conflicts

import mpi4py
mpi4py.rc.initialize = False
from mpi4py import MPI
from transformers import AutoConfig
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tt.moe.tt_prefill_transformer import TtPrefillTransformer
from models.demos.deepseek_v3_d_p.tt.tt_deepseek_prefill_pipeline import (
    TtDeepSeekPrefillPipeline,
    TtPrefillPipelineConfig,
)
from models.demos.deepseek_v3_d_p.utils.transformer_helpers import (
    create_hf_model,
    extract_tt_state_dict,
    tokenize_prompt_to_isl,
)

from shared_memory import PREFILL_MAX_TOKEN_IDS, SharedMemory

_shutdown = False

# -- Config constants (overridable via env vars) --
# Single Galaxy:  PREFILL_SP=8  PREFILL_TP=4  (1 rank, local 8x4 mesh)
# 4-Galaxy ring:  PREFILL_SP=32 PREFILL_TP=4  (4 ranks, global 32x4 mesh)
_sp = int(os.environ.get("PREFILL_SP", 32))
_tp = int(os.environ.get("PREFILL_TP", 4))
GLOBAL_MESH_SHAPE = (_sp, _tp)
NUM_LAYERS = int(os.environ.get("PREFILL_NUM_LAYERS", 61))
MAX_SEQ_LEN = int(os.environ.get("PREFILL_MAX_SEQ_LEN", 3200 * _sp))


def _handle_sigterm(signum, frame):
    global _shutdown
    _shutdown = True


def _is_shutdown() -> bool:
    return _shutdown


def _detect_world_size() -> int:
    for var in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "SLURM_NTASKS"):
        v = os.environ.get(var)
        if v and v.isdigit():
            return int(v)
    return 1


def _detect_rank() -> int:
    for var in ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "SLURM_PROCID", "RANK"):
        v = os.environ.get(var)
        if v and v.isdigit():
            return int(v)
    return 0


def _load_override_token_ids() -> list[int] | None:
    # 1) Pre-tokenized IDs take precedence.
    ids_path = os.environ.get("PREFILL_INPUT_FILE")
    if ids_path:
        with open(ids_path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data["token_ids"]
        if not isinstance(data, list) or not all(isinstance(x, int) for x in data):
            raise ValueError(
                f"PREFILL_INPUT_FILE {ids_path} must be a JSON list of ints or {{'token_ids': [...]}}"
            )
        return data

    # 2) Otherwise, tokenize a prompt file with the HF tokenizer.
    prompt_path = os.environ.get("PREFILL_INPUT_PROMPT_FILE")
    if not prompt_path:
        return None

    model_path = os.environ.get("DEEPSEEK_V3_HF_MODEL")
    if not model_path:
        raise ValueError(
            "PREFILL_INPUT_PROMPT_FILE requires DEEPSEEK_V3_HF_MODEL for tokenizer loading"
        )

    # Accept plain text or JSON with a "prompt" key (matches InfiniteBench cache format).
    with open(prompt_path) as f:
        raw = f.read()
    try:
        parsed = json.loads(raw)
        prompt_text = parsed["prompt"] if isinstance(parsed, dict) and "prompt" in parsed else raw
    except json.JSONDecodeError:
        prompt_text = raw

    isl = int(os.environ.get("PREFILL_ISL", MAX_SEQ_LEN))
    sp = GLOBAL_MESH_SHAPE[0]
    if isl % sp != 0:
        raise ValueError(f"PREFILL_ISL ({isl}) must be divisible by PREFILL_SP ({sp})")

    print(
        f"Rank 0: tokenizing {prompt_path} with HF tokenizer from {model_path} to ISL={isl}",
        file=sys.stderr,
    )
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    input_ids, _, _ = tokenize_prompt_to_isl(tok, max_isl=isl, prompt_text=prompt_text)
    return input_ids[0].tolist()


def _load_hf_config():
    """Load HuggingFace config from DEEPSEEK_V3_HF_MODEL env var."""
    model_path = os.environ.get("DEEPSEEK_V3_HF_MODEL")
    if not model_path:
        print("Warning: DEEPSEEK_V3_HF_MODEL not set, hf_config will be None", file=sys.stderr)
        return None
    print(f"Loading HF config from {model_path}", file=sys.stderr)
    return AutoConfig.from_pretrained(model_path, trust_remote_code=True)


def _open_mesh_device():
    """Open mesh device with appropriate fabric config for the mesh size."""
    sp = GLOBAL_MESH_SHAPE[0]
    if sp <= 8:
        # Single Galaxy: FABRIC_1D
        fabric_config = ttnn.FabricConfig.FABRIC_1D
    else:
        # Multi-Galaxy ring: FABRIC_2D
        fabric_config = ttnn.FabricConfig.FABRIC_2D

    fabric_router_config = create_fabric_router_config(
        max_payload_size=DeepSeekV3Config.EMB_SIZE
    )

    ttnn.set_fabric_config(
        fabric_config,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
        fabric_router_config,
    )

    return ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(*GLOBAL_MESH_SHAPE),
        worker_l1_size=1431568,
    )


# ================================================================
# Rank 0: coordinator
# ================================================================


def run_rank0_coordinator(
    pipeline: TtDeepSeekPrefillPipeline,
    comm,
    world_size: int,
    override_token_ids: list[int] | None,
) -> None:
    """Rank 0: Read from SHM (trigger), broadcast request, run prefill, return token to SHM.

    When override_token_ids is provided, SHM is still used as the "run now" trigger
    (task_id / response path) but the message's token_ids are ignored.
    """
    c2p_name = os.environ.get("TT_IPC_SHM_C2P")
    p2c_name = os.environ.get("TT_IPC_SHM_P2C")

    if not (c2p_name and p2c_name):
        print("Error: TT_IPC_SHM_C2P or TT_IPC_SHM_P2C not set", file=sys.stderr)
        sys.exit(1)

    print(f"Rank 0: Opening shared memory C2P={c2p_name}, P2C={p2c_name}", file=sys.stderr)
    if override_token_ids is not None:
        print(
            f"Rank 0: PREFILL_INPUT_FILE override active, using {len(override_token_ids)} "
            f"token_ids from file (SHM payload ignored; SHM is trigger-only)",
            file=sys.stderr,
        )

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
                    if world_size > 1:
                        comm.bcast(None, root=0)
                    break

                task_id = msg.task_id  # uint32_t per PR #3033

                # File mode: use the pre-loaded override as-is (caller controls exact ISL).
                # SHM mode: take what came in over SHM and zero-pad up to MAX_SEQ_LEN —
                # the shape the pipeline was compiled for.
                if override_token_ids is not None:
                    token_ids = override_token_ids
                    source = "file"
                else:
                    token_ids = list(msg.token_ids)
                    source = "shm"
                    if len(token_ids) < MAX_SEQ_LEN:
                        token_ids = token_ids + [0] * (MAX_SEQ_LEN - len(token_ids))

                print(
                    f"Rank 0: Received prefill request task_id={task_id}, "
                    f"num_tokens={len(token_ids)}, source={source}, "
                    f"tokens={token_ids[:5]}{'...' if len(token_ids) > 5 else ''}",
                    file=sys.stderr,
                )

                request = {
                    "token_ids": token_ids,
                    "slot_id": 0,  # TODO: extract from inference server protocol
                }

                # Only broadcast if there are worker ranks
                if world_size > 1:
                    comm.bcast(request, root=0)

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
# Ranks 1+: workers
# ================================================================


def run_worker_rank(rank: int, pipeline: TtDeepSeekPrefillPipeline, comm) -> None:
    """Ranks 1+: Receive request via MPI broadcast, participate in collective prefill."""
    print(f"Rank {rank}: Waiting for MPI broadcasts from rank 0...", file=sys.stderr)

    try:
        while not _shutdown:
            request = comm.bcast(None, root=0)
            if request is None:
                break

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

    # Detect launch topology from env *before* touching MPI. Single-rank runs
    # (e.g. one Galaxy, no tt-run/mpirun) skip init_distributed_context entirely
    # to match the pytest path, which never calls it.
    world_size = _detect_world_size()
    rank = _detect_rank()

    comm = None
    if world_size > 1:
        # Multi-rank: tt-run/MPI launched. init_distributed_context() calls
        # MPI_Init_thread -- must happen before any mpi4py operations
        # (mpi4py auto-init is disabled above).
        ttnn.init_distributed_context()
        comm = MPI.COMM_WORLD
        assert rank == comm.Get_rank(), f"env rank {rank} != MPI rank {comm.Get_rank()}"
        assert world_size == comm.Get_size(), f"env size {world_size} != MPI size {comm.Get_size()}"
    else:
        print("Single-rank mode: skipping MPI init / init_distributed_context", file=sys.stderr)

    print(f"Rank {rank}/{world_size}: Starting prefill runner (mesh={GLOBAL_MESH_SHAPE})", file=sys.stderr)

    # -- Open mesh device (all ranks) --
    mesh_device = _open_mesh_device()

    # -- Load HF config (all ranks need same config) --
    hf_config = _load_hf_config()
    hf_config.max_seq_len = MAX_SEQ_LEN

    # -- Build pipeline config --
    env_cache = os.environ.get("TT_DS_PREFILL_TTNN_CACHE")
    use_random_weights = os.environ.get("PREFILL_RANDOM_WEIGHTS", "0") == "1"

    # Mirror the layout produced by tests/conftest.py::weight_cache_path +
    # tests/test_prefill_transformer.py so the runner reads the same files the
    # pytest run wrote: $CACHE / deepseek_v3_d_p_{arch}_{N}dev / {sp}x{tp}
    if env_cache and not use_random_weights:
        arch = "bh" if is_blackhole() else "wh"
        num_devices = ttnn.get_num_devices()
        sp, tp = GLOBAL_MESH_SHAPE
        effective_cache_path = Path(env_cache) / f"deepseek_v3_d_p_{arch}_{num_devices}dev" / f"{sp}x{tp}"
        effective_cache_path.mkdir(parents=True, exist_ok=True)
    else:
        effective_cache_path = None

    pipeline_config = TtPrefillPipelineConfig(
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
        mesh_shape=GLOBAL_MESH_SHAPE,
        is_balanced=True,
        weight_cache_path=effective_cache_path,
    )

    # -- Build state_dict (all ranks) --
    experts_per_chip = 256 // (GLOBAL_MESH_SHAPE[0] * GLOBAL_MESH_SHAPE[1])
    if use_random_weights:
        print(f"Rank {rank}: Creating random weights (PREFILL_RANDOM_WEIGHTS=1)...", file=sys.stderr)
        hf_model = create_hf_model(hf_config, NUM_LAYERS)
        state_dict = extract_tt_state_dict(hf_model)
        del hf_model
        gc.collect()
        print(f"Rank {rank}: Random weights ready", file=sys.stderr)
    elif effective_cache_path and TtPrefillTransformer.check_cache_complete(
        effective_cache_path, NUM_LAYERS, experts_per_chip
    ):
        print(f"Rank {rank}: TTNN weight cache complete at {effective_cache_path}, loading from cache", file=sys.stderr)
        state_dict = {}
    else:
        print(f"Rank {rank}: No cache/weights configured, using empty state_dict", file=sys.stderr)
        state_dict = {}

    pipeline = TtDeepSeekPrefillPipeline(
        mesh_device=mesh_device,
        hf_config=hf_config,
        state_dict=state_dict,
        config=pipeline_config,
    )
    pipeline.compile()

    # -- Migration setup (rank 0 only, when decode runner is available) --
    # TODO: see migration stub in earlier commit for structure.

    override_token_ids = _load_override_token_ids()

    if world_size > 1:
        comm.Barrier()  # all ranks ready
    print(f"Rank {rank}: Setup complete, entering request loop", file=sys.stderr)

    # -- Run --
    if rank == 0:
        run_rank0_coordinator(pipeline, comm, world_size, override_token_ids)
    elif rank >= 1:
        run_worker_rank(rank, pipeline, comm)

    # -- Cleanup --
    del pipeline
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    ttnn.close_mesh_device(mesh_device)
    print(f"Rank {rank}: Shutdown complete", file=sys.stderr)


if __name__ == "__main__":
    main()
