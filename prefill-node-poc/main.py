# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Entry point for the prefill-node-poc. Uses a config object, DeepSeek V3 tokenizer,
concrete prompt strings, and runs the engine in a loop until main signals quit.

Supports two modes:
1. Standalone mode (default): Runs prefill engine with callbacks for KV cache
2. P/D mode (--pd-mode): Acts as prefill server, streams KV cache to decode node via MPI
"""

from __future__ import annotations

import argparse
import logging
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class AppConfig:
    """Hardcoded config for prefill-node-poc demo."""

    # Scheduler
    max_num_seqs: int = 4
    max_num_batched_tokens: int = 512
    num_kvcache_blocks: int | None = None
    block_size: int = 32
    # KV cache memory (used when num_kvcache_blocks is None)
    available_kv_cache_memory_gb: float = 12.0
    num_layers: int = 61  # full model layers for KV memory calc
    kvpe_dim: int = 576
    kv_cache_dtype_bytes: int = 1  # bfloat8 for DeepSeek; use 2 for bfloat16
    kv_tensors_per_layer: int = (
        1  # 1 for DeepSeek (combined KV/MLA); 2 for separate K and V
    )

    # Simulator / model
    vocab_size: int = 129280
    num_layers_sim: int = 4
    vocab_size_sim: int = 1000

    # Tokenizer (DeepSeek V3 on HuggingFace)
    tokenizer_model: str = "deepseek-ai/DeepSeek-V3"

    # Concrete prompts to tokenize and send (more than max_num_seqs to get waiting)
    prompts: list[str] = field(
        default_factory=lambda: [
            "What is the capital of France?",
            "Explain quantum computing in one sentence.",
            "Write a haiku about coding.",
            "List three benefits of open source software.",
            "What is the speed of light?",
            "Describe the water cycle briefly.",
        ]
    )

    # Run loop (main sleeps this long then signals engine to quit)
    run_seconds: float = 2.0
    verbose_kv: bool = False

    # P/D mode MPI config
    decode_rank: int = 1
    REQUEST_TAG: int = 100
    RESPONSE_TAG: int = 200
    KV_LAYER_TAG_BASE: int = 1000


def get_node_info() -> dict:
    """Gather node information for host identification."""
    info = {"hostname": socket.gethostname(), "fqdn": socket.getfqdn()}
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        info["primary_ip"] = s.getsockname()[0]
        s.close()
    except Exception:
        info["primary_ip"] = "unknown"
    try:
        with open("/etc/machine-id", "r") as f:
            info["machine_id"] = f.read().strip()[:12] + "..."
    except Exception:
        pass
    return info


def print_node_info(rank: int, role: str = "") -> None:
    """Print node identification info."""
    info = get_node_info()
    role_str = f" ({role})" if role else ""
    print(f"[Rank {rank}{role_str}] === NODE IDENTIFICATION ===")
    print(f"[Rank {rank}{role_str}]   Hostname: {info['hostname']}")
    print(f"[Rank {rank}{role_str}]   FQDN: {info['fqdn']}")
    print(f"[Rank {rank}{role_str}]   Primary IP: {info['primary_ip']}")
    if "machine_id" in info:
        print(f"[Rank {rank}{role_str}]   Machine ID: {info['machine_id']}")
    print(f"[Rank {rank}{role_str}] ==============================")


class KVCacheSender:
    """Handles MPI communication for streaming KV cache to decode node."""

    def __init__(self, comm, cfg: AppConfig):
        self.comm = comm
        self.cfg = cfg
        self.hostname = socket.gethostname()
        self._current_request_id: Optional[int] = None
        self._current_seq_len: int = 0
        self._layer_send_times: list[float] = []

    def _log(self, msg: str) -> None:
        print(f"[Prefill@{self.hostname}] {msg}")

    def wait_for_request(self) -> tuple[int, int]:
        """Wait for prefill request from decode node. Returns (request_id, seq_len)."""
        request_bytes = np.empty(9, dtype=np.uint8)
        self.comm.Recv(request_bytes, source=self.cfg.decode_rank, tag=self.cfg.REQUEST_TAG)
        msg_type, request_id, seq_len = struct.unpack("<BII", request_bytes.tobytes())
        self._current_request_id = request_id
        self._current_seq_len = seq_len
        self._layer_send_times = []
        self._log(f"Received request: id={request_id}, seq_len={seq_len}")
        return request_id, seq_len

    def send_response(self, num_layers: int, layer_size_bytes: int) -> None:
        """Send response header to decode node."""
        response = struct.pack(
            "<BIIII",
            2,  # PREFILL_RESPONSE
            self._current_request_id,
            num_layers,
            layer_size_bytes,
            0,  # status OK
        )
        self._log(f"Sending response: layers={num_layers}, layer_size={layer_size_bytes}")
        self.comm.Send(np.frombuffer(response, dtype=np.uint8), dest=self.cfg.decode_rank, tag=self.cfg.RESPONSE_TAG)

    def send_kv_layer(self, layer_idx: int, data: np.ndarray) -> None:
        """Send a single KV cache layer to decode node."""
        start = time.perf_counter()
        self.comm.Send(data, dest=self.cfg.decode_rank, tag=self.cfg.KV_LAYER_TAG_BASE + layer_idx)
        elapsed_ms = (time.perf_counter() - start) * 1000
        self._layer_send_times.append(elapsed_ms)
        if layer_idx == 0 or (layer_idx + 1) % 10 == 0:
            self._log(f"  Sent layer {layer_idx}: {elapsed_ms:.2f} ms")

    def finish_request(self) -> None:
        """Log summary after all layers sent."""
        if self._layer_send_times:
            total_ms = sum(self._layer_send_times)
            avg_ms = total_ms / len(self._layer_send_times)
            self._log(f"Request {self._current_request_id} complete: {len(self._layer_send_times)} layers, total={total_ms:.2f} ms, avg={avg_ms:.2f} ms/layer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prefill Node POC")
    parser.add_argument("--pd-mode", action="store_true", help="Run in P/D mode (MPI server)")
    parser.add_argument("--num-requests", type=int, default=5, help="Number of requests to handle in P/D mode")
    parser.add_argument("--run-seconds", type=float, default=2.0, help="Run time for standalone mode")
    parser.add_argument("--verbose-kv", action="store_true", help="Verbose KV cache logging")
    return parser.parse_args()


def run_standalone(cfg: AppConfig) -> None:
    """Run in standalone mode (original behavior)."""
    import ttnn
    from prefill_engine import PrefillEngine
    from prefill_simulator import KVCacheReference, PrefillConfig
    from scheduler import SchedulerConfig

    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_model, trust_remote_code=True)
    except Exception as e:
        logging.warning("Could not load DeepSeek tokenizer (%s), using dummy tokens", e)
        tokenizer = None

    def on_kv_ready(layer_idx: int, ref: KVCacheReference) -> None:
        if cfg.verbose_kv:
            print(f"  KV layer {layer_idx} ready for whole batch")

    def on_blocks_ready(layer_idx: int, req_id: object, blocks: list) -> None:
        if cfg.verbose_kv:
            print(f"  KV layer {layer_idx} blocks ready for req_id={req_id}, num_blocks={len(blocks)}")

    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 2))

    sched_config = SchedulerConfig(
        max_num_seqs=cfg.max_num_seqs,
        max_num_batched_tokens=cfg.max_num_batched_tokens,
        num_kvcache_blocks=cfg.num_kvcache_blocks,
        block_size=cfg.block_size,
        available_kv_cache_memory_gb=cfg.available_kv_cache_memory_gb,
        num_layers=cfg.num_layers,
        kvpe_dim=cfg.kvpe_dim,
        kv_cache_dtype_bytes=cfg.kv_cache_dtype_bytes,
        kv_tensors_per_layer=cfg.kv_tensors_per_layer,
    )
    num_kvcache_blocks = sched_config.get_num_kvcache_blocks()
    prefill_config = PrefillConfig(
        num_layers=cfg.num_layers_sim,
        vocab_size=cfg.vocab_size_sim,
        block_size=cfg.block_size,
        num_kvcache_blocks=num_kvcache_blocks,
    )
    engine = PrefillEngine(
        scheduler_config=sched_config,
        prefill_config=prefill_config,
        on_kv_cache_ready=on_kv_ready,
        on_kv_cache_blocks_ready=on_blocks_ready,
        mesh_device=mesh_device,
    )

    if tokenizer is not None:
        for i, text in enumerate(cfg.prompts):
            token_ids = tokenizer.encode(text, add_special_tokens=True)
            engine.add_request(token_ids, req_id=f"req_{i}")
            print(f"Added: {text[:50]}... -> {len(token_ids)} tokens")
    else:
        import random
        for i in range(len(cfg.prompts)):
            token_ids = [random.randint(0, prefill_config.vocab_size - 1) for _ in range(32)]
            engine.add_request(token_ids, req_id=f"req_{i}")
            print(f"Added dummy req_{i} ({len(token_ids)} tokens)")

    stop_event = threading.Event()
    loop_thread = threading.Thread(target=engine.run_loop, args=(stop_event,), daemon=True)
    loop_thread.start()
    print(f"Engine run_loop started; will stop in {cfg.run_seconds}s ...")
    time.sleep(cfg.run_seconds)
    stop_event.set()
    loop_thread.join(timeout=2.0)
    print("Engine stopped.")
    engine.cleanup()


def run_pd_mode(cfg: AppConfig, num_requests: int) -> None:
    """Run in P/D mode with actual prefill engine - requires ttnn/device.
    
    Uses true streaming: each KV layer is sent immediately when ready during prefill,
    not queued and sent after prefill completes.
    """
    import random
    import ttnn
    from prefill_engine import PrefillEngine
    from prefill_simulator import KVCacheReference, PrefillConfig
    from scheduler import SchedulerConfig
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    print_node_info(rank, "Prefill")
    comm.Barrier()

    if rank != 0:
        print(f"ERROR: Prefill node must be rank 0, got {rank}")
        return

    sender = KVCacheSender(comm, cfg)

    def on_kv_ready(layer_idx: int, ref: KVCacheReference) -> None:
        """Callback when KV layer is ready - send IMMEDIATELY for true streaming."""
        # Extract layer data from device tensor
        # TODO: Replace mock data with actual tensor read from ref.tensor
        num_blocks, _, block_size, kvpe_dim = ref.shape
        layer_bytes = num_blocks * block_size * kvpe_dim
        
        # For now, create mock data (will be replaced with actual tensor data)
        layer_data = np.full(layer_bytes, layer_idx % 256, dtype=np.uint8)
        
        # Send immediately - true streaming!
        sender.send_kv_layer(layer_idx, layer_data)
        
        if cfg.verbose_kv:
            print(f"  KV layer {layer_idx} streamed ({layer_bytes} bytes)")

    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 2))

    sched_config = SchedulerConfig(
        max_num_seqs=cfg.max_num_seqs,
        max_num_batched_tokens=cfg.max_num_batched_tokens,
        num_kvcache_blocks=cfg.num_kvcache_blocks,
        block_size=cfg.block_size,
        available_kv_cache_memory_gb=cfg.available_kv_cache_memory_gb,
        num_layers=cfg.num_layers,
        kvpe_dim=cfg.kvpe_dim,
        kv_cache_dtype_bytes=cfg.kv_cache_dtype_bytes,
        kv_tensors_per_layer=cfg.kv_tensors_per_layer,
    )
    num_kvcache_blocks = sched_config.get_num_kvcache_blocks()
    prefill_config = PrefillConfig(
        num_layers=cfg.num_layers_sim,
        vocab_size=cfg.vocab_size_sim,
        block_size=cfg.block_size,
        num_kvcache_blocks=num_kvcache_blocks,
    )
    engine = PrefillEngine(
        scheduler_config=sched_config,
        prefill_config=prefill_config,
        on_kv_cache_ready=on_kv_ready,
        on_kv_cache_blocks_ready=None,
        mesh_device=mesh_device,
    )

    print(f"[Prefill] P/D mode started (streaming), waiting for {num_requests} requests...")

    for req_num in range(num_requests):
        request_id, seq_len = sender.wait_for_request()
        comm.Barrier()

        # Calculate layer size upfront
        num_blocks = seq_len // cfg.block_size
        layer_size = num_blocks * cfg.block_size * cfg.kvpe_dim

        # Send response header BEFORE prefill starts
        sender.send_response(cfg.num_layers_sim, layer_size)

        # Run prefill - layers are streamed via callback DURING execution
        token_ids = [random.randint(0, prefill_config.vocab_size - 1) for _ in range(seq_len)]
        engine.add_request(token_ids, req_id=f"mpi_req_{request_id}")
        engine.step()  # Callback sends layers as they complete!

        sender.finish_request()
        engine.scheduler.release(list(engine.scheduler._running.values()))
        comm.Barrier()

    print("[Prefill] P/D mode finished")
    engine.cleanup()
    comm.Barrier()


def main() -> None:
    args = parse_args()
    cfg = AppConfig()
    cfg.run_seconds = args.run_seconds
    cfg.verbose_kv = args.verbose_kv

    if args.pd_mode:
        run_pd_mode(cfg, args.num_requests)
    else:
        run_standalone(cfg)


if __name__ == "__main__":
    main()
