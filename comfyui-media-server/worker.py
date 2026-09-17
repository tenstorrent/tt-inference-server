# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

import os
import time
import queue
import multiprocessing as mp

from utils.logger import setup_logger


def _serialize_pipeline_event(event) -> dict:
    """Map a tt_dit pipeline event dataclass to a JSON-friendly progress dict.

    Imported lazily by callers (the events module pulls in tt_dit). Returns None
    for event types we do not forward.
    """
    from models.tt_dit.pipelines.events import DenoiseStep, SectionEnd, SectionStart

    if isinstance(event, DenoiseStep):
        return {"type": "denoise_step", "step": int(event.step), "total": int(event.total)}
    if isinstance(event, SectionStart):
        return {"type": "section_start", "name": event.name}
    if isinstance(event, SectionEnd):
        return {"type": "section_end", "name": event.name}
    return None


def setup_sdxl_worker_environment(worker_id: int, config):
    """
    Set worker-specific environment variables for SDXL.

    Topology and descriptor wiring are driven by `config.board` via
    device_specs.BOARD_SPECS — every supported board sets
    TT_MESH_GRAPH_DESC_PATH unconditionally so the runtime never falls back
    to a CUSTOM cluster type missing its descriptor.
    """
    from device_specs import descriptor_path, get_board_spec

    # After the relocation out of tt-metal, cwd is this server's directory, not a
    # tt-metal checkout, so cwd is the wrong value for both of these. Respect what
    # the launcher exported; fall back to cwd for pre-relocation behaviour.
    os.environ.setdefault("TT_METAL_HOME", os.getcwd())
    os.environ.setdefault("PYTHONPATH", os.getcwd())

    # Set device visibility EARLY (before any ttnn operations)
    device_ids_str = ",".join(map(str, config.device_ids))
    os.environ["TT_VISIBLE_DEVICES"] = device_ids_str
    os.environ["TT_METAL_VISIBLE_DEVICES"] = device_ids_str

    # Mesh-graph descriptor — required by tt-metal whenever the cluster
    # falls back to CUSTOM (Blackhole P-series, partial-pop boards).
    os.environ["TT_MESH_GRAPH_DESC_PATH"] = descriptor_path(config.board)

    # Per-board extras (e.g. Galaxy's TT_MM_THROTTLE_PERF=5).
    os.environ.update(get_board_spec(config.board).extra_env_vars)


def setup_wan_worker_environment(worker_id: int, config):
    """Wan2.2: a single worker owns the full mesh. Topology is board-derived
    via device_specs (descriptor_path + extra_env_vars).
    """
    from device_specs import descriptor_path, get_board_spec

    # After the relocation out of tt-metal, cwd is this server's directory, not a
    # tt-metal checkout, so cwd is the wrong value for both of these. Respect what
    # the launcher exported; fall back to cwd for pre-relocation behaviour.
    os.environ.setdefault("TT_METAL_HOME", os.getcwd())
    os.environ.setdefault("PYTHONPATH", os.getcwd())

    device_ids_str = ",".join(map(str, config.device_ids))
    os.environ["TT_VISIBLE_DEVICES"] = device_ids_str
    os.environ["TT_METAL_VISIBLE_DEVICES"] = device_ids_str

    os.environ["TT_MESH_GRAPH_DESC_PATH"] = descriptor_path(config.board)
    os.environ.update(get_board_spec(config.board).extra_env_vars)


def setup_ltx_worker_environment(worker_id: int, config):
    """LTX-2.3: a single worker owns the full mesh. Topology is board-derived
    via device_specs, same as Wan.
    """
    from device_specs import descriptor_path, get_board_spec

    os.environ.setdefault("TT_METAL_HOME", os.getcwd())
    os.environ.setdefault("PYTHONPATH", os.getcwd())

    device_ids_str = ",".join(map(str, config.device_ids))
    os.environ["TT_VISIBLE_DEVICES"] = device_ids_str
    os.environ["TT_METAL_VISIBLE_DEVICES"] = device_ids_str

    os.environ["TT_MESH_GRAPH_DESC_PATH"] = descriptor_path(config.board)
    os.environ.update(get_board_spec(config.board).extra_env_vars)

    # launch_server.sh does not export this. Unset, every tilized weight load is a
    # cache miss and startup goes from ~40s to minutes.
    os.environ["TT_DIT_CACHE_DIR"] = config.tt_dit_cache_dir
    os.environ["LTX_QUANT"] = config.quant


def setup_sd35_worker_environment(worker_id: int, config):
    """
    Set worker-specific environment variables for SD3.5 (LoudBox, single worker
    owns the full 2x4 mesh).

    All 8 device IDs are made visible so the mesh device can be opened with
    the full 2x4 topology. No per-device isolation needed.
    """
    # After the relocation out of tt-metal, cwd is this server's directory, not a
    # tt-metal checkout, so cwd is the wrong value for both of these. Respect what
    # the launcher exported; fall back to cwd for pre-relocation behaviour.
    os.environ.setdefault("TT_METAL_HOME", os.getcwd())
    os.environ.setdefault("PYTHONPATH", os.getcwd())

    # SD3.5 needs the full 2x4 mesh (all 8 chips: 4 PCIe L + 4 ethernet R).
    # Do NOT restrict TT_VISIBLE_DEVICES — let UMD discover the full topology.
    # The shell may have set TT_VISIBLE_DEVICES to PCIe-only IDs; clear it here
    # so the system mesh auto-discovers all 8 chips correctly.
    os.environ.pop("TT_VISIBLE_DEVICES", None)
    os.environ.pop("TT_METAL_VISIBLE_DEVICES", None)


def device_worker_process(
    worker_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    warmup_signal_queue: mp.Queue,
    kernel_ready_queue: mp.Queue,
    error_queue: mp.Queue,
    config,
    progress_queue: mp.Queue = None,
):
    """
    Main worker process function that handles image generation inference tasks.

    Supports SDXL, SD3.5, Wan2.2 and LTX-2.3 via config type dispatch:
    - SD35Config  → SD35Runner  (2x4 mesh, single worker)
    - SDXLConfig  → SDXLRunner  (1x1 mesh, one worker per device)
    - WanConfig   → WanRunner   (board-derived mesh, single worker)
    - LTXConfig    → LTXRunner    (board-derived mesh, single worker)
    - LTXProConfig → LTXProRunner (same, guided one-stage sampler; checked first
      because it subclasses LTXConfig)

    All runner/ttnn imports are deferred to this function so the main server
    process never initializes ttnn (which would conflict with the child
    processes' device ownership).

    Args:
        worker_id: Worker process ID
        task_queue: Queue for incoming tasks — each item is (task_id, request_dict)
        result_queue: Queue for results — each item is a dict with task_id/images/inference_time
        warmup_signal_queue: Queue to signal warmup completion to the server
        kernel_ready_queue: Queue to signal kernel compilation complete (overlapped startup)
        error_queue: Queue for error reporting to the server
        config: SDXLConfig or SD35Config instance
    """
    logger = setup_logger(f"Worker-{worker_id}")
    logger.info(f"Worker {worker_id} starting...")

    try:
        # Determine model type and set up environment + runner
        # Imports are deferred here to avoid ttnn initialization in the main process
        from ltx_config import LTXConfig
        from ltx_pro_config import LTXProConfig
        from sd35_config import SD35Config
        from wan_config import WanConfig

        # LTXProConfig subclasses LTXConfig, so it must be tested first or a Pro
        # server would silently come up running the distilled pipeline.
        if isinstance(config, LTXProConfig):
            setup_ltx_worker_environment(worker_id, config)
            from ltx_pro_runner import LTXProRunner

            runner = LTXProRunner(worker_id, config)
        elif isinstance(config, LTXConfig):
            setup_ltx_worker_environment(worker_id, config)
            from ltx_runner import LTXRunner

            runner = LTXRunner(worker_id, config)
        elif isinstance(config, WanConfig):
            setup_wan_worker_environment(worker_id, config)
            from wan_runner import WanRunner

            runner = WanRunner(worker_id, config)
        elif isinstance(config, SD35Config):
            setup_sd35_worker_environment(worker_id, config)
            from sd35_runner import SD35Runner

            runner = SD35Runner(worker_id, config)
        else:
            setup_sdxl_worker_environment(worker_id, config)
            from sdxl_runner import SDXLRunner

            runner = SDXLRunner(worker_id, config)

        # Initialize device and load model with optional overlapped startup signal
        runner.initialize_device()
        runner.load_model(kernel_ready_queue=kernel_ready_queue)

        # Signal to server that this worker has completed full warmup
        warmup_signal_queue.put(worker_id)
        logger.info(f"Worker {worker_id} ready for inference")

        # Main processing loop
        while True:
            try:
                # Block with timeout so we can detect shutdown cleanly
                task = task_queue.get(timeout=1)

                if task is None:  # Shutdown signal
                    logger.info(f"Worker {worker_id} received shutdown signal")
                    break

                task_id, request = task
                op = request.get("op", "generate") if isinstance(request, dict) else "generate"
                logger.info(f"Worker {worker_id} processing task {task_id} (op={op})")

                # Run inference and measure wall-clock time
                start_time = time.time()

                if op == "generate":
                    # Existing full-pipeline path (unchanged behavior).
                    images = runner.run_inference([request])
                    inference_time = time.time() - start_time
                    result_queue.put({"task_id": task_id, "op": op, "images": images, "inference_time": inference_time})
                elif op == "av_generate":
                    # LTX-2.3: audio+video in one call. The runner returns encoded MP4
                    # bytes rather than frames, because the clip carries synchronized
                    # audio and the frames contract here has no audio channel.
                    if not hasattr(runner, "run_inference"):
                        raise RuntimeError(f"Runner {type(runner).__name__} does not support 'av_generate'")
                    av_kwargs = {}
                    # Same opt-in as denoise: stream only when the request asked for it
                    # and the runner accepts on_event, since the blocking endpoint never
                    # drains progress_queue.
                    if progress_queue is not None and request.get("stream_progress"):
                        import inspect

                        if "on_event" in inspect.signature(runner.run_inference).parameters:

                            def _on_av_event(event, _tid=task_id):
                                ev = _serialize_pipeline_event(event)
                                if ev is None:
                                    return
                                ev["task_id"] = _tid
                                try:
                                    progress_queue.put_nowait(ev)
                                except Exception:
                                    pass

                            av_kwargs["on_event"] = _on_av_event
                    video = runner.run_inference([request], **av_kwargs)[0]
                    inference_time = time.time() - start_time
                    result_queue.put(
                        {
                            "task_id": task_id,
                            "op": op,
                            "video": video,
                            "inference_time": inference_time,
                            # Per-adapter LoRA status. The staged-op branch below
                            # attaches this too, but does not cover av_generate.
                            "lora": getattr(runner, "_last_lora_status", None),
                        }
                    )
                elif op in ("denoise", "vae_decode", "vae_encode"):
                    # Additive staged ops (currently SDXL only).
                    if not hasattr(runner, op):
                        raise RuntimeError(f"Runner {type(runner).__name__} does not support staged op '{op}'")
                    if op == "denoise":
                        denoise_kwargs = {}
                        # Stream progress events only when the request opted in (the
                        # streaming endpoint sets stream_progress=True) AND the runner's
                        # denoise accepts on_event (WAN). The blocking /video/denoise path
                        # never drains progress_queue, so we must not emit for it.
                        if progress_queue is not None and request.get("stream_progress"):
                            import inspect

                            if "on_event" in inspect.signature(runner.denoise).parameters:

                                def _on_event(event, _tid=task_id):
                                    ev = _serialize_pipeline_event(event)
                                    if ev is None:
                                        return
                                    ev["task_id"] = _tid
                                    try:
                                        progress_queue.put_nowait(ev)
                                    except Exception:
                                        pass

                                denoise_kwargs["on_event"] = _on_event
                        tensor = runner.denoise(request, **denoise_kwargs)
                    elif op == "vae_decode":
                        tensor = runner.vae_decode(request["latent"])
                    else:  # vae_encode
                        tensor = runner.vae_encode(request["image"])
                    inference_time = time.time() - start_time
                    result_queue.put(
                        {
                            "task_id": task_id,
                            "op": op,
                            "tensor": tensor,
                            "inference_time": inference_time,
                            "lora": getattr(runner, "_last_lora_status", None),
                        }
                    )
                else:
                    raise RuntimeError(f"Unknown op '{op}'")

                logger.info(f"Task {task_id} completed in {inference_time:.2f}s")

            except queue.Empty:
                # Normal timeout — no task available, loop again
                continue
            except Exception as e:
                import traceback

                logger.error(f"Error processing task: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                error_queue.put({"worker_id": worker_id, "error": str(e)})

        # Cleanup
        runner.close_device()
        logger.info(f"Worker {worker_id} shutdown complete")

    except Exception as e:
        import traceback

        logger.error(f"Worker {worker_id} fatal error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        error_queue.put({"worker_id": worker_id, "error": f"Fatal: {str(e)}"})
