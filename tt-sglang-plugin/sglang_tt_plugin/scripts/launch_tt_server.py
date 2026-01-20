#!/usr/bin/env python3
"""
Launch SGLang server with TT-Metal plugin support.
Plugin MUST be imported before SGLang loads models.

Usage:
    sglang-tt-server --model-path meta-llama/Llama-3.1-8B-Instruct
"""

import os
import sys
import argparse

def setup_tt_environment():
    """Setup TT-Metal environment variables."""
    os.environ["VLLM_DEVICE_TYPE"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["VLLM_PLUGINS"] = ""
    os.environ["SGLANG_USE_CPU_ENGINE"] = "1"
    os.environ["LD_PRELOAD"] = "/lib/x86_64-linux-gnu/libnuma.so.1"
    os.environ["TRITON_CPU_ONLY"] = "1"
    os.environ["TRITON_INTERPRET"] = "1"
    # Set external model package for registry discovery
    os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "sglang_tt_plugin.models"

def main():
    """Main entry point for TT server launch."""
    # Force fork mode FIRST - before any multiprocessing imports
    # This makes child processes inherit the patched ModelRegistry
    import multiprocessing
    try:
        multiprocessing.set_start_method("fork", force=True)
        print("[TT-Plugin] Set multiprocessing start method to 'fork'", file=sys.stderr, flush=True)
    except RuntimeError as e:
        print(f"[TT-Plugin] Could not set fork mode: {e}", file=sys.stderr, flush=True)
    
    # Setup TT environment FIRST (before any imports)
    setup_tt_environment()
    
    # Import plugin to register TT models
    print("[TT-Plugin] Importing plugin...", file=sys.stderr, flush=True)
    try:
        import sglang_tt_plugin
        print(f"[TT-Plugin] Plugin version {sglang_tt_plugin.__version__} loaded", file=sys.stderr, flush=True)
    except ImportError as e:
        print(f"[TT-Plugin] Failed to load plugin: {e}", file=sys.stderr, flush=True)
        sys.exit(1)
    
    # Import SGLang modules - this populates ModelRegistry
    from sglang.srt.models.registry import ModelRegistry
    from sglang.srt.server_args import prepare_server_args
    from sglang.launch_server import run_server
    
    # Register TT models after SGLang is imported
    print("[TT-Plugin] Registering TT models...", file=sys.stderr, flush=True)
    sglang_tt_plugin.register_tt_models()
    
    # Parse arguments - pass through to SGLang with TT defaults
    # These mirror vLLM's LLM() constructor params for consistency
    parser = argparse.ArgumentParser(description="Launch SGLang server with TT-Metal support")
    parser.add_argument("--model-path", required=True, help="Model path (vLLM: model)")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=30000, help="Port number")
    
    # KV cache / memory management (critical for TT-Metal)
    parser.add_argument("--page-size", type=int, default=64, help="Block size for KV cache")
    parser.add_argument("--max-running-requests", type=int, default=32, help="Max batch size")
    parser.add_argument("--context-length", type=int, default=65536, help="Max sequence length")
    
    # TT-Metal specific settings
    parser.add_argument("--optimizations", default="performance", choices=["performance", "accuracy"],
                        help="TT-Metal optimization mode: 'performance' (fastest) or 'accuracy' (more precise)")
    
    # Other settings
    parser.add_argument("--log-level", default="info", help="Log level")
    parser.add_argument("--device", default="cpu", help="Device type (always cpu for TT)")
    parser.add_argument("--trust-remote-code", action="store_true", default=True, help="Trust remote code")
    parser.add_argument("--disable-overlap-schedule", action="store_true", default=True, help="Disable overlap schedule")
    
    args, remaining_args = parser.parse_known_args()
    
    # Build SGLang args
    sglang_args = [
        "--model-path", args.model_path,
        "--host", args.host,
        "--port", str(args.port),
        "--page-size", str(args.page_size),
        "--max-running-requests", str(args.max_running_requests),
        "--log-level", args.log_level,
        "--context-length", str(args.context_length),
        "--device", args.device,
    ]
    
    if args.trust_remote_code:
        sglang_args.append("--trust-remote-code")
    if args.disable_overlap_schedule:
        sglang_args.append("--disable-overlap-schedule")
    
    sglang_args.extend(remaining_args)
    
    # Set TT-Metal specific config via environment (not part of SGLang's server_args)
    os.environ["TT_METAL_OPTIMIZATIONS"] = args.optimizations
    
    print(f"[TT-Plugin] Starting server with args: {sglang_args}", file=sys.stderr, flush=True)
    print(f"[TT-Plugin] TT-Metal optimizations: {args.optimizations}", file=sys.stderr, flush=True)
    
    server_args = prepare_server_args(sglang_args)
    run_server(server_args)

if __name__ == "__main__":
    main()
