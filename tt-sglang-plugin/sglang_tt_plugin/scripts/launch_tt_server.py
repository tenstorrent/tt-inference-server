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
    parser = argparse.ArgumentParser(description="Launch SGLang server with TT-Metal support")
    parser.add_argument("--model-path", required=True, help="Model path")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=30000, help="Port number")
    parser.add_argument("--page-size", type=int, default=128, help="Page size")
    parser.add_argument("--max-running-requests", type=int, default=1, help="Max running requests")
    parser.add_argument("--log-level", default="info", help="Log level")
    parser.add_argument("--device", default="cpu", help="Device type (always cpu for TT)")
    parser.add_argument("--context-length", type=int, default=65536, help="Context length (max 65536 for N150)")
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
    ]
    
    if args.trust_remote_code:
        sglang_args.append("--trust-remote-code")
    if args.disable_overlap_schedule:
        sglang_args.append("--disable-overlap-schedule")
    
    sglang_args.extend(remaining_args)
    
    print(f"[TT-Plugin] Starting server with args: {sglang_args}", file=sys.stderr, flush=True)
    
    server_args = prepare_server_args(sglang_args)
    run_server(server_args)

if __name__ == "__main__":
    main()
