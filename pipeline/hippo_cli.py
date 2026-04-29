#!/usr/bin/env python3
"""
hippo_cli.py — Unified CLI entry point for Hippo Pipeline.

Usage:
    hippo-pipeline serve --model gemma-3-12b --mode pipeline --rank 0
    hippo-pipeline serve --model qwen3-4b --mode dflash
    hippo-pipeline serve --model qwen3-4b --mode standalone
    hippo-pipeline benchmark --model gemma-3-12b --mode pipeline --runs 3
    hippo-pipeline list-models
"""

import argparse
import os
import subprocess
import sys

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONF_PATH = os.path.join(SCRIPT_DIR, "hippo.conf.yaml")


def load_config():
    if os.path.exists(CONF_PATH):
        with open(CONF_PATH) as f:
            return yaml.safe_load(f)
    return {"models": {}, "defaults": {}, "hardware": {}}


def list_models(cfg):
    print("Available models:\n")
    for name, m in cfg.get("models", {}).items():
        modes = ", ".join(m.get("modes", []))
        print(f"  {name:20s}  {m.get('size_gb', '?')}GB  [{modes}]")
        for mode_name, mode_conf in m.items():
            if isinstance(mode_conf, dict) and "expected_tok_s" in mode_conf:
                print(f"    {mode_name}: ~{mode_conf['expected_tok_s']} tok/s")
    print()


def check_memory(cfg, model_name, mode):
    """Warn if model likely won't fit in RAM."""

    m = cfg.get("models", {}).get(model_name)
    if not m:
        return True

    cfg.get("hardware", {})
    mem_cfg = cfg.get("memory", {})
    safety = mem_cfg.get("safety_factor", 0.85)
    reserve = mem_cfg.get("system_reserve_gb", 4.0)
    overhead = mem_cfg.get("mlx_overhead_gb", 2.0)

    # Detect local RAM
    try:
        result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        ram_gb = int(result.stdout.strip()) / (1024 ** 3)
    except Exception:
        ram_gb = 16.0  # default assumption

    # Adjust safety factor for large memory machines
    if ram_gb >= 48:
        safety = 0.60  # 48GB+: generous, less conservative
    elif ram_gb >= 32:
        safety = 0.70  # 32GB+: moderately conservative

    usable = (ram_gb * safety) - reserve - overhead
    model_size = m.get("size_gb", 0)

    if mode == "pipeline":
        needed = model_size / 2  # shared across 2 machines
    elif mode == "dflash":
        draft_size = m.get("dflash", {}).get("draft_size_gb", 1.0)
        needed = model_size + draft_size  # target + draft on single machine
    else:
        needed = model_size

    if needed > usable:
        print(f"⚠️  Memory warning: {model_name} ({mode}) needs ~{needed:.1f}GB, "
              f"usable ~{usable:.1f}GB. May OOM.")
        return False
    print(f"✅ Memory OK: {model_name} ({mode}) needs ~{needed:.1f}GB, usable ~{usable:.1f}GB")
    return True


def serve(cfg, args):
    model = args.model
    mode = args.mode
    rank = args.rank

    # Validate model exists in config
    m = cfg.get("models", {}).get(model)
    if not m:
        print(f"❌ Unknown model: {model}")
        print("   Run 'hippo-pipeline list-models' to see available models.")
        sys.exit(1)

    # Validate mode
    if mode not in m.get("modes", []):
        print(f"❌ Model {model} does not support mode '{mode}'")
        print(f"   Supported: {', '.join(m.get('modes', []))}")
        sys.exit(1)

    # Memory check
    check_memory(cfg, model, mode)

    defaults = cfg.get("defaults", {})
    host = args.host or defaults.get("host", "0.0.0.0")
    port = args.port or defaults.get("port", 9998)
    rank0_host = args.rank0_host or defaults.get("rank0_host", "localhost")

    # Dispatch to appropriate runner
    if mode == "pipeline":
        if rank == 0:
            _serve_rank0_pipeline(model, host, port, rank0_host, cfg)
        else:
            _serve_rank1_pipeline(model, host, port, rank0_host, cfg)
    elif mode == "dflash":
        _serve_dflash(model, host, port, cfg)
    elif mode == "standalone":
        _serve_standalone(model, host, port, cfg)
    else:
        print(f"❌ Unknown mode: {mode}")
        sys.exit(1)


def _serve_rank0_pipeline(model, host, port, rank0_host, cfg):
    """Start Rank 0 for pipeline mode."""
    print(f"🚀 Starting R0 (pipeline) — model={model}")
    print(f"   Listening on {host}:{port+1}")
    print("   R1 will connect from configured IP")
    os.environ["PYTHONUNBUFFERED"] = "1"
    # Import and run rank0
    import asyncio

    from rank0 import rank0_serve
    asyncio.run(rank0_serve(host, port, rank0_host))


def _serve_rank1_pipeline(model, host, port, rank0_host, cfg):
    """Start Rank 1 for pipeline mode."""
    print(f"🚀 Starting R1 (pipeline) — model={model}")
    print(f"   Connecting to R0 at {rank0_host}:{port+1}")
    os.environ["PYTHONUNBUFFERED"] = "1"
    import asyncio

    from rank1 import rank1_serve
    asyncio.run(rank1_serve(host, port, rank0_host))


def _serve_dflash(model, host, port, cfg):
    """Start single-machine DFlash mode."""
    m = cfg.get("models", {}).get(model, {})
    dflash_cfg = m.get("dflash", {})

    # Resolve model paths
    target_model = os.path.expanduser(f"~/.cache/modelscope/{m.get('repo', 'Qwen/Qwen3-4B')}")
    draft_repo = dflash_cfg.get("draft_repo", "z-lab/Qwen3-4B-DFlash-b16")
    draft_model = os.path.expanduser(f"~/.cache/modelscope/{draft_repo}")

    print(f"🚀 Starting DFlash (single) — model={model}")
    print(f"   Target: {target_model}")
    print(f"   Draft:  {draft_model}")
    print(f"   Expected: ~{dflash_cfg.get('expected_tok_s', '?')} tok/s")
    os.environ["PYTHONUNBUFFERED"] = "1"
    from rank0_dflash import dflash_serve
    asyncio.run(dflash_serve(
        host=host, port=port,
        target_model=target_model,
        draft_model=draft_model,
        interactive=True,
    ))


def _serve_standalone(model, host, port, cfg):
    """Start single-machine standalone (no acceleration)."""
    print(f"🚀 Starting standalone — model={model}")
    print(f"   Run: python3 sharded_inference.py --model {model}")
    os.execv(sys.executable, [sys.executable, "sharded_inference.py", "--model", model])


def benchmark(cfg, args):
    """Run benchmark."""
    print(f"📊 Benchmark — model={args.model}, mode={args.mode}, runs={args.runs}")
    # Delegate to benchmark.py
    bench_args = ["python3", os.path.join(SCRIPT_DIR, "benchmark.py"),
                  "--model", args.model, "--runs", str(args.runs)]
    os.execv(sys.executable, bench_args)


def main():
    parser = argparse.ArgumentParser(
        prog="hippo-pipeline",
        description="Hippo Pipeline — Distributed LLM inference on Apple Silicon"
    )
    sub = parser.add_subparsers(dest="command")

    # serve
    serve_p = sub.add_parser("serve", help="Start inference server")
    serve_p.add_argument("--model", required=True, help="Model name (e.g. gemma-3-12b, qwen3-4b)")
    serve_p.add_argument("--mode", required=True,
                         choices=["standalone", "pipeline", "dflash"],
                         help="Inference mode")
    serve_p.add_argument("--rank", type=int, default=0,
                         help="Rank (0 or 1, only for pipeline mode)")
    serve_p.add_argument("--host", default=None)
    serve_p.add_argument("--port", type=int, default=None)
    serve_p.add_argument("--rank0-host", default=None,
                         help="R0 IP address (for R1 connection)")
    serve_p.add_argument("--prompt", default="Hello, how are you?",
                         help="Prompt for one-shot generation")
    serve_p.add_argument("--max-tokens", type=int, default=256,
                         help="Max tokens to generate")
    serve_p.add_argument("--temperature", type=float, default=0.0)
    serve_p.add_argument("--interactive", action="store_true",
                         help="Interactive mode (stdin loop)")
    serve_p.add_argument("--thunderbolt", action="store_true",
                         help="Use Thunderbolt IP instead of Wi-Fi")

    # benchmark
    bench_p = sub.add_parser("benchmark", help="Run performance benchmark")
    bench_p.add_argument("--model", required=True)
    bench_p.add_argument("--mode", default="pipeline")
    bench_p.add_argument("--runs", type=int, default=3)

    # list-models
    sub.add_parser("list-models", help="List available models and modes")

    args = parser.parse_args()
    cfg = load_config()

    if args.command == "serve":
        serve(cfg, args)
    elif args.command == "benchmark":
        benchmark(cfg, args)
    elif args.command == "list-models":
        list_models(cfg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
