#!/usr/bin/env python3
"""LLM Memory Budget Calculator — check if a model fits before loading.

Prevents OOM crashes by calculating memory requirements upfront.
Supports: single, dual-rpc, multi-rpc, pipeline (MLX Pipeline Parallelism).

Usage:
    python3 memory_budget.py --model-size 15.5 --mode single
    python3 memory_budget.py --model-size 15.5 --mode pipeline --world-size 2
    python3 memory_budget.py --model-size 15.5 --mode dual-rpc --tensor-split 0.5,0.5
    python3 memory_budget.py --model-path model.gguf --mode pipeline --world-size 2 --json
"""

import argparse
import json
import os
import subprocess
import sys


def get_available_memory_gb() -> float:
    """Get available memory (free + inactive) in GB. macOS only."""
    try:
        result = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5)
        lines = result.stdout.strip().split("\n")
        page_size = 16384  # macOS ARM64
        free_pages = inactive_pages = 0
        for line in lines:
            if "Pages free" in line:
                free_pages = int(line.split(":")[1].strip().rstrip("."))
            elif "Pages inactive" in line:
                inactive_pages = int(line.split(":")[1].strip().rstrip("."))
        return (free_pages + inactive_pages) * page_size / (1024 ** 3)
    except Exception:
        return 4.0  # P1-3: conservative fallback (was 13.0)


def get_total_memory_gb() -> float:
    """Get total physical memory in GB."""
    try:
        result = subprocess.run(["sysctl", "hw.memsize"], capture_output=True, text=True, timeout=5)
        return int(result.stdout.split(":")[1].strip()) / (1024 ** 3)
    except Exception:
        return 16.0


def estimate_kv_cache_gb(n_ctx: int = 2048, n_layers: int = 32, hidden_dim: int = 4096) -> float:
    """KV cache ≈ 2 * n_layers * n_ctx * hidden_dim * 2 bytes (FP16)."""
    return 2 * n_layers * n_ctx * hidden_dim * 2 / (1024 ** 3)


def get_model_size_gb(model_path: str) -> float:
    if model_path and os.path.isfile(model_path):
        return os.path.getsize(model_path) / (1024 ** 3)
    return 0.0


def calculate_budget(model_size_gb, mode="single", tensor_split=None,
                     world_size=2, n_ctx=2048, n_layers=32, hidden_dim=4096):
    available = get_available_memory_gb()
    total = get_total_memory_gb()
    kv_cache = estimate_kv_cache_gb(n_ctx, n_layers, hidden_dim)
    system_overhead = 1.5  # GB

    if mode == "pipeline":
        # MLX Pipeline: only loads assigned layers, NO full mmap
        per_device = model_size_gb / world_size
        embedding_overhead = max(0.2, model_size_gb * 0.05)  # P1-2: scale with model size
        peak = per_device * 1.3 + embedding_overhead + system_overhead
        steady = per_device + kv_cache + system_overhead
        feasible = peak < available
        deficit = max(0, peak - available)
        return {
            "model_size_gb": round(model_size_gb, 1),
            "mode": "pipeline",
            "world_size": world_size,
            "memory": {
                "total_physical_gb": round(total, 1),
                "available_gb": round(available, 1),
                "per_device_model_gb": round(per_device, 1),
                "loading_peak_gb": round(peak, 1),
                "steady_state_gb": round(steady, 1),
                "kv_cache_gb": round(kv_cache, 2),
            },
            "feasible": feasible,
            "deficit_gb": round(deficit, 1),
            "recommendation": _pipeline_recommendation(feasible, deficit, peak, available, world_size),
        }

    # --- llama.cpp modes (single / dual-rpc / multi-rpc) ---
    if mode == "single":
        local_ratio = 1.0
        n_workers = 0
    else:
        if tensor_split is None:
            tensor_split = [0.5, 0.5]
        local_ratio = tensor_split[0]
        n_workers = len(tensor_split) - 1

    # CRITICAL: mmap peak = FULL file, regardless of tensor_split
    # P1-1 fix: single mode also has 1.2x loading peak
    real_peak = model_size_gb * 1.2 + kv_cache + system_overhead
    local_model = model_size_gb * local_ratio
    steady_state = local_model + kv_cache + system_overhead

    feasible = real_peak < available
    deficit = max(0, real_peak - available)

    return {
        "model_size_gb": round(model_size_gb, 1),
        "mode": mode,
        "tensor_split": tensor_split,
        "local_ratio": local_ratio,
        "n_rpc_workers": n_workers,
        "memory": {
            "total_physical_gb": round(total, 1),
            "available_gb": round(available, 1),
            "mmap_peak_gb": round(model_size_gb, 1),
            "loading_peak_gb": round(real_peak, 1),
            "steady_state_gb": round(steady_state, 1),
            "kv_cache_gb": round(kv_cache, 2),
        },
        "feasible": feasible,
        "deficit_gb": round(deficit, 1),
        "recommendation": _rpc_recommendation(feasible, deficit, mode, model_size_gb, available),
    }


def _pipeline_recommendation(feasible, deficit, peak, available, world_size):
    if feasible:
        return f"✅ Pipeline OK. Per-device peak ~{peak:.1f}GB, available {available:.1f}GB."
    return (f"❌ Pipeline NOT feasible. Need {deficit:.1f}GB more per device. "
            f"Try: smaller quant, more devices (current: {world_size}), or stop other services.")


def _rpc_recommendation(feasible, deficit, mode, model_size, available):
    if feasible:
        return f"✅ Model should load. Peak ~{model_size + 2:.1f}GB, available {available:.1f}GB."
    if model_size > available:
        return (f"❌ GGUF mmap ({model_size:.1f}GB) exceeds available ({available:.1f}GB). "
                f"Even RPC won't help — mmap loads FULL file. "
                f"Solutions: (1) smaller quant, (2) stop services, (3) use MLX Pipeline mode.")
    return (f"⚠️ Tight budget. Peak {model_size + 2:.1f}GB vs {available:.1f}GB. "
            f"May work if other services stopped. Recommend: stop Ollama, close apps.")


def print_report(budget):
    m = budget["memory"]
    print("=" * 60)
    print("LLM Memory Budget Report")
    print("=" * 60)
    print(f"  Model size:      {budget['model_size_gb']:.1f} GB")
    print(f"  Mode:            {budget['mode']}")
    if budget["mode"] == "pipeline":
        print(f"  World size:      {budget['world_size']}")
        print(f"  Per device:      {m['per_device_model_gb']:.1f} GB")
    else:
        if budget.get("tensor_split"):
            print(f"  Tensor split:    {budget['tensor_split']}")
            print(f"  Local ratio:     {budget.get('local_ratio', 1) * 100:.0f}%")
    print()
    print("Memory:")
    print(f"  Physical RAM:    {m['total_physical_gb']:.1f} GB")
    print(f"  Available:       {m['available_gb']:.1f} GB")
    print(f"  Loading peak:    {m['loading_peak_gb']:.1f} GB")
    print(f"  Steady state:    {m['steady_state_gb']:.1f} GB")
    print(f"  KV cache:        {m['kv_cache_gb']:.2f} GB")
    if budget["mode"] != "pipeline":
        print(f"  Mmap peak:       {m['mmap_peak_gb']:.1f} GB ⚠️ (full file)")
    print()
    if budget["feasible"]:
        print("  ✅ FEASIBLE")
    else:
        print(f"  ❌ NOT FEASIBLE — Deficit: {budget['deficit_gb']:.1f} GB")
    print(f"\n  {budget['recommendation']}")
    print("=" * 60)


def main():
    p = argparse.ArgumentParser(description="LLM Memory Budget Calculator")
    p.add_argument("--model-size", type=float, help="Model size in GB")
    p.add_argument("--model-path", type=str, help="Path to model file (auto-detect size)")
    p.add_argument("--mode", choices=["single", "dual-rpc", "multi-rpc", "pipeline"], default="single")
    p.add_argument("--tensor-split", type=str, help="Comma-separated ratios, e.g. 0.5,0.5")
    p.add_argument("--world-size", type=int, default=2, help="Devices for pipeline mode")
    p.add_argument("--n-ctx", type=int, default=2048)
    p.add_argument("--n-layers", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=4096)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    model_size = args.model_size or 0
    if args.model_path:
        model_size = get_model_size_gb(args.model_path)
    if not model_size:
        print("Error: provide --model-size or --model-path", file=sys.stderr)
        sys.exit(1)

    tensor_split = [float(x) for x in args.tensor_split.split(",")] if args.tensor_split else None

    budget = calculate_budget(
        model_size_gb=model_size, mode=args.mode, tensor_split=tensor_split,
        world_size=args.world_size, n_ctx=args.n_ctx,
        n_layers=args.n_layers, hidden_dim=args.hidden_dim,
    )

    if args.json:
        print(json.dumps(budget, indent=2))
    else:
        print_report(budget)
    sys.exit(0 if budget["feasible"] else 1)


if __name__ == "__main__":
    main()
