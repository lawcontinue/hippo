#!/usr/bin/env python3
"""LLM Memory Budget Calculator — check if a model fits before loading.

Prevents OOM crashes by calculating memory requirements upfront.
Based on real crash data from Hippo distributed inference testing.

Usage:
    python3 memory_budget.py --model-size 15.5 --mode dual-rpc --tensor-split 0.5,0.5
    python3 memory_budget.py --model-size 4.6 --mode single
    python3 memory_budget.py --model-path ~/.hippo/models/model.gguf --mode dual-rpc --rpc-workers 2
"""

import argparse
import os
import sys
import subprocess
import json
from pathlib import Path


def get_available_memory_gb() -> float:
    """Get available memory (free + inactive/reclaimable) in GB. macOS only."""
    try:
        result = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5)
        lines = result.stdout.strip().split("\n")
        page_size = 16384  # macOS ARM64
        free_pages = 0
        inactive_pages = 0
        for line in lines:
            if "Pages free" in line:
                free_pages = int(line.split(":")[1].strip().rstrip("."))
            elif "Pages inactive" in line:
                inactive_pages = int(line.split(":")[1].strip().rstrip("."))
        available_bytes = (free_pages + inactive_pages) * page_size
        return available_bytes / (1024 ** 3)
    except Exception:
        # Fallback: guess 13GB for 16GB Mac
        return 13.0


def get_total_memory_gb() -> float:
    """Get total physical memory in GB."""
    try:
        result = subprocess.run(["sysctl", "hw.memsize"], capture_output=True, text=True, timeout=5)
        mem_bytes = int(result.stdout.split(":")[1].strip())
        return mem_bytes / (1024 ** 3)
    except Exception:
        return 16.0


def estimate_kv_cache_gb(n_ctx: int, n_layers: int = 32, hidden_dim: int = 4096) -> float:
    """Estimate KV cache memory for given context length.

    KV cache ≈ 2 * n_layers * n_ctx * hidden_dim * 2 bytes (FP16)
    """
    kv_bytes = 2 * n_layers * n_ctx * hidden_dim * 2
    return kv_bytes / (1024 ** 3)


def get_model_size_gb(model_path: str) -> float:
    """Get GGUF model file size in GB."""
    if model_path and os.path.isfile(model_path):
        return os.path.getsize(model_path) / (1024 ** 3)
    return 0.0


# Known model memory profiles (from real testing)
MODEL_PROFILES = {
    "deepseek-r1-8b-q4": {"size_gb": 4.6, "n_layers": 32, "hidden": 4096, "tested_single": True},
    "qwen3.6-35b-a3b-q3km": {"size_gb": 15.5, "n_layers": 64, "hidden": 2048, "tested_single": False},
    "qwen3.6-35b-a3b-q4km": {"size_gb": 20.6, "n_layers": 64, "hidden": 2048, "tested_single": False},
}


def calculate_budget(
    model_size_gb: float,
    mode: str = "single",
    tensor_split: list[float] = None,
    n_ctx: int = 2048,
    n_layers: int = 32,
    hidden_dim: int = 4096,
    peak_factor: float = 1.2,
) -> dict:
    """Calculate memory budget for model loading.

    Args:
        model_size_gb: GGUF file size in GB
        mode: 'single', 'dual-rpc', or 'multi-rpc'
        tensor_split: list of ratios [local, remote1, ...]
        n_ctx: context window size
        n_layers: model layers
        hidden_dim: hidden dimension
        peak_factor: loading peak multiplier (1.2 = 20% headroom)

    Returns:
        dict with budget analysis
    """
    available = get_available_memory_gb()
    total = get_total_memory_gb()
    kv_cache = estimate_kv_cache_gb(n_ctx, n_layers, hidden_dim)

    if mode == "single":
        local_ratio = 1.0
        n_workers = 0
    elif mode == "dual-rpc":
        if tensor_split is None:
            tensor_split = [0.5, 0.5]
        local_ratio = tensor_split[0]
        n_workers = len(tensor_split) - 1
    else:  # multi-rpc
        if tensor_split is None:
            n_workers = 1
            local_ratio = 0.5
        else:
            local_ratio = tensor_split[0]
            n_workers = len(tensor_split) - 1

    # Memory calculations
    # Phase 1: mmap entire file (virtual memory)
    mmap_peak = model_size_gb

    # Phase 2: steady state after distribution
    local_model = model_size_gb * local_ratio
    steady_state = local_model + kv_cache + 1.5  # 1.5GB system overhead

    # Phase 3: loading peak (worst case)
    loading_peak = model_size_gb * peak_factor * local_ratio + kv_cache + 1.5

    # Actually, the real peak is mmap of the FULL file (even with RPC)
    # because llama-cpp mmaps the entire GGUF before splitting
    real_peak = model_size_gb + kv_cache + 1.5

    feasible = real_peak < available
    deficit = max(0, real_peak - available)

    return {
        "model_size_gb": model_size_gb,
        "mode": mode,
        "local_ratio": local_ratio,
        "n_rpc_workers": n_workers,
        "tensor_split": tensor_split,
        "memory": {
            "total_physical_gb": round(total, 1),
            "available_gb": round(available, 1),
            "mmap_peak_gb": round(mmap_peak, 1),
            "loading_peak_gb": round(real_peak, 1),
            "steady_state_gb": round(steady_state, 1),
            "kv_cache_gb": round(kv_cache, 2),
        },
        "feasible": feasible,
        "deficit_gb": round(deficit, 1),
        "recommendation": _get_recommendation(feasible, deficit, mode, model_size_gb, available),
    }


def _get_recommendation(feasible: bool, deficit: float, mode: str, model_size: float, available: float) -> str:
    if feasible:
        return f"✅ Model should load. Peak ~{model_size + 2:.1f}GB, available {available:.1f}GB."
    if mode == "single" and deficit > 0:
        return (
            f"❌ Not enough memory for single-machine loading. "
            f"Need {deficit:.1f}GB more. Try: dual-rpc with tensor_split=[0.3,0.7] "
            f"or use a smaller quantization."
        )
    if mode in ("dual-rpc", "multi-rpc") and deficit > 0:
        # Check if even mmap fits
        if model_size > available:
            return (
                f"❌ GGUF mmap alone ({model_size:.1f}GB) exceeds available memory ({available:.1f}GB). "
                f"Even RPC split won't help — the loading process mmaps the FULL file. "
                f"Solutions: (1) Use smaller quant (IQ3_XXS instead of Q3_K_M), "
                f"(2) Stop other services (Ollama, etc.) to free RAM, "
                f"(3) Use llama-cli instead of llama-cpp-python (lower overhead)."
            )
        return (
            f"⚠️ Tight budget. Peak {model_size + 2:.1f}GB vs {available:.1f}GB available. "
            f"May work if other services are stopped first. "
            f"Recommend: stop Ollama, use llama-cli, close other apps."
        )
    return "Unknown situation."


def print_report(budget: dict):
    """Print human-readable budget report."""
    m = budget["memory"]
    print("=" * 60)
    print("LLM Memory Budget Report")
    print("=" * 60)
    print(f"  Model size:      {budget['model_size_gb']:.1f} GB")
    print(f"  Mode:            {budget['mode']}")
    if budget["tensor_split"]:
        print(f"  Tensor split:    {budget['tensor_split']}")
        print(f"  Local ratio:     {budget['local_ratio']*100:.0f}%")
        print(f"  RPC workers:     {budget['n_rpc_workers']}")
    print()
    print("Memory Budget:")
    print(f"  Physical RAM:    {m['total_physical_gb']:.1f} GB")
    print(f"  Available:       {m['available_gb']:.1f} GB (free + reclaimable)")
    print(f"  KV cache:        {m['kv_cache_gb']:.2f} GB")
    print(f"  Mmap peak:       {m['mmap_peak_gb']:.1f} GB ⚠️ (full file mapped)")
    print(f"  Loading peak:    {m['loading_peak_gb']:.1f} GB")
    print(f"  Steady state:    {m['steady_state_gb']:.1f} GB")
    print()
    if budget["feasible"]:
        print(f"  ✅ FEASIBLE — Peak fits in available memory")
    else:
        print(f"  ❌ NOT FEASIBLE — Deficit: {budget['deficit_gb']:.1f} GB")
    print()
    print(f"  {budget['recommendation']}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="LLM Memory Budget Calculator")
    parser.add_argument("--model-size", type=float, help="GGUF model size in GB")
    parser.add_argument("--model-path", type=str, help="Path to GGUF file (auto-detect size)")
    parser.add_argument("--mode", choices=["single", "dual-rpc", "multi-rpc"], default="single")
    parser.add_argument("--tensor-split", type=str, help="Comma-separated ratios, e.g. 0.5,0.5")
    parser.add_argument("--n-ctx", type=int, default=2048, help="Context window size")
    parser.add_argument("--n-layers", type=int, default=32, help="Model layers")
    parser.add_argument("--hidden-dim", type=int, default=4096, help="Hidden dimension")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    # Get model size
    model_size = args.model_size or 0
    if args.model_path:
        model_size = get_model_size_gb(args.model_path)
    if not model_size:
        print("Error: provide --model-size or --model-path", file=sys.stderr)
        sys.exit(1)

    # Parse tensor split
    tensor_split = None
    if args.tensor_split:
        tensor_split = [float(x) for x in args.tensor_split.split(",")]

    # Calculate
    budget = calculate_budget(
        model_size_gb=model_size,
        mode=args.mode,
        tensor_split=tensor_split,
        n_ctx=args.n_ctx,
        n_layers=args.n_layers,
        hidden_dim=args.hidden_dim,
    )

    if args.json:
        print(json.dumps(budget, indent=2))
    else:
        print_report(budget)

    # Exit code: 0 = feasible, 1 = not feasible
    sys.exit(0 if budget["feasible"] else 1)


if __name__ == "__main__":
    main()
