#!/usr/bin/env python3
"""
R0 Draft Runner — DFlash 集成 PoC

功能：
1. 加载 target model（Qwen3-27B-4bit 前半层）
2. 加载 draft model（DFlash）
3. Draft 生成 + 验证
4. 输出验证通过的 tokens 到 R1

作者：忒弥斯 🔮 + Code 💻
创建时间：2026-04-27 15:55
"""

from __future__ import annotations

import argparse
import json
import os

# DFlash 依赖
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parents[2] / "research" / "dflash-mlx"))
from dflash_mlx.api import DFlashGenerator


@dataclass
class DraftResult:
    """Draft 生成结果"""
    prompt_tokens: int
    draft_tokens: int
    accepted_tokens: int
    acceptance_rate: float
    draft_time_s: float
    verify_time_s: float
    total_time_s: float
    peak_memory_gb: float
    output_tokens: list[int]
    output_text: str


class R0DraftRunner:
    """R0 Draft Runner（DFlash 集成）"""

    def __init__(
        self,
        target_model: str,
        draft_model: str,
        draft_block_size: int = 16,
        verify_mode: str = "parallel-replay",
    ):
        """
        初始化 R0 Draft Runner

        Args:
            target_model: Target model 路径（Qwen3-27B-4bit）
            draft_model: Draft model 路径（DFlash draft）
            draft_block_size: Draft block size（默认 16）
            verify_mode: 验证模式（parallel-replay 推荐）
        """
        self.target_model = target_model
        self.draft_model = draft_model
        self.draft_block_size = draft_block_size
        self.verify_mode = verify_mode

        print("[R0] Loading DFlash generator...")
        print(f"[R0]   Target: {target_model}")
        print(f"[R0]   Draft: {draft_model}")

        self.generator = DFlashGenerator(
            target_model=target_model,
            draft_model=draft_model,
        )

        print("[R0] Generator loaded!")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> DraftResult:
        """
        Draft 生成 + 验证

        Args:
            prompt: 输入 prompt
            max_new_tokens: 最大生成 tokens
            temperature: 温度（0.0 = 贪婪）

        Returns:
            DraftResult
        """
        print("[R0] Generating draft...")
        print(f"[R0]   Prompt length: {len(prompt)} chars")
        print(f"[R0]   Max tokens: {max_new_tokens}")
        print(f"[R0]   Temperature: {temperature}")

        # 重置内存峰值
        mx.reset_peak_memory()

        # 计时
        start_time = time.perf_counter()

        # 生成
        result = self.generator.generate(
            prompt_text=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            speculative_tokens=self.draft_block_size,
            verify_mode=self.verify_mode,
            reset_peak_memory=True,
        )

        elapsed = time.perf_counter() - start_time

        # 提取指标
        metrics = result.metrics
        peak_memory_gb = mx.get_peak_memory() / 1024**3

        draft_result = DraftResult(
            prompt_tokens=metrics.get("prompt_tokens", 0),
            draft_tokens=sum(metrics.get("acceptance_lengths", [])),
            accepted_tokens=len(result.generated_tokens),
            acceptance_rate=metrics.get("avg_acceptance_length", 0) / self.draft_block_size,
            draft_time_s=metrics.get("prefill_time_s", 0),
            verify_time_s=metrics.get("decode_time_s", 0),
            total_time_s=elapsed,
            peak_memory_gb=peak_memory_gb,
            output_tokens=result.generated_tokens,
            output_text=result.text,
        )

        return draft_result

    def benchmark(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        num_runs: int = 3,
    ) -> dict[str, Any]:
        """
        Benchmark（多次运行）

        Args:
            prompt: 输入 prompt
            max_new_tokens: 最大生成 tokens
            temperature: 温度
            num_runs: 运行次数

        Returns:
            Benchmark 结果
        """
        print("[R0] Benchmarking...")
        print(f"[R0]   Runs: {num_runs}")
        print(f"[R0]   Max tokens: {max_new_tokens}")

        results = []

        for i in range(num_runs):
            print(f"\n[R0] Run {i+1}/{num_runs}")

            result = self.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )

            results.append(result)

            # 打印单次结果
            print(f"[R0]   Generated: {result.accepted_tokens} tokens")
            print(f"[R0]   Time: {result.total_time_s:.2f}s")
            print(f"[R0]   tok/s: {result.accepted_tokens / result.total_time_s:.2f}")
            print(f"[R0]   AR: {result.acceptance_rate:.2%}")
            print(f"[R0]   Memory: {result.peak_memory_gb:.2f}GB")

        # 汇总统计
        summary = {
            "num_runs": num_runs,
            "avg_tokens": sum(r.accepted_tokens for r in results) / num_runs,
            "avg_time_s": sum(r.total_time_s for r in results) / num_runs,
            "avg_tps": sum(r.accepted_tokens / r.total_time_s for r in results) / num_runs,
            "avg_ar": sum(r.acceptance_rate for r in results) / num_runs,
            "max_memory_gb": max(r.peak_memory_gb for r in results),
            "results": [
                {
                    "tokens": r.accepted_tokens,
                    "time_s": r.total_time_s,
                    "tps": r.accepted_tokens / r.total_time_s,
                    "ar": r.acceptance_rate,
                    "memory_gb": r.peak_memory_gb,
                }
                for r in results
            ],
        }

        # 打印汇总
        print("\n[R0] === Benchmark Summary ===")
        print(f"[R0] Avg tokens: {summary['avg_tokens']:.1f}")
        print(f"[R0] Avg time: {summary['avg_time_s']:.2f}s")
        print(f"[R0] Avg tok/s: {summary['avg_tps']:.2f}")
        print(f"[R0] Avg AR: {summary['avg_ar']:.2%}")
        print(f"[R0] Max memory: {summary['max_memory_gb']:.2f}GB")

        return summary


def main():
    parser = argparse.ArgumentParser(description="R0 Draft Runner — DFlash PoC")
    parser.add_argument(
        "--target-model",
        default="/Users/deepsearch/.cache/modelscope/Qwen/Qwen3-4B",
        help="Target model path",
    )
    parser.add_argument(
        "--draft-model",
        default="/Users/deepsearch/.cache/modelscope/z-lab/Qwen3-4B-DFlash-b16",
        help="Draft model path",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Write a Python function to implement quicksort.",
        help="Input prompt",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Max new tokens",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=3,
        help="Benchmark runs",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON",
    )

    args = parser.parse_args()

    # 创建 runner
    runner = R0DraftRunner(
        target_model=args.target_model,
        draft_model=args.draft_model,
    )

    # Benchmark
    summary = runner.benchmark(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        num_runs=args.num_runs,
    )

    # Crit 条件检查
    print("\n[R0] === Crit 条件检查 ===")

    # 条件 1: 内存余量 ≥ 1.5GB
    # MLX 的 get_peak_memory() 已经包含模型内存
    # 可用余量 = 总内存 - 系统预留 - MLX 峰值内存
    total_memory_gb = 48.0  # M4 Max
    system_reserved_gb = 20.0  # 系统预留（macOS + 窗口服务）
    mlx_peak_memory_gb = summary["max_memory_gb"]  # MLX 峰值（包含模型）
    available_memory_gb = total_memory_gb - system_reserved_gb - mlx_peak_memory_gb
    memory_ok = available_memory_gb >= 1.5

    print("[R0] 条件 1: 内存余量 ≥ 1.5GB")
    print(f"[R0]   总内存: {total_memory_gb:.1f}GB")
    print(f"[R0]   系统预留: {system_reserved_gb:.1f}GB")
    print(f"[R0]   MLX 峰值: {mlx_peak_memory_gb:.2f}GB（包含模型）")
    print(f"[R0]   可用余量: {available_memory_gb:.1f}GB")
    print(f"[R0]   结果: {'✅ 通过' if memory_ok else '❌ 失败'}")

    # 条件 2: Draft 验证延迟 < 150ms/step
    avg_step_time_ms = (summary["avg_time_s"] / summary["avg_tokens"]) * 1000
    latency_ok = avg_step_time_ms < 150

    print("\n[R0] 条件 2: Draft 验证延迟 < 150ms/step")
    print(f"[R0]   平均延迟: {avg_step_time_ms:.1f}ms/step")
    print(f"[R0]   结果: {'✅ 通过' if latency_ok else '❌ 失败'}")

    # 条件 3: Acceptance rate ≥ 60%
    ar_ok = summary["avg_ar"] >= 0.6

    print("\n[R0] 条件 3: Acceptance rate ≥ 60%")
    print(f"[R0]   平均 AR: {summary['avg_ar']:.2%}")
    print(f"[R0]   结果: {'✅ 通过' if ar_ok else '❌ 失败'}")

    # 总体结果
    all_ok = memory_ok and latency_ok and ar_ok
    print("\n[R0] === 总体结果 ===")
    print(f"[R0] Crit 条件: {'✅ 全部通过' if all_ok else '❌ 部分失败'}")

    if args.json:
        summary["crit_check"] = {
            "memory_ok": memory_ok,
            "latency_ok": latency_ok,
            "ar_ok": ar_ok,
            "all_ok": all_ok,
        }
        print(json.dumps(summary, indent=2))


async def dflash_serve(
    host: str = "0.0.0.0",
    port: int = 9998,
    target_model: str | None = None,
    draft_model: str | None = None,
    prompt: str = "Hello, how are you?",
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    interactive: bool = False,
):
    """
    DFlash serve mode — single-machine accelerated inference.

    Can be used as:
      1. One-shot generation (interactive=False)
      2. Interactive loop (interactive=True, reads from stdin)
    """
    # Defaults from config
    if target_model is None:
        target_model = os.path.expanduser("~/.cache/modelscope/Qwen/Qwen3-4B")
    if draft_model is None:
        draft_model = os.path.expanduser("~/.cache/modelscope/z-lab/Qwen3-4B-DFlash-b16")

    runner = R0DraftRunner(target_model=target_model, draft_model=draft_model)

    if not interactive:
        # One-shot mode
        result = runner.generate(prompt=prompt, max_new_tokens=max_new_tokens, temperature=temperature)
        print(f"\n{'='*60}")
        print(f"📝 Output: {result.output_text}")
        print(f"   Tokens: {result.accepted_tokens} | AR: {result.acceptance_rate:.2%} | "
              f"Time: {result.total_time_s:.2f}s | "
              f"Speed: {result.accepted_tokens/result.total_time_s:.1f} tok/s")
        print(f"{'='*60}")
        return result

    # Interactive mode
    print("\n🚀 DFlash interactive mode (Ctrl+C to quit)")
    print(f"   Model: {target_model}")
    print("   Speed: ~42 tok/s expected")
    print()

    while True:
        try:
            user_input = input("👤 ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("👋 Bye!")
                break

            result = runner.generate(
                prompt=user_input,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            print(f"🔮 {result.output_text}")
            print(f"   [{result.accepted_tokens} tokens, {result.accepted_tokens/result.total_time_s:.1f} tok/s]")
            print()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Bye!")
            break


if __name__ == "__main__":
    main()
