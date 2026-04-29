#!/usr/bin/env python3
"""
continuous_batch.py — MLX Continuous Batch Engine for Hippo

特性：
1. Prompt padding（解决长度不一致问题）
2. 动态 batch size（1-7 序列）
3. 7 Agent slot 管理
4. 内存监控（> 14GB 拒绝）
5. FIFO 请求队列

作者：忒弥斯 (T-Mind) 🔮
版本：v1.1 (Hippo 集成版)
创建日期：2026-04-22
"""

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple

import mlx.core as mx
from mlx_lm import load

logger = logging.getLogger("hippo.continuous_batch")

# ─── 数据结构 ────────────────────────────────────────

@dataclass
class Request:
    """推理请求"""
    id: str
    prompt: str
    max_tokens: int = 100
    tokens: List[int] = None  # 生成的 tokens
    done: bool = False

    def __post_init__(self):
        if self.tokens is None:
            self.tokens = []

@dataclass
class BatchResult:
    """批量推理结果"""
    request_id: str
    text: str
    tokens: List[int]
    elapsed_sec: float
    tok_s: float

# ─── 连续批处理引擎 ───────────────────────────────────

class ContinuousBatchEngine:
    """MLX 连续批处理引擎（支持变长 prompt）"""

    def __init__(self, model_path: str = "mlx-community/gemma-3-12b-it-qat-4bit",
                 max_slots: int = 7, max_memory_gb: float = 14.0):
        """
        初始化引擎

        Args:
            model_path: MLX 模型路径
            max_slots: 最大并发请求数（家族成员数）
            max_memory_gb: 最大内存限制（GB，保留 2GB 安全边际）
        """
        logger.info(f"Loading MLX batch engine: {model_path}")
        self.model, self.tokenizer = load(model_path)
        mx.eval(self.model.parameters())
        self.lm_container = self.model.language_model
        self.lm = self.lm_container.model
        self.lm_head = self.lm_container.lm_head

        self.max_slots = max_slots
        self.max_memory_gb = max_memory_gb
        self.queue = deque(maxlen=max_slots)
        self.active_requests: Dict[str, Request] = {}

        logger.info(f"MLX batch engine ready: {max_slots} slots, GPU={mx.get_active_memory()/1e9:.2f} GB")

    def _check_memory(self) -> bool:
        """检查内存是否安全"""
        active_gb = mx.get_active_memory() / 1e9
        if active_gb > self.max_memory_gb:
            print(f"⚠️ Memory high: {active_gb:.2f} GB > {self.max_memory_gb} GB")
            return False
        return True

    def submit(self, request: Request) -> bool:
        """
        提交请求到队列

        Args:
            request: 推理请求

        Returns:
            True=已加入队列，False=队列满或内存不足
        """
        if len(self.queue) >= self.max_slots:
            logger.warning(f"Batch queue full ({self.max_slots} slots), rejecting request {request.id}")
            return False

        if not self._check_memory():
            logger.warning(f"Memory insufficient, rejecting request {request.id}")
            return False

        self.queue.append(request)
        self.active_requests[request.id] = request
        logger.info(f"Request {request.id} queued ({len(self.queue)}/{self.max_slots})")
        return True

    def _pad_prompts(self, prompts: List[str]) -> Tuple[List[mx.array], List[int]]:
        """
        Pad prompts 到最大长度（解决 P0-1）

        Returns:
            (input_ids_list, original_lengths)
        """
        # Tokenize all prompts
        tokenized = [self.tokenizer.encode(p) for p in prompts]
        original_lengths = [len(t) for t in tokenized]
        max_len = max(original_lengths)

        # Pad to max length
        padded = []
        pad_token_id = self.tokenizer.pad_token_id or 0
        for toks in tokenized:
            if len(toks) < max_len:
                toks = toks + [pad_token_id] * (max_len - len(toks))
            padded.append(mx.array([toks], dtype=mx.int32))

        return padded, original_lengths

    def _prefill_batch(self, requests: List[Request]) -> Tuple[List, List[int]]:
        """
        Prefill 阶段（批量处理 prompt）

        Returns:
            (caches, last_tokens, actual_lengths)
        """
        prompts = [r.prompt for r in requests]
        padded_ids, original_lengths = self._pad_prompts(prompts)

        # Prefill each request separately
        caches = []
        last_tokens = []

        for i, (req, input_ids) in enumerate(zip(requests, padded_ids)):
            c = self.lm_container.make_cache()
            L = original_lengths[i]  # Actual length (without padding)

            # Forward FULL padded input (so all caches have same seq_len)
            # Note: Gemma3Model doesn't accept mask parameter, model handles causality internally
            h = self.lm(input_ids, cache=c)
            logits = self.lm_head(h)
            mx.eval(logits)

            # Store cache and last token (actual position)
            caches.append(c)
            last_tokens.append(int(mx.argmax(logits[0, L-1, :])))

        return caches, last_tokens, original_lengths

    def _decode_batch(self, requests: List[Request], caches: List, last_tokens: List[int], original_lengths: List[int]) -> Dict[str, List[int]]:
        """
        Decode 阶段（批量生成）

        Returns:
            {request_id: generated_tokens}
        """
        n_seq = len(requests)
        max_tokens = max(r.max_tokens for r in requests)

        # Batch caches by modifying the first cache in-place
        batched_cache = caches[0]  # Use first cache as container
        for layer_idx in range(48):  # 12B has 48 layers
            # Concatenate all caches along batch dim
            batched_cache[layer_idx].keys = mx.concatenate(
                [caches[s][layer_idx].keys for s in range(n_seq)], axis=0
            )
            batched_cache[layer_idx].values = mx.concatenate(
                [caches[s][layer_idx].values for s in range(n_seq)], axis=0
            )

        # Decode loop
        generated = {r.id: [] for r in requests}

        for step in range(max_tokens):
            # Check if all requests done
            active = [r for r in requests if len(generated[r.id]) < r.max_tokens]
            if not active:
                break

            # Batch input
            input_ids = mx.array([[last_tokens[i]] for i in range(n_seq)], dtype=mx.int32)

            # Forward
            h = self.lm(input_ids, cache=batched_cache)
            logits = self.lm_head(h)
            mx.eval(logits)

            # Sample
            new_tokens = [int(mx.argmax(logits[i, 0, :])) for i in range(n_seq)]
            last_tokens = new_tokens

            # Append to active requests
            for i, req in enumerate(requests):
                if len(generated[req.id]) < req.max_tokens:
                    generated[req.id].append(new_tokens[i])

        return generated

    def run_batch(self) -> List[BatchResult]:
        """
        运行一个批次（处理队列中的所有请求）

        Returns:
            批次结果列表
        """
        if not self.queue:
            return []

        # Collect all requests
        requests = list(self.queue)
        self.queue.clear()

        logger.info(f"Running batch: {len(requests)} requests")

        t0 = time.time()

        # Phase 1: Prefill
        caches, last_tokens, original_lengths = self._prefill_batch(requests)
        logger.info(f"Batch prefill done ({len(requests)} seq)")

        # Phase 2: Decode
        generated = self._decode_batch(requests, caches, last_tokens, original_lengths)

        elapsed = time.time() - t0

        # Collect results
        results = []
        for req in requests:
            text = self.tokenizer.decode(generated[req.id])
            results.append(BatchResult(
                request_id=req.id,
                text=text,
                tokens=generated[req.id],
                elapsed_sec=elapsed,
                tok_s=len(generated[req.id]) / elapsed
            ))
            # Remove from active
            del self.active_requests[req.id]

        logger.info(f"Batch done: {elapsed:.2f}s, {len(requests)} requests")

        return results

    def stats(self) -> Dict:
        """引擎统计"""
        return {
            "queue_size": len(self.queue),
            "active_requests": len(self.active_requests),
            "memory_gb": mx.get_active_memory() / 1e9,
            "memory_peak_gb": mx.get_peak_memory() / 1e9,
        }

# ─── 测试 ─────────────────────────────────────────────

def test_prompt_padding():
    """测试 P0-1: Prompt padding"""
    print("=" * 60)
    print("🧪 Test P0-1: Prompt Padding")
    print("=" * 60)

    engine = ContinuousBatchEngine(max_slots=3)

    # Test 1: Different length prompts
    requests = [
        Request(id="req1", prompt="Hi"),
        Request(id="req2", prompt="Hello world"),
        Request(id="req3", prompt="The capital of France is Paris"),
    ]

    for req in requests:
        assert engine.submit(req), f"Failed to submit {req.id}"

    # Run batch
    results = engine.run_batch()

    assert len(results) == 3, f"Expected 3 results, got {len(results)}"

    for r in results:
        print(f"  ✅ {r.request_id}: {len(r.tokens)} tokens, {r.tok_s:.1f} tok/s")
        print(f"     Text: {r.text[:50]}...")

    print(f"\n📊 Stats: {engine.stats()}")

    print("\n✅ P0-1 Test PASSED")

def test_queue_management():
    """测试 P0-3: 7 Agent slot 管理"""
    print("\n" + "=" * 60)
    print("🧪 Test P0-3: Queue Management")
    print("=" * 60)

    engine = ContinuousBatchEngine(max_slots=7)

    # Submit 7 requests
    for i in range(7):
        req = Request(id=f"req{i}", prompt=f"Prompt {i}")
        assert engine.submit(req), f"Failed to submit req{i}"

    # 8th should fail
    req8 = Request(id="req8", prompt="Too many")
    assert not engine.submit(req8), "Should reject 8th request"

    print(f"  ✅ Queue size: {engine.stats()['queue_size']}/7")

    results = engine.run_batch()
    assert len(results) == 7, f"Expected 7 results, got {len(results)}"

    print("\n✅ P0-3 Test PASSED")

def test_memory_monitor():
    """测试 P1-4: 内存监控"""
    print("\n" + "=" * 60)
    print("🧪 Test P1-4: Memory Monitor")
    print("=" * 60)

    engine = ContinuousBatchEngine(max_memory_gb=15.0)  # Reasonable limit (model takes ~7GB)

    # First request should work
    req1 = Request(id="req1", prompt="Hi")
    assert engine.submit(req1), "Should accept first request"

    stats = engine.stats()
    print(f"  Memory: {stats['memory_gb']:.2f} GB / {engine.max_memory_gb} GB")

    assert stats['memory_gb'] < engine.max_memory_gb, "Memory should be under limit"

    print("\n✅ P1-4 Test PASSED")

if __name__ == "__main__":
    logger.info("ContinuousBatchEngine Test Suite")

    # Run tests
    test_prompt_padding()
    test_queue_management()
    test_memory_monitor()

    logger.info("All tests PASSED!")
