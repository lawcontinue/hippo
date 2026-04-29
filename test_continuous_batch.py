#!/usr/bin/env python3
"""test_continuous_batch.py — verify multi-sequence batch decode works.

Tests:
1. Single request baseline (tok/s)
2. 2 concurrent requests 
3. 4 concurrent requests
4. 7 concurrent requests (full family)

Key metric: does total throughput scale with batch size?
"""

import ctypes
import os
import time

import numpy as np

MODEL_PATH = os.path.expanduser(
    "~/.ollama/models/blobs/sha256-f8eba201522ab44b79bc54166126bfaf836111ff4cbf2d13c59c3b57da10573b"
)


def test_single_baseline():
    """Test 1: Single request baseline using create_completion."""
    from llama_cpp import Llama
    
    print("📦 Model: DeepSeek-R1-Distill-Llama-8B-Q4_K_M")
    model = Llama(model_path=MODEL_PATH, n_ctx=2048, n_batch=512, n_gpu_layers=-1, verbose=False)
    
    prompt = "The capital of France is"
    n_tokens = 100
    
    t0 = time.time()
    result = model.create_completion(prompt, max_tokens=n_tokens, temperature=0, stream=False)
    elapsed = time.time() - t0
    
    actual = result["usage"].get("completion_tokens", n_tokens)
    tok_per_s = actual / elapsed if elapsed > 0 else 0
    text = result["choices"][0]["text"]
    
    print(f"  单请求基线: {tok_per_s:.1f} tok/s ({elapsed:.2f}s, {actual} tokens)")
    print(f"  生成: {text[:80]}...")
    
    return model, tok_per_s


def test_batch_decode(model):
    """Test 2-5: Multi-sequence batch decode via llama_batch."""
    from llama_cpp import (
        llama_batch_free,
        llama_batch_init,
        llama_decode,
        llama_get_logits_ith,
        llama_memory_seq_rm,
    )
    
    ctx = model.ctx
    n_vocab = model.n_vocab()
    
    prompts = [
        "The capital of France is",
        "2 + 2 equals",
        "The largest planet is",
        "Water boils at",
        "Python is a programming",
        "The speed of light is",
        "Gravity pulls objects",
    ]
    
    results = {}
    
    for n_concurrent in [1, 2, 4, 7]:
        print(f"\n{'='*50}")
        print(f"🔬 {n_concurrent} 并发请求 — Batch Decode")
        print(f"{'='*50}")
        
        # Clear all sequences
        for sid in range(7):
            llama_memory_seq_rm(ctx, sid, 0, -1)
        
        # Phase 1: Prefill all sequences
        sequences = []
        for seq_id in range(n_concurrent):
            prompt = prompts[seq_id]
            tokens = model.tokenize(prompt.encode(), add_bos=True)
            n_prompt = len(tokens)
            
            batch = llama_batch_init(n_prompt, 0)
            for i, tok in enumerate(tokens):
                batch.token[i] = tok
                batch.pos[i] = i
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = seq_id
                batch.logits[i] = 1 if i == n_prompt - 1 else 0
            batch.n_tokens = n_prompt
            
            ret = llama_decode(ctx, batch)
            llama_batch_free(batch)
            
            if ret != 0:
                print(f"  ❌ Prefill seq {seq_id} failed: ret={ret}")
                break
            
            # Sample first token from last position logits
            logits_ptr = llama_get_logits_ith(ctx, ctypes.c_int(n_prompt - 1))
            logits = np.ctypeslib.as_array(logits_ptr, shape=(n_vocab,)).copy()
            first_tok = int(np.argmax(logits))
            
            sequences.append({
                "seq_id": seq_id,
                "pos": n_prompt,
                "generated": [first_tok],
            })
        
        # Phase 2: Batch decode
        n_decode = 100
        t0 = time.time()
        
        for step in range(n_decode):
            batch = llama_batch_init(n_concurrent, 0)
            for i, seq in enumerate(sequences):
                batch.token[i] = seq["generated"][-1]
                batch.pos[i] = seq["pos"]
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = seq["seq_id"]
                batch.logits[i] = 1
            batch.n_tokens = n_concurrent
            
            ret = llama_decode(ctx, batch)
            llama_batch_free(batch)
            
            if ret != 0:
                print(f"  ❌ Decode failed at step {step}: ret={ret}")
                break
            
            # Sample for each sequence
            for i, seq in enumerate(sequences):
                logits_ptr = llama_get_logits_ith(ctx, ctypes.c_int(i))
                logits = np.ctypeslib.as_array(logits_ptr, shape=(n_vocab,)).copy()
                tok = int(np.argmax(logits))
                seq["generated"].append(tok)
                seq["pos"] += 1
        
        elapsed = time.time() - t0
        
        # Results
        total_tokens = n_decode * n_concurrent
        total_tok_per_s = total_tokens / elapsed if elapsed > 0 else 0
        per_seq_tok_per_s = total_tok_per_s / n_concurrent
        step_ms = (elapsed / n_decode) * 1000
        
        results[n_concurrent] = {
            "total_tok_per_s": total_tok_per_s,
            "per_seq_tok_per_s": per_seq_tok_per_s,
            "step_ms": step_ms,
            "elapsed": elapsed,
        }
        
        print(f"  ✅ 总吞吐: {total_tok_per_s:.1f} tok/s")
        print(f"  ✅ 每序列: {per_seq_tok_per_s:.1f} tok/s")
        print(f"  ⏱️  Step: {step_ms:.1f}ms")
        print(f"  📊 加速比: {total_tok_per_s / results[1]['total_tok_per_s']:.2f}x (vs 1序列)")
        
        # Sample output
        for i, seq in enumerate(sequences[:2]):
            text = model.detokenize(seq["generated"][:30]).decode("utf-8", errors="replace")
            print(f"  Seq {i}: {text[:60]}...")
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 汇总")
    print(f"{'='*50}")
    print(f"  {'并发':>4s} | {'总吞吐':>8s} | {'每序列':>8s} | {'Step':>6s} | {'加速':>4s}")
    print(f"  {'─'*4}─┼─{'─'*8}─┼─{'─'*8}─┼─{'─'*6}─┼─{'─'*4}")
    base = results[1]["total_tok_per_s"]
    for n in [1, 2, 4, 7]:
        r = results[n]
        speedup = r["total_tok_per_s"] / base
        print(f"  {n:4d} | {r['total_tok_per_s']:8.1f} | {r['per_seq_tok_per_s']:8.1f} | {r['step_ms']:6.1f}ms | {speedup:.2f}x")
    
    return results


if __name__ == "__main__":
    print("=" * 50)
    print("🔬 Continuous Batching 性能测试")
    print("=" * 50)
    
    model, baseline = test_single_baseline()
    results = test_batch_decode(model)
