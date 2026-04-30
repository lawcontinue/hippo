#!/usr/bin/env python3
"""
Speculative Decoding for Hippo Pipeline Parallelism.

R0 drafts K tokens using its local 24 layers + lm_head.
R1 batch-verifies all K tokens.
R0 accepts longest matching prefix.

Expected: 2.5-3.5x speedup (7.9 → 20-28 tok/s).
"""
import sys, os, time, asyncio, glob
sys.stdout.reconfigure(line_buffering=True)  # Force line buffering
sys.path.insert(0, os.path.dirname(__file__))
os.environ['MXNET_GPU_MEM_POOL_TYPE'] = '0'

import mlx.core as mx
import numpy as np
from shard import ShardMetadata
from shard_loader import load_shard_weights, load_tokenizer
from sharded_inference import (
    forward_layer, rms_norm, compress_logits_topk, sample_from_topk,
    HIDDEN_SIZE, VOCAB_SIZE, LOGITS_TOP_K, GROUP_SIZE, BITS,
    TensorSender, TensorReceiver, frame_to_mlx,
)
from tcp_transport import TensorFrame


def load_lm_head():
    """Load lm_head weights from second safetensors file."""
    cache = os.path.expanduser('~/.cache/huggingface/hub')
    model_id = 'mlx-community/gemma-3-12b-it-qat-4bit'
    model_dir = glob.glob(os.path.join(cache, f'models--{model_id.replace("/", "--")}/snapshots/*'))[0]
    f2 = sorted(glob.glob(f'{model_dir}/*.safetensors'))[1]
    extra = mx.load(f2)
    lm_w = extra['language_model.lm_head.weight']
    lm_s = extra['language_model.lm_head.scales']
    lm_b = extra['language_model.lm_head.biases']
    norm_w = extra['language_model.model.norm.weight']
    mx.eval(lm_w, lm_s, lm_b, norm_w)
    return lm_w, lm_s, lm_b, norm_w


def draft_tokens(last_token, kv_cache, weights, embed_full, lm_w, lm_s, lm_b, norm_w, 
                 offset, K=3, temperature=0.0):
    """
    Draft K tokens using R0's 24 layers + lm_head.
    Returns: (draft_tokens: list[int], draft_hiddens: list[mx.array])
    """
    draft_tokens_list = []
    draft_hiddens = []
    
    token = last_token
    for k in range(K):
        inp = mx.array([[token]])
        h = embed_full[inp] * mx.array(HIDDEN_SIZE**0.5, dtype=mx.float16)
        for i in range(0, 24):  # R0's 24 layers
            h = forward_layer(h, i, weights, kv_cache=kv_cache, offset=offset + k)
        # Save hidden state for verification (lazy - no eval yet)
        draft_hiddens.append(h)
        # Local lm_head to get draft token
        last_h = h[:, -1:, :]
        last_h = rms_norm(last_h, norm_w)
        logits = mx.quantized_matmul(last_h, lm_w, lm_s, lm_b, group_size=GROUP_SIZE, bits=BITS)
        mx.eval(logits)
        mx.synchronize()
        # Greedy sample for draft
        token = int(mx.argmax(logits, axis=-1).item())
        draft_tokens_list.append(token)
    
    return draft_tokens_list, draft_hiddens


async def rank0_sd_main(host='169.254.2.225', port=9998, prompt='The capital of France is', 
                        max_tokens=50, K=3, temperature=0.0):
    """R0 main with Speculative Decoding."""
    
    # Load R0 weights
    print(f"=== R0 Speculative Decoding (K={K}) ===")
    t0 = time.time()
    shard = ShardMetadata(model_id='mlx-community/gemma-3-12b-it-qat-4bit', 
                          start_layer=0, end_layer=24, n_layers=48, device_rank=0, world_size=2)
    weights = load_shard_weights(shard.model_id, shard)
    
    # Load lm_head for local drafting
    lm_w, lm_s, lm_b, norm_w = load_lm_head()
    print(f"✅ Weights loaded ({time.time()-t0:.1f}s), GPU: {mx.get_active_memory()/1024**3:.2f} GB")
    
    # Load tokenizer and dequantize embedding
    tokenizer = load_tokenizer(shard.model_id)
    from sharded_inference import dequantize_weight
    embed_full = dequantize_weight(
        weights['language_model.model.embed_tokens.weight'],
        weights['language_model.model.embed_tokens.scales'],
        weights['language_model.model.embed_tokens.biases'],
    )
    mx.eval(embed_full)
    print(f"   Embedding: {embed_full.shape}")
    
    # Connect to R1
    print(f"\n连接 Rank 1 ({host}:{port})...")
    sender = TensorSender(host, port)
    await sender.connect()
    receiver = TensorReceiver(host="0.0.0.0", port=port+1)
    await receiver.start()
    print(f"✅ 连接建立")
    
    # Tokenize prompt
    tokens = tokenizer.encode(prompt)
    print(f"\nPrompt: '{prompt}' ({len(tokens)} tokens)")
    
    # === Prefill ===
    print(f"\nPrefill...")
    t_prefill = time.time()
    kv_cache_r0 = {}
    
    inp = mx.array([tokens])
    h = embed_full[inp] * mx.array(HIDDEN_SIZE**0.5, dtype=mx.float16)
    for i in range(0, 24):
        h = forward_layer(h, i, weights, kv_cache=kv_cache_r0, offset=0)
    mx.eval(h)
    mx.synchronize()
    
    # Send prefill hidden to R1
    await sender.send(h, rank=0)
    frame = await receiver.recv(timeout=300)
    recv_data = frame_to_mlx(frame)
    mx.eval(recv_data)
    
    top_k_recv = recv_data.shape[-1] // 2
    indices = recv_data[:, :, :top_k_recv].astype(mx.int32)
    values = recv_data[:, :, top_k_recv:]
    first_token = sample_from_topk(indices, values, temperature, 1.0)
    
    t_prefill_end = time.time()
    print(f"✅ Prefill: {t_prefill_end - t_prefill:.1f}s, first token: {first_token} ({tokenizer.decode([first_token])!r})")
    
    # === SD Decode Loop ===
    print(f"\nDecode (SD K={K}, max {max_tokens} tokens)...")
    
    generated_tokens = [first_token]
    total_draft = 0
    total_accepted = 0
    total_steps = 0
    
    # --- SD Quality Monitor (inspired by Zhipu "Scaling Pain") ---
    # Track per-step accept lengths for anomaly detection.
    # Patterns from Zhipu's production data:
    #   - Garbled output → spec_accept_length drops to near 0
    #   - Repetition → spec_accept_rate spikes above 0.96
    # Thresholds (initial, Apple Silicon may need tuning):
    SPEC_ACCEPT_LEN_WARN = 1.4   # Warn if avg accept length < 1.4 over window
    SPEC_ACCEPT_RATE_WARN = 0.96  # Warn if accept rate > 0.96 (repetition)
    MONITOR_WINDOW = 20           # Rolling window size
    accept_history = []           # Per-step accept lengths
    quality_warnings = 0
    
    eos_token_id = tokenizer.eos_token_id if hasattr(tokenizer, 'eos_token_id') else 1
    
    while len(generated_tokens) < max_tokens:
        t_step = time.time()
        
        # Phase 1: Draft K tokens using R0's local model
        t_draft = time.time()
        draft_toks, draft_hiddens = draft_tokens(
            generated_tokens[-1], kv_cache_r0, weights, embed_full,
            lm_w, lm_s, lm_b, norm_w,
            offset=len(tokens) + len(generated_tokens) - 1,
            K=K, temperature=temperature
        )
        t_draft = time.time() - t_draft
        
        # Phase 2: Batch send K hidden states to R1
        # Stack hiddens: (K, 1, 1, hidden_size) → send as batch
        batch_hidden = mx.concatenate(draft_hiddens, axis=1)  # (1, K, hidden_size)
        mx.eval(batch_hidden)
        
        t_send = time.time()
        await sender.send(batch_hidden, rank=0)
        t_send = time.time() - t_send
        
        # Phase 3: R1 batch verify → recv K sets of logits
        t_recv = time.time()
        frame = await receiver.recv(timeout=300)
        t_recv = time.time() - t_recv
        
        recv_data = frame_to_mlx(frame)
        mx.eval(recv_data)
        
        # recv_data shape: (1, K, 2*top_k) — K sets of top-k logits
        # Parse and verify
        top_k_recv = recv_data.shape[-1] // 2
        
        accepted = 0
        for k in range(K):
            indices_k = recv_data[:, k:k+1, :top_k_recv].astype(mx.int32)
            values_k = recv_data[:, k:k+1, top_k_recv:]
            target_token = sample_from_topk(indices_k, values_k, temperature, 1.0)
            
            if target_token == draft_toks[k]:
                # Draft accepted!
                generated_tokens.append(target_token)
                accepted += 1
                if target_token == eos_token_id:
                    break
            else:
                # Draft rejected — use target token
                generated_tokens.append(target_token)
                # Need to rollback KV cache: remove K - accepted - 1 entries
                # We added K entries during draft, need to remove the rejected ones
                _rollback_kv_cache(kv_cache_r0, remove_count=K - accepted - 1)
                break
        
        total_draft += K
        total_accepted += accepted
        total_steps += 1
        
        # --- SD Quality Monitor: track accept length ---
        accept_history.append(accepted)
        if len(accept_history) > MONITOR_WINDOW:
            accept_history.pop(0)
        
        # Check quality anomalies
        if len(accept_history) >= 5:  # Need minimum data
            recent_avg_len = sum(accept_history[-MONITOR_WINDOW:]) / len(accept_history[-MONITOR_WINDOW:])
            recent_rate = sum(accept_history[-MONITOR_WINDOW:]) / (len(accept_history[-MONITOR_WINDOW:]) * K) if K > 0 else 0
            
            # Garbled output signal: accept length consistently near 0
            if n_total > 128 and recent_avg_len < SPEC_ACCEPT_LEN_WARN:
                quality_warnings += 1
                print(f"  ⚠️ QUALITY WARNING #{quality_warnings}: spec_accept_length={recent_avg_len:.2f} < {SPEC_ACCEPT_LEN_WARN} "
                      f"(possible garbled output / KV cache corruption)")
            
            # Repetition signal: accept rate suspiciously high
            if recent_rate > SPEC_ACCEPT_RATE_WARN:
                quality_warnings += 1
                print(f"  ⚠️ QUALITY WARNING #{quality_warnings}: spec_accept_rate={recent_rate:.2%} > {SPEC_ACCEPT_RATE_WARN:.0%} "
                      f"(possible repetition loop)")
        
        elapsed = time.time() - t_step
        n_total = len(generated_tokens)
        tps = (n_total - 1) / (time.time() - t_prefill_end) if (time.time() - t_prefill_end) > 0 else 0
        
        if total_steps <= 5 or total_steps % 10 == 0:
            text = tokenizer.decode([generated_tokens[-1]], skip_special_tokens=True)
            print(f"  Step {total_steps}: draft={K} accepted={accepted} | "
                  f"draft_t={t_draft*1000:.0f}ms send={t_send*1000:.0f}ms recv={t_recv*1000:.0f}ms | "
                  f"{tps:.1f} tok/s | {text!r}", flush=True)
        
        if generated_tokens[-1] == eos_token_id:
            print(f"  EOS at token {n_total}")
            break
    
    # Summary
    total_time = time.time() - t_prefill_end
    n_gen = len(generated_tokens) - 1
    accept_rate = total_accepted / total_draft if total_draft > 0 else 0
    
    print(f"\n=== SD Summary ===")
    print(f"Tokens: {n_gen} in {total_time:.1f}s = {n_gen/total_time:.1f} tok/s")
    print(f"Acceptance rate: {accept_rate:.1%} ({total_accepted}/{total_draft})")
    avg_accept_len = sum(accept_history) / len(accept_history) if accept_history else 0
    print(f"Avg accept length: {avg_accept_len:.2f}")
    print(f"Quality warnings: {quality_warnings}")
    if quality_warnings > 0:
        print(f"⚠️ Output quality may be degraded — review generated text carefully")
    print(f"Steps: {total_steps}")
    print(f"Text: {tokenizer.decode(generated_tokens, skip_special_tokens=True)}")
    
    await sender.close()
    await receiver.stop()


def _rollback_kv_cache(kv_cache, remove_count):
    """Remove last N entries from KV cache (for rejected draft tokens)."""
    if remove_count <= 0:
        return
    for key in kv_cache:
        k, v = kv_cache[key]
        # Shape: (n_heads, 1, seq_len, head_dim)
        seq_len = k.shape[2]
        new_len = max(0, seq_len - remove_count)
        kv_cache[key] = (k[:, :, :new_len, :], v[:, :, :new_len, :])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='169.254.2.225')
    parser.add_argument('--port', type=int, default=9998)
    parser.add_argument('--prompt', default='The capital of France is')
    parser.add_argument('--max-tokens', type=int, default=50)
    parser.add_argument('--K', type=int, default=3, help='Speculative draft depth')
    parser.add_argument('--temperature', type=float, default=0.0)
    args = parser.parse_args()
    
    asyncio.run(rank0_sd_main(
        host=args.host, port=args.port, prompt=args.prompt,
        max_tokens=args.max_tokens, K=args.K, temperature=args.temperature
    ))
