#!/usr/bin/env python3
"""
test_hippo_batch_integration.py — Hippo Batch API 集成测试

作者：忒弥斯 (T-Mind) 🔮
创建日期：2026-04-22
"""

import time

import requests

# Hippo 服务器配置
BASE_URL = "http://localhost:11434"
MODEL_NAME = "PetrosStav/gemma3-tools:12b"  # Gemma3-12B Q4_K_M

def test_batch_submit():
    """测试 P2-1: 提交请求到 batch queue"""
    print("=" * 60)
    print("🧪 Test P2-1: Batch Submit")
    print("=" * 60)

    # Submit 3 requests
    request_ids = []
    for i in range(3):
        response = requests.post(
            f"{BASE_URL}/api/batch/submit",
            json={
                "model": MODEL_NAME,
                "prompt": f"Prompt {i}",
                "max_tokens": 10
            }
        )
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        request_ids.append(data["request_id"])
        print(f"  ✅ Request {i} submitted: {data['request_id']}, queue position: {data['queue_position']}")

    print("\n✅ P2-1 Test PASSED")
    return request_ids

def test_batch_run(request_ids):
    """测试 P2-2: 运行 batch"""
    print("\n" + "=" * 60)
    print("🧪 Test P2-2: Batch Run")
    print("=" * 60)

    t0 = time.time()

    response = requests.post(
        f"{BASE_URL}/api/batch/run",
        json={"model": MODEL_NAME}
    )

    elapsed = time.time() - t0

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()

    results = data["results"]
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"

    print(f"  ✅ Batch done: {elapsed:.2f}s")
    for r in results:
        print(f"  ✅ {r['request_id']}: {len(r['tokens'])} tokens, {r['tok_s']:.1f} tok/s")

    print("\n✅ P2-2 Test PASSED")

def test_batch_stats():
    """测试 P2-3: Batch stats"""
    print("\n" + "=" * 60)
    print("🧪 Test P2-3: Batch Stats")
    print("=" * 60)

    response = requests.get(
        f"{BASE_URL}/api/batch/stats",
        params={"model": MODEL_NAME}
    )

    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    data = response.json()

    print(f"  Queue size: {data['queue_size']}")
    print(f"  Active requests: {data['active_requests']}")
    print(f"  Memory: {data['memory_gb']:.2f} GB")
    print(f"  Memory peak: {data['memory_peak_gb']:.2f} GB")

    print("\n✅ P2-3 Test PASSED")

def test_queue_full():
    """测试 P2-4: Queue 满（7 个请求）"""
    print("\n" + "=" * 60)
    print("🧪 Test P2-4: Queue Full")
    print("=" * 60)

    # Submit 7 requests
    for i in range(7):
        response = requests.post(
            f"{BASE_URL}/api/batch/submit",
            json={
                "model": MODEL_NAME,
                "prompt": f"Queue test {i}",
                "max_tokens": 10
            }
        )
        assert response.status_code == 200, f"Request {i} should succeed"

    # 8th should fail
    response = requests.post(
        f"{BASE_URL}/api/batch/submit",
        json={
            "model": MODEL_NAME,
            "prompt": "Too many",
            "max_tokens": 10
        }
    )
    assert response.status_code == 503, f"Expected 503, got {response.status_code}"
    print("  ✅ 8th request rejected (queue full)")

    # Run batch to clear
    requests.post(f"{BASE_URL}/api/batch/run", json={"model": MODEL_NAME})

    print("\n✅ P2-4 Test PASSED")

if __name__ == "__main__":
    print("🔬 Hippo Batch API Integration Test Suite\n")

    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/api/tags")
        if response.status_code != 200:
            print("❌ Hippo server not responding")
            exit(1)
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Hippo server")
        print("   Start Hippo first: cd /Users/deepsearch/.openclaw/workspace/hippo && python3 -m hippo.cli")
        exit(1)

    # Run tests
    request_ids = test_batch_submit()
    test_batch_run(request_ids)
    test_batch_stats()
    test_queue_full()

    print("\n" + "=" * 60)
    print("🎉 All integration tests PASSED!")
    print("=" * 60)
