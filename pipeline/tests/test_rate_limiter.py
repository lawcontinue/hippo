#!/usr/bin/env python3
"""Tests for RateLimiter and input sanitization in hippo_api."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hippo_api import RateLimiter


def test_rate_limiter_allows_under_limit():
    rl = RateLimiter(max_requests=3, window_s=60)
    assert rl.allow("client1") is True
    assert rl.allow("client1") is True
    assert rl.allow("client1") is True


def test_rate_limiter_blocks_over_limit():
    rl = RateLimiter(max_requests=3, window_s=60)
    rl.allow("client1")
    rl.allow("client1")
    rl.allow("client1")
    assert rl.allow("client1") is False


def test_rate_limiter_isolates_clients():
    rl = RateLimiter(max_requests=2, window_s=60)
    rl.allow("client1")
    rl.allow("client1")
    assert rl.allow("client2") is True


def test_rate_limiter_window_expiry():
    rl = RateLimiter(max_requests=2, window_s=0)  # 0s window = instant expiry
    rl.allow("client1")
    rl.allow("client1")
    # Window expired, should allow again
    assert rl.allow("client1") is True


if __name__ == "__main__":
    tests = [t for t in dir() if t.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            globals()[t]()
            print(f"  [OK] {t}")
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {t}: {e}")
        except Exception as e:
            print(f"  [ERR] {t}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
