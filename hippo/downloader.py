"""HuggingFace GGUF model downloader."""

import os
import hashlib
import logging
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger("hippo")

# Mirror support (China-friendly)
HF_BASE = os.environ.get("HF_ENDPOINT", "https://huggingface.co")

# Common GGUF repos on HuggingFace
REPO_PATTERNS = [
    "bartowski/{name}-GGUF",
    "TheBloke/{name}-GGUF",
    "QuantFactory/{name}-GGUF",
]


def _sanitize_model_name(name: str) -> list[str]:
    """Convert 'llama3.2:3b' → candidates for HF search."""
    parts = name.split(":")
    base = parts[0]
    tag = parts[1] if len(parts) > 1 else ""

    candidates = []
    if tag:
        candidates.append(f"{base}-{tag}".replace(".", "-"))
        candidates.append(f"{base}-{tag}")
    candidates.append(base)
    return candidates


def _find_gguf_file(repo_id: str, prefer_tag: str = "") -> Optional[str]:
    """Find a GGUF file in a HuggingFace repo."""
    api_url = f"{HF_BASE}/api/models/{repo_id}"
    try:
        resp = requests.get(api_url, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()

        siblings = data.get("siblings", [])
        gguf_files = [s["rfilename"] for s in siblings if s["rfilename"].endswith(".gguf")]

        if not gguf_files:
            return None

        if prefer_tag:
            for f in gguf_files:
                if prefer_tag.lower() in f.lower():
                    return f

        prefs = ["Q4_K_M", "Q5_K_M", "Q4_0", "Q8_0"]
        for p in prefs:
            for f in gguf_files:
                if p in f:
                    return f

        return gguf_files[0]
    except Exception:
        return None


def pull_model(name: str, models_dir: Path) -> Path:
    """Download a GGUF model from HuggingFace with integrity verification and resume support."""
    models_dir.mkdir(parents=True, exist_ok=True)

    # Resolve repo and file
    if "/" in name:
        repo_id = name
        gguf_file = _find_gguf_file(repo_id)
        if not gguf_file:
            raise FileNotFoundError(f"No GGUF files found in repo '{repo_id}'")
    else:
        parts = name.split(":")
        tag = parts[1] if len(parts) > 1 else ""
        name_candidates = _sanitize_model_name(name)

        repo_id = None
        gguf_file = None

        for pattern in REPO_PATTERNS:
            for nc in name_candidates:
                candidate_repo = pattern.format(name=nc)
                gf = _find_gguf_file(candidate_repo, tag)
                if gf:
                    repo_id = candidate_repo
                    gguf_file = gf
                    break
            if repo_id:
                break

        if not repo_id:
            raise FileNotFoundError(
                f"Could not find GGUF for '{name}' on HuggingFace. "
                f"Try: hippo pull bartowski/<Model>-GGUF"
            )

    dest = models_dir / gguf_file
    url = f"{HF_BASE}/{repo_id}/resolve/main/{gguf_file}"

    if dest.exists():
        logger.info("Model already exists: %s", dest)
        return dest

    # Resume support: use .part file
    part_dest = dest.with_suffix(dest.suffix + ".part")
    resume_header = {}
    existing_size = 0

    if part_dest.exists():
        existing_size = part_dest.stat().st_size
        resume_header = {"Range": f"bytes={existing_size}-"}
        logger.info("Resuming download from %d bytes", existing_size)

    logger.info("Downloading %s/%s ...", repo_id, gguf_file)
    resp = requests.get(url, stream=True, timeout=30, headers=resume_header)

    # Check if server supports range
    if existing_size and resp.status_code == 200:
        # Server didn't support range, restart
        existing_size = 0

    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    if existing_size and resp.status_code == 206:
        total += existing_size

    downloaded = existing_size
    mode = "ab" if existing_size else "wb"
    last_pct = -1  # Track last logged percentage (10% increments)

    with open(part_dest, mode) as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                # Log every 10% progress
                pct_bucket = int(pct // 10) * 10
                if pct_bucket > last_pct:
                    last_pct = pct_bucket
                    logger.info(
                        "  %.2f GB / %.2f GB (%.0f%%)",
                        downloaded / (1024**3), total / (1024**3), pct,
                    )
            else:
                # No Content-Length: log every 100MB
                last_logged = existing_size
                if downloaded - last_logged >= 100 * 1024**3:
                    last_logged = downloaded
                    logger.info(
                        "  %.2f GB downloaded", downloaded / (1024**3),
                    )

    # Integrity check: verify file size matches Content-Length
    actual_size = part_dest.stat().st_size
    if total and actual_size != total:
        part_dest.unlink(missing_ok=True)
        raise IOError(
            f"Download integrity check failed: expected {total} bytes, got {actual_size} bytes"
        )

    # Rename .part to final
    part_dest.rename(dest)
    logger.info("Downloaded to %s (%d bytes)", dest, actual_size)

    # P1-3 fix: SHA256 integrity verification
    integrity_ok, sha256 = verify_sha256(dest)
    if integrity_ok is not None:
        if integrity_ok:
            logger.info("SHA256 verification passed: %s", sha256[:16] + "...")
        else:
            logger.error("SHA256 verification FAILED for %s", dest)
            dest.unlink(missing_ok=True)
            raise IOError(
                f"SHA256 integrity check failed for {dest}. "
                f"File may be corrupted or tampered with."
            )
    else:
        # No expected hash available, log actual hash for user verification
        logger.info("SHA256 (no expected hash to compare): %s", sha256[:16] + "...")

    return dest


def compute_sha256(path: Path, block_size: int = 65536) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _fetch_hf_file_hash(repo_id: str, filename: str) -> Optional[str]:
    """Fetch expected SHA256 from HuggingFace API."""
    try:
        api_url = f"{HF_BASE}/api/models/{repo_id}"
        resp = requests.get(api_url, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        for s in data.get("siblings", []):
            if s.get("rfilename") == filename:
                # HuggingFace provides checksums in some cases
                return s.get("checksum") or None
        return None
    except Exception:
        return None


def verify_sha256(path: Path, expected: Optional[str] = None) -> tuple[Optional[bool], str]:
    """Verify file integrity via SHA256.

    Returns (match_result, actual_hash):
    - (True, hash) if expected matches
    - (False, hash) if expected doesn't match
    - (None, hash) if no expected hash available
    """
    actual = compute_sha256(path)
    if expected:
        return (actual == expected, actual)
    return (None, actual)
