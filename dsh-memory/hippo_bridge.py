# -*- coding: utf-8 -*-
"""
hippo_bridge.py — JSON bridge between an agent runtime (DSH preset plugin) and
the hippo VectorStore, keeping the embedding model warm in a persistent process.

Two modes:
  one-shot : read one JSON command from stdin, write one JSON result to stdout.
  serve    : ``python hippo_bridge.py --serve`` — newline-delimited JSON loop.
             Each request  {"rid": N, "op": ...} gets one response line
             {"rid": N, "ok": true, ...} / {"rid": N, "ok": false, "error": ...}.
             NOTE: the correlation key is `rid`, never `id` — `id` belongs to
             business ops (delete/update) and must not be clobbered.

Ops: store | recall | list | delete | update | count | decay | rebuild

Configuration (env):
  HIPPO_MEMORY_DB       memory db path  (default ~/.dsh/hippo-memory.db)
  HIPPO_MEMORY_MODE     auto | hybrid | sparse   (auto = hybrid when the
                        local embedding model exists, else sparse fallback)
  HIPPO_EMBED_MODEL_PATH  local embedding model dir
                        (default <script_dir>/models/bge-small-zh-v1.5)
  HIPPO_REPO_DIR        hippo source checkout root (default: auto-detect —
                        the repo root when this file lives inside the repo,
                        else the script's own directory)

Memory schema per document (metadata JSON):
  source: user | model | inference | system | verified
  confidence: 0..1        tags: [str]      session: optional writer id
"""
import io
import json
import os
import sys

# Force UTF-8 stdio on Windows so Chinese memory text survives the pipe.
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _detect_repo_dir():
    """Locate the hippo source root: env override, in-repo layout, or script dir."""
    env = os.environ.get("HIPPO_REPO_DIR")
    if env:
        return env
    parent = os.path.dirname(_SCRIPT_DIR)  # this file lives in <repo>/dsh-memory/
    if os.path.isdir(os.path.join(parent, "hippo", "embedding")):
        return parent
    return _SCRIPT_DIR


# The embedding model path must be set BEFORE hippo.embedding.engine is imported
# (it reads HIPPO_EMBED_MODEL_PATH at module import time).
_DEFAULT_MODEL_DIR = os.path.join(_SCRIPT_DIR, "models", "bge-small-zh-v1.5")
if not os.environ.get("HIPPO_EMBED_MODEL_PATH") and os.path.isdir(_DEFAULT_MODEL_DIR):
    os.environ["HIPPO_EMBED_MODEL_PATH"] = _DEFAULT_MODEL_DIR

HIPPO_REPO = _detect_repo_dir()
if HIPPO_REPO not in sys.path:
    sys.path.insert(0, HIPPO_REPO)

from hippo.embedding import VectorStore  # noqa: E402
from hippo.embedding.memory_safety import (  # noqa: E402
    add_with_source,
    decay_low_confidence,
    search_with_confidence,
)

DB_PATH = os.environ.get(
    "HIPPO_MEMORY_DB",
    os.path.join(os.path.expanduser("~"), ".dsh", "hippo-memory.db"),
)
MODE_ENV = os.environ.get("HIPPO_MEMORY_MODE", "auto").lower()


def _doc_to_json(doc):
    meta = doc.metadata or {}
    return {
        "id": doc.id,
        "text": doc.text,
        "score": round(doc.score, 4),
        "source": meta.get("source", "model"),
        "confidence": meta.get("confidence", 0.5),
        "tags": meta.get("tags", []),
        "session": meta.get("session"),
        "created_at": meta.get("created_at"),
    }


def _open_store():
    """Open VectorStore in the configured mode. Returns (store, mode, engine)."""
    want_hybrid = MODE_ENV in ("hybrid", "auto")
    model_ready = bool(os.environ.get("HIPPO_EMBED_MODEL_PATH"))
    if want_hybrid and model_ready:
        try:
            from hippo.embedding.engine import EmbeddingEngine

            engine = EmbeddingEngine()
            engine.embed("预热 warmup")  # fail fast + warm the model
            store = VectorStore(DB_PATH, mode="hybrid", embedding_engine=engine)
            return store, "hybrid", engine
        except Exception:
            if MODE_ENV == "hybrid":
                raise
    return VectorStore(DB_PATH, mode="sparse"), "sparse", None


def execute(store, engine, cmd):
    op = cmd.get("op")

    if op == "store":
        text = (cmd.get("text") or "").strip()
        if not text:
            return {"ok": False, "error": "text is empty"}
        meta = {}
        if cmd.get("tags"):
            meta["tags"] = list(cmd["tags"])
        if cmd.get("session"):
            meta["session"] = str(cmd["session"])
        doc_id = add_with_source(
            store,
            text,
            source=cmd.get("source", "model"),
            confidence=cmd.get("confidence"),
            metadata=meta,
        )
        return {"ok": True, "id": doc_id, "count": store.count()}

    if op == "recall":
        query = (cmd.get("query") or "").strip()
        if not query:
            return {"ok": False, "error": "query is empty"}
        top_k = int(cmd.get("top_k", 5))
        min_conf = float(cmd.get("min_confidence", 0.0))
        docs = search_with_confidence(store, query, top_k=top_k, min_confidence=min_conf)
        return {"ok": True, "results": [_doc_to_json(d) for d in docs]}

    if op == "list":
        limit = int(cmd.get("limit", 200))
        offset = int(cmd.get("offset", 0))
        rows = store.execute(
            "SELECT id, text, metadata, created_at FROM documents "
            "ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        items = []
        for doc_id, text, meta_json, created_at in rows:
            meta = json.loads(meta_json) if meta_json else {}
            items.append({
                "id": doc_id,
                "text": text,
                "source": meta.get("source", "model"),
                "confidence": meta.get("confidence", 0.5),
                "tags": meta.get("tags", []),
                "session": meta.get("session"),
                "created_at": created_at,
            })
        return {"ok": True, "total": store.count(), "items": items}

    if op == "delete":
        doc_id = int(cmd["id"])
        ok = store.delete(doc_id)
        return {"ok": bool(ok), "count": store.count()}

    if op == "update":
        doc_id = int(cmd["id"])
        meta = store.get_metadata(doc_id)
        if meta is None:
            return {"ok": False, "error": f"doc {doc_id} not found"}
        if "confidence" in cmd and cmd["confidence"] is not None:
            meta["confidence"] = max(0.0, min(1.0, float(cmd["confidence"])))
        if "source" in cmd and cmd["source"]:
            meta["source"] = str(cmd["source"])
        if "tags" in cmd and cmd["tags"] is not None:
            meta["tags"] = list(cmd["tags"])
        ok = store.update_metadata(doc_id, meta)
        return {"ok": bool(ok)}

    if op == "count":
        return {"ok": True, "count": store.count(), "db": DB_PATH, "mode": store.mode}

    if op == "decay":
        threshold = float(cmd.get("threshold", 0.6))
        days_old = int(cmd.get("days_old", 7))
        n = decay_low_confidence(store, threshold=threshold, days_old=days_old)
        return {"ok": True, "decayed": n}

    if op == "rebuild":
        if engine is None:
            return {"ok": False, "error": "sparse 模式无向量可重建"}
        n = store.rebuild_embeddings()
        return {"ok": True, "rebuilt": n}

    return {"ok": False, "error": f"unknown op: {op}"}


def _run_one(store, engine, cmd):
    try:
        return execute(store, engine, cmd)
    except Exception as exc:  # noqa: BLE001 — bridge must always emit JSON
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def serve():
    store, mode, engine = _open_store()
    sys.stdout.write(json.dumps(
        {"ok": True, "ready": True, "mode": mode, "db": DB_PATH},
        ensure_ascii=False,
    ) + "\n")
    sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except json.JSONDecodeError as exc:
            result = {"ok": False, "error": f"bad JSON: {exc}"}
            sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
            sys.stdout.flush()
            continue
        result = _run_one(store, engine, cmd)
        result["rid"] = cmd.get("rid")
        sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    store.close()


def one_shot():
    raw = sys.stdin.read()
    cmd = json.loads(raw)
    store, _mode, engine = _open_store()
    try:
        return _run_one(store, engine, cmd)
    finally:
        store.close()


if __name__ == "__main__":
    if "--serve" in sys.argv:
        try:
            serve()
        except Exception as exc:  # startup failure — emit JSON then exit
            sys.stdout.write(json.dumps(
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                ensure_ascii=False,
            ) + "\n")
            sys.stdout.flush()
            sys.exit(1)
    else:
        result = one_shot()
        sys.stdout.write(json.dumps(result, ensure_ascii=False))
        sys.stdout.flush()
