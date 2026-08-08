"""
Hippo Importers — bulk import CSV and JSON into VectorStore.

Dependencies: csv, json (stdlib)
"""

from __future__ import annotations

import csv
import json
import os
from typing import List, Optional

from .store import VectorStore

__all__ = ["import_csv", "import_json"]


def _validate_path(path: str) -> str:
    """Resolve and validate file path; reject path traversal.

    Security: must check the *original* path for '..' BEFORE normpath
    resolves it away. normpath('/tmp/../etc/passwd') → '/etc/passwd'
    which contains no '..', making a post-normpath check useless.
    """
    if ".." in path:
        raise ValueError(f"Path traversal not allowed: {path}")
    return os.path.abspath(path)


def import_csv(
    store: VectorStore,
    path: str,
    text_col: str,
    meta_cols: Optional[List[str]] = None,
) -> List[int]:
    """Import rows from a CSV file into the store. Returns list of doc IDs."""
    meta_cols = meta_cols or []
    abs_path = _validate_path(path)
    ids: List[int] = []
    with open(abs_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row[text_col]
            meta = {k: row[k] for k in meta_cols if k in row}
            ids.append(store.add(text, meta))
    return ids


def import_json(
    store: VectorStore,
    path: str,
    text_key: str,
    meta_key: Optional[str] = None,
) -> List[int]:
    """Import records from a JSON file (array of objects). Returns list of doc IDs."""
    abs_path = _validate_path(path)
    with open(abs_path, encoding="utf-8") as f:
        records = json.load(f)

    ids: List[int] = []
    for rec in records:
        text = rec[text_key]
        meta = rec.get(meta_key, {}) if meta_key else {}
        ids.append(store.add(text, meta))
    return ids
