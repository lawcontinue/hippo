# v0.5 Design: Agent Memory Layer

> Status: **Draft** — pending review before v0.5 implementation begins
> Author: lawcontinue
> Created: 2026-06-28
> Related: v0.3 `memory_safety.py` (M0 source tagging + confidence-weighted search)

## 1. Motivation

v0.3's `memory_safety` module provides the primitives: `add_with_source`, `search_with_confidence`, `decay_low_confidence`. v0.5 builds the **agent memory layer** on top — defining what an episodic memory is, how the LLM reads/writes it, and how memories evolve over time.

## 2. Schema Specification

### Required Fields (metadata JSON)

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | str | Session/conversation ID |
| `created_at` | ISO 8601 | Creation timestamp (UTC) |
| `kind` | str | Memory type: `fact` / `decision` / `preference` / `event` / `reflection` |
| `source` | str | Origin: `user` / `model` / `verified` / `external` / `system` |
| `confidence` | float (0-1) | Trust score, default by source type |
| `archive_id` | str (optional) | Links to session archive on consolidation |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `expires_at` | ISO 8601 | TTL-based expiry |
| `tags` | list[str] | Free-form tags for categorical filtering |
| `related_ids` | list[int] | Document IDs of related memories |
| `last_decayed` | ISO 8601 | Last confidence decay timestamp |
| `reviewed_by` | str | Who confirmed this memory |
| `reviewed_at` | ISO 8601 | When it was confirmed |

## 3. Layered Design: Working Memory vs Long-term Memory

### Working Memory (short-term, high-density)

- **Lifecycle**: Within a single session (minutes to hours)
- **Storage**: In-memory `VectorStore` instance, persisted on session end
- **Density**: Every exchange is stored — high recall, low precision
- **Decay**: Full confidence during session; on session end, eligible memories are promoted to long-term

### Long-term Memory (persistent, curated)

- **Lifecycle**: Across sessions (days to months)
- **Storage**: SQLite-backed `VectorStore`, persisted to disk
- **Density**: Only promoted memories — low recall, high precision
- **Decay**: `decay_low_confidence()` runs periodically; `source=verified` and `source=user` are exempt

### Promotion Criteria (Working → Long-term)

1. `source` is `user` or `verified` — always promoted
2. `confidence >= 0.7` after session-end review — promoted
3. Referenced by ≥2 subsequent exchanges within the session — promoted
4. Everything else — discarded (or archived with `confidence *= 0.5`)

## 4. Write Workflow (LLM → Memory)

### When to Write

| Trigger | `kind` | Example |
|---------|--------|---------|
| User states a fact | `fact` | "I'm a lawyer in Beijing" |
| User makes a decision | `decision` | "Let's use sparse mode" |
| User expresses preference | `preference` | "I prefer concise answers" |
| Significant event occurs | `event` | "Deployed v0.3.2 to production" |
| Agent reflects on pattern | `reflection` | "User consistently asks about Rust" |

### Format

The LLM produces a structured write intent:

```json
{
  "text": "User prefers dark mode for code examples",
  "kind": "preference",
  "source": "user",
  "confidence": 1.0
}
```

The memory layer calls `add_with_source()` with appropriate metadata. The LLM does **not** directly call `store.add()` — it goes through the memory layer API.

## 5. Read Workflow (Memory → LLM Context)

### Ranking

1. BM25/hybrid search returns candidates (`top_k * 3`)
2. Confidence re-ranking: `adjusted_score = score * (1 - w + w * confidence)` where `w=0.3` (configurable)
3. Recency boost: memories from the current session get +0.1 score
4. Final `top_k` returned

### Conflict Resolution

When two memories contradict (same `kind`, same `tags`, different values):
- Higher `confidence` wins
- If confidence is equal, more recent `created_at` wins
- Both are returned to the LLM with a `_conflict` flag — the LLM decides

## 6. Relationship to `memory_safety` (M0)

`memory_safety` is the **foundation**, not the replacement. v0.5 extends it:

| `memory_safety` (M0) | v0.5 Memory Layer |
|----------------------|--------------------|
| `add_with_source()` | Called internally by memory layer's `remember()` |
| `search_with_confidence()` | Called internally by memory layer's `recall()` |
| `decay_low_confidence()` | Runs on a schedule (session end or periodic cron) |
| `tag_memory()` | Used for promotion (model → verified) |
| `log_behavioral_signal()` | Feeds into reflection triggers |

**Upgrade path**: Existing code using `memory_safety` directly continues to work. The v0.5 layer is an optional higher-level API.

## 7. API Sketch (Subject to Change)

```python
from embedding.memory_safety import MemoryLayer

mem = MemoryLayer(store=VectorStore("agent_memory.db", mode="hybrid"))

# Write
mem.remember("User is a Java developer", kind="fact", source="user")
mem.remember("Might be interested in Kotlin", kind="reflection", source="model", confidence=0.4)

# Read (injects into LLM context)
context = mem.recall("what programming languages", top_k=5, session_id="sess_001")

# Promote working → long-term at session end
mem.consolidate(session_id="sess_001")

# Periodic maintenance
mem.decay(threshold=0.3, days_old=7)
```

## 8. Open Questions

1. **Vector dimensions for episodic memory**: Should `session_id` be a dense vector dimension (learned) or remain metadata filter?
2. **Conflict detection**: BM25 similarity threshold for "same topic" detection — needs empirical tuning
3. **Multi-agent sharing**: If multiple agents share a memory store, how do we handle conflicting writes?
4. **Privacy boundary**: Some memories should never be promoted to long-term (PII, sensitive). Need a `privacy` field?

## References

- v0.3 `memory_safety.py` implementation
- Issue #5 (this design doc's origin)
-熔炉 #99 (four-layer memory safety architecture)
-熔炉 #108 (stake-based trust routing + batch confirmation)
