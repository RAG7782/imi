# IMI Concurrency Model

> Design document — defines isolation semantics before implementation.

## Phase 1: Single-Writer, Multiple-Reader

### Assumptions
- One `IMISpace` instance owns the database at a time
- Multiple readers (e.g. navigate queries) are safe within the same process
- Multi-writer (multiple agents) deferred to Phase 2

### Write Semantics

**Append-only versioning.** Every mutation to a `MemoryNode` is an INSERT with an incremented `version` number. The current state of a node is always:

```sql
SELECT DISTINCT ON (node_id) *
FROM memory_nodes
WHERE node_id = ? AND store_name = ? AND NOT is_deleted
ORDER BY node_id, version DESC
```

No UPDATEs, no DELETEs on data rows. "Deletion" is a new version with `is_deleted = TRUE`.

**Events are append-only.** The `memory_events` table is INSERT-only, never updated or deleted.

### Transaction Boundaries

Each `IMISpace.save()` call wraps all stores in a single transaction:

```
BEGIN
  INSERT INTO memory_nodes (episodic nodes)
  INSERT INTO memory_nodes (semantic nodes)
  INSERT INTO temporal_contexts (upsert)
  INSERT INTO anchors (upsert)
COMMIT
```

If the process crashes mid-save, the transaction rolls back. The previous complete state remains intact.

### Connection Pool

- `min_size=2, max_size=10`
- Serialize data outside the connection context to minimize hold time
- `SET synchronous_commit = off` for write throughput (writes buffered in WAL, durable on OS crash but not on DB process crash — acceptable tradeoff for memory data that can be re-encoded)

### Read Isolation

Readers use default `READ COMMITTED` isolation. A reader may see a partially-saved state within the same transaction only if it reads from a different connection. Since Phase 1 is single-writer, this is not a concern.

---

## Phase 2: Multi-Writer (Future)

### Advisory Locks

Per-store advisory locks for write serialization:

```sql
-- Before writing to episodic store:
SELECT pg_advisory_lock(hashtext('imi_episodic'));
-- Write...
SELECT pg_advisory_unlock(hashtext('imi_episodic'));
```

### Version Conflict Resolution

Two writers cannot conflict on the same row because writes are append-only. The UNIQUE constraint `(node_id, store_name, version)` prevents duplicate versions. If writer B tries to INSERT version N+1 while writer A already did, B gets a constraint violation and retries with N+2.

### Multi-Space Isolation

Each `IMISpace` gets a `space_id` (namespace). Different spaces never interfere:

```sql
WHERE space_id = ? AND node_id = ? AND store_name = ?
```

This enables multiple agents with separate memory spaces on the same database.

### Shared Memory Spaces

For agents that share a memory space (collaborative memory):
- Read: no locking needed
- Write (encode): advisory lock on `space_id + store_name`
- Dream (maintenance): exclusive advisory lock on `space_id` (no concurrent dreaming)

---

## Decision Log

| Decision | Rationale |
|----------|-----------|
| Append-only, no UPDATE | Enables version history, eliminates write conflicts, natural for TSDB |
| `synchronous_commit = off` | 2-5x write throughput; acceptable risk for memory data |
| NOT UNLOGGED tables | Memories are long-lived state, not ephemeral checkpoints |
| Single-writer Phase 1 | Simplifies implementation; multi-writer is additive, not refactor |
| Advisory locks Phase 2 | Lightweight, no schema change needed, per-resource granularity |
