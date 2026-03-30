"""Storage backends for IMI — abstract interface + JSON and SQLite implementations.

The StorageBackend ABC defines the persistence contract. VectorStore remains
the in-memory search engine; the backend only handles durable storage.

Backends:
  - JSONBackend: drop-in replacement for the original JSON file persistence
  - SQLiteBackend: O(1) inserts, zero-infra, WAL mode, FTS5 search (recommended)
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import Any

from imi.events import MemoryEvent
from imi.node import MemoryNode
from imi.observe import timed
from imi.temporal import TemporalContext

logger = logging.getLogger("imi.storage")


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class StorageBackend(ABC):
    """Abstract persistence backend for IMI memory spaces."""

    # --- Node CRUD ---

    @abstractmethod
    def put_node(self, store_name: str, node: MemoryNode) -> None:
        """Persist a single node (insert or update)."""

    @abstractmethod
    def put_nodes(self, store_name: str, nodes: list[MemoryNode]) -> None:
        """Persist a batch of nodes (full store snapshot)."""

    @abstractmethod
    def get_node(self, store_name: str, node_id: str) -> MemoryNode | None:
        """Retrieve a single node by ID (latest version)."""

    @abstractmethod
    def remove_node(self, store_name: str, node_id: str) -> None:
        """Mark a node as deleted."""

    @abstractmethod
    def get_all_nodes(self, store_name: str) -> list[MemoryNode]:
        """Retrieve all current nodes for a store."""

    # --- Anchors ---

    @abstractmethod
    def put_anchors(self, anchors: dict[str, list[dict]]) -> None:
        """Persist anchor data. Keys are node_ids, values are lists of anchor dicts."""

    @abstractmethod
    def get_anchors(self) -> dict[str, list[dict]]:
        """Retrieve all anchors as {node_id: [anchor_dict, ...]}."""

    # --- Temporal ---

    @abstractmethod
    def put_temporal(self, contexts: dict[str, TemporalContext]) -> None:
        """Persist temporal context data."""

    @abstractmethod
    def get_temporal(self) -> dict[str, TemporalContext]:
        """Retrieve all temporal contexts."""

    # --- Events ---

    def log_event(self, event: MemoryEvent) -> None:
        """Log a mutation event. Default: no-op."""

    def query_events(
        self,
        event_type: str | None = None,
        node_id: str | None = None,
        since: float | None = None,
        limit: int = 100,
    ) -> list[MemoryEvent]:
        """Query events. Default: empty."""
        return []

    # --- Advanced temporal queries (TSDB-native overrides) ---

    def query_by_time_range(
        self,
        start: float,
        end: float,
        store_name: str | None = None,
    ) -> list[MemoryNode]:
        """Return nodes created within [start, end] timestamp range."""
        nodes = []
        stores = [store_name] if store_name else ["episodic", "semantic"]
        for sn in stores:
            for n in self.get_all_nodes(sn):
                if start <= n.created_at <= end:
                    nodes.append(n)
        return nodes

    def query_by_session(self, session_id: str) -> list[str]:
        """Return node IDs belonging to a session."""
        return [
            nid
            for nid, ctx in self.get_temporal().items()
            if ctx.session_id == session_id
        ]

    # --- Versioning ---

    def get_node_history(
        self, store_name: str, node_id: str
    ) -> list[MemoryNode]:
        """Return all versions of a node, newest first. Default: current only."""
        node = self.get_node(store_name, node_id)
        return [node] if node else []

    # --- Bulk export/import for migration ---

    @abstractmethod
    def export_all(self) -> dict[str, Any]:
        """Export full state as a dict."""

    @abstractmethod
    def import_all(self, data: dict[str, Any]) -> None:
        """Import full state from a dict."""

    # --- Lifecycle ---

    def setup(self) -> None:
        """One-time initialization (create tables, dirs, etc.)."""

    def close(self) -> None:
        """Release resources."""


# ---------------------------------------------------------------------------
# JSON Backend — wraps the original file persistence
# ---------------------------------------------------------------------------


class JSONBackend(StorageBackend):
    """Persists IMI state as JSON files — identical behavior to the original."""

    def __init__(self, persist_dir: str | Path) -> None:
        self.persist_dir = Path(persist_dir)

    def setup(self) -> None:
        self.persist_dir.mkdir(parents=True, exist_ok=True)

    # --- Nodes ---

    @timed("json.put_node")
    def put_node(self, store_name: str, node: MemoryNode) -> None:
        nodes = self.get_all_nodes(store_name)
        by_id = {n.id: n for n in nodes}
        by_id[node.id] = node
        self.put_nodes(store_name, list(by_id.values()))

    @timed("json.put_nodes")
    def put_nodes(self, store_name: str, nodes: list[MemoryNode]) -> None:
        path = self.persist_dir / f"{store_name}.json"
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        data = [n.to_dict() for n in nodes]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    @timed("json.get_node")
    def get_node(self, store_name: str, node_id: str) -> MemoryNode | None:
        for n in self.get_all_nodes(store_name):
            if n.id == node_id:
                return n
        return None

    @timed("json.remove_node")
    def remove_node(self, store_name: str, node_id: str) -> None:
        nodes = [n for n in self.get_all_nodes(store_name) if n.id != node_id]
        self.put_nodes(store_name, nodes)

    @timed("json.get_all_nodes")
    def get_all_nodes(self, store_name: str) -> list[MemoryNode]:
        path = self.persist_dir / f"{store_name}.json"
        if not path.exists():
            return []
        data = json.loads(path.read_text())
        return [MemoryNode.from_dict(d) for d in data]

    # --- Anchors ---

    @timed("json.put_anchors")
    def put_anchors(self, anchors: dict[str, list[dict]]) -> None:
        path = self.persist_dir / "anchors.json"
        path.write_text(json.dumps(anchors, ensure_ascii=False, indent=2))

    @timed("json.get_anchors")
    def get_anchors(self) -> dict[str, list[dict]]:
        path = self.persist_dir / "anchors.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text())

    # --- Temporal ---

    @timed("json.put_temporal")
    def put_temporal(self, contexts: dict[str, TemporalContext]) -> None:
        path = self.persist_dir / "temporal.json"
        data = {nid: ctx.to_dict() for nid, ctx in contexts.items()}
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    @timed("json.get_temporal")
    def get_temporal(self) -> dict[str, TemporalContext]:
        path = self.persist_dir / "temporal.json"
        if not path.exists():
            return {}
        data = json.loads(path.read_text())
        return {nid: TemporalContext.from_dict(d) for nid, d in data.items()}

    # --- Events ---

    def log_event(self, event: MemoryEvent) -> None:
        path = self.persist_dir / "events.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")

    def query_events(
        self,
        event_type: str | None = None,
        node_id: str | None = None,
        since: float | None = None,
        limit: int = 100,
    ) -> list[MemoryEvent]:
        path = self.persist_dir / "events.jsonl"
        if not path.exists():
            return []
        events = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            evt = MemoryEvent.from_dict(json.loads(line))
            if event_type and evt.event_type != event_type:
                continue
            if node_id and evt.node_id != node_id:
                continue
            if since and evt.timestamp < since:
                continue
            events.append(evt)
            if len(events) >= limit:
                break
        return events

    # --- Export/Import ---

    @timed("json.export_all")
    def export_all(self) -> dict[str, Any]:
        return {
            "episodic": [n.to_dict() for n in self.get_all_nodes("episodic")],
            "semantic": [n.to_dict() for n in self.get_all_nodes("semantic")],
            "anchors": self.get_anchors(),
            "temporal": {
                nid: ctx.to_dict() for nid, ctx in self.get_temporal().items()
            },
        }

    @timed("json.import_all")
    def import_all(self, data: dict[str, Any]) -> None:
        self.setup()
        if "episodic" in data:
            nodes = [MemoryNode.from_dict(d) for d in data["episodic"]]
            self.put_nodes("episodic", nodes)
        if "semantic" in data:
            nodes = [MemoryNode.from_dict(d) for d in data["semantic"]]
            self.put_nodes("semantic", nodes)
        if "anchors" in data:
            self.put_anchors(data["anchors"])
        if "temporal" in data:
            contexts = {
                nid: TemporalContext.from_dict(d)
                for nid, d in data["temporal"].items()
            }
            self.put_temporal(contexts)


# ---------------------------------------------------------------------------
# SQLite Backend — zero-infra, O(1) inserts, WAL mode, FTS5
# ---------------------------------------------------------------------------

SQLITE_SCHEMA = """
-- Main node storage: versioned, append-only (same model as TSDB)
CREATE TABLE IF NOT EXISTS memory_nodes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id     TEXT NOT NULL,
    store_name  TEXT NOT NULL,
    version     INTEGER NOT NULL DEFAULT 1,
    data        TEXT NOT NULL,           -- JSON
    embedding   BLOB,                    -- float32 array as bytes
    is_deleted  INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL,
    inserted_at REAL NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_sqlite_nodes_version
    ON memory_nodes (node_id, store_name, version);
CREATE INDEX IF NOT EXISTS idx_sqlite_nodes_latest
    ON memory_nodes (node_id, store_name, version DESC);
CREATE INDEX IF NOT EXISTS idx_sqlite_nodes_created
    ON memory_nodes (created_at DESC)
    WHERE NOT is_deleted;
CREATE INDEX IF NOT EXISTS idx_sqlite_nodes_store
    ON memory_nodes (store_name, inserted_at DESC)
    WHERE NOT is_deleted;

-- Append-only event log
CREATE TABLE IF NOT EXISTS memory_events (
    event_id     TEXT NOT NULL,
    event_type   TEXT NOT NULL,
    node_id      TEXT NOT NULL DEFAULT '',
    store_name   TEXT NOT NULL DEFAULT '',
    node_version INTEGER NOT NULL DEFAULT 0,
    metadata     TEXT NOT NULL DEFAULT '{}',  -- JSON
    timestamp    REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sqlite_events_node
    ON memory_events (node_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_sqlite_events_type
    ON memory_events (event_type, timestamp DESC);

-- Temporal contexts
CREATE TABLE IF NOT EXISTS temporal_contexts (
    node_id            TEXT PRIMARY KEY,
    session_id         TEXT NOT NULL DEFAULT '',
    sequence_pos       INTEGER NOT NULL DEFAULT 0,
    timestamp          REAL NOT NULL,
    temporal_neighbors TEXT NOT NULL DEFAULT '[]'  -- JSON array
);

CREATE INDEX IF NOT EXISTS idx_sqlite_temporal_session
    ON temporal_contexts (session_id);

-- Anchors
CREATE TABLE IF NOT EXISTS anchors (
    node_id    TEXT NOT NULL,
    anchor_idx INTEGER NOT NULL,
    data       TEXT NOT NULL,  -- JSON
    PRIMARY KEY (node_id, anchor_idx)
);
"""

# FTS5 virtual table for full-text search on seeds/summaries
SQLITE_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    node_id UNINDEXED,
    store_name UNINDEXED,
    seed,
    summary_orbital,
    summary_medium,
    summary_detailed,
    tokenize='porter unicode61'
);
"""


class SQLiteBackend(StorageBackend):
    """Zero-infra persistence with O(1) inserts, WAL mode, and optional FTS5.

    Solves the JSONBackend O(n) put_node problem while requiring zero external
    infrastructure. Append-only versioned storage (same model as TimescaleDB).
    """

    def __init__(
        self,
        db_path: str | Path,
        enable_fts: bool = True,
    ) -> None:
        import sqlite3

        self.db_path = Path(db_path)
        self.enable_fts = enable_fts
        self._conn: sqlite3.Connection | None = None

    def _get_conn(self) -> "sqlite3.Connection":
        import sqlite3

        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            # WAL mode for concurrent reads
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA cache_size=-64000")  # 64MB
        return self._conn

    def setup(self) -> None:
        conn = self._get_conn()
        conn.executescript(SQLITE_SCHEMA)
        if self.enable_fts:
            conn.executescript(SQLITE_FTS_SCHEMA)
        conn.commit()
        logger.info("SQLite schema ready at %s", self.db_path)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # --- Helpers ---

    def _next_version(self, conn: Any, node_id: str, store_name: str) -> int:
        row = conn.execute(
            "SELECT COALESCE(MAX(version), 0) + 1 FROM memory_nodes "
            "WHERE node_id = ? AND store_name = ?",
            (node_id, store_name),
        ).fetchone()
        return row[0] if row else 1

    @staticmethod
    def _embedding_to_bytes(emb: Any) -> bytes | None:
        if emb is None:
            return None
        import numpy as np

        return np.asarray(emb, dtype=np.float32).tobytes()

    @staticmethod
    def _bytes_to_embedding(b: bytes | None) -> Any:
        if b is None:
            return None
        import numpy as np

        return np.frombuffer(b, dtype=np.float32).copy()

    def _row_to_node(self, row: Any) -> MemoryNode:
        d = json.loads(row["data"])
        emb = self._bytes_to_embedding(row["embedding"])
        if emb is not None:
            d["embedding"] = emb.tolist()
        return MemoryNode.from_dict(d)

    # --- FTS helpers ---

    def _fts_index_node(self, conn: Any, node: MemoryNode, store_name: str) -> None:
        if not self.enable_fts:
            return
        d = node.to_dict()
        conn.execute(
            "INSERT INTO memory_fts (node_id, store_name, seed, "
            "summary_orbital, summary_medium, summary_detailed) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                node.id,
                store_name,
                d.get("seed", ""),
                d.get("summary_orbital", ""),
                d.get("summary_medium", ""),
                d.get("summary_detailed", ""),
            ),
        )

    def search_fts(
        self, query: str, store_name: str | None = None, limit: int = 20
    ) -> list[tuple[str, float]]:
        """Full-text search on seeds/summaries. Returns [(node_id, rank), ...]."""
        conn = self._get_conn()
        if store_name:
            rows = conn.execute(
                "SELECT node_id, rank FROM memory_fts "
                "WHERE memory_fts MATCH ? AND store_name = ? "
                "ORDER BY rank LIMIT ?",
                (query, store_name, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT node_id, rank FROM memory_fts "
                "WHERE memory_fts MATCH ? ORDER BY rank LIMIT ?",
                (query, limit),
            ).fetchall()
        return [(row["node_id"], row["rank"]) for row in rows]

    # --- Nodes ---

    @timed("sqlite.put_node")
    def put_node(self, store_name: str, node: MemoryNode) -> None:
        conn = self._get_conn()
        d = node.to_dict()
        embedding = d.pop("embedding", None)
        version = self._next_version(conn, node.id, store_name)
        now = time.time()
        conn.execute(
            "INSERT INTO memory_nodes "
            "(node_id, store_name, version, data, embedding, created_at, inserted_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                node.id,
                store_name,
                version,
                json.dumps(d, ensure_ascii=False),
                self._embedding_to_bytes(embedding),
                node.created_at,
                now,
            ),
        )
        self._fts_index_node(conn, node, store_name)
        conn.commit()

    @timed("sqlite.put_nodes")
    def put_nodes(self, store_name: str, nodes: list[MemoryNode]) -> None:
        if not nodes:
            return
        conn = self._get_conn()
        now = time.time()
        for node in nodes:
            d = node.to_dict()
            embedding = d.pop("embedding", None)
            version = self._next_version(conn, node.id, store_name)
            conn.execute(
                "INSERT INTO memory_nodes "
                "(node_id, store_name, version, data, embedding, created_at, inserted_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    node.id,
                    store_name,
                    version,
                    json.dumps(d, ensure_ascii=False),
                    self._embedding_to_bytes(embedding),
                    node.created_at,
                    now,
                ),
            )
            self._fts_index_node(conn, node, store_name)
        conn.commit()

    @timed("sqlite.get_node")
    def get_node(self, store_name: str, node_id: str) -> MemoryNode | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT data, embedding, is_deleted FROM memory_nodes "
            "WHERE node_id = ? AND store_name = ? "
            "ORDER BY version DESC LIMIT 1",
            (node_id, store_name),
        ).fetchone()
        if not row or row["is_deleted"]:
            return None
        return self._row_to_node(row)

    @timed("sqlite.remove_node")
    def remove_node(self, store_name: str, node_id: str) -> None:
        conn = self._get_conn()
        version = self._next_version(conn, node_id, store_name)
        now = time.time()
        conn.execute(
            "INSERT INTO memory_nodes "
            "(node_id, store_name, version, data, embedding, is_deleted, created_at, inserted_at) "
            "VALUES (?, ?, ?, '{}', NULL, 1, ?, ?)",
            (node_id, store_name, version, now, now),
        )
        conn.commit()

    @timed("sqlite.get_all_nodes")
    def get_all_nodes(self, store_name: str) -> list[MemoryNode]:
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT data, embedding FROM (
                SELECT data, embedding, is_deleted,
                       ROW_NUMBER() OVER (PARTITION BY node_id ORDER BY version DESC) AS rn
                FROM memory_nodes
                WHERE store_name = ?
            ) latest
            WHERE rn = 1 AND NOT is_deleted
            """,
            (store_name,),
        ).fetchall()
        return [self._row_to_node(row) for row in rows]

    # --- Anchors ---

    @timed("sqlite.put_anchors")
    def put_anchors(self, anchors: dict[str, list[dict]]) -> None:
        conn = self._get_conn()
        conn.execute("DELETE FROM anchors")
        for node_id, anchor_list in anchors.items():
            for idx, anchor_data in enumerate(anchor_list):
                conn.execute(
                    "INSERT INTO anchors (node_id, anchor_idx, data) VALUES (?, ?, ?)",
                    (node_id, idx, json.dumps(anchor_data, ensure_ascii=False)),
                )
        conn.commit()

    @timed("sqlite.get_anchors")
    def get_anchors(self) -> dict[str, list[dict]]:
        conn = self._get_conn()
        result: dict[str, list[dict]] = {}
        rows = conn.execute(
            "SELECT node_id, data FROM anchors ORDER BY node_id, anchor_idx"
        ).fetchall()
        for row in rows:
            nid = row["node_id"]
            if nid not in result:
                result[nid] = []
            result[nid].append(json.loads(row["data"]))
        return result

    # --- Temporal ---

    @timed("sqlite.put_temporal")
    def put_temporal(self, contexts: dict[str, TemporalContext]) -> None:
        conn = self._get_conn()
        for nid, ctx in contexts.items():
            conn.execute(
                """
                INSERT OR REPLACE INTO temporal_contexts
                    (node_id, session_id, sequence_pos, timestamp, temporal_neighbors)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    nid,
                    ctx.session_id,
                    ctx.sequence_pos,
                    ctx.timestamp,
                    json.dumps(ctx.temporal_neighbors),
                ),
            )
        conn.commit()

    @timed("sqlite.get_temporal")
    def get_temporal(self) -> dict[str, TemporalContext]:
        conn = self._get_conn()
        result = {}
        rows = conn.execute("SELECT * FROM temporal_contexts").fetchall()
        for row in rows:
            result[row["node_id"]] = TemporalContext(
                timestamp=row["timestamp"],
                session_id=row["session_id"],
                sequence_pos=row["sequence_pos"],
                temporal_neighbors=json.loads(row["temporal_neighbors"]),
            )
        return result

    # --- Events ---

    @timed("sqlite.log_event")
    def log_event(self, event: MemoryEvent) -> None:
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO memory_events "
            "(event_id, event_type, node_id, store_name, node_version, metadata, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                event.event_id,
                event.event_type,
                event.node_id,
                event.store_name,
                event.node_version,
                json.dumps(event.metadata, ensure_ascii=False),
                event.timestamp,
            ),
        )
        conn.commit()

    def query_events(
        self,
        event_type: str | None = None,
        node_id: str | None = None,
        since: float | None = None,
        limit: int = 100,
    ) -> list[MemoryEvent]:
        conn = self._get_conn()
        query = "SELECT * FROM memory_events WHERE 1=1"
        params: list[Any] = []
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        if node_id:
            query += " AND node_id = ?"
            params.append(node_id)
        if since:
            query += " AND timestamp >= ?"
            params.append(since)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [
            MemoryEvent(
                event_id=row["event_id"],
                event_type=row["event_type"],
                node_id=row["node_id"],
                store_name=row["store_name"],
                timestamp=row["timestamp"],
                metadata=json.loads(row["metadata"]),
                node_version=row["node_version"],
            )
            for row in rows
        ]

    # --- Advanced temporal queries ---

    @timed("sqlite.query_by_time_range")
    def query_by_time_range(
        self,
        start: float,
        end: float,
        store_name: str | None = None,
    ) -> list[MemoryNode]:
        conn = self._get_conn()
        if store_name:
            rows = conn.execute(
                """
                SELECT data, embedding FROM (
                    SELECT data, embedding, is_deleted,
                           ROW_NUMBER() OVER (PARTITION BY node_id ORDER BY version DESC) AS rn
                    FROM memory_nodes
                    WHERE created_at BETWEEN ? AND ? AND store_name = ?
                ) latest
                WHERE rn = 1 AND NOT is_deleted
                """,
                (start, end, store_name),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT data, embedding FROM (
                    SELECT data, embedding, is_deleted,
                           ROW_NUMBER() OVER (PARTITION BY node_id ORDER BY version DESC) AS rn
                    FROM memory_nodes
                    WHERE created_at BETWEEN ? AND ?
                ) latest
                WHERE rn = 1 AND NOT is_deleted
                """,
                (start, end),
            ).fetchall()
        return [self._row_to_node(row) for row in rows]

    @timed("sqlite.query_by_session")
    def query_by_session(self, session_id: str) -> list[str]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT node_id FROM temporal_contexts WHERE session_id = ? "
            "ORDER BY sequence_pos",
            (session_id,),
        ).fetchall()
        return [row["node_id"] for row in rows]

    # --- Versioning ---

    @timed("sqlite.get_node_history")
    def get_node_history(
        self, store_name: str, node_id: str
    ) -> list[MemoryNode]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT data, embedding FROM memory_nodes "
            "WHERE node_id = ? AND store_name = ? "
            "ORDER BY version DESC",
            (node_id, store_name),
        ).fetchall()
        return [self._row_to_node(row) for row in rows]

    # --- Export/Import ---

    @timed("sqlite.export_all")
    def export_all(self) -> dict[str, Any]:
        return {
            "episodic": [n.to_dict() for n in self.get_all_nodes("episodic")],
            "semantic": [n.to_dict() for n in self.get_all_nodes("semantic")],
            "anchors": self.get_anchors(),
            "temporal": {
                nid: ctx.to_dict() for nid, ctx in self.get_temporal().items()
            },
        }

    @timed("sqlite.import_all")
    def import_all(self, data: dict[str, Any]) -> None:
        if "episodic" in data:
            nodes = [MemoryNode.from_dict(d) for d in data["episodic"]]
            self.put_nodes("episodic", nodes)
        if "semantic" in data:
            nodes = [MemoryNode.from_dict(d) for d in data["semantic"]]
            self.put_nodes("semantic", nodes)
        if "anchors" in data:
            self.put_anchors(data["anchors"])
        if "temporal" in data:
            contexts = {
                nid: TemporalContext.from_dict(d)
                for nid, d in data["temporal"].items()
            }
            self.put_temporal(contexts)
