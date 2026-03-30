"""Bridge between AGORA-OS KB (NAOs) and IMI (MemoryNodes).

Syncs knowledge between:
- AGORA KB: semantic memory (facts, procedures, relations) via SQLite
- IMI: episodic memory (experiences, decisions, temporal context)

Two-way sync:
  KB → IMI: Import NAOs as MemoryNodes (for temporal/affordance enrichment)
  IMI → KB: Export consolidated patterns as NAOs (for knowledge graph)

Relation mapping:
  AGORA KB                    IMI Graph
  ─────────                   ─────────
  causes                  →   CAUSAL
  temporal_precedes       →   CAUSAL (label="precedes")
  results_from            →   CAUSAL (label="results_from")
  supports, similar_to    →   SIMILAR
  related_to, part_of     →   CO_OCCURRENCE
  contradicts             →   (no equivalent — logged as metadata)

Usage:
    from integrations.agora_os.bridge import AgoraIMIBridge

    bridge = AgoraIMIBridge(
        kb_db="~/.claude/plugins/data/agora-os/kb.sqlite",
        imi_db="~/.claude/plugins/data/imi/agora-memory.db",
    )

    # Import all KB NAOs into IMI
    stats = bridge.import_naos_to_imi()

    # Export IMI patterns back to KB
    stats = bridge.export_patterns_to_kb()

    # Unified search across both
    results = bridge.unified_search("circuit breaker pattern")
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from imi.graph import EdgeType


# AGORA KB relation → IMI edge type mapping
RELATION_TO_EDGE = {
    # Causal
    "causes": (EdgeType.CAUSAL, "causes"),
    "temporal_precedes": (EdgeType.CAUSAL, "precedes"),
    "results_from": (EdgeType.CAUSAL, "results_from"),
    "inferred_from": (EdgeType.CAUSAL, "inferred_from"),
    # Similar
    "supports": (EdgeType.SIMILAR, "supports"),
    "similar_to": (EdgeType.SIMILAR, "similar_to"),
    "refines": (EdgeType.SIMILAR, "refines"),
    "version_of": (EdgeType.SIMILAR, "version_of"),
    "complements": (EdgeType.SIMILAR, "complements"),
    # Co-occurrence
    "related_to": (EdgeType.CO_OCCURRENCE, "related_to"),
    "part_of": (EdgeType.CO_OCCURRENCE, "part_of"),
    "is_a": (EdgeType.CO_OCCURRENCE, "is_a"),
    "requires_truth_of": (EdgeType.CO_OCCURRENCE, "requires"),
    "cites": (EdgeType.CO_OCCURRENCE, "cites"),
    # Domain-specific → CO_OCCURRENCE
    "heir_of": (EdgeType.CO_OCCURRENCE, "heir_of"),
    "obligates": (EdgeType.CO_OCCURRENCE, "obligates"),
    "evidence_for": (EdgeType.CO_OCCURRENCE, "evidence_for"),
    "precedent_for": (EdgeType.CO_OCCURRENCE, "precedent_for"),
    "controls": (EdgeType.CO_OCCURRENCE, "controls"),
}

# IMI edge type → best AGORA KB relation
EDGE_TO_RELATION = {
    EdgeType.CAUSAL: "causes",
    EdgeType.SIMILAR: "similar_to",
    EdgeType.CO_OCCURRENCE: "related_to",
}


@dataclass
class SyncStats:
    """Stats from a sync operation."""
    nodes_imported: int = 0
    edges_imported: int = 0
    nodes_exported: int = 0
    edges_exported: int = 0
    errors: int = 0

    def __str__(self) -> str:
        return (
            f"Sync: imported {self.nodes_imported} nodes + {self.edges_imported} edges, "
            f"exported {self.nodes_exported} nodes + {self.edges_exported} edges, "
            f"{self.errors} errors"
        )


class AgoraIMIBridge:
    """Bidirectional bridge between AGORA KB and IMI."""

    def __init__(
        self,
        kb_db: str = "~/.claude/plugins/data/agora-os/kb.sqlite",
        imi_db: str = "~/.claude/plugins/data/imi/agora-memory.db",
    ):
        self.kb_db_path = Path(kb_db).expanduser()
        self.imi_db_path = str(Path(imi_db).expanduser())
        self._space = None

    def _get_space(self):
        if self._space is None:
            from imi.space import IMISpace
            self._space = IMISpace.from_sqlite(self.imi_db_path)
        return self._space

    def _get_kb_conn(self) -> sqlite3.Connection:
        if not self.kb_db_path.exists():
            raise FileNotFoundError(f"AGORA KB not found: {self.kb_db_path}")
        return sqlite3.connect(str(self.kb_db_path))

    def import_naos_to_imi(
        self,
        domain: str | None = None,
        min_confidence: int = 0,
        nao_types: list[str] | None = None,
    ) -> SyncStats:
        """Import NAOs from AGORA KB into IMI as episodic memories.

        Args:
            domain: Filter by domain (None = all)
            min_confidence: Minimum confidence score (0-100)
            nao_types: Filter by NAO type (fact, claim, procedure, etc.)
        """
        stats = SyncStats()
        space = self._get_space()
        conn = self._get_kb_conn()

        try:
            # Build query
            query = "SELECT id, type, content, confidence, created_at FROM naos WHERE confidence >= ?"
            params: list[Any] = [min_confidence]

            if domain:
                query += " AND json_extract(metadata, '$.domain') = ?"
                params.append(domain)
            if nao_types:
                placeholders = ",".join(["?" for _ in nao_types])
                query += f" AND type IN ({placeholders})"
                params.extend(nao_types)

            cursor = conn.execute(query, params)
            existing_ids = {n.id for n in space.episodic.nodes}

            for row in cursor:
                nao_id, nao_type, content, confidence, created_at = row
                # Skip if already imported
                tag_id = f"nao:{nao_id}"
                if any(tag_id in n.tags for n in space.episodic.nodes):
                    continue

                try:
                    space.encode(
                        content,
                        tags=["agora-kb", nao_type, f"nao:{nao_id}", f"confidence:{confidence}"],
                        source="agora-kb",
                        context_hint=f"AGORA KB {nao_type} (confidence={confidence})",
                    )
                    stats.nodes_imported += 1
                except Exception:
                    stats.errors += 1

            # Import relations as graph edges
            rel_query = "SELECT from_nao_id, to_nao_id, relation_type FROM relations"
            if min_confidence > 0:
                rel_query += f" WHERE confidence >= {min_confidence}"

            for row in conn.execute(rel_query):
                from_id, to_id, rel_type = row
                mapping = RELATION_TO_EDGE.get(rel_type)
                if mapping:
                    edge_type, label = mapping
                    # Find IMI node IDs for these NAOs
                    from_nodes = [n for n in space.episodic.nodes if f"nao:{from_id}" in n.tags]
                    to_nodes = [n for n in space.episodic.nodes if f"nao:{to_id}" in n.tags]
                    if from_nodes and to_nodes:
                        space.graph.add_edge(
                            from_nodes[0].id, to_nodes[0].id,
                            edge_type, label=label,
                        )
                        stats.edges_imported += 1

        finally:
            conn.close()

        return stats

    def export_patterns_to_kb(self) -> SyncStats:
        """Export IMI consolidated patterns back to AGORA KB as NAOs.

        Only exports semantic memories (consolidated by dream()) that
        don't already exist in the KB.
        """
        stats = SyncStats()
        space = self._get_space()

        if not self.kb_db_path.exists():
            return stats

        conn = self._get_kb_conn()
        try:
            for node in space.semantic.nodes:
                # Skip if already has a KB tag
                if any(t.startswith("nao:") for t in node.tags):
                    continue

                # Create as a "claim" NAO (consolidated pattern, not verified fact)
                try:
                    import uuid
                    nao_id = str(uuid.uuid4())[:26]  # ULID-like
                    conn.execute(
                        """INSERT INTO naos (id, type, content, confidence, metadata, created_at, updated_at)
                           VALUES (?, 'claim', ?, ?, ?, datetime('now'), datetime('now'))""",
                        (
                            nao_id,
                            node.seed[:2000],
                            int(node.mass * 100),  # mass → confidence
                            json.dumps({
                                "extraction_method": "imi-consolidation",
                                "domain": "general",
                                "requires_validation": True,
                                "validation_status": "unreviewed",
                                "source": "imi-dream",
                            }),
                        ),
                    )
                    # Tag the IMI node so we don't re-export
                    node.tags.append(f"nao:{nao_id}")
                    stats.nodes_exported += 1
                except Exception:
                    stats.errors += 1

            conn.commit()

            # Export graph edges as relations
            for edge in space.graph.edges:
                rel_type = EDGE_TO_RELATION.get(edge.edge_type, "related_to")
                from_nodes = [n for n in space.episodic.nodes if n.id == edge.source]
                to_nodes = [n for n in space.episodic.nodes if n.id == edge.target]

                if from_nodes and to_nodes:
                    from_nao = next((t[4:] for t in from_nodes[0].tags if t.startswith("nao:")), None)
                    to_nao = next((t[4:] for t in to_nodes[0].tags if t.startswith("nao:")), None)
                    if from_nao and to_nao:
                        try:
                            conn.execute(
                                """INSERT OR IGNORE INTO relations
                                   (from_nao_id, to_nao_id, relation_type, confidence, strength)
                                   VALUES (?, ?, ?, ?, 'medium')""",
                                (from_nao, to_nao, rel_type, int(edge.weight * 100)),
                            )
                            stats.edges_exported += 1
                        except Exception:
                            stats.errors += 1

            conn.commit()
        finally:
            conn.close()

        return stats

    def unified_search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Search across both AGORA KB and IMI, merged and ranked.

        Returns results from both systems with source attribution.
        """
        results = []
        space = self._get_space()

        # IMI search (episodic + semantic)
        nav = space.navigate(query, top_k=top_k)
        for m in nav.memories:
            results.append({
                "source": "imi",
                "score": m["score"],
                "content": m["content"],
                "type": "episodic",
                "tags": m.get("tags", []),
                "affordances": m.get("affordances", []),
            })

        # KB search (if available)
        if self.kb_db_path.exists():
            conn = self._get_kb_conn()
            try:
                # Simple content search (KB doesn't have embeddings)
                keywords = query.lower().split()
                where_clauses = [f"LOWER(content) LIKE '%{kw}%'" for kw in keywords[:5]]
                if where_clauses:
                    sql = f"SELECT id, type, content, confidence FROM naos WHERE {' OR '.join(where_clauses)} ORDER BY confidence DESC LIMIT ?"
                    for row in conn.execute(sql, (top_k,)):
                        nao_id, nao_type, content, confidence = row
                        results.append({
                            "source": "agora-kb",
                            "score": confidence / 100.0,
                            "content": content,
                            "type": nao_type,
                            "tags": [f"nao:{nao_id}", nao_type],
                            "affordances": [],
                        })
            finally:
                conn.close()

        # Sort by score (IMI scores and KB confidence are both 0-1)
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
