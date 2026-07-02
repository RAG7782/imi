"""canary.py — Continuity Canary: silent-drift detector for IMI memory.

Spec: ~/experimentos/specs/2026-06-14-imi-hmem-positional-index.md (CANÁRIO)
Pre-mortem cause #4 (escore 20): "memória ruim não grita". A canary turns
silent retrieval drift into a loud, boot-time signal.

PORQUÊ this is the detector BEFORE the mechanism:
  In a memory system the failure mode is silent — retrieval returns something
  plausible, just not the right thing. Before adding ANY new retrieval path
  (H-MEM recursive search), we need a fixed set of anchor queries whose targets
  are KNOWN, run on every boot. If an anchor stops returning its target, the
  retrieval broke — and we hear about it the same day, not weeks later.

DESIGN — robust by construction:
  - Uses LEXICAL (FTS5) search, NOT the embedder. The detector must not share
    the failure mode it detects: if Ollama is down, semantic search silently
    degrades, but the canary still runs (DCI lexical path, mcp_server._lexical_search).
  - Anchors are (distinctive_token, expected_node_id) pairs. The token is a
    rare lexical string (SHA, unique tag, dated phrase) that FTS5 matches exactly.
  - A seed anchor set lives in ~/.imi/canary_anchors.json. If absent, derive a
    candidate set from high-salience dated nodes and WRITE it once (then a human
    reviews/freezes it). Auto-derivation is bootstrap, not source of truth.

Exit contract (for boot hook):
  - all anchors hit  -> status "ok",       exit 0
  - any anchor misses-> status "drift",     exit 1  (LOUD: boot block surfaces it)
  - cannot run       -> status "unavailable",exit 2  (FTS missing / no anchors)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_ANCHORS_PATH = Path.home() / ".imi" / "canary_anchors.json"
_DEFAULT_DB = Path.home() / "experimentos/tools/imi/imi_memory.db"


@dataclass
class Anchor:
    """One canary probe: a distinctive token that MUST retrieve expected_id."""
    token: str          # rare lexical string (FTS5-matchable): SHA, tag, dated phrase
    expected_id: str    # node id (12-char) that must appear in top_k for `token`
    note: str = ""      # human-readable description of what this memory is


@dataclass
class CanaryReport:
    total: int = 0
    hits: int = 0
    misses: list[dict] = field(default_factory=list)  # [{token, expected_id, got_ids}]
    status: str = "ok"  # ok | drift | unavailable
    reason: str = ""

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    def __str__(self) -> str:
        if self.status == "unavailable":
            return f"[CANARY] unavailable — {self.reason}"
        line = (
            f"[CANARY] {self.status} — {self.hits}/{self.total} "
            f"anchors hit ({self.hit_rate:.0%})"
        )
        if self.misses:
            line += "\n  MISSES (retrieval drift):"
            for m in self.misses:
                line += f"\n    ✗ '{m['token']}' expected {m['expected_id']} — got {m['got_ids']}"
        return line


def load_anchors(path: Path = _ANCHORS_PATH) -> list[Anchor]:
    """Load the frozen anchor set. Empty list if file absent (caller derives)."""
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return [Anchor(**a) for a in data]


def derive_candidate_anchors(db_path: Path = _DEFAULT_DB, n: int = 20,
                             space: Any = None) -> list[Anchor]:
    """Bootstrap: derive candidate anchors from high-salience nodes.

    NOT source of truth — writes a candidate file for a human to review/freeze.

    A GOOD anchor token is DISTINCTIVE: it must retrieve its own node at rank 1
    via FTS5. A frequent tag ("gravar", "agentes") matches dozens of nodes and
    the target sinks — that is a bad anchor (lesson learned 2026-06-14: the first
    derivation used tags and got 10% hit-rate, all false positives). So each
    candidate token is VALIDATED against lexical search before being accepted:
    the token is kept only if expected_id is rank-1 for it. Self-validating
    derivation — only anchors that already pass become anchors.
    """
    import sqlite3

    if space is None:
        from imi.space import IMISpace
        space = IMISpace.from_sqlite(str(db_path))

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT n.node_id, n.data FROM memory_nodes n "
        "INNER JOIN (SELECT node_id, MAX(version) mv FROM memory_nodes "
        "WHERE is_deleted=0 GROUP BY node_id) l "
        "ON n.node_id=l.node_id AND n.version=l.mv WHERE n.is_deleted=0 "
        "ORDER BY n.node_id"
    ).fetchall()
    conn.close()

    candidates: list[Anchor] = []
    for r in rows:
        if not r["data"]:
            continue
        try:
            d = json.loads(r["data"])
        except (json.JSONDecodeError, TypeError):
            continue
        affect = d.get("affect") or {}
        sal = affect.get("salience", 0) if isinstance(affect, dict) else 0
        if sal < 0.9:
            continue
        nid = r["node_id"][:12]
        summary = d.get("summary_orbital") or d.get("summary_medium") or ""
        # Candidate tokens: multi-word tags (more distinctive) + the node-id prefix
        # itself (the most distinctive token possible — always rank-1 if indexed).
        tags = d.get("tags") or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        token_pool = [t for t in tags if t and len(t) >= 10]  # long tags = rarer
        token_pool.append(nid)  # node-id prefix: maximally distinctive fallback

        for token in token_pool:
            # VALIDATE: does this token retrieve expected_id at rank 1?
            try:
                hits = _validate_token(space, token, nid)
            except RuntimeError:
                break  # FTS unavailable — abort derivation
            if hits:
                candidates.append(Anchor(token=token, expected_id=nid, note=summary[:60]))
                break  # one good anchor per node is enough
        if len(candidates) >= n:
            break
    return candidates


def _validate_token(space: Any, token: str, expected_id: str) -> bool:
    """True iff `token` retrieves `expected_id` at rank 1 via lexical search."""
    hits = lexical_search(space, token, 1)
    return bool(hits) and hits[0].get("id", "")[:12] == expected_id


def lexical_search(space: Any, query: str, top_k: int) -> list[dict]:
    """Embedder-free FTS5 search — the canary's own retrieval path.

    Inlined (not imported from mcp_server) on purpose: the canary must not pull
    in FastMCP/the whole MCP server just to run a query. This is the same FTS5
    logic mcp_server._lexical_search uses, kept dependency-light so the detector
    runs anywhere (CI, boot hook, no `mcp` package). Survives Ollama outage —
    it never touches the embedder.
    """
    backend = getattr(space, "backend", None)
    if backend is None or not hasattr(backend, "search_fts"):
        raise RuntimeError(
            "lexical mode requires SQLiteBackend with FTS5; "
            f"active backend: {type(backend).__name__ if backend else None}"
        )
    # FTS5 treats -, :, . as syntax; quote a single bare term to force literal match.
    fts_query = query if " " in query.strip() else f'"{query.strip()}"'
    try:
        raw = backend.search_fts(fts_query, limit=top_k * 3)
    except Exception as e:  # noqa: BLE001 — offending value logged, not swallowed
        sys.stderr.write(f"[canary:lexical] search_fts failed for {fts_query!r}: {e}\n")
        return []
    out = []
    for node_id, rank in raw[:top_k]:
        out.append({"id": node_id, "score": round(float(-rank), 3)})
    return out


def run_canary(space: Any, anchors: list[Anchor], top_k: int = 10) -> CanaryReport:
    """Run each anchor through LEXICAL search; verify expected_id in top_k.

    `space` is an open IMI Space. Uses the embedder-free FTS5 path so the
    canary survives an Ollama outage (the very thing it must detect around).
    """
    report = CanaryReport(total=len(anchors))
    if not anchors:
        report.status = "unavailable"
        report.reason = (
            "no anchors — run derive_candidate_anchors and freeze ~/.imi/canary_anchors.json"
        )
        return report

    for a in anchors:
        try:
            hits = lexical_search(space, a.token, top_k)
        except RuntimeError as e:  # FTS5 unavailable — detector itself can't run
            report.status = "unavailable"
            report.reason = str(e)
            return report
        got_ids = [h.get("id", "")[:12] for h in hits]
        if a.expected_id in got_ids:
            report.hits += 1
        else:
            report.misses.append(
                {"token": a.token, "expected_id": a.expected_id, "got_ids": got_ids[:5]}
            )

    report.status = "ok" if not report.misses else "drift"
    return report


def main() -> int:
    """CLI: python -m imi.canary [--derive] [--db PATH].

    Exit 0=ok, 1=drift (LOUD), 2=unavailable. Designed for the boot hook.
    """
    import argparse

    p = argparse.ArgumentParser(description="IMI Continuity Canary — silent-drift detector")
    p.add_argument("--db", type=Path, default=_DEFAULT_DB)
    p.add_argument("--derive", action="store_true",
                   help="Derive candidate anchors from the DB and write the seed file")
    p.add_argument("--top-k", type=int, default=10)
    args = p.parse_args()

    if args.derive:
        cands = derive_candidate_anchors(args.db)
        _ANCHORS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _ANCHORS_PATH.write_text(
            json.dumps([{"token": a.token, "expected_id": a.expected_id, "note": a.note}
                        for a in cands], ensure_ascii=False, indent=2)
        )
        print(f"[CANARY] derived {len(cands)} candidate anchors → {_ANCHORS_PATH}")
        print("  Review and freeze this file. Then run without --derive.")
        return 0

    # Same factory production uses (mcp_server._get_space → IMISpace.from_sqlite).
    # No embedder/llm passed: the canary is lexical-only by design (FTS5, no Ollama).
    from imi.space import IMISpace

    space = IMISpace.from_sqlite(str(args.db))
    report = run_canary(space, load_anchors(), top_k=args.top_k)
    print(report)
    return {"ok": 0, "drift": 1, "unavailable": 2}[report.status]


if __name__ == "__main__":
    sys.exit(main())
