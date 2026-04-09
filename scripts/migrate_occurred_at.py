#!/usr/bin/env python3
"""
Migrate existing IMI nodes to include occurred_at.

Strategy: infer occurred_at from the `source` field in each node's data.
Source fields like "session-7abr" map to known dates.
For nodes without a session source, use created_at as occurred_at.

Usage:
    python scripts/migrate_occurred_at.py [--db PATH] [--dry-run]
"""

import argparse
import json
import sqlite3
from datetime import datetime, timezone, timedelta

# São Paulo timezone (UTC-3)
SP_TZ = timezone(timedelta(hours=-3))

# Source → occurred_at mapping (inferred from handoffs and session dates)
# Format: source_prefix → (year, month, day)
SOURCE_DATE_MAP = {
    "session-6abr": (2026, 4, 6),
    "session-6abr-2026": (2026, 4, 6),
    "session-7abr": (2026, 4, 7),
    "session-7abr-2": (2026, 4, 7),
    "session-7abr-b": (2026, 4, 7),
    "session-7abr-cc-analysis": (2026, 4, 7),
    "session-7abr-final": (2026, 4, 7),
    "session-7abr-legal-tech-pt2": (2026, 4, 7),
    "session-7abr-pt2": (2026, 4, 7),
    "session-7abr-pt3": (2026, 4, 7),
    "session-8abr": (2026, 4, 8),
    "session-8abr-gravar": (2026, 4, 8),
    "session-8-9abr": (2026, 4, 8),  # spans 8-9, anchor to 8
    "session-9abr": (2026, 4, 9),
    "session-9abr-backup": (2026, 4, 9),
    "session-9abr-benchmarks": (2026, 4, 9),
    "session-9abr-deploy": (2026, 4, 9),
    "session-9abr-e2e": (2026, 4, 9),
    "session-9abr-final": (2026, 4, 9),
    "session-9abr-gravar": (2026, 4, 9),
    "session-9abr-integration": (2026, 4, 9),
    "session-9abr-investidor": (2026, 4, 9),
    "session-9abr-monetizacao": (2026, 4, 9),
    "session-9abr-neuroclaw": (2026, 4, 9),
    "session-9abr-opportunities": (2026, 4, 9),
    "session-9abr-papers": (2026, 4, 9),
    "session-9abr-PI": (2026, 4, 9),
    "session-9abr-v5update": (2026, 4, 9),
}


def infer_occurred_at(source: str, created_at: float) -> float:
    """Infer when a memory event actually occurred."""
    if not source:
        return created_at  # No source info → assume real-time

    # Direct match
    if source in SOURCE_DATE_MAP:
        y, m, d = SOURCE_DATE_MAP[source]
        # Use 14:00 local time as default anchor (middle of workday)
        dt = datetime(y, m, d, 14, 0, 0, tzinfo=SP_TZ)
        return dt.timestamp()

    # Prefix match (e.g., "session-7abr-something-new")
    for prefix, (y, m, d) in sorted(SOURCE_DATE_MAP.items(), key=lambda x: -len(x[0])):
        if source.startswith(prefix):
            dt = datetime(y, m, d, 14, 0, 0, tzinfo=SP_TZ)
            return dt.timestamp()

    # Non-session sources (claude-code, test, etc.) → use created_at
    return created_at


def migrate(db_path: str, dry_run: bool = False):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")

    # Get latest version of each episodic node (by MAX(id), not MAX(version))
    rows = conn.execute("""
        SELECT mn.id, mn.node_id, 0 as version, mn.data, mn.created_at
        FROM memory_nodes mn
        INNER JOIN (
            SELECT node_id, MAX(id) as max_id
            FROM memory_nodes
            WHERE store_name = 'episodic' AND is_deleted = 0
            GROUP BY node_id
        ) latest ON mn.id = latest.max_id
    """).fetchall()

    updated = 0
    skipped = 0
    by_source = {}

    for row_id, node_id, version, data_json, created_at in rows:
        data = json.loads(data_json)

        # Skip if already has occurred_at
        if data.get("occurred_at") is not None:
            skipped += 1
            continue

        source = data.get("source", "")
        occurred_at = infer_occurred_at(source, created_at)
        data["occurred_at"] = occurred_at

        # Track stats
        date_str = datetime.fromtimestamp(occurred_at).strftime("%Y-%m-%d")
        by_source.setdefault(source or "(empty)", []).append(date_str)

        if not dry_run:
            conn.execute(
                "UPDATE memory_nodes SET data = ? WHERE id = ?",
                (json.dumps(data, ensure_ascii=False), row_id),
            )
        updated += 1

    if not dry_run:
        conn.commit()

    # Report
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Migration complete:")
    print(f"  Updated: {updated}")
    print(f"  Skipped (already has occurred_at): {skipped}")
    print(f"\n  Source → inferred date:")
    for src, dates in sorted(by_source.items()):
        unique_dates = sorted(set(dates))
        print(f"    {src}: {len(dates)} nodes → {', '.join(unique_dates)}")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate IMI nodes with occurred_at")
    parser.add_argument("--db", default="imi_memory.db", help="Path to IMI SQLite DB")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    args = parser.parse_args()

    migrate(args.db, dry_run=args.dry_run)
