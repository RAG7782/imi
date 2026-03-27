"""Bidirectional migration between IMI storage backends.

Usage:
    python -m imi.migrate json2tsdb --json-dir ./imi_data --conn-string postgresql://imi:imi_dev@localhost:5433/imi
    python -m imi.migrate tsdb2json --json-dir ./imi_export --conn-string postgresql://imi:imi_dev@localhost:5433/imi
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from imi.events import MIGRATE_IN, MIGRATE_OUT, MemoryEvent
from imi.storage import JSONBackend, TimescaleDBBackend

logger = logging.getLogger("imi.migrate")


def json_to_timescale(json_dir: Path, conn_string: str) -> int:
    """Migrate existing JSON persistence to TimescaleDB.

    Returns number of nodes migrated.
    """
    json_backend = JSONBackend(json_dir)
    tsdb_backend = TimescaleDBBackend(conn_string)
    tsdb_backend.setup()

    data = json_backend.export_all()
    tsdb_backend.import_all(data)

    count = len(data.get("episodic", [])) + len(data.get("semantic", []))

    tsdb_backend.log_event(
        MemoryEvent(
            event_type=MIGRATE_IN,
            node_id="*",
            store_name="*",
            metadata={
                "source": str(json_dir),
                "node_count": count,
                "episodic": len(data.get("episodic", [])),
                "semantic": len(data.get("semantic", [])),
                "anchors": len(data.get("anchors", {})),
                "temporal": len(data.get("temporal", {})),
            },
        )
    )

    logger.info("Migrated %d nodes from %s to TimescaleDB", count, json_dir)
    tsdb_backend.close()
    return count


def timescale_to_json(conn_string: str, json_dir: Path) -> int:
    """Export TimescaleDB state to JSON directory.

    Returns number of nodes exported.
    """
    tsdb_backend = TimescaleDBBackend(conn_string)
    json_backend = JSONBackend(json_dir)
    json_backend.setup()

    data = tsdb_backend.export_all()
    json_backend.import_all(data)

    count = len(data.get("episodic", [])) + len(data.get("semantic", []))

    tsdb_backend.log_event(
        MemoryEvent(
            event_type=MIGRATE_OUT,
            node_id="*",
            store_name="*",
            metadata={
                "target": str(json_dir),
                "node_count": count,
            },
        )
    )

    logger.info("Exported %d nodes from TimescaleDB to %s", count, json_dir)
    tsdb_backend.close()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate IMI data between storage backends"
    )
    parser.add_argument(
        "direction",
        choices=["json2tsdb", "tsdb2json"],
        help="Migration direction",
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        required=True,
        help="Path to JSON persistence directory",
    )
    parser.add_argument(
        "--conn-string",
        type=str,
        required=True,
        help="TimescaleDB connection string",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.direction == "json2tsdb":
        count = json_to_timescale(args.json_dir, args.conn_string)
    else:
        count = timescale_to_json(args.conn_string, args.json_dir)

    print(f"Done. {count} nodes migrated.")


if __name__ == "__main__":
    main()
