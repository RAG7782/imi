"""memory_worth.py — MemoryWorth: análise e auditoria de salience dinâmica (IMI-E02 S06).

Módulo standalone que expõe:
  1. Funções de análise: distribuição de salience, nós em risco de sub-utilização
  2. CLI de auditoria: `python3 -m imi.memory_worth [--report] [--decay-dry-run]`

Filosofia:
  - O loop de feedback fecha via im_feedback() no MCP server
  - Este módulo NÃO altera salience — apenas analisa e reporta
  - Fonte única de verdade para regras de MemoryWorth

Regras de salience:
  ┌─────────────────────────────────────────────────────────────────┐
  │ Positive outcome: Δs = magnitude × 0.1 × (1 - current)        │
  │ Negative outcome: Δs = −magnitude × 0.1 × current             │
  │ Access (touch):   Δs = 0.05 × log2(access_count + 1) + base   │
  │ Cap: [0.1, 0.95] para todos os updates                        │
  └─────────────────────────────────────────────────────────────────┘

Baseado em:
  - Damasio: somatic marker hypothesis (affect modulates memory)
  - Ebbinghaus: forgetting curve (decaimento por não-uso)
  - Bayesian update: aprendizado incremental com prior
  - Power law of practice: benefício marginal decrescente com acesso
"""

from __future__ import annotations

import json
import sqlite3
import sys
from dataclasses import dataclass, field
from math import log2
from pathlib import Path
from typing import Iterator

# ── Caminhos canônicos ───────────────────────────────────────────────────────
_HOME = Path.home()
_IMI_DB = _HOME / "experimentos/tools/imi/imi_memory.db"


# ── Tipos de análise ─────────────────────────────────────────────────────────

@dataclass
class NodeWorth:
    """Perfil de MemoryWorth para um único nó."""
    node_id: str
    store: str
    salience: float
    access_count: int
    valence: float
    tags: list[str]
    summary: str

    @property
    def tier(self) -> str:
        """Classifica o nó segundo sua salience atual."""
        if self.salience >= 0.8:
            return "HIGH"
        elif self.salience >= 0.6:
            return "MEDIUM"
        elif self.salience >= 0.4:
            return "LOW"
        else:
            return "AT_RISK"

    @property
    def projected_after_positive(self) -> float:
        """Salience projetada após um feedback positivo de magnitude 0.5."""
        delta = 0.5 * 0.1 * (1.0 - self.salience)
        return min(0.95, self.salience + delta)

    @property
    def projected_after_negative(self) -> float:
        """Salience projetada após um feedback negativo de magnitude 0.5."""
        delta = 0.5 * 0.1 * self.salience
        return max(0.1, self.salience - delta)

    @property
    def projected_after_access(self) -> float:
        """Salience projetada após mais um acesso (via touch/update_dynamic)."""
        next_count = self.access_count + 1
        # base_salience não recuperável do DB sem histórico — usar salience atual
        # como aproximação conservadora
        boost = 0.05 * log2(next_count + 1)
        return min(0.95, self.salience + boost)


@dataclass
class WorthReport:
    """Relatório agregado de MemoryWorth para um espaço de memória."""
    total_nodes: int = 0
    high_count: int = 0       # salience >= 0.8
    medium_count: int = 0     # 0.6 <= salience < 0.8
    low_count: int = 0        # 0.4 <= salience < 0.6
    at_risk_count: int = 0    # salience < 0.4
    zero_access_count: int = 0  # nenhum acesso registrado
    mean_salience: float = 0.0
    median_salience: float = 0.0
    nodes_at_risk: list[NodeWorth] = field(default_factory=list)  # AT_RISK, sample


# ── Leitura do DB ─────────────────────────────────────────────────────────────

def _iter_nodes(db_path: Path = _IMI_DB) -> Iterator[tuple[str, str, dict]]:
    """Itera nós vivos (versão mais recente, não-deletados) do IMI SQLite.

    Yields: (node_id, store_name, data_dict)
    """
    if not db_path.exists():
        raise FileNotFoundError(f"IMI DB não encontrado: {db_path}")

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("""
            SELECT n.node_id, n.store_name, n.data
            FROM memory_nodes n
            INNER JOIN (
                SELECT node_id, MAX(version) AS max_ver
                FROM memory_nodes
                WHERE is_deleted = 0
                GROUP BY node_id
            ) latest ON n.node_id = latest.node_id AND n.version = latest.max_ver
            WHERE n.is_deleted = 0
            ORDER BY n.node_id
        """).fetchall()
    finally:
        conn.close()

    for row in rows:
        try:
            data = json.loads(row["data"]) if row["data"] else {}
        except (json.JSONDecodeError, TypeError):
            continue
        yield row["node_id"], row["store_name"], data


def _extract_worth(node_id: str, store: str, data: dict) -> NodeWorth:
    """Extrai NodeWorth de um dict de dados do DB."""
    affect = data.get("affect") or {}
    if isinstance(affect, str):
        try:
            affect = json.loads(affect)
        except Exception:
            affect = {}

    tags_raw = data.get("tags", [])
    if isinstance(tags_raw, str):
        try:
            tags = json.loads(tags_raw)
        except Exception:
            tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
    else:
        tags = tags_raw or []

    summary = (
        data.get("summary_orbital")
        or data.get("summary_medium")
        or data.get("content")
        or data.get("seed")
        or ""
    )
    if isinstance(summary, str):
        summary = summary[:80].replace("\n", " ")

    return NodeWorth(
        node_id=node_id[:12],
        store=store,
        salience=float(affect.get("salience", 0.5)),
        access_count=int(data.get("access_count", 0)),
        valence=float(affect.get("valence", 0.0)),
        tags=tags[:6],
        summary=summary,
    )


# ── Análise ───────────────────────────────────────────────────────────────────

def analyze(db_path: Path = _IMI_DB, max_at_risk: int = 10) -> WorthReport:
    """Analisa distribuição de MemoryWorth de todos os nós.

    Fluxo (step1→step2→step3→output):
        step1: ler todos os nós vivos do DB
        step2: calcular tier e contadores
        step3: computar estatísticas agregadas
        output: WorthReport com distribuição completa
    """
    report = WorthReport()
    saliences: list[float] = []
    at_risk_nodes: list[NodeWorth] = []

    for node_id, store, data in _iter_nodes(db_path):
        nw = _extract_worth(node_id, store, data)
        report.total_nodes += 1
        saliences.append(nw.salience)

        if nw.tier == "HIGH":
            report.high_count += 1
        elif nw.tier == "MEDIUM":
            report.medium_count += 1
        elif nw.tier == "LOW":
            report.low_count += 1
        else:
            report.at_risk_count += 1
            if len(at_risk_nodes) < max_at_risk:
                at_risk_nodes.append(nw)

        if nw.access_count == 0:
            report.zero_access_count += 1

    if saliences:
        report.mean_salience = sum(saliences) / len(saliences)
        sorted_s = sorted(saliences)
        mid = len(sorted_s) // 2
        report.median_salience = (
            sorted_s[mid] if len(sorted_s) % 2 == 1
            else (sorted_s[mid - 1] + sorted_s[mid]) / 2
        )

    report.nodes_at_risk = at_risk_nodes
    return report


# ── CLI ───────────────────────────────────────────────────────────────────────

def _print_report(report: WorthReport) -> None:
    print("=" * 60)
    print("  MemoryWorth Report — IMI-E02 S06")
    print("=" * 60)
    print(f"  Total nós:          {report.total_nodes}")
    print(f"  Salience média:     {report.mean_salience:.3f}")
    print(f"  Salience mediana:   {report.median_salience:.3f}")
    print()
    print("  Distribuição por tier:")
    print(f"    HIGH   (≥0.8): {report.high_count:4d}  {'█' * min(report.high_count, 40)}")
    print(f"    MEDIUM (≥0.6): {report.medium_count:4d}  {'█' * min(report.medium_count, 40)}")
    print(f"    LOW    (≥0.4): {report.low_count:4d}  {'█' * min(report.low_count, 40)}")
    print(f"    AT_RISK(<0.4): {report.at_risk_count:4d}  {'█' * min(report.at_risk_count, 40)}")
    print()
    print(f"  Zero acessos: {report.zero_access_count} nós (candidatos a decaimento)")
    print()

    if report.nodes_at_risk:
        print("  Nós AT_RISK (amostra):")
        for nw in report.nodes_at_risk:
            tags_str = ", ".join(nw.tags[:3]) or "sem tags"
            print(
                f"    [{nw.node_id}] sal={nw.salience:.2f} "
                f"acc={nw.access_count} tags=[{tags_str}]"
            )
            print(f"      {nw.summary[:60]}...")
    print("=" * 60)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="MemoryWorth — análise de salience dinâmica do IMI (S06)"
    )
    parser.add_argument(
        "--db", type=Path, default=_IMI_DB,
        help=f"Path do DB SQLite (default: {_IMI_DB})"
    )
    parser.add_argument(
        "--report", action="store_true", default=True,
        help="Exibir relatório de distribuição (default: True)"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output como JSON (para integração programática)"
    )
    parser.add_argument(
        "--at-risk-limit", type=int, default=10,
        help="Máximo de nós AT_RISK no relatório (default: 10)"
    )
    args = parser.parse_args()

    try:
        report = analyze(db_path=args.db, max_at_risk=args.at_risk_limit)
    except FileNotFoundError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps({
            "total_nodes": report.total_nodes,
            "mean_salience": round(report.mean_salience, 4),
            "median_salience": round(report.median_salience, 4),
            "distribution": {
                "HIGH": report.high_count,
                "MEDIUM": report.medium_count,
                "LOW": report.low_count,
                "AT_RISK": report.at_risk_count,
            },
            "zero_access_count": report.zero_access_count,
            "nodes_at_risk": [
                {
                    "id": nw.node_id,
                    "salience": nw.salience,
                    "access_count": nw.access_count,
                    "tags": nw.tags,
                    "summary": nw.summary,
                    "projected_after_positive": round(nw.projected_after_positive, 4),
                }
                for nw in report.nodes_at_risk
            ],
        }, ensure_ascii=False, indent=2))
    else:
        _print_report(report)


if __name__ == "__main__":
    main()
