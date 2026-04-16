#!/usr/bin/env python3
"""
imi_boot_semantic.py — IMI Bootstrap Protocol v3.0 (Semantic + Typed Slots)
============================================================
Substitui leitura estática de arquivos por busca semântica real no IMI.

Algoritmo:
  1. Abre imi_memory.db diretamente (sem MCP server, zero latência extra)
  2. Seleciona top-N memórias por score composto diferenciado por store:
     - episodic: salience * fade_resist * recency_weight
     - semantic: score_episodic * min(2.0, log2(episode_count))  [boost]
  3. Aplica Positional Reorder v3 com slots explícitos (Liu 2023):
     - Slot 0-1: top-2 episódicos recentes (hot context)
     - Slot 2-3: top-2 padrões semânticos (knowledge distilled)
     - Slot 4-6: top-3 por score misto (bordas = máx atenção LLM)
  4. Classifica cada memória em LP/CP/IM conforme regras da membrane
  5. Monta bloco <imi_boot> com seções separadas (episodic / semantic / misto)
  6. Persiste em ~/.imi_boot_cache

Fail-safe: qualquer exceção → fallback para cache existente ou bloco vazio.
Cache TTL: 4h (14400s). Se fresco, imprime e sai sem reconstruir.

Uso:
  python3 ~/.claude/imi_boot_semantic.py
"""

import json
import math
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Configuração ──────────────────────────────────────────────────────────────

IMI_DB          = Path.home() / "experimentos/tools/imi/imi_memory.db"
KNOWLEDGE_FILE  = Path.home() / "experimentos/KNOWLEDGE.md"
HANDOFFS_DIR    = Path.home() / ".aiox/memory/handoffs"
CACHE_FILE      = Path.home() / ".imi_boot_cache"
LOG_FILE        = Path.home() / ".claude/imi_boot.log"
CACHE_TTL       = 14400          # 4 horas
TOP_N           = 7              # memórias para positional reorder
TOKEN_TARGET    = 480            # tokens aproximados no output (300-600 range Karpathy)
RECENCY_HALF_LIFE_DAYS = 14.0   # salience decai 50% em 14 dias sem acesso
SEMANTIC_BOOST_ENABLED = os.environ.get("IMI_SEMANTIC_BOOST", "1") != "0"

# Domínios LP vs CP conforme membrane.yaml
LP_TAGS = {
    "decision", "architecture", "breaking-change", "milestone", "pattern",
    "rule", "publication", "benchmark", "ip-protection", "framework-injection",
    "artesanato-digital", "fcm", "imi", "aiox",
}
LP_KEYWORDS = [
    "FI", "framework injection", "densidade semiótica", "benchmark",
    "p<0.05", "IC=", "extended thinking", "reasoning scaffold",
]


def log(msg: str) -> None:
    try:
        with open(LOG_FILE, "a") as f:
            ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass


def cache_is_fresh() -> bool:
    if not CACHE_FILE.exists():
        return False
    age = time.time() - CACHE_FILE.stat().st_mtime
    return age < CACHE_TTL


def recency_weight(last_accessed: float | None, created_at: float) -> float:
    """
    Exponential decay baseado em last_accessed (ou created_at como fallback).
    w = 2^(-days_elapsed / half_life)
    """
    ref = last_accessed or created_at or time.time()
    days_elapsed = (time.time() - ref) / 86400.0
    return 2.0 ** (-days_elapsed / RECENCY_HALF_LIFE_DAYS)


def composite_score(affect: dict, last_accessed: float | None, created_at: float) -> float:
    """
    strength = salience * fade_resist (se disponível) ou salience
    final   = strength * recency_weight
    """
    sal   = affect.get("salience", 0.5)
    fade  = affect.get("fade_resist", sal)          # fallback para salience
    strength = sal * fade
    return strength * recency_weight(last_accessed, created_at)


# ── S01: Parser de episode_count ──────────────────────────────────────────────

_EPISODE_RE = re.compile(r"consolidated\s+from\s+(\d+)\s+episodes?", re.IGNORECASE)

def parse_episode_count(data: dict) -> int:
    """Extrai N de strings como 'consolidated from 85 episodes'.

    Verifica source, summary_orbital e summary_medium em ordem.
    Fallback: 1 (sem boost).

    Verify: parse_episode_count({"source": "consolidated from 85 episodes"}) → 85
    """
    for field in ("source", "summary_orbital", "summary_medium"):
        text = data.get(field) or ""
        m = _EPISODE_RE.search(text)
        if m:
            return int(m.group(1))
    return 1


# ── S02: Score diferenciado por store ─────────────────────────────────────────

def boosted_score(base_score: float, store_name: str, data: dict) -> float:
    """Aplica boost logarítmico a padrões semânticos consolidados.

    Boost = min(2.0, log2(max(2, episode_count)))
    log2(85) ≈ 6.4 → cap em 2.0 | log2(40) ≈ 5.3 → cap em 2.0 | log2(2) = 1.0

    Só ativo se SEMANTIC_BOOST_ENABLED=True.
    """
    if not SEMANTIC_BOOST_ENABLED or store_name != "semantic":
        return base_score
    n = parse_episode_count(data)
    boost = min(2.0, math.log2(max(2, n)))
    boosted = base_score * boost
    log(f"semantic boost: {data.get('id', '?')[:8]} {base_score:.4f} → {boosted:.4f} ({n} episodes, ×{boost:.2f})")
    return boosted


def classify_tier(data: dict) -> str:
    """Classifica memória em LP, CP ou IM conforme membrane rules."""
    tags = set(data.get("tags", []))
    content = (data.get("summary_medium") or data.get("summary_orbital") or "").lower()

    if tags & LP_TAGS:
        return "LP"
    if any(kw.lower() in content for kw in LP_KEYWORDS):
        return "LP"
    return "CP"


def positional_reorder(memories: list[dict]) -> list[dict]:
    """
    Liu 2023 v2 (legado — mantido para compatibilidade).
    Top-2 → posições 0,1 | Next-2 → posições -2,-1 | resto → meio
    """
    n = len(memories)
    if n <= 4:
        return memories
    top2   = memories[:2]
    next2  = memories[2:4]
    middle = memories[4:]
    return top2 + middle + next2


# ── S03: Positional Reorder v3 com slots explícitos ──────────────────────────

def positional_reorder_v3(memories: list[dict]) -> list[dict]:
    """Reorder v3: slots tipados para maximizar atenção LLM (Liu 2023).

    Layout de saída (7 slots, bordas = máxima atenção):
      [0-1] episodic recentes — hot context (início do bloco)
      [2-4] misto por score   — contexto geral (meio, menor atenção)
      [5-6] semantic patterns — conhecimento consolidado (fim do bloco)

    Padrões semânticos no fim aproveitam o efeito recency (primacy-recency).
    Episódicos recentes no início aproveitam o efeito primacy.

    Verify: output deve conter pelo menos 1 memória com store='semantic'
    quando houver padrões disponíveis.
    """
    episodic = [m for m in memories if m.get("store") != "semantic"]
    semantic = [m for m in memories if m.get("store") == "semantic"]

    slot_episodic = episodic[:2]
    slot_semantic = semantic[:2]
    # Misto: tudo que não foi para os slots de borda
    used = set(id(m) for m in slot_episodic + slot_semantic)
    slot_mixed = [m for m in memories if id(m) not in used][:3]

    result = slot_episodic + slot_mixed + slot_semantic
    log(
        f"v3 composition: {len(slot_episodic)} episodic + {len(slot_mixed)} mixed + "
        f"{len(slot_semantic)} semantic | semantic_ratio={len(slot_semantic)/max(1,len(result)):.2f}"
    )
    return result


def _score_rows(rows: list, store_name: str) -> list[tuple[float, dict]]:
    """Converte rows SQLite em lista (score, data) para um store específico."""
    scored = []
    for row in rows:
        data_str = row[2] if len(row) > 2 else row[1]
        try:
            data: dict[str, Any] = json.loads(data_str)
        except Exception:
            continue

        affect = data.get("affect") or {}
        sal = affect.get("salience", data.get("salience", 0.5))
        sal_threshold = 0.40 if store_name == "semantic" else 0.65
        if sal < sal_threshold:
            continue

        summary = data.get("summary_orbital") or data.get("summary_medium") or ""
        if not summary or len(summary) < 20:
            continue

        base = composite_score(
            affect if affect else {"salience": sal},
            data.get("last_accessed"),
            data.get("created_at", time.time()),
        )
        final = boosted_score(base, store_name, data)
        data["store"] = store_name
        scored.append((final, data))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def fetch_top_memories(conn: sqlite3.Connection, n: int = TOP_N) -> list[dict]:
    """Busca memórias com slots garantidos por store (v3).

    Garante representação semântica independentemente do score absoluto:
    - 2 slots reservados para semantic (knowledge distilled)
    - n-2 slots para episodic (hot context)

    Retorna lista com campo 'store' injetado para positional_reorder_v3.
    """
    cur = conn.cursor()

    cur.execute("""
        SELECT node_id, store_name, data FROM memory_nodes
        WHERE is_deleted = 0 AND store_name = 'episodic'
        ORDER BY id DESC LIMIT 300
    """)
    ep_scored = _score_rows(cur.fetchall(), "episodic")

    cur.execute("""
        SELECT node_id, store_name, data FROM memory_nodes
        WHERE is_deleted = 0 AND store_name = 'semantic'
        ORDER BY id DESC LIMIT 200
    """)
    sem_scored = _score_rows(cur.fetchall(), "semantic")

    # Slots garantidos: 2 semantic + (n-2) episodic
    sem_slots = max(0, min(2, len(sem_scored)))
    ep_slots  = n - sem_slots

    top_ep  = [d for _, d in ep_scored[:ep_slots]]
    top_sem = [d for _, d in sem_scored[:sem_slots]]

    log(f"fetch: {len(top_ep)} episodic + {len(top_sem)} semantic (pool: {len(ep_scored)} ep, {len(sem_scored)} sem)")
    return top_ep + top_sem


def format_memory(data: dict, tier: str) -> str:
    """Formata uma memória para o bloco de boot (1-2 linhas)."""
    summary = data.get("summary_orbital") or (data.get("summary_medium") or "")[:120]
    tags    = data.get("tags", [])[:4]
    tag_str = " ".join(f"#{t}" for t in tags) if tags else ""
    return f"[{tier}] {summary}  {tag_str}".strip()


def read_knowledge_snippet() -> str:
    """Lê o primeiro bloco H2 do KNOWLEDGE.md (marco mais recente)."""
    if not KNOWLEDGE_FILE.exists():
        return ""
    lines, in_block, block = KNOWLEDGE_FILE.read_text().splitlines(), False, []
    for line in lines:
        if line.startswith("## ") and not in_block:
            in_block = True
            block.append(line)
        elif line.startswith("## ") and in_block:
            break
        elif in_block:
            block.append(line)
        if len(block) > 10:
            break
    return "\n".join(block[:9])


def read_handoff_snippet() -> str:
    """Lê as primeiras linhas úteis do handoff mais recente."""
    if not HANDOFFS_DIR.exists():
        return ""
    handoffs = sorted(HANDOFFS_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not handoffs:
        return ""
    text = handoffs[0].read_text().splitlines()
    # Procura a primeira seção relevante
    snippet, capturing = [], False
    for line in text:
        if any(kw in line for kw in ["## O que foi feito", "## Estado", "## Próximos"]):
            capturing = True
        if capturing:
            snippet.append(line)
        if len(snippet) >= 8:
            break
    return "\n".join(snippet) if snippet else "\n".join(text[:8])


def build_cache() -> str:
    """Constrói o bloco <imi_boot> completo."""
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Memórias do IMI com scoring diferenciado e slots tipados
    episodic_lines: list[str] = []
    semantic_lines: list[str] = []
    mixed_lines:    list[str] = []
    semantic_ratio = 0.0

    try:
        if not IMI_DB.exists():
            raise FileNotFoundError(f"IMI DB não encontrado: {IMI_DB}")
        conn = sqlite3.connect(str(IMI_DB))
        memories = fetch_top_memories(conn, TOP_N)
        conn.close()
        reordered = positional_reorder_v3(memories)

        # Separar por store para seções distintas no output
        ep_slot  = [m for m in reordered if m.get("store") != "semantic"][:2]
        sem_slot = [m for m in reordered if m.get("store") == "semantic"][:2]
        used     = set(id(m) for m in ep_slot + sem_slot)
        mix_slot = [m for m in reordered if id(m) not in used]

        for data in ep_slot:
            episodic_lines.append(format_memory(data, classify_tier(data)))
        for data in sem_slot:
            n_ep = parse_episode_count(data)
            prefix = f"[PATTERN] consolidated from {n_ep} episodes — " if n_ep > 1 else "[PATTERN] "
            summary = data.get("summary_orbital") or (data.get("summary_medium") or "")[:120]
            tags    = " ".join(f"#{t}" for t in data.get("tags", [])[:4])
            semantic_lines.append(f"[LP] {prefix}{summary}  {tags}".strip())
        for data in mix_slot:
            mixed_lines.append(format_memory(data, classify_tier(data)))

        total = len(reordered)
        semantic_ratio = len(sem_slot) / max(1, total)
        log(f"v3 done: {len(ep_slot)} episodic + {len(mix_slot)} mixed + {len(sem_slot)} semantic | ratio={semantic_ratio:.2f}")

        # Alerta se padrões não aparecem (S05)
        if semantic_ratio == 0.0:
            log("WARN: semantic_ratio=0.0 — padrões semânticos ausentes do boot")

    except Exception as e:
        log(f"WARN: falha na busca semântica: {e}")
        episodic_lines = ["(busca semântica indisponível — usar im_sts no boot)"]

    knowledge = read_knowledge_snippet()
    handoff   = read_handoff_snippet()

    # Monta seções separadas (S04)
    ep_section  = "\n".join(f"  {i+1}. {l}" for i, l in enumerate(episodic_lines))
    sem_section = "\n".join(f"  {i+1}. {l}" for i, l in enumerate(semantic_lines))
    mix_section = "\n".join(f"  {i+1}. {l}" for i, l in enumerate(mixed_lines))

    has_semantic = bool(semantic_lines)
    semantic_block = f"""
## Conhecimento Consolidado (semantic patterns)
{sem_section}""" if has_semantic else ""

    block = f"""<imi_boot ts="{ts}" strategy="semantic_v3">
## Memórias Recentes (episodic — primacy)
{ep_section or "  (nenhuma)"}
{semantic_block}
## Contexto Misto
{mix_section or "  (nenhum)"}

## Último Marco de Conhecimento
{knowledge or "(KNOWLEDGE.md não encontrado)"}

## Estado da Sessão Anterior
{handoff or "(Nenhum handoff encontrado)"}

## Instruções de Boot para o Agente
- Executar im_sts + cv_wk antes de responder à primeira mensagem
- Episodic = contexto recente | Semantic = conhecimento destilado (bordas = máx atenção)
- [LP] = Long-term Permanente (conhecimento estrutural)  [CP] = Curta-Permanente (estado ativo)
- Usar im_sact(entidade, top_k=5) para qualquer entidade detectada na conversa
</imi_boot>"""

    return block


def main() -> None:
    if cache_is_fresh():
        log("cache hit (< 4h)")
        print(CACHE_FILE.read_text())
        return

    log("cache miss — rebuilding (semantic v2)")
    try:
        block = build_cache()
    except Exception as e:
        log(f"ERROR build_cache: {e}")
        # Fallback: cache antigo se existir
        if CACHE_FILE.exists():
            log("usando cache antigo como fallback")
            print(CACHE_FILE.read_text())
        return

    CACHE_FILE.write_text(block)
    log(f"cache rebuilt — {len(block)} bytes")
    print(block)


if __name__ == "__main__":
    main()
