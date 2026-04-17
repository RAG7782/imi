"""IMI MCP Server — Token-Optimized (CoD Pass 2)

Wrapper that imports original IMI tools but re-registers them with:
- Shortened tool names (im_enc, im_nav, im_drm, im_sact, im_sts, im_glnk)
- Compressed descriptions (CoD Pass 2 — max density, no arg repetition)
- No server instructions (documented in CLAUDE.md)

Original server: ~/experimentos/tools/imi/imi/mcp_server.py
This file does NOT modify the original — it wraps it.
"""

import json
import os
import sys
import time as _time_module
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# --- Server setup (no instructions = 0 tokens) ---

mcp = FastMCP("imi", port=int(os.environ.get("IMI_PORT", "8080")))

# --- Singleton with TTL + diagnostics (IMI-E05 S01/S02/S05) ---

_space = None
_space_loaded_at: float = 0.0
_sqlite_load_count: int = 0
_nav_latencies: list[float] = []   # ring buffer p50/p95
_SPACE_TTL = float(os.environ.get("IMI_SPACE_TTL", "3600"))  # 1h default
_LOG_FILE = Path.home() / ".claude" / "imi_boot.log"


def _log(msg: str) -> None:
    try:
        with open(_LOG_FILE, "a") as f:
            ts = _time_module.strftime("%H:%M:%S", _time_module.localtime())
            f.write(f"[{ts}][mcp] {msg}\n")
    except Exception:
        pass


def _record_latency(ms: float) -> None:
    _nav_latencies.append(ms)
    if len(_nav_latencies) > 100:
        _nav_latencies.pop(0)


def _percentile(data: list[float], pct: int) -> float | None:
    if not data:
        return None
    sorted_d = sorted(data)
    idx = int(len(sorted_d) * pct / 100)
    return round(sorted_d[min(idx, len(sorted_d) - 1)], 1)


def _get_space():
    global _space, _space_loaded_at, _sqlite_load_count
    now = _time_module.monotonic()
    if _space is None or (now - _space_loaded_at) > _SPACE_TTL:
        t0 = _time_module.monotonic()
        from imi.space import IMISpace
        db_path = os.environ.get("IMI_DB", "imi_memory.db")
        _space = IMISpace.from_sqlite(db_path)
        _space_loaded_at = now
        _sqlite_load_count += 1
        elapsed = (_time_module.monotonic() - t0) * 1000
        _log(
            f"from_sqlite() #{_sqlite_load_count} — {elapsed:.1f}ms | "
            f"episodic={len(_space.episodic.nodes)} semantic={len(_space.semantic.nodes)}"
        )
    return _space

# --- Tools (CoD Pass 2 descriptions) ---

def _find_related_intentions(space, tag_list: list[str], experience: str) -> list[dict]:
    """IMI-E04 S06: Find pending intentions with tag or keyword overlap.

    Matches by: (1) direct tag intersection, (2) keyword presence in intention content.
    Returns intentions sorted by overlap_score DESC, max 3.
    """
    exp_words = set(w.lower() for w in experience.split() if len(w) > 4)
    tag_set = set(t.lower() for t in (tag_list or []))

    matches = []
    for node in space.episodic.nodes:
        if not (node.summary_orbital.startswith("[INTENT:") or
                node.summary_orbital.startswith("[DONE:")):
            continue
        try:
            data = json.loads(node.seed)
        except Exception:
            continue
        if data.get("node_type") != "intention" or data.get("status") != "pending":
            continue

        intent_tags = set(t.lower() for t in (data.get("tags") or []))
        intent_words = set(w.lower() for w in data.get("content", "").split() if len(w) > 4)

        tag_overlap = len(tag_set & intent_tags)
        word_overlap = len(exp_words & intent_words)
        overlap_score = tag_overlap * 2 + word_overlap  # tags contam mais

        if overlap_score >= 1:
            matches.append({
                "id": node.id,
                "content": data.get("content", "")[:120],
                "project": data.get("project", ""),
                "deadline": data.get("deadline", ""),
                "overlap_score": overlap_score,
                "tag_matches": list(tag_set & intent_tags),
            })

    matches.sort(key=lambda x: x["overlap_score"], reverse=True)
    return matches[:3]


@mcp.tool()
def im_enc(
    experience: str,
    tags: str = "",
    source: str = "",
    context_hint: str = "",
    occurred_at: str = "",
) -> str:
    """Store new memory from experience"""
    import time as _time
    from datetime import datetime, timezone

    space = _get_space()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None

    event_timestamp = None
    occurred_float = None
    if occurred_at:
        try:
            dt = datetime.fromisoformat(occurred_at.replace("Z", "+00:00"))
            event_timestamp = dt.timestamp()
            occurred_float = event_timestamp
        except ValueError:
            pass

    node = space.encode(
        experience, tags=tag_list, source=source,
        context_hint=context_hint, timestamp=event_timestamp,
    )
    node.occurred_at = occurred_float

    # IMI-E04 S06: Check for pending intentions with tag/keyword overlap
    related_intentions = _find_related_intentions(space, tag_list or [], experience)

    result = {
        "id": node.id,
        "summary": node.summary_medium,
        "tags": node.tags,
        "affect": {
            "salience": node.affect.salience if node.affect else 0,
            "valence": node.affect.valence if node.affect else 0,
        },
        "affordances": [str(a) for a in node.affordances[:3]],
        "mass": round(node.mass, 3),
        "total_memories": len(space.episodic),
    }
    if related_intentions:
        result["pending_intentions_hint"] = related_intentions
        _log(f"im_enc S06: {len(related_intentions)} related intention(s) detected for node {node.id[:8]}")

    return json.dumps(result, ensure_ascii=False, indent=2)


def _mw_score_from_node(node) -> float:
    """Extrai Memory Worth score do seed JSON do nó. Retorna 0.5 (neutral) se não existe."""
    try:
        if node.seed:
            d = json.loads(node.seed)
            if "mw_score" in d:
                return float(d["mw_score"])
    except Exception:
        pass
    return 0.5  # prior neutro para nós sem histórico


def _affordance_max_confidence(node) -> float:
    """Retorna a maior confidence de affordance do nó."""
    try:
        affs = node.affordances or []
        if not affs:
            return 0.5
        return max(
            (float(a.confidence) if hasattr(a, "confidence") else 0.5)
            for a in affs
        )
    except Exception:
        return 0.5


@mcp.tool()
def im_nav(
    query: str,
    top_k: int = 10,
    zoom: str = "medium",
    context: str = "",
    relevance_weight: float = -1,
    positional_optimize: bool = True,
    mode: str = "semantic",
) -> str:
    """Search memories with adaptive relevance and graph expansion.

    mode="semantic"  — padrão: ranking por cosine similarity (comportamento original)
    mode="utility"   — IMI-E01 S03: Two-Phase Retrieval
                       Phase 1: filtra candidatos com score semântico >= MIN_SCORE
                       Phase 2: rerank por MW × affordance_max_confidence
                       Ref: MemRL (arXiv:2601.03192) + MemoryWorth (arXiv:2604.12007)
    """
    t0 = _time_module.monotonic()
    space = _get_space()
    rw = None if relevance_weight < 0 else relevance_weight

    nav = space.navigate(
        query, zoom=zoom, top_k=top_k if mode == "semantic" else top_k * 3,
        context=context, relevance_weight=rw, positional_optimize=positional_optimize,
    )

    rw_used, intent_obj = space.adaptive_rw.classify_with_info(query)

    raw_memories = nav.memories[: top_k * 3 if mode == "utility" else top_k]

    if mode == "utility":
        # Phase 1: filtrar candidatos com score semântico >= 0.62 (MIN_SCORE)
        MIN_SEMANTIC = 0.62
        candidates = [m for m in raw_memories if m.get("score", 0) >= MIN_SEMANTIC]
        if not candidates:
            candidates = raw_memories  # fallback: sem filtro se nenhum passa

        # Phase 2: rerank por MW × affordance_max_confidence
        def utility_score(m: dict) -> float:
            node = space.episodic.get(m.get("id", "")) or space.semantic.get(m.get("id", ""))
            mw = _mw_score_from_node(node) if node else 0.5
            aff = _affordance_max_confidence(node) if node else 0.5
            return mw * aff

        candidates.sort(key=utility_score, reverse=True)
        final_memories = candidates[:top_k]
    else:
        final_memories = raw_memories[:top_k]

    memories = []
    for m in final_memories:
        mem = {"score": round(m["score"], 3), "content": m["content"],
               "id": m.get("id", ""), "tags": m.get("tags", [])}
        if m.get("affordances"): mem["affordances"] = m["affordances"][:2]
        if m.get("affect_str"): mem["affect"] = m["affect_str"]
        if mode == "utility":
            node = space.episodic.get(m.get("id", "")) or space.semantic.get(m.get("id", ""))
            mem["mw_score"] = _mw_score_from_node(node) if node else 0.5
        memories.append(mem)

    elapsed_ms = (_time_module.monotonic() - t0) * 1000
    _record_latency(elapsed_ms)
    _log(f"im_nav '{query[:40]}' — {elapsed_ms:.1f}ms | hits={len(memories)} zoom={zoom} mode={mode}")

    result = {
        "query": query, "intent": intent_obj.name,
        "relevance_weight_used": round(rw_used if rw is None else rw, 3),
        "zoom": zoom, "hits": len(memories),
        "retrieval_mode": mode,
        "memories": memories,
        "_perf_ms": round(elapsed_ms, 1),
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def im_drm() -> str:
    """Consolidation cycle — cluster memories into patterns"""
    space = _get_space()
    report = space.dream()
    result = {
        "nodes_processed": report.nodes_processed,
        "clusters_formed": report.clusters_formed,
        "patterns_extracted": report.patterns_extracted,
        "convergence": {
            "energy": round(space.annealing.energy_history[-1], 4) if space.annealing.energy_history else None,
            "steps": space.annealing.iteration,
            "converged": space.annealing.converged,
        },
        "total_episodic": len(space.episodic),
        "total_semantic": len(space.semantic),
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def im_sact(action_query: str, top_k: int = 5) -> str:
    """Find memories by enabled actions (affordance search)"""
    space = _get_space()
    results = space.search_affordances(action_query, top_k=top_k)
    items = [{
        "action": r["action"], "confidence": round(r["confidence"], 2),
        "conditions": r["conditions"], "similarity": round(r["similarity"], 3),
        "memory_summary": r["memory_summary"][:200], "node_id": r["node_id"],
    } for r in results]
    return json.dumps({"query": action_query, "results": items}, ensure_ascii=False, indent=2)


@mcp.tool()
def im_sts() -> str:
    """Memory space statistics"""
    space = _get_space()
    graph_stats = space.graph.stats()
    result = {
        "episodic_count": len(space.episodic),
        "semantic_count": len(space.semantic),
        "total_memories": len(space.episodic) + len(space.semantic),
        "graph": {
            "total_edges": graph_stats["total_edges"],
            "edge_types": graph_stats.get("by_type", {}),
            "nodes_with_edges": graph_stats.get("nodes_with_edges", 0),
        },
        "annealing": {
            "steps": space.annealing.iteration,
            "converged": space.annealing.converged,
            "energy": round(space.annealing.energy_history[-1], 4) if space.annealing.energy_history else None,
        },
        "persist_dir": str(space.persist_dir) if space.persist_dir else None,
        "tiers": space.tier_stats(),
        "l0_l1_preview": space.get_l0_l1(),
        "l0_l1_tokens": len(space.get_l0_l1()) // 4,
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def im_glnk(source_id: str, target_id: str, edge_type: str = "causal", label: str = "") -> str:
    """Add edge between memories in graph"""
    from imi.graph import EdgeType
    space = _get_space()
    type_map = {"causal": EdgeType.CAUSAL, "co_occurrence": EdgeType.CO_OCCURRENCE, "similar": EdgeType.SIMILAR}
    et = type_map.get(edge_type, EdgeType.CAUSAL)
    space.graph.add_edge(source_id, target_id, et, label=label)
    if space.persist_dir: space.save()
    return json.dumps({
        "status": "ok", "edge": f"{source_id} --[{edge_type}]--> {target_id}",
        "label": label, "total_edges": space.graph.stats()["total_edges"],
    }, ensure_ascii=False, indent=2)


@mcp.tool()
def im_int(
    content: str,
    context: str,
    project: str = "",
    deadline: str = "",
    confidence: float = 0.85,
    tags: str = "",
    source: str = "",
) -> str:
    """Store pending intention with deadline and context"""
    import uuid as _uuid
    from datetime import datetime, timezone

    space = _get_space()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    if project and project not in tag_list:
        tag_list.insert(0, project.lower())

    node_id = f"intent_{_uuid.uuid4().hex[:8]}"
    now = _time_module.time()

    deadline_ts = None
    if deadline:
        try:
            deadline_ts = datetime.fromisoformat(deadline.replace("Z", "+00:00")).timestamp()
        except ValueError:
            pass

    data = {
        "id": node_id,
        "node_type": "intention",
        "content": content,
        "context": context,
        "project": project,
        "deadline": deadline,
        "deadline_ts": deadline_ts,
        "confidence": confidence,
        "status": "pending",
        "fulfilled_by": None,
        "blocked_by": None,
        "tags": tag_list,
        "created_at": now,
        "salience": 0.85,
        "source": source or "im_int",
        "affect": {"salience": 0.85, "fade_resist": 0.95},
        "summary_orbital": f"[INTENT:{project}] {content[:120]}",
        "summary_medium": f"{content} | {context[:200]}",
    }

    mn = _intention_to_node(data)
    # Persiste via backend e mantém em memória no VectorStore
    space.backend.put_node("episodic", mn)
    space.episodic.nodes.append(mn)
    space.episodic._dirty = True

    _log(f"im_int created: {node_id} | project={project} deadline={deadline}")

    return json.dumps({
        "id": node_id, "content": content,
        "project": project, "deadline": deadline,
        "status": "pending", "tags": tag_list,
    }, ensure_ascii=False, indent=2)


def _intention_to_node(data: dict):
    """Converte dict de intenção em MemoryNode.

    O JSON completo fica em `content` para recuperação fiel via im_int_list.
    summary_orbital = preview legível (node_type como prefixo para filtragem).
    """
    from imi.node import MemoryNode
    from imi.affect import AffectiveTag
    # AffectiveTag: salience + valence + arousal (fade_resist é property calculada)
    # arousal=0.8 → fade_resistance = salience*(0.3+0.7*emotional_intensity) ≈ 0.85*0.86 ≈ 0.73
    affect = AffectiveTag(salience=data.get("salience", 0.85), valence=0.5, arousal=0.8)
    return MemoryNode(
        id=data["id"],
        seed=json.dumps(data, ensure_ascii=False),       # JSON completo em seed para recuperação
        original=data["content"],                        # texto legível em original
        summary_medium=data.get("summary_medium", data["content"][:200]),
        summary_orbital=data.get("summary_orbital", f"[INTENT:{data.get('project','')}] {data['content'][:80]}"),
        tags=data.get("tags", []),
        source=data.get("source", "im_int"),
        created_at=data.get("created_at", _time_module.time()),
        affect=affect,
    )


@mcp.tool()
def im_int_fulfill(
    intent_id: str,
    fulfilled_by: str = "",
    notes: str = "",
) -> str:
    """Mark intention as fulfilled, optionally linking to the completing memory node"""
    from imi.graph import EdgeType

    space = _get_space()

    # Busca o nó pelo ID (episodic store — in-memory first, then backend)
    node = space.episodic.get(intent_id)
    if node is None:
        node = space.backend.get_node("episodic", intent_id)

    if node is None:
        return json.dumps({"error": f"intent_id not found: {intent_id}"}, ensure_ascii=False)

    # Verifica que é intenção
    try:
        data = json.loads(node.seed)
    except Exception:
        data = {}
    if data.get("node_type") != "intention":
        return json.dumps({"error": f"{intent_id} is not an intention node"}, ensure_ascii=False)

    # Atualiza campos de fulfillment no JSON interno
    data["status"] = "fulfilled"
    data["fulfilled_by"] = fulfilled_by or None
    data["notes"] = notes
    data["fulfilled_at"] = _time_module.time()
    node.seed = json.dumps(data, ensure_ascii=False)
    node.summary_orbital = node.summary_orbital.replace("[INTENT:", "[DONE:")

    # Persiste atualização
    space.backend.put_node("episodic", node)

    # Cria edge causal se fulfilled_by é node_id válido
    edge_created = False
    if fulfilled_by:
        try:
            space.graph.add_edge(fulfilled_by, intent_id, EdgeType.CAUSAL, label="fulfills")
            edge_created = True
        except Exception as e:
            _log(f"im_int_fulfill edge WARN: {e}")

    _log(f"im_int_fulfill: {intent_id} → fulfilled | by={fulfilled_by} edge={edge_created}")

    return json.dumps({
        "status": "fulfilled", "intent_id": intent_id,
        "fulfilled_by": fulfilled_by, "edge_created": edge_created,
    }, ensure_ascii=False, indent=2)


@mcp.tool()
def im_int_list(
    project: str = "",
    status: str = "pending",
    top_k: int = 10,
) -> str:
    """List intentions filtered by project and status, ordered by deadline"""
    space = _get_space()

    intentions = []
    for node in space.episodic.nodes:
        # Intenções são identificadas pelo prefixo [INTENT: no summary_orbital
        # ou pelo node_type no JSON de content
        if not (node.summary_orbital.startswith("[INTENT:") or node.summary_orbital.startswith("[DONE:")):
            continue
        try:
            data = json.loads(node.seed)
        except Exception:
            continue
        if data.get("node_type") != "intention":
            continue
        if status and data.get("status", "pending") != status:
            continue
        if project and data.get("project", "").upper() != project.upper():
            continue
        intentions.append({
            "id": node.id,
            "content": data.get("content", ""),
            "context": data.get("context", ""),
            "project": data.get("project", ""),
            "deadline": data.get("deadline", ""),
            "deadline_ts": data.get("deadline_ts"),
            "status": data.get("status", "pending"),
            "confidence": data.get("confidence", 0.85),
            "tags": node.tags,
            "created_at": node.created_at,
        })

    # Ordenar: deadline ASC (None vai para o fim), depois salience DESC
    def sort_key(i):
        dt = i.get("deadline_ts") or float("inf")
        return (dt, -i.get("confidence", 0.85))

    intentions.sort(key=sort_key)
    items = intentions[:top_k]

    _log(f"im_int_list: {len(items)} intentions | project={project} status={status}")

    return json.dumps({
        "total": len(intentions), "filtered": len(items),
        "status": status, "project": project,
        "intentions": items,
    }, ensure_ascii=False, indent=2)


@mcp.tool()
def im_feedback(
    node_ids: list[str],
    outcome: str,
    context: str = "",
    magnitude: float = 0.5,
) -> str:
    """MemoryWorth S06 — Feedback de outcome para ajuste dinâmico de salience.

    Recebe o resultado de usar memórias (positivo/negativo/neutro) e ajusta
    salience + valence via Bayesian update. Fecha o loop: memórias que levam
    a bons outcomes ganham salience; memórias que levam a erros perdem.

    Args:
        node_ids:  Lista de IDs de nós usados (retornados pelo im_nav).
        outcome:   "positive" | "negative" | "neutral"
        context:   Contexto do uso (opcional — enriquece o log de aprendizado).
        magnitude: Força do feedback 0.0–1.0 (default 0.5 = feedback normal).

    Fórmula de update (Bayesian-inspired):
        positive → salience += magnitude * 0.1 * (1 - current_salience)  [ceil 0.95]
        negative → salience -= magnitude * 0.1 * current_salience         [floor 0.1]
        neutral  → salience unchanged (apenas registra acesso)

    Valence também é ajustada:
        positive → valence += 0.05 * magnitude  [ceil +0.9]
        negative → valence -= 0.05 * magnitude  [floor -0.9]
    """
    outcome = outcome.strip().lower()
    if outcome not in ("positive", "negative", "neutral"):
        return json.dumps({
            "error": f"outcome deve ser 'positive', 'negative' ou 'neutral' — recebido: '{outcome}'"
        })

    if not node_ids:
        return json.dumps({"error": "node_ids não pode ser vazio"})

    space = _get_space()
    updated = []
    not_found = []

    for node_id in node_ids:
        # Buscar nó em ambas as stores (episodic e semantic)
        node = space.episodic.get(node_id) or space.semantic.get(node_id)
        if node is None:
            not_found.append(node_id)
            continue

        old_salience = node.affect.salience if node.affect else 0.5
        old_valence = node.affect.valence if node.affect else 0.0

        if node.affect is None:
            from imi.affect import AffectiveTag
            node.affect = AffectiveTag()

        if outcome == "positive":
            # Regra assintótica: quanto mais alta a salience, menor o ganho marginal
            delta_s = magnitude * 0.1 * (1.0 - node.affect.salience)
            node.affect.salience = min(0.95, node.affect.salience + delta_s)
            delta_v = 0.05 * magnitude
            node.affect.valence = min(0.9, node.affect.valence + delta_v)
        elif outcome == "negative":
            # Decaimento proporcional: memórias com alta salience decaem mais
            delta_s = magnitude * 0.1 * node.affect.salience
            node.affect.salience = max(0.1, node.affect.salience - delta_s)
            delta_v = 0.05 * magnitude
            node.affect.valence = max(-0.9, node.affect.valence - delta_v)
        # neutral: sem mudança em affect, mas touch() registra o acesso

        node.touch()  # registra acesso + boost logarítmico via update_dynamic

        updated.append({
            "id": node_id,
            "salience_before": round(old_salience, 3),
            "salience_after": round(node.affect.salience, 3),
            "valence_before": round(old_valence, 3),
            "valence_after": round(node.affect.valence, 3),
            "access_count": node.access_count,
        })

    # Persistir mudanças
    if updated:
        space.save()

    _log(
        f"im_feedback outcome={outcome} mag={magnitude:.2f} | "
        f"updated={len(updated)} not_found={len(not_found)}"
    )

    return json.dumps({
        "outcome": outcome,
        "magnitude": magnitude,
        "context": context[:200] if context else "",
        "updated": updated,
        "not_found": not_found,
        "saved": bool(updated),
    }, ensure_ascii=False, indent=2)


@mcp.tool()
def im_mw_update(
    node_ids: list[str],
    outcome: str,
    session_id: str = "",
) -> str:
    """IMI-E01 S02: Update Memory Worth counters (success/failure) for given nodes.

    Memory Worth (MW) = success_count / max(1, success_count + failure_count)
    Ref: arXiv:2604.12007 — correlação 0.89 com utilidade real.

    Args:
        node_ids: IDs dos nós usados nesta sessão
        outcome:  "success" | "failure"
        session_id: identificador da sessão para auditoria (opcional)

    Returns:
        {updated: N, mw_scores: {node_id: float}, session_id: str}

    Verify: im_mw_update(["node_id"], "success") → mw_scores shows MW > 0
    """
    outcome = outcome.strip().lower()
    if outcome not in ("success", "failure"):
        return json.dumps({"error": f"outcome deve ser 'success' ou 'failure', recebido: '{outcome}'"})

    space = _get_space()
    updated: dict[str, float] = {}
    not_found: list[str] = []

    for node_id in node_ids:
        node = space.episodic.get(node_id) or space.semantic.get(node_id)
        if node is None:
            # Tentar via backend direto
            node = space.backend.get_node("episodic", node_id) or space.backend.get_node("semantic", node_id)
        if node is None:
            not_found.append(node_id)
            continue

        # MW counters vivem no JSON do seed/data (schema-free — sem ALTER TABLE)
        try:
            seed_data = json.loads(node.seed) if node.seed else {}
        except Exception:
            seed_data = {}

        if outcome == "success":
            seed_data["mw_success"] = seed_data.get("mw_success", 0) + 1
        else:
            seed_data["mw_failure"] = seed_data.get("mw_failure", 0) + 1

        sc = seed_data.get("mw_success", 0)
        fc = seed_data.get("mw_failure", 0)
        mw = sc / max(1, sc + fc)
        seed_data["mw_score"] = round(mw, 4)
        seed_data["mw_last_updated"] = _time_module.time()

        node.seed = json.dumps(seed_data, ensure_ascii=False)
        updated[node_id] = round(mw, 4)

    if updated:
        space.save()

    _log(f"im_mw_update outcome={outcome} | updated={len(updated)} not_found={len(not_found)} session={session_id}")

    return json.dumps({
        "outcome": outcome,
        "updated": len(updated),
        "mw_scores": updated,
        "not_found": not_found,
        "session_id": session_id,
    }, ensure_ascii=False, indent=2)


@mcp.tool()
def im_perf() -> str:
    """MCP server performance metrics and health check (IMI-E05 S05)"""
    space_ok = _space is not None
    age = (_time_module.monotonic() - _space_loaded_at) if space_ok else None
    result = {
        "space_loaded": space_ok,
        "space_age_seconds": round(age, 1) if age is not None else None,
        "space_ttl_seconds": _SPACE_TTL,
        "episodic_count": len(_space.episodic.nodes) if space_ok else 0,
        "semantic_count": len(_space.semantic.nodes) if space_ok else 0,
        "from_sqlite_calls_this_session": _sqlite_load_count,
        "nav_samples": len(_nav_latencies),
        "p50_nav_ms": _percentile(_nav_latencies, 50),
        "p95_nav_ms": _percentile(_nav_latencies, 95),
        "verdict": (
            "OK — singleton ativo, latência nominal"
            if _sqlite_load_count <= 1 and (_percentile(_nav_latencies, 95) or 0) < 500
            else "WARN — checar log ~/.claude/imi_boot.log"
        ),
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


# --- Entry point ---

def main():
    transport = os.environ.get("IMI_TRANSPORT", "stdio")
    if "--transport" in sys.argv:
        idx = sys.argv.index("--transport")
        transport = sys.argv[idx + 1]
    mcp.run(transport=transport)

if __name__ == "__main__":
    main()
