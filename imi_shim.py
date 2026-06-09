#!/usr/bin/env python3
"""imi_shim — shim CLI determinístico para as tools IMI, SEM dependência de MCP.

PORQUÊ: o Pi (e crons, e qualquer harness não-MCP) não falam o protocolo MCP. Esta
ferramenta colapsa a dependência de MCP num único comando de linha — exatamente o
padrão "comando > texto livre" e "skills/CLI > MCP" defendido pelo projeto. Reusa a
MESMA lógica de negócio que o mcp_server expõe (IMISpace + secure_encode + busca
lexical), respeitando o mesmo contrato de env (IMI_DB, IMI_STORAGE_DIR, IMI_CRYPTO).

Viabiliza ~/pi-imi-bridge.ts, cujos call-sites esperam:
    imi im_sts
    imi im_enc --experience "<txt>" --tags "<a,b>"
    imi im_nav --mode lexical "<termo>"

Ponto de fluxo: argv -> parse -> resolve IMISpace (singleton de processo) -> verbo -> JSON em stdout.

SEGURANÇA: escrita passa por secure_encode (sanitizer + crypto opt-in via IMI_CRYPTO=1),
o mesmo caminho do MCP — herda o secret-scanning da ST-IMI-SECSCAN. Falha graciosa:
qualquer erro vira JSON {"ok": false, "error": ...} em stdout + exit code 1, nunca stack-trace cru.

Uso headless (Pi/cron): saída sempre JSON de uma linha — parseável sem heurística.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

# O DB e o storage seguem o MESMO contrato do mcp_server (~/.mcp.json). Defaults espelham
# o diretório do projeto para que o shim funcione mesmo sem env explícito.
_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("IMI_DB", os.path.join(_PROJECT_DIR, "imi_memory.db"))

# LLM backend determinístico: força Ollama local (phi4-mini) por padrão.
# PORQUÊ: o auto-detect de create_llm_from_env (llm.py:149) escolhe ClaudeLLM se
# ANTHROPIC_API_KEY estiver setada — mas a key da org está DESABILITADA (HTTP 400) e
# 'anthropic' nem está instalado no venv. Sem este override, im_enc via shim quebra com
# ModuleNotFoundError. Ollama local é grátis, offline e é o backend correto para um shim CLI.
# Override explícito (não setdefault) para vencer uma key herdada e quebrada no ambiente.
if not os.environ.get("IMI_LLM_BACKEND"):
    os.environ["IMI_LLM_BACKEND"] = "ollama"


def _resolve_space() -> Any:
    """Carrega o IMISpace do sqlite — único ponto de acesso ao store (espelha _get_space do MCP)."""
    from imi.space import IMISpace

    db_path = os.environ["IMI_DB"]
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"IMI_DB não encontrado: {db_path!r} — esperava o sqlite do IMI. "
            f"Defina IMI_DB ou rode o shim a partir do diretório do projeto."
        )
    return IMISpace.from_sqlite(db_path)


def verb_status(_args: argparse.Namespace) -> dict[str, Any]:
    """im_sts — estado do store: contagem por tier + preview. Reusa IMISpace.stats()."""
    space = _resolve_space()
    stats = space.stats()
    return {
        "ok": True,
        "verb": "im_sts",
        "stats": stats,
        "episodic": len(space.episodic.nodes),
        "semantic": len(space.semantic.nodes),
    }


def verb_encode(args: argparse.Namespace) -> dict[str, Any]:
    """im_enc — grava memória pelo MESMO caminho do MCP (secure_encode → sanitizer + crypto)."""
    experience = (args.experience or "").strip()
    if not experience:
        raise ValueError("im_enc exige --experience não-vazio (o texto da memória a persistir)")

    space = _resolve_space()
    tag_list = [t.strip() for t in (args.tags or "").split(",") if t.strip()] or None

    # Mesmo padrão do mcp_server.im_enc: tenta secure_encode (sanitizer+crypto), cai para encode cru.
    try:
        from imi.integrations.crypto_layer import secure_encode

        node = secure_encode(space, experience, tags=tag_list, source=args.source or "")
        path = "secure_encode"
    except ImportError:
        node = space.encode(experience, tags=tag_list, source=args.source or "")
        path = "encode(sem-sanitizer)"

    # IMISpace.encode() já chama self.save() ao final (space.py:282) quando há backend/persist_dir —
    # a escrita sobrevive ao processo sem chamada extra. Verificado por leitura (G7).
    return {
        "ok": True,
        "verb": "im_enc",
        "id": node.id,
        "summary": getattr(node, "summary_medium", ""),
        "tags": node.tags,
        "write_path": path,
    }


def _lexical_fts(space: Any, query: str, top_k: int) -> list[dict[str, Any]]:
    """Busca lexical FTS5 sem embedder. Inlined de mcp_server._lexical_search para evitar
    arrastar a dependência 'mcp' (não instalada fora do `uv run` do servidor). Mesma lógica."""
    backend = getattr(space, "backend", None)
    if backend is None or not hasattr(backend, "search_fts"):
        raise RuntimeError(
            f"lexical mode requer SQLiteBackend com FTS5; backend ativo: "
            f"{type(backend).__name__ if backend else None}"
        )
    # FTS5 trata -, :, . como sintaxe; termo único cru vira frase literal entre aspas.
    fts_query = query if " " in query.strip() else f'"{query.strip()}"'
    raw = backend.search_fts(fts_query, limit=top_k * 3)
    hits: list[dict[str, Any]] = []
    for node_id, rank in raw[:top_k]:
        node = backend.get_node("episodic", node_id) or backend.get_node("semantic", node_id)
        if node is None:
            continue
        d = node.to_dict()
        content = d.get("summary_medium") or d.get("seed") or d.get("summary_orbital") or ""
        hits.append({"score": round(float(-rank), 3), "id": node_id, "content": content[:200]})
    return hits


def verb_navigate(args: argparse.Namespace) -> dict[str, Any]:
    """im_nav — busca. mode=lexical usa FTS5 (sem embedder); senão navigate() semântico."""
    query = (args.query or "").strip()
    if not query:
        raise ValueError("im_nav exige um termo de busca posicional")

    space = _resolve_space()

    if args.mode == "lexical":
        hits = _lexical_fts(space, query, args.top_k)
    else:
        # navigate() retorna NavigationResult, cujo .memories é list[dict] (space.py:60). G7.
        nav = space.navigate(query, top_k=args.top_k)
        hits = [
            {"score": round(m.get("score", 0.0), 3), "store": m.get("store", "?"),
             "content": m.get("content", "")[:200]}
            for m in nav.memories
        ]

    return {"ok": True, "verb": "im_nav", "mode": args.mode, "count": len(hits), "hits": hits}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="imi", description="Shim CLI para as tools IMI (sem MCP).")
    sub = parser.add_subparsers(dest="verb", required=True)

    sub.add_parser("im_sts", help="Estado do store IMI (tiers + preview).")

    p_enc = sub.add_parser("im_enc", help="Grava uma memória (passa pelo sanitizer/crypto).")
    p_enc.add_argument("--experience", required=True, help="Texto da memória a persistir.")
    p_enc.add_argument("--tags", default="", help="Tags separadas por vírgula.")
    p_enc.add_argument("--source", default="", help="Origem da memória (opcional).")

    p_nav = sub.add_parser("im_nav", help="Busca memórias (semântica ou lexical/FTS5).")
    p_nav.add_argument("query", help="Termo de busca.")
    p_nav.add_argument("--mode", default="semantic", choices=["semantic", "lexical"])
    p_nav.add_argument("--top-k", type=int, default=10, dest="top_k")

    return parser


_DISPATCH = {"im_sts": verb_status, "im_enc": verb_encode, "im_nav": verb_navigate}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = _DISPATCH[args.verb](args)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as err:
        # Erro NUNCA engolido (Agent Legibility): inclui o ofensor e o verbo no JSON de saída.
        print(json.dumps(
            {"ok": False, "verb": args.verb, "error": str(err), "type": type(err).__name__},
            ensure_ascii=False,
        ))
        return 1


if __name__ == "__main__":
    sys.exit(main())
