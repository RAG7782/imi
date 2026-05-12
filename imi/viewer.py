#!/usr/bin/env python3
"""viewer.py — IMI Web Viewer (porta 7891).

Inspetor local do banco de memórias IMI. Zero dependências externas — stdlib pura.
Lê diretamente do SQLite (~/experimentos/tools/imi/imi_memory.db).

Funcionalidades:
  - Lista nós por tier/salience (paginado, 30/página)
  - Busca FTS5 full-text
  - Detalhe de nó (todas as propriedades)
  - Filtro por store/source
  - Deletar nó (soft-delete: is_deleted=1)
  - Stats gerais (contagem por tier/store/source)
  - API JSON: /api/stats

Uso:
  python3 ~/experimentos/imi/imi/viewer.py
  python3 -m imi.viewer
  # Abre http://localhost:7891
"""

from __future__ import annotations

import html as html_mod
import json
import sqlite3
import time
import urllib.parse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

HOME = Path.home()
DB = HOME / "experimentos/tools/imi/imi_memory.db"
PORT = 7891
HOST = "127.0.0.1"

# ── DB helpers ────────────────────────────────────────────────────────────────


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(DB), timeout=5.0)
    c.row_factory = sqlite3.Row
    return c


def _parse_node(row: sqlite3.Row) -> dict:
    try:
        data = json.loads(row["data"])
    except Exception:
        data = {}
    return {
        "node_id": row["node_id"],
        "store_name": row["store_name"],
        "is_deleted": row["is_deleted"],
        "created_at": row["created_at"],
        "inserted_at": row["inserted_at"],
        **data,
    }


def _total() -> int:
    with _conn() as c:
        r = c.execute("SELECT COUNT(*) FROM memory_nodes WHERE is_deleted=0").fetchone()
        return r[0] if r else 0


def _fmt_ts(ts) -> str:
    if not ts:
        return "—"
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(ts)


def _sal_class(s: float) -> str:
    if s >= 0.6:
        return "high"
    if s >= 0.45:
        return "mid"
    return "low"


def _e(s) -> str:
    """HTML-escape a value."""
    return html_mod.escape(str(s) if s is not None else "")


# ── HTML components ───────────────────────────────────────────────────────────

CSS = """
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', monospace;
       background: #0d1117; color: #e6edf3; font-size: 14px; }
a { color: #58a6ff; text-decoration: none; }
a:hover { text-decoration: underline; }
.nav { background: #161b22; border-bottom: 1px solid #30363d;
       padding: 12px 24px; display: flex; align-items: center; gap: 20px; }
.nav h1 { font-size: 17px; color: #f0f6fc; }
.badge { background: #21262d; border: 1px solid #30363d; border-radius: 12px;
         padding: 2px 10px; font-size: 12px; color: #8b949e; }
.container { max-width: 1100px; margin: 0 auto; padding: 24px; }
.stats-row { display: grid; grid-template-columns: repeat(auto-fit,minmax(130px,1fr));
             gap: 10px; margin-bottom: 20px; }
.sc { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px; }
.sc .lbl { font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: .5px; }
.sc .val { font-size: 26px; font-weight: 700; margin-top: 4px; }
.filters { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 12px; align-items: center; }
.flabel { font-size: 11px; color: #8b949e; }
.fb { background: #21262d; border: 1px solid #30363d; border-radius: 20px;
      padding: 3px 12px; font-size: 12px; color: #8b949e; cursor: pointer; }
.fb.on { background: #1f6feb; border-color: #388bfd; color: #fff; }
.cards { display: flex; flex-direction: column; gap: 8px; }
.card { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
        padding: 12px 14px; }
.card:hover { border-color: #58a6ff; }
.meta { font-size: 11px; color: #8b949e; display: flex; gap: 8px;
        flex-wrap: wrap; margin-bottom: 4px; }
.orbital { font-size: 14px; color: #f0f6fc; font-weight: 500; }
.medium  { font-size: 12px; color: #8b949e; margin-top: 3px; }
.tags { display: flex; gap: 4px; flex-wrap: wrap; margin-top: 6px; }
.tag  { background: #21262d; border: 1px solid #30363d; border-radius: 12px;
        padding: 1px 8px; font-size: 11px; color: #8b949e; }
.etag { background: #1a2332; border: 1px solid #1f6feb; border-radius: 12px;
        padding: 1px 8px; font-size: 11px; color: #58a6ff; }
.pill { padding: 1px 8px; border-radius: 12px; font-size: 11px; font-weight: 600; }
.high { background: #1a3a1a; color: #3fb950; }
.mid  { background: #2d2a1a; color: #d29922; }
.low  { background: #2d1a1a; color: #f85149; }
.tier-pill { background: #1f2937; color: #60a5fa; padding: 1px 8px;
             border-radius: 12px; font-size: 11px; }
.pag { display: flex; gap: 8px; margin-top: 20px; align-items: center; }
.pag a { background: #21262d; border: 1px solid #30363d; border-radius: 6px;
         padding: 5px 14px; color: #8b949e; }
.sbar { display: flex; gap: 8px; margin-bottom: 18px; }
.sbar input { flex: 1; background: #21262d; border: 1px solid #30363d; border-radius: 6px;
              padding: 7px 12px; color: #e6edf3; font-size: 14px; }
.sbar input:focus { outline: none; border-color: #58a6ff; }
.sbar button { background: #238636; border: 1px solid #2ea043; border-radius: 6px;
               padding: 7px 16px; color: #fff; cursor: pointer; }
.dtable { width: 100%; border-collapse: collapse; }
.dtable td { padding: 7px 0; border-bottom: 1px solid #21262d; vertical-align: top; }
.dtable td:first-child { color: #8b949e; width: 150px; padding-right: 12px; }
pre { background: #21262d; border-radius: 6px; padding: 10px; font-size: 12px;
      white-space: pre-wrap; word-break: break-all; color: #e6edf3; }
.delbtn { background: #da3633; border: 1px solid #f85149; border-radius: 6px;
          padding: 6px 14px; color: #fff; cursor: pointer; font-size: 13px; }
.delbtn:hover { background: #b91c1c; }
.stitle { font-size: 12px; color: #8b949e; text-transform: uppercase; letter-spacing: .5px;
          margin: 18px 0 10px; }
.panel { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 8px; }
.empty { text-align: center; padding: 50px; color: #484f58; }
</style>
"""


def page_wrap(title: str, body: str, total: int) -> str:
    nav = f"""<div class="nav">
      <h1>🧠 IMI</h1>
      <span class="badge">{total} memórias</span>
      <a href="/">Início</a>
      <a href="/search">Busca</a>
      <a href="/stats">Stats</a>
    </div>"""
    return f"""<!DOCTYPE html><html lang="pt-BR">
<head><meta charset="utf-8"><title>{_e(title)}</title>{CSS}</head>
<body>{nav}<div class="container">{body}</div></body></html>"""


# ── Page builders ─────────────────────────────────────────────────────────────


def build_index(qs: dict) -> str:
    page = max(1, int(qs.get("page", ["1"])[0]))
    store = qs.get("store", [""])[0]
    source = qs.get("source", [""])[0]
    per = 30
    offset = (page - 1) * per

    where, params = ["is_deleted=0"], []
    if store:
        where.append("store_name=?")
        params.append(store)
    if source:
        where.append("json_extract(data,'$.source')=?")
        params.append(source)
    wh = " AND ".join(where)

    total = _total()
    with _conn() as c:
        rows = c.execute(
            f"SELECT * FROM memory_nodes WHERE {wh} ORDER BY inserted_at DESC LIMIT ? OFFSET ?",
            params + [per, offset],
        ).fetchall()
        stores = [
            r[0]
            for r in c.execute("SELECT DISTINCT store_name FROM memory_nodes WHERE is_deleted=0")
        ]
        sources = [
            r[0]
            for r in c.execute(
                "SELECT DISTINCT json_extract(data,'$.source') FROM memory_nodes WHERE is_deleted=0"
            )
            if r[0]
        ]

    nodes = [_parse_node(r) for r in rows]
    pages = max(1, (total + per - 1) // per)

    def qs_with(**kw) -> str:
        d = {"store": store, "source": source}
        d.update(kw)
        return "?" + "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in d.items() if v)

    # Filters
    store_f = (
        f'<a href="{qs_with(page=1, store="")}" class="fb {"on" if not store else ""}">Todos</a>'
        + "".join(
            f'<a href="{qs_with(page=1, store=s)}" '
            f'class="fb {"on" if store == s else ""}">{_e(s)}</a>'
            for s in stores
        )
    )
    src_f = (
        f'<a href="{qs_with(page=1, source="")}" class="fb {"on" if not source else ""}">Todos</a>'
        + "".join(
            f'<a href="{qs_with(page=1, source=s)}" '
            f'class="fb {"on" if source == s else ""}">'
            f"{_e(s.replace('hook:', ''))}</a>"
            for s in sources
        )
    )

    # Cards
    cards = ""
    for n in nodes:
        sal = float((n.get("affect") or {}).get("salience", 0))
        sc = _sal_class(sal)
        tier = n.get("tier") or "—"
        orb = _e(n.get("summary_orbital") or str(n.get("seed", ""))[:80] or "(sem orbital)")
        med = _e((n.get("summary_medium") or "")[:110])
        tags = n.get("tags") or []
        ents = n.get("entities") or []
        src = _e(n.get("source", "—"))
        ts = _fmt_ts(n.get("created_at"))
        tag_html = "".join(f'<span class="tag">{_e(t)}</span>' for t in tags[:5])
        ent_html = "".join(f'<span class="etag">{_e(e)}</span>' for e in ents[:4])
        cards += f"""<div class="card">
          <div class="meta">
            <span class="pill {sc}">{sal:.2f}</span>
            <span class="tier-pill">{_e(str(tier))}</span>
            <span>{src}</span><span>{ts}</span>
          </div>
          <div class="orbital"><a href="/node/{_e(n["node_id"])}">{orb}</a></div>
          {'<div class="medium">' + med + "</div>" if med else ""}
          <div class="tags">{ent_html}{tag_html}</div>
        </div>"""

    if not cards:
        cards = '<div class="empty">Nenhuma memória.</div>'

    pag = '<div class="pag">'
    if page > 1:
        pag += f'<a href="{qs_with(page=page - 1)}">← Anterior</a>'
    pag += f'<span style="color:#8b949e">Pág {page}/{pages}</span>'
    if page < pages:
        pag += f'<a href="{qs_with(page=page + 1)}">Próxima →</a>'
    pag += "</div>"

    body = f"""
    <div class="stats-row">
      <div class="sc"><div class="lbl">Total</div><div class="val">{total}</div></div>
    </div>
    <div class="filters"><span class="flabel">Store:</span>{store_f}</div>
    <div class="filters"><span class="flabel">Source:</span>{src_f}</div>
    <div class="cards">{cards}</div>{pag}"""
    return page_wrap("IMI Viewer", body, total)


def build_detail(node_id: str) -> str:
    with _conn() as c:
        row = c.execute("SELECT * FROM memory_nodes WHERE node_id=?", (node_id,)).fetchone()
    if not row:
        return page_wrap(
            "IMI — não encontrado", f"<p>Nó {_e(node_id)} não encontrado.</p>", _total()
        )

    n = _parse_node(row)
    sal = float((n.get("affect") or {}).get("salience", 0))
    sc = _sal_class(sal)

    def dr(label: str, value) -> str:
        if value is None or value == "" or value == []:
            return ""
        if isinstance(value, (list, dict)):
            value = json.dumps(value, ensure_ascii=False, indent=2)
        return f"<tr><td>{_e(label)}</td><td><pre>{_e(str(value))}</pre></td></tr>"

    trows = "".join(
        [
            dr("node_id", n.get("node_id")),
            dr("store", n.get("store_name")),
            dr("source", n.get("source")),
            dr("tier", n.get("tier")),
            dr("salience", f"{sal:.3f} ({sc})"),
            dr("created", _fmt_ts(n.get("created_at"))),
            dr("orbital", n.get("summary_orbital")),
            dr("medium", n.get("summary_medium")),
            dr("seed", n.get("seed")),
            dr("tags", n.get("tags")),
            dr("entities", n.get("entities")),
            dr("affect", n.get("affect")),
            dr("file_path", n.get("file_path")),
        ]
    )

    orig = _e(n.get("original") or n.get("seed") or "")
    orig_html = f'<div class="stitle">Conteúdo</div><pre>{orig}</pre>' if orig else ""

    body = f"""
    <div style="margin-bottom:14px"><a href="/" style="color:#8b949e">← Voltar</a></div>
    <div class="stitle">Propriedades</div>
    <div class="panel"><table class="dtable">{trows}</table></div>
    {orig_html}
    <div style="margin-top:20px">
      <form method="post" action="/node/{_e(node_id)}/delete"
            onsubmit="return confirm('Deletar esta memória?')">
        <button type="submit" class="delbtn">🗑 Deletar (soft-delete)</button>
      </form>
    </div>"""
    return page_wrap(f"IMI — {node_id[:12]}", body, _total())


def build_search(qs: dict) -> str:
    q = qs.get("q", [""])[0].strip()
    total = _total()
    nodes: list[dict] = []

    if q:
        with _conn() as c:
            try:
                rows = c.execute(
                    """SELECT mn.* FROM memory_nodes mn
                       JOIN memory_fts mf ON mn.node_id = mf.node_id
                       WHERE memory_fts MATCH ? AND mn.is_deleted=0 LIMIT 50""",
                    (q,),
                ).fetchall()
            except Exception:
                rows = c.execute(
                    "SELECT * FROM memory_nodes WHERE data LIKE ? AND is_deleted=0 LIMIT 50",
                    (f"%{q}%",),
                ).fetchall()
        nodes = [_parse_node(r) for r in rows]

    cards = ""
    for n in nodes:
        sal = float((n.get("affect") or {}).get("salience", 0))
        sc = _sal_class(sal)
        orb = _e(n.get("summary_orbital") or str(n.get("seed", ""))[:80] or "(sem orbital)")
        med = _e((n.get("summary_medium") or "")[:100])
        ents = n.get("entities") or []
        tags = n.get("tags") or []
        ent_html = "".join(f'<span class="etag">{_e(e)}</span>' for e in ents[:3])
        tag_html = "".join(f'<span class="tag">{_e(t)}</span>' for t in tags[:4])
        cards += f"""<div class="card">
          <div class="meta">
            <span class="pill {sc}">{sal:.2f}</span>
            <span>{_e(n.get("source", "—"))}</span>
            <span>{_fmt_ts(n.get("created_at"))}</span>
          </div>
          <div class="orbital"><a href="/node/{_e(n["node_id"])}">{orb}</a></div>
          {'<div class="medium">' + med + "</div>" if med else ""}
          <div class="tags">{ent_html}{tag_html}</div>
        </div>"""

    if q and not cards:
        cards = '<div class="empty">Sem resultados para esta busca.</div>'

    body = f"""
    <form method="get" action="/search">
      <div class="sbar">
        <input name="q" value="{_e(q)}" placeholder="Buscar memórias (FTS5)..." autofocus>
        <button type="submit">Buscar</button>
      </div>
    </form>
    {'<div class="stitle">Resultados (' + str(len(nodes)) + ")</div>" if q else ""}
    <div class="cards">{cards}</div>"""
    return page_wrap("IMI — Busca", body, total)


def build_stats() -> str:
    total = _total()
    with _conn() as c:
        by_store = c.execute(
            "SELECT store_name, COUNT(*) FROM memory_nodes WHERE is_deleted=0 GROUP BY store_name"
        ).fetchall()
        by_source = c.execute(
            "SELECT json_extract(data,'$.source'), COUNT(*) "
            "FROM memory_nodes WHERE is_deleted=0 "
            "GROUP BY json_extract(data,'$.source') ORDER BY COUNT(*) DESC"
        ).fetchall()
        by_tier = c.execute(
            "SELECT json_extract(data,'$.tier'), COUNT(*) "
            "FROM memory_nodes WHERE is_deleted=0 "
            "GROUP BY json_extract(data,'$.tier') ORDER BY COUNT(*) DESC"
        ).fetchall()
        recent = c.execute(
            "SELECT COUNT(*) FROM memory_nodes WHERE is_deleted=0 AND inserted_at > ?",
            (time.time() - 86400,),
        ).fetchone()[0]

    def tbl(rows, headers) -> str:
        h = "".join(
            f"<th style='text-align:left;padding:6px 10px;color:#8b949e'>{_e(h)}</th>"
            for h in headers
        )
        b = "".join(
            "<tr>"
            + "".join(
                "<td style='padding:6px 10px;border-bottom:1px solid #21262d'>"
                f"{_e(str(v or '—'))}</td>"
                for v in r
            )
            + "</tr>"
            for r in rows
        )
        return (
            "<table style='width:100%;border-collapse:collapse'>"
            f"<thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"
        )

    body = f"""
    <div class="stats-row">
      <div class="sc"><div class="lbl">Total</div><div class="val">{total}</div></div>
      <div class="sc"><div class="lbl">Últimas 24h</div><div class="val">{recent}</div></div>
      <div class="sc"><div class="lbl">Stores</div><div class="val">{len(by_store)}</div></div>
      <div class="sc"><div class="lbl">Sources</div><div class="val">{len(by_source)}</div></div>
    </div>
    <div class="stitle">Por Store</div>
    <div class="panel">{tbl(by_store, ["Store", "Count"])}</div>
    <div class="stitle">Por Source</div>
    <div class="panel">{tbl(by_source, ["Source", "Count"])}</div>
    <div class="stitle">Por Tier</div>
    <div class="panel">{tbl(by_tier, ["Tier", "Count"])}</div>"""
    return page_wrap("IMI — Stats", body, total)


def build_api_stats() -> bytes:
    total = _total()
    with _conn() as c:
        by_store = dict(
            c.execute(
                "SELECT store_name, COUNT(*) FROM memory_nodes "
                "WHERE is_deleted=0 GROUP BY store_name"
            ).fetchall()
        )
        recent = c.execute(
            "SELECT COUNT(*) FROM memory_nodes WHERE is_deleted=0 AND inserted_at > ?",
            (time.time() - 86400,),
        ).fetchone()[0]
    return json.dumps(
        {"total": total, "by_store": by_store, "last_24h": recent}, ensure_ascii=False
    ).encode()


# ── HTTP handler ──────────────────────────────────────────────────────────────


class IMIHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silencioso — sem logs de request no terminal

    def _send(
        self, body: str | bytes, ct: str = "text/html; charset=utf-8", status: int = 200
    ) -> None:
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        path = parsed.path

        try:
            if path == "/":
                self._send(build_index(qs))
            elif path == "/search":
                self._send(build_search(qs))
            elif path == "/stats":
                self._send(build_stats())
            elif path == "/api/stats":
                self._send(build_api_stats(), ct="application/json")
            elif path.startswith("/node/") and not path.endswith("/delete"):
                node_id = path.split("/node/")[1].strip("/")
                self._send(build_detail(node_id))
            else:
                self._send(b"Not found", status=404)
        except Exception as e:
            self._send(f"<pre>Erro interno: {_e(str(e))}</pre>", status=500)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path.startswith("/node/") and path.endswith("/delete"):
            node_id = path.split("/node/")[1].rstrip("/delete").strip("/")
            node_id = node_id.replace("/delete", "")
            # Parse node_id correctly
            parts = path.split("/")
            # /node/{id}/delete → parts = ['', 'node', '{id}', 'delete']
            if len(parts) == 4 and parts[3] == "delete":
                node_id = parts[2]
                with _conn() as c:
                    c.execute("UPDATE memory_nodes SET is_deleted=1 WHERE node_id=?", (node_id,))
                    c.commit()
            self.send_response(302)
            self.send_header("Location", "/")
            self.end_headers()
        else:
            self._send(b"Not found", status=404)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    if not DB.exists():
        print(f"[IMI Viewer] ERRO: banco não encontrado em {DB}")
        print("  Verifique: ~/experimentos/tools/imi/imi_memory.db")
        return

    total = _total()
    print(f"[IMI Viewer] {total} memórias no banco")
    print(f"[IMI Viewer] Abrindo http://{HOST}:{PORT}")
    print("[IMI Viewer] Ctrl+C para parar\n")

    server = HTTPServer((HOST, PORT), IMIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[IMI Viewer] Encerrado.")


if __name__ == "__main__":
    main()
