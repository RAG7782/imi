# Epic: IMI-FAST — MCP Server com State Caching + ANN Index

> **ID:** IMI-E05
> **Status:** Ready
> **Prioridade:** P2 (P1 após 500K+ nós)
> **Depende de:** nenhuma
> **Fundamentação técnica:** FAISS (Meta AI) + Springdrift long-running agent pattern (arXiv:2604.04660)
> **Problema resolvido:** IMISpace.from_sqlite() potencialmente recarrega 236K embeddings a cada tool call — não escala para 1M+ nós

---

## Objetivo

Garantir que o MCP server mantenha o IMISpace em memória entre tool calls (singleton estável) e substituir o linear scan O(N) de embeddings por um índice ANN (Approximate Nearest Neighbor) O(log N). Com isso, latência de `im_nav` passa de potencialmente segundos para < 50ms, e o sistema escala para 1M+ nós sem degradação.

## Diagnóstico necessário (Story S01)

Antes de implementar, verificar se o problema realmente existe: o `_space = None` singleton em `mcp_server.py` pode já estar funcionando corretamente se o processo MCP server vive entre tool calls. Medir latência real antes de otimizar.

## Stories

### S01: Diagnóstico de latência e estado do singleton
Antes de qualquer otimização, medir o estado real:

```python
# Adicionar timing a cada tool call no mcp_server.py
import time
t0 = time.monotonic()
space = _get_space()
t1 = time.monotonic()
log(f"_get_space() took {(t1-t0)*1000:.1f}ms, episodic={len(space.episodic)}")
```

Verificar:
1. `_space` é `None` no início de CADA tool call ou persiste entre calls?
2. `from_sqlite()` é chamado 1x (startup) ou N×?
3. Latência atual de `im_nav` end-to-end: medir com `time.monotonic()`

- **Verify:** log `~/.claude/imi_boot.log` mostra latência e quantas vezes `_get_space()` chama `from_sqlite()`
- **Critério de GO/NO-GO:** se `from_sqlite()` < 2 calls por sessão e latência < 200ms → S02-S04 são opcionais. Se > 2 calls ou > 500ms → crítico, executar tudo.

### S02: Garantir singleton robusto com warm-up
Se diagnóstico confirmar recarregamento frequente:

```python
# mcp_server.py — singleton com warm-up explícito
_space: IMISpace | None = None
_space_loaded_at: float = 0.0
SPACE_TTL = 3600  # recarregar após 1h (para pegar novos encodes de outras sessões)

def _get_space() -> IMISpace:
    global _space, _space_loaded_at
    now = time.monotonic()
    if _space is None or (now - _space_loaded_at) > SPACE_TTL:
        db_path = os.environ.get("IMI_DB", ...)
        _space = IMISpace.from_sqlite(db_path)
        _space_loaded_at = now
        log(f"IMISpace loaded: {len(_space.episodic)} episodic, {len(_space.semantic)} semantic")
    return _space
```

- TTL de 1h: garante que novos encodes (de outros processos) sejam visíveis
- **Verify:** após warm-up, 10 chamadas consecutivas de `im_nav` mostram `from_sqlite()` chamado 1x no log

### S03: Índice FAISS para ANN search
Substituir linear scan por FAISS `IndexFlatIP` (exact inner product = cosine sim com embeddings normalizados):

```python
# imi/spatial.py ou novo imi/ann_index.py
import faiss
import numpy as np

class ANNIndex:
    """Approximate Nearest Neighbor index sobre embeddings do IMISpace."""
    
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # Inner Product = cosine se normalizado
        self.node_ids: list[str] = []        # mapeamento posição → node_id
    
    def add(self, node_id: str, embedding: np.ndarray) -> None:
        emb = embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(emb)
        self.index.add(emb)
        self.node_ids.append(node_id)
    
    def search(self, query_emb: np.ndarray, top_k: int = 10) -> list[tuple[str, float]]:
        q = query_emb.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, top_k)
        return [(self.node_ids[i], float(scores[0][j])) 
                for j, i in enumerate(indices[0]) if i >= 0]
    
    def rebuild_from_space(self, space: IMISpace) -> None:
        """Constrói índice a partir do IMISpace carregado."""
        ...
    
    def add_incremental(self, node_id: str, embedding: bytes) -> None:
        """Adiciona nó novo sem rebuild completo (no im_enc)."""
        ...
```

- Rebuild completo: apenas no warm-up (1x por sessão)
- Adições incrementais: `add_incremental()` chamado em cada `im_enc`
- Fallback: se FAISS não instalado, usar linear scan existente
- **Verify:** `im_nav("KONA")` com FAISS ativo retorna mesmos top-5 que linear scan (± 1 por ANN approximation), com latência < 50ms para 236K nós

### S04: Write-ahead buffer para im_enc
Evitar `space.save()` síncrono a cada encode (I/O pesado):

```python
_write_buffer: list[MemoryNode] = []
_last_flush: float = 0.0
BUFFER_SIZE = 10
FLUSH_INTERVAL = 60  # segundos

def _flush_buffer(space: IMISpace) -> None:
    global _write_buffer, _last_flush
    if not _write_buffer:
        return
    # Batch write
    for node in _write_buffer:
        space.backend.save_node(node)
    _write_buffer.clear()
    _last_flush = time.monotonic()
    log(f"Flushed {len(_write_buffer)} nodes to SQLite")
```

- Flush automático: quando buffer >= 10 ou >= 60s sem flush
- Flush obrigatório: antes de `im_drm` e no shutdown
- **Verify:** 10 chamadas consecutivas de `im_enc` → apenas 1 `space.save()` no log SQLite

### S05: Tool im_perf() para monitoramento
Novo MCP tool para inspecionar saúde e performance do servidor:

```python
@mcp.tool()
def im_perf() -> str:
    """MCP server performance metrics and health check"""
    return {
        "space_loaded": _space is not None,
        "space_age_seconds": time.monotonic() - _space_loaded_at,
        "episodic_count": len(_space.episodic) if _space else 0,
        "semantic_count": len(_space.semantic) if _space else 0,
        "ann_index_size": ann_index.index.ntotal if ann_index else 0,
        "write_buffer_size": len(_write_buffer),
        "p50_nav_ms": percentile(_nav_latencies, 50) if _nav_latencies else None,
        "p95_nav_ms": percentile(_nav_latencies, 95) if _nav_latencies else None,
        "from_sqlite_calls_this_session": _sqlite_load_count,
    }
```

- Logar latências de `im_nav` em ring buffer de 100 entradas
- **Verify:** `im_perf()` retorna `from_sqlite_calls_this_session` = 1 após múltiplas tool calls

## Métricas de sucesso

- `from_sqlite()` calls por sessão: baseline N → alvo = 1
- `im_nav` latência P95: baseline ? ms → alvo < 100ms
- `im_enc` latência P95: baseline ? ms → alvo < 200ms (com buffer)
- Memória RAM do processo MCP: < 600MB para 236K nós (budget: 2.5 bytes/dim/nó)

## Critério de priorização

Executar S01 imediatamente. Se `from_sqlite_calls_this_session` > 1 ou P95 > 500ms → escalar para P1 e executar S02-S05. Caso contrário, manter como P2 e revisar quando nós > 500K.

## Referências

- FAISS (Meta AI): github.com/facebookresearch/faiss — ANN search em RAM
- arXiv:2604.04660 — Springdrift: long-running agent process pattern
- Kuzu (embeddable graph): kuzudb.com — alternativa para graph layer se Qdrant for pesado
