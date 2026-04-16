# Epic: IMI-DOMAIN — Boot Contextual por Projeto (CategoryRAG)

> **ID:** IMI-E03
> **Status:** Ready
> **Prioridade:** P1
> **Depende de:** IMI-E02 (Positional Reorder v3 facilita a composição, mas não é blocker)
> **Fundamentação científica:** Synthius-Mem CategoryRAG (arXiv:2604.11563) — 94.37% LoCoMo, 21.79ms latência
> **Problema resolvido:** 5 domínios em paralelo num espaço único. Boot traz contexto de OXÉ quando você quer KONA.

---

## Objetivo

Implementar **detecção de domínio ativo** + **filtro de memórias por domínio** no boot e no delta nav. Quando você abre uma sessão de KONA, as primeiras memórias do boot devem ser de KONA, não de OXÉ (que tem maior volume e salience média).

Inspirado no Synthius-Mem que organiza memórias em 6 domínios cognitivos com CategoryRAG (filtro por categoria antes do top-K). Adaptado ao seu ecossistema de 5 projetos primários.

## Taxonomia de domínios

```python
DOMAIN_TAXONOMY = {
    "OXE": {
        "tags": {"oxe", "qdrant", "juris-repo", "ingestao", "vps-alan",
                 "open-webui", "docker", "corpus", "ollama", "api-gateway"},
        "paths": ["/experimentos/tools/oxe", "/opt/oxe"],
        "handoff_keywords": ["oxe", "ingestão", "qdrant", "sprint"],
    },
    "KONA": {
        "tags": {"kona", "1time", "xp-investimentos", "carf", "impugnacao",
                 "art43-ctn", "upfront", "rfb", "fato-gerador", "tax-litigation"},
        "paths": ["/analises", "/AGAC"],
        "handoff_keywords": ["kona", "1time", "impugnação", "rfb"],
    },
    "SYMBIONT": {
        "tags": {"symbiont", "deerflow", "v0.3.0", "v0.4.1", "orchestrator",
                 "wave", "multi-agent", "dispatcher", "bridge"},
        "paths": ["/experimentos/tools/symbiont"],
        "handoff_keywords": ["symbiont", "deerflow", "orchestrator"],
    },
    "RESEARCH": {
        "tags": {"paper", "densidade-semiotica", "sigma-star", "l2ds",
                 "modal", "zenodo", "benchmark", "experimento", "a1", "a2", "e16"},
        "paths": ["/experimentos"],
        "handoff_keywords": ["paper", "experimento", "densidade", "sigma"],
    },
    "JURIDICO": {
        "tags": {"protecao-patrimonial", "holding", "familia-bilionaria", "lca",
                 "caso-juridico", "successoes", "sociedade", "advogado"},
        "paths": ["/analises/juridico"],
        "handoff_keywords": ["proteção", "holding", "família", "caso jurídico"],
    },
    "AIOX": {
        "tags": {"aiox", "ps-skill", "story", "epic", "deploy", "imi",
                 "clawvault", "gravar", "handoff", "sessao"},
        "paths": ["/.aiox", "/.claude"],
        "handoff_keywords": ["aiox", "imi", "gravar", "handoff"],
    },
}
```

## Boot composition com domain awareness

```
Detecção (prioridade decrescente):
  1. CWD: path match contra DOMAIN_TAXONOMY[domain]["paths"]
  2. HANDOFF.yaml: campo "project" ou keyword match
  3. Tag frequency: top-10 tags dos últimos 5 encodes → domain por overlap
  4. Fallback: "GLOBAL" (comportamento atual)

Composição do boot:
  4 slots domain-aware  → memórias do domínio detectado (score × 1.5 boost)
  2 slots cross-domain  → top memórias globais (mantém cross-pollination)
  1 slot intention      → intenção pendente mais urgente (IMI-E04, se ativo)

Output: <imi_boot domain="KONA" detection_method="handoff_keyword">
```

## Stories

### S01: Domain taxonomy como módulo Python
Criar `~/.claude/imi_domains.py` com a DOMAIN_TAXONOMY dict + função `detect_domain(cwd: str, handoff_path: str | None) -> tuple[str, str]`.

- Retorna `(domain_name, detection_method)` onde method é `"cwd" | "handoff" | "tag_frequency" | "global"`
- Para tag_frequency: ler últimos 5 nós inseridos no SQLite e calcular overlap
- **Verify:** `python3 -c "from imi_domains import detect_domain; print(detect_domain('/Users/renato/analises', None))"` → `('KONA', 'cwd')`

### S02: Função domain_filter em fetch_top_memories()
Adicionar parâmetro `domain: str | None = None` em `fetch_top_memories()`.

Quando domain != None e domain != "GLOBAL":
```python
domain_tags = DOMAIN_TAXONOMY[domain]["tags"]
# Filtrar nós que tenham pelo menos 1 tag em domain_tags
# Em SQLite: JSON_EACH para iterar tags no campo data
# Performance: índice FTS já cobre — usar memory_fts para tag matching
```

- Boost de 1.5× no score de nós dentro do domínio detectado
- Não filtrar exclusivamente — manter 2 slots cross-domain
- **Verify:** com domain="KONA", `fetch_top_memories()` retorna >= 3 nós com tags kona/1time/art43

### S03: Integração em imi_boot_semantic.py
Chamar `detect_domain(CWD, HANDOFFS_DIR/latest)` no início de `build_cache()`.

- Passar domain para `fetch_top_memories()`
- Incluir `domain="{detected}"` e `detection_method="{method}"` no tag `<imi_boot>`
- Adicionar ao log: `[boot] domain detected: KONA via handoff_keyword`
- **Verify:** bloco `<imi_boot>` tem atributo `domain` diferente de "GLOBAL" quando CWD é `/analises`

### S04: Integração em imi_delta_nav.py
Detectar domínio da mensagem atual via tag matching nas entidades detectadas.

```python
# Se entidade "KONA" detectada → domain = "KONA"
# Busca semântica filtrada por domain antes de busca global
# Se domain hit count >= 2: retornar somente domain-filtered results
# Caso contrário: mesclar domain + global
```

- **Verify:** mensagem "qual o status do KONA?" → delta nav retorna apenas memórias KONA

### S05: Override manual de domínio
Adicionar suporte a `AIOX_ACTIVE_DOMAIN=KONA` como env var ou arquivo `~/.aiox_domain`.

```bash
echo "KONA" > ~/.aiox_domain  # força domínio para próxima sessão
```

- `detect_domain()` verifica arquivo antes de qualquer heurística
- Arquivo deletado automaticamente após 1 boot (single-use override)
- **Verify:** `echo "RESEARCH" > ~/.aiox_domain && python3 imi_boot_semantic.py` → boot com `domain="RESEARCH"`

## Métricas de sucesso

- Domain detection accuracy: testar em 10 sessões históricas, verificar se domínio detectado corresponde ao projeto trabalhado (alvo: 80%+)
- Cross-domain contamination: boot deve ter <= 2 memórias de domínio diferente do ativo
- Latência: detect_domain() deve completar em < 100ms

## Referências

- arXiv:2604.11563 — Synthius-Mem: CategoryRAG + 6 domínios cognitivos, 94.37% LoCoMo
- arXiv:2309.02427 — CoALA: taxonomia de 4 tipos de memória (working/episodic/semantic/procedural)
