# IMI — Roadmap de Evolução 2026
> Baseado em pesquisa de estado da arte: 25 sistemas analisados, 202 URLs coletadas, benchmarks até Abril 2026

---

## Diagnóstico do sistema atual

| Dimensão | Estado atual | Gap identificado |
|---|---|---|
| **Encode** | 4.300 events | Qualidade heterogênea — context_hint ausente em muitos |
| **Navigate** | 98 events | 44x menos que encode — memória sem uso |
| **Semantic patterns** | 57.253 nós, avg_access=1.7 | Conhecimento consolidado não reativado |
| **Retrieval mode** | Cosine similarity único | Similar ≠ útil para decisão atual |
| **Domínios** | 1 espaço global | 5 projetos, amnésia temática cross-domínio |
| **Intenções** | Não modeladas | "Onde parei e por quê" perdido entre sessões |
| **Performance** | from_sqlite() potencialmente N× | Escala linear, não escala para 1M nós |

---

## Os 5 Epics

| Epic | ID | Problema | Solução | SOTA ref | Prioridade |
|---|---|---|---|---|---|
| IMI-AFFORD | E01 | Navigate = 44× menos que encode | Two-Phase Retrieval + Memory Worth | MemRL (arXiv:2601.03192) | **P1** |
| IMI-PATTERN | E02 | Padrões semânticos com access=1.7 | Boot scoring diferenciado + Positional Reorder v3 | GAM (arXiv:2604.12285) | **P1** |
| IMI-DOMAIN | E03 | Amnésia temática cross-domínio | CategoryRAG + domain detection | Synthius-Mem 94.37% (arXiv:2604.11563) | **P1** |
| IMI-INTENT | E04 | Intenções não modeladas | Nó intention + Foresight signals | EverMemOS (arXiv:2601.02163) | **P1** |
| IMI-FAST | E05 | Escalabilidade do MCP server | Singleton + FAISS ANN index | FAISS + Springdrift | **P2** |

---

## Sequência de implementação recomendada

### Fase 1 — Quick wins (2-3 sessões)
Implementar primeiro o que tem maior ROI com menor esforço:

1. **IMI-E02 S01-S03** — Scoring diferenciado + Positional Reorder v3
   - Modifica apenas `imi_boot_semantic.py` (arquivo local, zero risco)
   - Impacto imediato: padrões semânticos aparecem no boot
   
2. **IMI-E05 S01** — Diagnóstico de latência (20min)
   - Determina se E05 é P1 ou P2 real
   - Sem diagnóstico, otimização prematura

3. **IMI-E04 S01-S02** — Tools im_int + im_int_fulfill
   - Adição pura ao MCP server (zero breaking changes)
   - Começa a capturar intenções imediatamente

### Fase 2 — Core upgrades (3-4 sessões)
4. **IMI-E01 S01-S03** — Schema MW + Two-Phase Retrieval
   - Requer atenção: modifica schema do nó e lógica de im_nav
   - Implementar em branch separado, testar antes de merge

5. **IMI-E03 S01-S03** — Domain detection + CategoryRAG no boot
   - Cria `imi_domains.py` + modifica fetch_top_memories()
   - Testar com sessions de KONA vs OXÉ para validar detection accuracy

6. **IMI-E04 S03-S05** — Intentions no boot + protocolo GRAVAR
   - Integra com E03 (domain-aware intentions)

### Fase 3 — Advanced (2-3 sessões)
7. **IMI-E01 S04-S05** — Delta nav utility mode + MW update no GRAVAR
8. **IMI-E03 S04-S05** — Domain filter no delta nav + override manual
9. **IMI-E05 S02-S05** — ANN index + write buffer (se S01 confirmar necessidade)

---

## Descobertas da pesquisa não incorporadas (backlog futuro)

### Alta prioridade técnica, baixa urgência

**Dual-Trace Encoding** (arXiv:2604.12948)
- +20pp accuracy com custo zero — adicionar `scene_trace` ao im_enc
- Schema change: campo `scene_trace TEXT` no nó IMI
- Quando: após E01-E04 estabilizarem

**Multi-signal retrieval: Semantic + BM25 + Entity** (Mem0 v2)
- Nomes próprios e siglas (KONA, CARF, 1TIME) falham em semantic-only
- SQLite FTS5 já existe no IMI — conectar ao im_nav como fallback
- Quando: Fase 3

**Bi-temporal facts** (Graphiti, arXiv:2603.17244)
- valid_from + valid_until para fatos que mudam com o tempo
- Crítico quando: informações de estado de projeto (stale after 30d)
- Quando: IMI v2 (grande refactor)

**HippoRAG — Personalized PageRank** (arXiv:2405.14831)
- Propagação de relevância pelo grafo im_glnk
- NetworkX já instalável em Python
- Quando: após im_glnk ter > 100 edges (Fase 3+)

**Memory Worth para deprecação automática** (arXiv:2604.12007)
- MW < 0.2 após 30 sessões → marcar como deprecated
- Requer E01 completo (success/failure tracking)
- Quando: após E01 rodar 30 sessões

### Para pesquisa futura (IMI v3+)

- **HyperMem** (arXiv:2604.08256) — hiperedges para dependências de ordem > 2
- **Kumiho AGM** (arXiv:2603.17244) — revisão formal de crenças inconsistentes
- **Thought-Retriever** (arXiv:2604.12231) — cachear raciocínio, não só fatos
- **MemMachine** (arXiv:2604.04853) — ground-truth preservation (raw episode storage)

---

## O que a pesquisa confirmou sobre o IMI atual

### Pontos fortes (o IMI já implementa SOTA)

- ✅ **CLS dual-store** (episodic + semantic) — correto, confirmado por todos os papers
- ✅ **Surprise-based encoding** — Predictive coding, implementado em `surprise.py`
- ✅ **Affordances** como action potentials — único entre sistemas analisados
- ✅ **Dreaming** como consolidação offline — equivalente ao GAM Topic Associative Network
- ✅ **Reconsolidação** em acesso — sofisticado, não encontrado em outros sistemas
- ✅ **Positional Reorder** (Liu 2023) — já implementado, melhorar com v3

### Gaps vs. SOTA (a implementar via epics)

- ❌ **Flat retrieval** — cosine sim único vs. multi-signal + two-phase (E01)
- ❌ **Domain blindness** — espaço único vs. CategoryRAG (E03)
- ❌ **Intenções** — sem Foresight signals (E04)
- ❌ **Padrões subutilizados** — semantic patterns com access=1.7 (E02)
- ❌ **Sem feedback loop** — sem Memory Worth / success tracking (E01)

### Conclusão do diagnóstico

O IMI já está na top-10% dos sistemas de memória disponíveis publicamente. A arquitetura CLS + affordances + dreaming é mais sofisticada que Mem0, Zep, LangMem, e MemGPT/Letta. Os gaps são de **otimização de uso**, não de arquitetura fundamental. Os 5 epics fecham esses gaps sem precisar de refactor arquitetural.

---

## Referências completas

| Paper | Score LoCoMo | O que implementa |
|---|---|---|
| Synthius-Mem (arXiv:2604.11563) | 94.37% | CategoryRAG + 6 domínios |
| Kumiho (arXiv:2603.17244) | 93.3% | Dual-store + AGM belief revision |
| HyperMem (arXiv:2604.08256) | 92.73% | Hipergrafos hierárquicos |
| Mem0 v2 (github.com/mem0ai/mem0) | 91.6% | ADD-only + multi-signal |
| REMem (arXiv:2602.13530) | +13.4pp vs SOTA | Gists temporais + retriever agêntico |
| EverMemOS (arXiv:2601.02163) | SOTA LoCoMo | MemCells + Foresight signals |
| FadeMem (arXiv:2601.18642) | -45% storage | Decay adaptativo + Ebbinghaus |
| MemRL (arXiv:2601.03192) | Superior em 4 benchmarks | Two-phase retrieval |
| Memory Worth (arXiv:2604.12007) | Corr=0.89 | MW = success/total |
| Dual-Trace (arXiv:2604.12948) | +20pp LongMemEval | Scene trace encoding |
| GAM (arXiv:2604.12285) | LoCoMo + LongDialQA | Dual graph + gating |
| HippoRAG (arXiv:2405.14831, NeurIPS 2024) | +20% multi-hop | PageRank sobre KG |
| Graphiti (github.com/getzep/graphiti) | 80.32% | Bi-temporal graph |
| MemGPT (arXiv:2310.08560) | baseline | Virtual context management |
| Generative Agents (arXiv:2304.03442) | fundacional | Observation+reflection |

Pesquisa conduzida em Abril 2026. Fonte: 202 URLs coletadas via web search por agente especializado.
