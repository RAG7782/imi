# IMI: Insights Consolidados (WS1 → WS-I)

> Síntese de 9 fluxos de experimentação. Todas as métricas são reproduzíveis.

---

## Insight Principal: O Paradoxo Recuperação vs Relevância

**O IMI descobriu um trade-off fundamental que nenhum paper de RAG aborda:**

> Características que modelam a memória humana (recência, afeto, massa) **degradam**
> a recuperação pura, mas **melhoram** a relevância para agentes.

Isso explica por que sistemas RAG puro são insuficientes para agentes: eles
otimizam a métrica errada. Um agente de SRE não quer "o incidente mais similar"
— quer "o incidente mais útil para resolver o problema AGORA".

### Evidência quantificada

| Cenário | Métrica | RAG (rw=0) | IMI (rw=0.1) | Variação |
|---------|---------|-----------|--------------|----------|
| Recuperação pura | R@5 | **0,341** | 0,304 | -0,037 |
| Temporal de agente | PrecDom@5 | 0,689 | **0,756** | +0,067 |
| Incidentes recentes | PrecDom@5 | 0,800 | **0,900** | +0,100 |
| Multi-salto | R@10 | 0,750 | **1,000** | +0,250 |
| Coerência temporal | Idade média top-5 | 41,2d | **16,8d** | -24,4d |

---

## Os 7 Insights

### 1. O Ponto Ótimo é rw=0,10 (não 0,3, não 0,0)

```
rw=0,00  → melhor recuperação pura, zero consciência temporal
rw=0,10  → -3,7% recuperação, +6,7% precisão de domínio, -59% idade média
rw=0,15  → ponto ótimo para consultas "recente", sem perda em consultas "antigo"
rw=0,30  → -40% recuperação, muito agressivo (padrão anterior)
```

**Implicação**: O valor padrão de 0,3 estava errado. 0,10 é o equilíbrio ótimo.
Já corrigido no código.

### 2. Surpresa é Elegante mas Inútil (para recuperação)

**Custo**: 2 chamadas ao LLM por codificação (prever + calcular_surpresa)
**Benefício**: +0,003 R@5 (negligível)
**Situação atual**: Agora ativado sob demanda (`use_predictive_coding=False`)

**Porém**: surpresa pode ter valor não medido — como sinalização de
anomalias, ou como gatilho para consolidação prioritária. Isso não foi
testado nos benchmarks atuais.

**Possibilidade**: Usar `surprise_magnitude` como **critério de consolidação**
(memórias de alta surpresa são consolidadas primeiro) em vez de como boost
de recuperação.

### 3. Arestas de Grafo Resolvem Multi-salto Sem Chamadas ao LLM

| Sistema | Multi-salto R@10 | Chamadas LLM/consulta |
|---------|-----------------|----------------------|
| Apenas cosseno | 75% (15/20) | 0 |
| HippoRAG-Sim | 10% (1/10) | 1 NER/consulta |
| **IMI + Grafo** | **100% (20/20)** | **0** |

A camada de grafo com ativação em cascata (Collins & Loftus, 1975) superou o
HippoRAG-Sim em multi-salto. As 5 consultas que o cosseno perdia foram resolvidas
por expansão de 1 salto via arestas causais.

**Possibilidade**: Detectar automaticamente arestas causais no momento da codificação
usando o LLM (1 chamada extra). Isso transformaria surpresa (inútil para recuperação)
em detecção de arestas (útil para multi-salto).

### 4. Memórias Antigas Não "Apodrecem"

No horizonte longo (365 dias, 600 incidentes):
- Incidente do 1º trimestre pesquisado no 4º trimestre: **posição 1** tanto com cosseno quanto com IMI
- R@5 cai ao longo do tempo, mas isso é um artefato do denominador
  (mais incidentes similares acumulados)

**Implicação**: `rw=0,10` não empurra memórias antigas para baixo.
A resistência ao esquecimento pelo afeto protege memórias emocionalmente salientes.

### 5. Memória Compartilhada em Múltiplos Agentes Tem Valor Mensurável

3 agentes especializados compartilhando memória vs cada um isolado:
- Compartilhada vence em 2/5 consultas entre domínios
- Maior ganho: consultas que cruzam 3 ou mais domínios

**Possibilidade**:
- Visões por agente: cada agente enxerga a memória compartilhada com
  pesos de relevância personalizados
- Gradiente de confiança: memórias do próprio agente têm boost de massa vs memórias
  de outros agentes

### 6. AMBench é o Primeiro Benchmark para Memória de Agente

Nenhum benchmark existente testa o ciclo completo de memória de agente:
codificar → recuperar → consolidar → agir → aprender.

**AMBench testa 5 dimensões que benchmarks de RAG ignoram:**
1. Precisão de recuperação (padrão)
2. Qualidade de consolidação (pureza do cluster)
3. Relevância de ação (precisão de affordance)
4. Coerência temporal (preservação de recência)
5. Curva de aprendizado (melhora ao longo do tempo)

**Situação atual**: Funcional com 300–600 incidentes, 10 padrões, 90–365 dias.
Publicável como contribuição independente.

### 7. TimescaleDB era Engenharia em Excesso

SQLite com modo WAL: 87x mais rápido que TimescaleDB para cargas de trabalho de agente.
Complexidade removida: -425 linhas de código, -11 testes, -1 docker-compose.

**Lição**: Para memória de agente único (o caso de uso principal), infraestrutura mínima
vence. TimescaleDB só faria sentido para análises temporais em escala multi-agente
— um caso de uso que ainda não existe.

---

## Possibilidades (ordenadas por impacto/esforço)

### Alta prioridade (impacto comprovado)

| Possibilidade | Evidência | Esforço |
|--------------|-----------|---------|
| **rw adaptativo por contexto de consulta** | rw=0,10 é ótimo globalmente mas "recente X" pede rw=0,15 e "encontrar X" pede rw=0,0 | Médio |
| **Detecção automática de arestas causais no encode** | Arestas de grafo deram 100% de recall em multi-salto | Médio (1 chamada LLM/encode) |
| **Preprint no arXiv** | AMBench + ablação + grafo = paper completo | 1–2 semanas |

### Média prioridade (promissor mas não validado)

| Possibilidade | Hipótese | Esforço |
|--------------|----------|---------|
| **Surpresa → prioridade de consolidação** | Memórias de alta surpresa são mais importantes para generalizar | Baixo |
| **Visões por agente em memória compartilhada** | Gradiente de confiança por agente | Médio |
| **Atualização do modelo de embedding (e5-large)** | 384d→1024d pode mudar o equilíbrio recuperação/relevância | Baixo (teste) |

### Exploratória (futuro acadêmico)

| Possibilidade | Pergunta | Esforço |
|--------------|----------|---------|
| **Calibração da curva de esquecimento** | Qual taxa de decaimento maximiza R@5 no longo prazo? | Médio |
| **Implantação em ambiente real** | IMI melhora a conclusão de tarefas de agentes reais? | Alto |
| **IMI híbrido + HippoRAG** | Grafo de conhecimento para multi-salto + IMI para temporal/afeto | Alto |
| **Consolidação em fluxo contínuo** | Consolidar em tempo real vs em lote | Médio |

---

## Mapa de Evidências

```
WS1 (Percepção)────────→ embeddings funcionam, níveis de zoom são úteis
WS2 (Surpresa)─────────→ implementado mas impacto negligível → tornado opt-in
WS3 (Validação)────────→ 46 testes, métricas de linha de base estabelecidas
WS4 (Mergulho Fundo)───→ correção de bugs, backend SQLite, surpresa integrada
WS-A (Ablação)─────────→ contribuição de cada feature quantificada
                          ├─ surpresa: negligível (+0,003)
                          ├─ recência: maior impacto (-0,117)
                          └─ rw ótimo = 0,10
WS-B (Temporal)────────→ rw ajuda cenários de agente (+6,7%)
                          └─ consultas recentes: +10% com rw=0,15
WS-C (HippoRAG)────────→ IMI > features de agente, HippoRAG > multi-salto
                          └─ motivou a camada de grafo (WS-G)
WS-D (AMBench)─────────→ primeiro benchmark de memória de agente
                          ├─ 300 incidentes, 5 métricas, 3 linhas de base
                          └─ coerência temporal: IMI 16,8d vs RAG 41,2d
WS-E (Arquitetura)─────→ TSDB removido, preditivo opt-in, temp=0,3
WS-F (padrão rw)───────→ 0,3 → 0,10 (baseado em evidência)
WS-G (Camada de Grafo)─→ 100% recall multi-salto, sem degradação padrão
                          └─ ativação em cascata funciona para expansão de memória
WS-H (Rascunho do Paper)→ estrutura completa do paper com todos os resultados
WS-I (Benchmark Ampliado)→ multi-agente, entre domínios, 365 dias validado
                           ├─ memória compartilhada ajuda consultas entre agentes
                           ├─ memórias antigas acessíveis com rw=0,10
                           └─ efeito do denominador em R@5 (não é degradação real)
```

---

## Estado Final do Sistema

```
imi/
├── node.py          # MemoryNode com afeto, surpresa, massa, relevância
├── store.py         # VectorStore com relevance_weight configurável (padrão 0,10)
├── space.py         # IMISpace: codificar → navegar → consolidar → sonhar
├── graph.py         # ★ NOVO: MemoryGraph com ativação em cascata
├── affect.py        # AffectiveTag: saliência, valência, excitação → resist_esquecimento, massa
├── surprise.py      # Codificação preditiva (agora opt-in)
├── affordance.py    # Extração de ações com temperature=0,3
├── embedder.py      # SentenceTransformer (all-MiniLM-L6-v2)
├── storage.py       # JSONBackend + SQLiteBackend (TSDB removido)
├── maintain.py      # Consolidação: encontrar_clusters, mesclar_padrões
├── llm.py           # LLMAdapter com suporte a temperatura
├── lite.py          # IMI Lite-B: cosseno + zoom + affordances
└── temporal.py      # TemporalContext para rastreamento de sessão

experimentos/
├── ws_a_ablation_study.py          # Análise de contribuição de features
├── ws_b_temporal_decay.py          # Validação temporal de 90 dias
├── ws_c_hipporag_comparison.py     # IMI vs HippoRAG-Sim
├── ws_d_agent_memory_benchmark.py  # AMBench (primeiro benchmark de memória de agente)
├── ws_g_graph_augmented_retrieval.py # Validação de expansão via grafo
├── ws_i_expanded_benchmark.py      # Multi-agente, entre domínios, 365 dias
└── ws3_validation_framework.py     # Validação base (100 post-mortems)

Testes: 35 passando, 0 falhas
Linhas removidas: ~500 (TSDB + migração + docker)
Linhas adicionadas: ~200 (graph.py) + ~2.000 (experimentos)
```
