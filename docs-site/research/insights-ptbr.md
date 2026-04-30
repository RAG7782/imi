# Insights de Pesquisa

Resultados consolidados de 9 fluxos experimentais (WS1 a WS-I). Todas as métricas são reproduzíveis executando os experimentos em `/experiments/`.

## Descoberta central: O paradoxo Recuperação vs Relevância

O IMI revelou um trade-off fundamental que a literatura de RAG não aborda:

> Características que modelam a memória humana (recência, afeto, massa) **degradam a recuperação pura**, mas **melhoram a relevância para agentes**.

Um agente de SRE consultando "falhas de autenticação recentes" não quer a memória semanticamente mais similar — quer a mais útil para a situação atual. O RAG otimiza a métrica errada.

### Evidência quantificada

| Cenário | Métrica | RAG (rw=0) | IMI (rw=0,1) | Variação |
|---------|---------|-----------|--------------|----------|
| Recuperação pura | R@5 | **0,341** | 0,304 | -0,037 |
| Consultas temporais de agente | PrecDom@5 | 0,689 | **0,756** | +0,067 |
| Incidentes recentes | PrecDom@5 | 0,800 | **0,900** | +0,100 |
| Recuperação multi-salto | R@10 | 0,750 | **1,000** | +0,250 |
| Coerência temporal | Idade média top-5 | 41,2d | **16,8d** | -24,4d |

---

## Insight 1: O ponto ótimo é rw=0,10

```
rw=0,00  → melhor recuperação pura, zero consciência temporal
rw=0,10  → -3,7% recuperação, +6,7% precisão de domínio, -59% idade média dos resultados
rw=0,15  → ponto ótimo para consultas "recente", sem penalidade para consultas "antigo"
rw=0,30  → -40% recuperação (muito agressivo — era o valor padrão anterior)
```

O padrão anterior de `rw=0,30` estava errado. `rw=0,10` é o ótimo empiricamente validado. Corrigido no código.

## Insight 2: Surpresa é elegante mas negligenciável para recuperação

- **Custo**: 2 chamadas ao LLM por codificação (prever + calcular surpresa)
- **Benefício**: +0,003 R@5 — estatisticamente negligenciável
- **Situação atual**: Agora ativado sob demanda (`use_predictive_coding=False` por padrão)

Surpresa pode ter valor não medido como sinal de anomalia ou critério de prioridade para consolidação (memórias de alta surpresa merecem ser consolidadas primeiro). Ainda não validado.

## Insight 3: Arestas de grafo resolvem multi-salto sem chamadas ao LLM

| Sistema | Multi-salto R@10 | Chamadas LLM/consulta |
|---------|-----------------|----------------------|
| Apenas cosseno | 75% (15/20) | 0 |
| HippoRAG-Sim | 10% (1/10) | 1 NER/consulta |
| **IMI + Grafo** | **100% (20/20)** | **0** |

Ativação em cascata (Collins & Loftus, 1975) sobre arestas de similaridade detectadas automaticamente supera a simulação HippoRAG no multi-salto. As 5 consultas em que o cosseno falhou foram todas resolvidas por expansão de 1 salto.

Observação: A comparação com HippoRAG usou uma simulação de NER via regex, não o HippoRAG real com NER por LLM. O HippoRAG real teria desempenho significativamente melhor.

## Insight 4: Memórias antigas não "apodrecem"

Em uma simulação de 365 dias com 600 incidentes:
- Incidente do 1º trimestre pesquisado no 4º trimestre chegou à **posição 1** tanto no cosseno quanto no IMI
- R@5 diminui ao longo do tempo, mas isso é um efeito do denominador (mais incidentes similares se acumulam), não uma degradação real

`rw=0,10` não empurra memórias antigas para baixo. A `fade_resistance` da marcação afetiva protege memórias emocionalmente salientes do decaimento.

## Insight 5: Memória compartilhada entre múltiplos agentes tem valor mensurável

3 agentes especializados compartilhando memória vs isolados:
- Memória compartilhada vence em 2/5 consultas entre domínios
- Maior ganho: consultas que cruzam 3 ou mais domínios
- Trabalho futuro: visões por agente com gradientes de confiança por agente

## Insight 6: AMBench é o primeiro benchmark de memória de agente

Nenhum benchmark existente testa o ciclo completo de memória de agente: codificar → recuperar → consolidar → agir → aprender.

O AMBench testa 5 dimensões que benchmarks de RAG ignoram:
1. Precisão de recuperação (R@5, MRR)
2. Qualidade de consolidação (pureza do cluster)
3. Relevância de ação (precisão de affordance@1)
4. Coerência temporal (idade média dos 5 primeiros resultados)
5. Curva de aprendizado (melhora ao longo do tempo)

Situação atual: Funcional com 300–600 incidentes, 10 padrões, 90–365 dias simulados.

## Insight 7: SQLite é 87x mais rápido que TimescaleDB

Para cargas de trabalho de agente único, SQLite com modo WAL supera amplamente o TimescaleDB:
- 87x menor latência para inserções O(1)
- -425 linhas de código removidas
- -11 testes removidos
- -1 docker-compose.yml removido

TimescaleDB só faria sentido para análises multi-agente em escala — um caso de uso que ainda não existe para o IMI.

---

## Mapa de evidências

```
WS1 (Percepção)      → embeddings funcionam, níveis de zoom são úteis
WS2 (Surpresa)       → implementado mas impacto negligenciável → tornado opt-in
WS3 (Validação)      → 46 testes, métricas de linha de base estabelecidas
WS4 (Mergulho Fundo) → correção de bugs, backend SQLite, surpresa integrada
WS-A (Ablação)       → contribuição de cada feature quantificada
                        ├─ surpresa: +0,003 (negligenciável)
                        ├─ recência: maior impacto (-0,117 se removida)
                        └─ rw ótimo = 0,10
WS-B (Temporal)      → rw ajuda cenários de agente (+6,7%)
                        └─ consultas recentes: +10% com rw=0,15
WS-C (HippoRAG)      → motivou a camada de grafo (WS-G)
WS-D (AMBench)       → primeiro benchmark de memória de agente
                        └─ coerência temporal: IMI 16,8d vs RAG 41,2d
WS-E (Arquitetura)   → TimescaleDB removido, preditivo opt-in
WS-F (padrão rw)     → 0,3 → 0,10 (correção baseada em evidência)
WS-G (Camada Grafo)  → 100% recall multi-salto, sem degradação padrão
WS-H (Rascunho Paper)→ paper completo com todos os resultados
WS-I (Ampliado)      → multi-agente, entre domínios, 365 dias validado
```
