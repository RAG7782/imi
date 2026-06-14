# ADR 0001 — H-MEM: índice como hit + gate de paridade-com-flat

> **Status:** Aceito · **Data:** 2026-06-14 · **Decisor:** Renato (com Claude, research G7)
> **Contexto técnico:** spec `~/experimentos/specs/2026-06-14-imi-hmem-positional-index.md`
> **Commits:** `0718aae` (Opção A), `d5f7591` (gate independente), `329807c` (collapsed-net + paridade)

## Contexto

O IMI ganhou retrieval recursivo H-MEM (arXiv:2507.22925): uma árvore `layer/parent_id/child_ptrs`
populada na consolidação, descida top-down por ponteiros pai→filho. O objetivo é busca
`O((a+k·fanout)·D)` em vez de flat `O(N·D)` — ~100× menos comparações no store atual.

Antes de promover `mode=hierarchical` a default, a spec (§4.5 passo 7) exige um gate quantitativo
medido sob shadow-mode. Ao construir o harness de medição (`scripts/hmem_passo7.py`), duas questões
de design emergiram dos DADOS, não de teoria. Este ADR registra ambas e por quê.

## Decisão 1 — Nó-índice é roteador E hit válido (Opção A)

**Decisão.** No `recursive_retrieve`, um nó-índice (pattern consolidado, `layer<3`) é candidato a
hit, além de ser roteador da descida. Não retornamos apenas episódios (L3).

**Por quê (dados).** Medido no store real: **28% das queries reais querem um pattern node como
top-1**. No IMI, patterns SÃO conhecimento destilado consultável — descartá-los do resultado
degradaria 28% dos retrievals.

**Trade-off.** Desvia do paper H-MEM puro (que retorna só episódios, tratando índices como puro
roteamento). Aceitamos o desvio: fidelidade ao IMI > pureza ao paper, com evidência empírica.

**Efeito medido.** `top1_divergence` vs flat caiu de 15% → 5%.

## Decisão 2 — Gate de promoção = paridade-com-flat (não recall absoluto)

**Decisão.** O gate de promoção exige `recall_hier ≥ recall_flat − ε` (paridade, ε=2pp), NÃO
`recall ≥ 90%` absoluto. Pure-flat recall é medido como referência de não-regressão. Promoção
também exige canário 100% + árvore populada + (sob shadow) divergência baixa por 2 semanas.

**Por quê (investigação G6, em snapshots do store real).** A barra de 90% absoluto é INATINGÍVEL
por qualquer retrieval:
- O teto de recall **NÃO é cobertura**: 18/20 alvos dos anchors ESTÃO na árvore.
- **NÃO é tuning de k_topo**: o pai do alvo rankeava #51 de 56 (sim centroide 0.07 vs sim
  folha 0.67) — poda no topo por centroide lossy.
- **Pure FLAT também trava em ~80%**: 4/20 anchors têm nota cujo embedding mal casa com o próprio
  alvo (sim 0.27–0.39). O teto é **embedder + qualidade dos anchors**, não a arquitetura H-MEM.

Logo, "iterar até recall ≥ 90%" tunaria o knob errado. O gate honesto e atingível é: o hierárquico
não regride vs o baseline confiável (flat), ao custo de ~100× menos comparações. Hoje: 80% = 80%,
`parity_ok=True`, `PROMOTE_OK=True`.

**Trade-off.** Paridade prova não-regressão, não correção absoluta. A correção absoluta fica
limitada pelo embedder e é rastreada separadamente (ver Trabalho Futuro).

## Decisão 2b — Collapsed-tree safety net (mecanismo que viabiliza a paridade)

Para o hierárquico ALCANÇAR a paridade (estava em 75% vs flat 80%), adicionamos um safety net
RAPTOR-style (`IMI_HMEM_COLLAPSED`, default ON): além da descida, um sweep flat BOUNDED das
folhas-episódio IN-TREE é mesclado no pool de candidatos. Recupera a folha forte cujo ramo foi
podado por um centroide fraco. ASSERT-6 cobre órfãos FORA da árvore; isto cobre folhas DENTRO da
árvore que a descida pulou. Efeito: recall 75% → 80% (= flat). Latência ~inalterada (sweep sobre
embeddings já em memória).

## Decisão 3 — Validade estatística: Wilson CI em vez de corte rígido

N=20 anchors é pequeno: um corte rígido de `<2%` tolera 0 erros e é hipersensível a um único
quase-empate. O harness reporta o intervalo de confiança de Wilson 95% da taxa de match top-1,
não um ponto frágil. (reparo de boas práticas #2)

## Alternativas consideradas e rejeitadas

- **Opção B (só episódios, purista):** rejeitada — perde os 28% de queries que querem pattern.
- **Multi-vector centroid (medoid + spread):** simulado, recuperava só até 80% (= collapsed),
  mais complexo. Collapsed-net entrega o mesmo com menos código.
- **Manter recall absoluto 90% + bloquear até ter oracle:** rejeitada — adia o H-MEM
  indefinidamente por um problema (embedder) que não é dele. Adotamos "ambos": paridade agora,
  oracle depois.

## Consequências

- ✅ H-MEM promovível por um critério honesto e atingível (não-regressão a ~100× menos custo).
- ✅ Default permanece `semantic` (flat) até 2 semanas de shadow + canário 100%.
- ⚠️ O recall absoluto (~80%) é limitado pelo embedder — não é resolvível por retrieval.
- 🔬 Flags: `IMI_HMEM_RETRIEVAL`, `IMI_HMEM_PROMOTE`, `IMI_HMEM_SHADOW`, `IMI_HMEM_COLLAPSED`.

## Trabalho futuro (rastreado)

**Oracle semântico real.** Os 20 anchors do canário são LEXICAIS (congelados para FTS5), fracos
como ground-truth semântico (4/20 com sim<0.4 ao próprio alvo). Para medir recall ABSOLUTO de
forma confiável, construir um conjunto de avaliação a partir de queries reais de uso (não os
anchors lexicais). Ver `docs/adr/0001-...` §Trabalho Futuro e o item de backlog correspondente.
