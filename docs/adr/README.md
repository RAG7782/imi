# Architecture Decision Records (ADR)

Decisões arquiteturais do IMI com contexto, alternativas e consequências — para que o
próximo agente/humano entenda o PORQUÊ, não só o O QUÊ. Formato leve (Michael Nygard).

| # | Título | Status | Data |
|---|--------|--------|------|
| [0001](0001-hmem-index-as-hit-and-parity-gate.md) | H-MEM: índice como hit + gate de paridade-com-flat | Aceito | 2026-06-14 |

## Backlog de decisões / trabalho futuro

- **Oracle semântico real (aberto):** os anchors do canário são lexicais (FTS5), fracos como
  ground-truth semântico — 4/20 têm sim<0.4 ao próprio alvo. O recall absoluto do retrieval
  (~80% teto) é limitado por isso, não pela arquitetura. Construir um eval-set a partir de
  queries reais de uso para medir recall absoluto de forma confiável. Bloqueia: medição de
  correção absoluta (não bloqueia a promoção, que usa paridade-com-flat). Ref: ADR 0001 §2.
