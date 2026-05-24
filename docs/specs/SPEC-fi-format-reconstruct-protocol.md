---
spec_id: SPEC-IMI-FI-001
status: accepted
date: 2026-05-24
deciders:
  - renato
  - claude-opus-4-7
confidence: 0.93
addresses_audit_gap: T2.2 (Spec gap - formato .fi.md para reconstruct_protocol)
adr_ref: ~/experimentos/.aiox/stories/ADR-004.adr.md
plan_ref: ~/.claude/plans/gentle-riding-dusk.md (Fase 4, Onda 2)
target_artifact: ~/experimentos/tools/imi/imi/fi/reconstruct_protocol.fi.md
applies_to: Onda 2 / Fase 4 (Reconstruct Worker)
status_in_plan: artefato a criar (não existe ainda; este doc é o contrato de formato)
---

# SPEC-IMI-FI-001 — Formato `.fi.md` para `reconstruct_protocol`

## Contexto

A Fase 4 do plano IMI Persistence v4 (Onda 2, condicional) introduz o **Reconstruct Worker** — processo background que, para nós inativos >90d, reconstroi conteúdo a partir do seed + neighbors context via phi4-mini local e calcula `fidelity_score = cosine(emb(reconstructed), emb(original))`. **Nunca apaga, só anota.**

O AIP cross-pollination da sessão 2026-05-24 (agente background frente 2) propôs **hook FI**: codificar o protocolo de reconstrução (orbital→medium→detailed → fidelity_check) como **Framework Injection declarativo testável** em vez de heurística in-code Python. Isso permite Têmpera adversarial (`/tempera` ou `/forja-pro`) sobre o próprio protocolo, evita protocolo travado em código C-like, e permite versionamento de protocolo independente do código.

O ataque T2.2 da auditoria final `/forja` Rota C exigiu **definir formalmente o formato do arquivo `reconstruct_protocol.fi.md`** antes de chegar à Fase 4 — caso contrário a Fase 4 começa com retrabalho de design de formato.

## Decisão de formato

**Usar o template oficial da skill `/fig construir`** (8 blocos obrigatórios) + frontmatter de governance herdado de `/fig governanca`. Justificativa:

1. **Padrão já estabelecido** — todas as FIs do ecossistema do usuário seguem os 8 blocos (Identidade, Princípios Inegociáveis, Definições Críticas, Protocolo de Raciocínio, Cercas e Exclusões, Critérios de Qualidade, Arquitetura de Delivery, Autoauditoria). Romper o padrão criaria FI órfã não-temperável.
2. **Têmpera reutilizável** — `/tempera`, `/forja-pro`, `/forja Rota C` esperam encontrar os 8 blocos para atacar; FI fora do template não passa pelos ataques estruturados.
3. **Governance herdada** — frontmatter da ficha de governança (Identificação, Propósito, Contexto de Uso, Riscos, Limites de Confiança) dá rastreabilidade exigida pelo CMP envelope.
4. **Operacionalidade > novidade** — formato livre teria que ser justificado por ganho concreto; o ganho não existe (protocolo de reconstrução é exatamente o tipo de instrução semântico-estratégica que motivou o template FI).

**Decisão**: formato livre **rejeitado**. Template oficial **adotado**.

## Schema do arquivo `imi/fi/reconstruct_protocol.fi.md`

Estrutura obrigatória, em ordem:

### 1. Frontmatter YAML (governance)

```yaml
---
fi_id: FI-IMI-RECONSTRUCT-001
fi_version: <X.Y>           # bump por edição substantiva, não cosmética
fi_status: <draft|temperada|production>
fi_temperada: <true|false>
fi_ic: <0.00-1.00>          # IC pós-/tempera; null se status=draft
fi_author: <claude-opus-4-7 | gemini-2.5-flash | llama-3.3-70b | renato>
fi_temperada_em: <YYYY-MM-DD ou null>
fi_target_model: <phi4-mini-local | outro>
fi_input_contract:          # contrato de entrada esperado
  - seed: <string, content truncado do node, ≤ 500 chars>
  - neighbors: <list[str], top-3 vizinhos por edge_weight do grafo>
  - tier_hint: <hard|soft|speculative>
fi_output_contract:         # contrato de saída esperado
  - reconstructed: <string, texto reconstruído>
  - confidence: <0.00-1.00, autodeclarada pelo modelo>
  - failure_mode: <string|null, se modelo se recusa a reconstruir>
fi_metric: cosine(embedding(reconstructed), embedding(original))
fi_metric_threshold: 0.92   # ≥ 0.92 = indistinguível qualitativamente
fi_write_scope:             # hard rule: worker SÓ escreve nestas colunas
  - fidelity_score
  - worker_skip
fi_forbidden_scope:         # hard rule: worker NUNCA escreve nestas
  - seed
  - summary_orbital
  - summary_medium
  - summary_detailed
  - embedding
plan_ref: ~/.claude/plans/gentle-riding-dusk.md
adr_ref: ~/experimentos/.aiox/stories/ADR-004.adr.md
spec_ref: ~/experimentos/tools/imi/docs/specs/SPEC-fi-format-reconstruct-protocol.md
---
```

### 2. Os 8 blocos obrigatórios

Conforme `~/.aiox/skills/fig/fig-construir.md` (BLOCO B), na ordem exata:

**Bloco 1 — Identidade** (papel do modelo, missão de reconstrução, escopo do IMI)

**Bloco 2 — Princípios Inegociáveis** (zero-loss invariant; nunca inventar; preferir "não sei" a confabulação; preservar campos não-tocáveis)

**Bloco 3 — Definições Críticas** (seed, summary_orbital/medium/detailed, neighbors, fidelity_score, tier hard/soft/speculative, worker_skip)

**Bloco 4 — Protocolo de Raciocínio** (orbital→medium→detailed em 3 passos com checkpoints de validação cruzada; quando interromper)

**Bloco 5 — Cercas e Exclusões** (nunca escrever em colunas fora de fi_write_scope; nunca rodar se phi4-mini timeout; nunca fabricar neighbor se grafo retornou < 3)

**Bloco 6 — Critérios de Qualidade** (cosine ≥ 0.92 = ótimo; 0.70-0.92 = aceitável com flag "reconstructed"; <0.70 = manter Warm original como Hard)

**Bloco 7 — Arquitetura de Delivery** (formato JSON estrito conforme fi_output_contract; campos obrigatórios; ordem)

**Bloco 8 — Autoauditoria** (checklist pré-output: princípio zero-loss? cerca de escopo? fidelity self-declared coerente? failure_mode preenchido se applicable?)

### 3. Histórico de versões

Tabela ao final:

```markdown
## Histórico

| Versão | Data | Autor | Mudança | Têmpera IC |
|---|---|---|---|---|
| 0.1 | YYYY-MM-DD | claude-opus-4-7 | Draft inicial | - |
| 0.2 | YYYY-MM-DD | tempera multi-modal | Pós-/tempera 22 ataques (18 legítimos) | 0.91 |
| 1.0 | YYYY-MM-DD | renato | Production após validação 100 nós | 0.91 |
```

## Esqueleto pronto-para-preencher

A Fase 4 deve criar `~/experimentos/tools/imi/imi/fi/reconstruct_protocol.fi.md` partindo deste esqueleto (substituir `<...>` por conteúdo real):

```markdown
---
fi_id: FI-IMI-RECONSTRUCT-001
fi_version: 0.1
fi_status: draft
fi_temperada: false
fi_ic: null
fi_author: <quem-está-criando>
fi_temperada_em: null
fi_target_model: phi4-mini-local
fi_input_contract:
  - seed: <string, ≤ 500 chars>
  - neighbors: <list[str], top-3>
  - tier_hint: <hard|soft|speculative>
fi_output_contract:
  - reconstructed: <string>
  - confidence: <0.00-1.00>
  - failure_mode: <string|null>
fi_metric: cosine(embedding(reconstructed), embedding(original))
fi_metric_threshold: 0.92
fi_write_scope:
  - fidelity_score
  - worker_skip
fi_forbidden_scope:
  - seed
  - summary_orbital
  - summary_medium
  - summary_detailed
  - embedding
plan_ref: ~/.claude/plans/gentle-riding-dusk.md
adr_ref: ~/experimentos/.aiox/stories/ADR-004.adr.md
spec_ref: ~/experimentos/tools/imi/docs/specs/SPEC-fi-format-reconstruct-protocol.md
---

# FI-IMI-RECONSTRUCT-001 — Protocolo de Reconstrução de Nó IMI

## Bloco 1 — Identidade

Você é <persona específica em termos do domínio IMI>.
Sua função é <missão precisa>.
Você opera dentro do domínio de <escopo: nó memória IMI inativo >90d, reconstrução por phi4-mini local, restrição zero-loss>.

## Bloco 2 — Princípios Inegociáveis

1. <princípio 1 — por que é inegociável>
2. ...

## Bloco 3 — Definições Críticas

- seed = <definição operacional>
- summary_orbital = <...>
- ...

## Bloco 4 — Protocolo de Raciocínio

1. <passo 1: lê seed e neighbors>
2. <passo 2: reconstrói summary_orbital>
3. <passo 3: expande para medium, depois detailed>
4. <checkpoint: validação cruzada com neighbors antes de declarar confidence>

## Bloco 5 — Cercas e Exclusões

NUNCA faça:
- <proibição: escrever em coluna fora de fi_write_scope>
- ...

SEMPRE faça:
- <obrigação: declarar failure_mode se não conseguir reconstruir>
- ...

## Bloco 6 — Critérios de Qualidade

Seu output é bom quando:
- cosine ≥ 0.92 vs embedding original
- confidence autodeclarada ≥ 0.85
- nenhum campo do output está vazio sem failure_mode preenchido

Seu output é ruim quando:
- <antipadrão 1: confabula neighbor inexistente>
- ...

## Bloco 7 — Arquitetura de Delivery

Entregue seu output como JSON estrito:
```json
{
  "reconstructed": "<texto>",
  "confidence": 0.00,
  "failure_mode": null
}
```

## Bloco 8 — Autoauditoria

Antes de entregar, verifique:
- [ ] Honrei o princípio zero-loss (não sobrescrevi nada)
- [ ] Respeitei fi_write_scope e fi_forbidden_scope
- [ ] Declarei failure_mode se applicable
- [ ] confidence autodeclarada é coerente com qualidade do reconstructed

## Histórico

| Versão | Data | Autor | Mudança | Têmpera IC |
|---|---|---|---|---|
| 0.1 | <data> | <autor> | Draft inicial conforme SPEC-IMI-FI-001 | - |
```

## Gates e validação

- **Gate de criação** (Fase 4 Onda 2): arquivo `reconstruct_protocol.fi.md` deve passar por `/fig-validar` antes de ser instanciado pelo Reconstruct Worker
- **Gate de produção**: arquivo deve passar por `/tempera` ou `/forja-pro` multi-modal antes de virar `fi_status: production`; IC mínimo 0.85
- **Gate de mudança**: qualquer edit do `.fi.md` em produção exige bump `fi_version` (X.Y → X.Y+1 se cosmético; X.Y → X+1.0 se Bloco 4/5/6 mudou)
- **Gate cross-canal**: bump de `fi_version` exige nota no Histórico + atualização do plano `~/.claude/plans/gentle-riding-dusk.md` (Fase 4) + edge IMI causal (mesma política do ADR-004)

## Decisão NÃO tomada (registrar)

- **Não definir Bloco 1-8 agora** — esta spec define o **formato do contêiner**, não o conteúdo da FI. O conteúdo será preenchido na própria Fase 4, com input do estado real do worker e dados de 30+ dias pós-Onda 1.
- **Não criar a pasta `imi/fi/` agora** — só criar quando Onda 2 for ativada (ADR-008 futuro). Esta spec fica em `docs/specs/` até lá.
- **Não acoplar a um modelo específico além de phi4-mini** — `fi_target_model` permite trocar para outro local (qwen3, llama3.2 VPS) sem mudar o `.fi.md`, desde que o modelo suporte o contrato de entrada/saída.

## Links

- ADR governance: `~/experimentos/.aiox/stories/ADR-004.adr.md`
- Plano congelado: `~/.claude/plans/gentle-riding-dusk.md` (Fase 4)
- Skill /fig-construir (template dos 8 blocos): `~/.aiox/skills/fig/fig-construir.md`
- Skill /fig-governanca (ficha técnica): `~/.aiox/skills/fig/fig-governanca.md`
- Skill /fig-validar (gate pré-produção): `~/.aiox/skills/fig/fig-validar.md`
- Paper FI (paper7): `~/experimentos/research/papers/paper7-framework-injection/paper.tex`
- IMI marco v92: `~/experimentos/KNOWLEDGE.md`
