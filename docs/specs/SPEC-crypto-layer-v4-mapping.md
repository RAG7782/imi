---
spec_id: SPEC-IMI-CRYPTO-V4-001
status: accepted
date: 2026-05-24
deciders:
  - renato
  - claude-opus-4-7
confidence: 0.93
addresses_audit_gap: T2.3 (Mapear papel de crypto_layer.py no v4)
adr_ref: ~/experimentos/.aiox/stories/ADR-004.adr.md
plan_ref: ~/.claude/plans/gentle-riding-dusk.md (Onda 2 / Fase 2)
target_artifact: ~/experimentos/tools/imi/imi/integrations/crypto_layer.py
classification: no-change-spec  # vira ADR-005 SE comportamento mudar; por ora é spec de compatibilidade
---

# SPEC-IMI-CRYPTO-V4-001 — Papel de `crypto_layer.py` no IMI v4

## Contexto

O ataque T2.3 da auditoria final `/forja` Rota C exigiu mapear formalmente como `~/experimentos/tools/imi/imi/integrations/crypto_layer.py` se comporta no v4 — Tier-Hot continua igual; Tier-Warm e Tier-Cold (Onda 2) precisam **persistir `node.original` encriptado conforme convenção atual**. Sem este mapeamento, há risco de:

- (a) Tier-Warm gravar plaintext acidentalmente por desconhecimento do prefixo `[ENC:v1]`
- (b) Tier-Cold (audit trail eterno) vazar PII reverso por usar `node.original` sem passar pelo `decrypt_experience` reverso
- (c) Reconstruct Worker (Fase 4) ler `node.original` ainda criptografado e tratar como plaintext, gerando reconstrução de gibberish

## Comportamento atual do `crypto_layer.py` (v3, sessão 2026-05-24)

### Pipeline ativo (`IMI_CRYPTO=1`)
1. `secure_encode(space, experience, ...)` é o ponto de entrada
2. **Sanitização PII** (sanitizer_wrapper) — remove dados sensíveis ANTES do encode → `processed`
3. **Encode no IMI** com `processed` (plaintext sanitizado): summarizers, embedder, seed recebem texto legível
4. **Criptografia AES-256-GCM** — protege APENAS o campo `node.original`, prefixado com `[ENC:v1]`
5. **Audit log** append-only em `~/.imi/crypto_audit.jsonl` (apenas metadados — `ts, node_id[:12], pii_count, risk_score, key_fp, sanitizer`)

### Pipeline passivo (`IMI_CRYPTO=0`, default)
- `secure_encode` é pass-through transparente para `space.encode()` — zero overhead
- `node.original` permanece em plaintext

### Garantias formais
- **G-CRYPTO-1**: nunca persiste chave em disco. Chave vem de `IMI_CRYPTO_KEY` (hex 64) ou `IMI_CRYPTO_SECRET` (PBKDF2). Sem chave configurada com `IMI_CRYPTO=1`: **erro explícito** (não cria ephemeral — fix M3 evita perda de dados em restart).
- **G-CRYPTO-2**: nunca loga conteúdo no audit. Apenas metadados.
- **G-CRYPTO-3**: campo `summary_*` e `embedding` sempre em plaintext (decisão do M3 fix — antes ciphertext ia para summarizers e retornava "conteúdo criptografado").
- **G-CRYPTO-4**: `decrypt_experience()` é idempotente em legacy (texto sem prefixo `[ENC:v1]` retorna sem modificação).

### Campos do nó: ciphertext vs plaintext

| Campo | Estado físico (com `IMI_CRYPTO=1`) | Lido por |
|---|---|---|
| `node.original` | **Ciphertext** `[ENC:v1]<hex>` | `dialect.py:298`, `mcp_server.py:121,135,674` |
| `node.seed` | Plaintext (sanitizado) | summarizers, dialect, mcp_server |
| `node.summary_orbital` | Plaintext | boot bundle, navigate |
| `node.summary_medium` | Plaintext | navigate, search |
| `node.summary_detailed` | Plaintext | drm, navigate |
| `node.embedding` | Plaintext (float vector) | vector search |
| `node.tags` | Plaintext (sanitizadas) | filter, navigate |

**Observação crítica**: somente `node.original` é encriptado. Todo o resto do nó está em plaintext sanitizado (sem PII). Isso significa que **um nó sanitizado mas com `node.original` em ciphertext é seguro mesmo se o ciphertext for vazado**, desde que a sanitização do `seed` tenha sido efetiva.

## Comportamento no v4 — mapeamento por Tier

### Tier-Hot (SQLite + WAL — Onda 1, Fases 0+1+1.5)

**Mudança**: **NENHUMA**. Hot continua igual ao v3.

- `secure_encode` segue como ponto de entrada
- `node.original` segue prefixado `[ENC:v1]` quando `IMI_CRYPTO=1`
- Audit log em `~/.imi/crypto_audit.jsonl` segue inalterado
- Decrypt em `im_nav`/`im_drm` segue via `decrypt_experience()`

**Spec Constraint SC-CRYPTO-1 (Onda 1)**:
- `dirty_tracking` (Fase 1) **não pode olhar dentro de `node.original`** para decidir dirty — usar timestamp ou hash do `summary_orbital` (plaintext). Razão: olhar dentro de ciphertext exigiria decrypt sincrono no save path, regredindo latência.
- `compact_versions(keep_versions=∞, dirty_only=True)` (Fase 1) **deve preservar byte-equal o campo `original`** entre versões — invariante zero-loss garante que `[ENC:v1]<hex>` não muda entre versões físicas duplicadas.
- `FTS dedup` (Fase 1) **não pode indexar `node.original` em ciphertext** — FTS usa `node.summary_*` (já o caso em v3); manter convenção.
- `_fts_index_node` em `storage.py:452-468` deve continuar usando `summary_medium` e `summary_detailed` (plaintext), **nunca `original`**.

### Tier-Warm (Parquet + Zstd content-addressable — Onda 2, Fase 2)

**Mudança**: **NENHUMA NO COMPORTAMENTO**, mas **adição de hard rule**.

Tier-Warm é arquive write-once: nós inativos migrados de Hot para Warm. O dump inicial e migrations subsequentes **devem preservar `node.original` em ciphertext** se estava encriptado em Hot.

**Spec Constraint SC-CRYPTO-2 (Onda 2 / Fase 2)**:
- `warm_store.py` (a criar) deve serializar `node.original` em Parquet **sem decrypt prévio** — bytes do ciphertext vão direto para Parquet
- Delta-chain (bsdiff4) opera sobre **ciphertext** se Hot estava encriptado — diffs entre 2 ciphertexts AES-GCM produzem ratio próximo a 1.0 (zstd não comprime ciphertext de forma significativa). **Implicação**: nós com `IMI_CRYPTO=1` não se beneficiam de delta-chain; aceitar custo de storage.
- **Decisão arquitetural**: NÃO decriptar para fazer delta-chain. Razão: (1) violaria zero-knowledge em repouso; (2) plaintext em Parquet seria nova superfície de leak; (3) ganho marginal de storage não justifica.
- Coluna Parquet `original_encrypted: bool` deve registrar se o blob estava prefixado `[ENC:v1]` (útil para audit/diagnóstico).

**Trade-off aceito**: usuários com `IMI_CRYPTO=1` têm Warm tier maior que usuários com `IMI_CRYPTO=0`. Documentar em RFD-IMI-v4-001.

### Tier-Cold (JSONL.zst append-only — Onda 2, Fase 2)

**Mudança**: **NENHUMA NO COMPORTAMENTO**.

Tier-Cold é o audit trail eterno — toda mutação observável vai para `~/.imi/cold/events-YYYY-MM.jsonl.zst`. Para nós que foram criados via `secure_encode`, o evento Cold contém **`node.original` em ciphertext** (mesmo formato que Hot).

**Spec Constraint SC-CRYPTO-3 (Onda 2 / Fase 2)**:
- `cold_log.py` (a criar) **nunca** pode chamar `decrypt_experience()` antes de escrever — Cold é write-once e deve refletir o estado físico de Hot byte-equal.
- Merkle-root por 1000 eventos (paper IMI Theory §6.3) calculado sobre ciphertext quando applicable. Verificação de integridade não exige decrypt.
- Disaster recovery: restaurar a partir de Cold mantém ciphertext; sem chave AES, dados ficam permanentemente perdidos para conteúdo de `node.original`, mas `summary_*` (plaintext sanitizado) permanecem recuperáveis. **Documentar como trade-off de segurança em RFD-IMI-v4-000**.

### Reconstruct Worker (Onda 2, Fase 4)

**Mudança**: **CRÍTICA — adicionar hard rule de decrypt**.

O Reconstruct Worker lê seed + neighbors para reconstruir conteúdo via phi4-mini. Se o worker ler `node.original` em ciphertext, ele tentará "reconstruir gibberish" e retornará lixo com confidence alta — caso de falha silenciosa catastrófica.

**Spec Constraint SC-CRYPTO-4 (Onda 2 / Fase 4)**:
- `reconstruct_worker.py` (a criar) **deve consumir `node.seed` (plaintext sanitizado)**, NUNCA `node.original`
- Se algum acesso a `node.original` for necessário para validação cruzada, **deve passar por `decrypt_experience()` primeiro**
- Hard rule em código: import explicit + linter rule (ruff custom check) que falha se `reconstruct_worker.py` referencia `node.original` sem `decrypt_experience()` na mesma expressão
- Documentar em `reconstruct_protocol.fi.md` (formato SPEC-IMI-FI-001) Bloco 5 (Cercas e Exclusões): "NUNCA leia `node.original` cru — sempre passe por `decrypt_experience()`"

## Decisão sobre ADR-005

Conforme requisito T2.3 do ADR-004: **"mapear papel de `crypto_layer.py` no v4 durante Fase 0 (ADR-005 se comportamento mudar)"**.

**Análise**:
- Tier-Hot: **0 mudanças** no comportamento de `crypto_layer.py`
- Tier-Warm: **0 mudanças** no comportamento de `crypto_layer.py` (mudanças são em `warm_store.py` — novo arquivo)
- Tier-Cold: **0 mudanças** no comportamento de `crypto_layer.py` (mudanças são em `cold_log.py` — novo arquivo)
- Reconstruct Worker: **0 mudanças** no comportamento de `crypto_layer.py` (mudanças são em `reconstruct_worker.py` — novo arquivo + linter rule)

**Conclusão**: o comportamento de `crypto_layer.py` **NÃO MUDA** no v4. As 4 Spec Constraints (SC-CRYPTO-1 a SC-CRYPTO-4) são contratos sobre **consumidores** do crypto layer (Hot dirty-tracking, Warm store, Cold log, Reconstruct worker), não sobre o crypto layer em si.

**Decisão**: **NÃO criar ADR-005**. Esta spec (SPEC-IMI-CRYPTO-V4-001) é o registro "no-change ADR" exigido pelo T2.3 — documenta explicitamente que o comportamento se mantém, e estabelece as Spec Constraints que os novos módulos do v4 devem honrar para preservar essa estabilidade.

**Quando ADR-005 deve ser criada (gatilho condicional)**:
- Se durante Fase 2 a equipe decidir trocar AES-256-GCM por outro algoritmo (ex: ChaCha20-Poly1305) → ADR-005 obrigatória
- Se for adicionada rotação automática de chave (key rotation) → ADR-005 obrigatória
- Se o `[ENC:v1]` for incrementado para `[ENC:v2]` por mudança de formato de cipher → ADR-005 obrigatória
- Se decidir que Warm tier **decripta** antes de delta-chain (rejeitado nesta spec) → ADR-005 obrigatória

## Spec Constraints consolidadas (a propagar para RFD-IMI-v4-000)

| ID | Constraint | Aplica em |
|---|---|---|
| SC-CRYPTO-1 | Dirty tracking, compact_versions e FTS dedup operam sobre plaintext (`summary_*`), nunca `node.original` em ciphertext | Hot (Fase 1) |
| SC-CRYPTO-2 | `warm_store.py` serializa `node.original` em Parquet sem decrypt; coluna `original_encrypted: bool` registra estado | Warm (Fase 2) |
| SC-CRYPTO-3 | `cold_log.py` é write-once byte-equal a Hot; nunca chama `decrypt_experience` antes de escrever | Cold (Fase 2) |
| SC-CRYPTO-4 | `reconstruct_worker.py` consome `node.seed` (plaintext sanitizado); referência a `node.original` exige `decrypt_experience` no mesmo statement (linter rule custom) | Reconstruct (Fase 4) |

## Riscos identificados (a adicionar ao Risk Register se promover)

| Risco | Prob | Sev | Mitigação |
|---|---|---|---|
| Tier-Warm de usuário com `IMI_CRYPTO=1` cresce mais que esperado (delta-chain não comprime ciphertext) | Média | Baixa | Documentar trade-off em RFD-IMI-v4-001; oferecer flag opcional `warm_decrypt_for_delta=true` em release futuro (rejeitado nesta spec) |
| Disaster recovery sem chave AES perde `node.original` permanentemente | Baixa | Alta | Documentar em RFD-IMI-v4-000 como trade-off de segurança; `summary_*` (plaintext) sobrevive ao recovery |
| Reconstruct Worker lê `node.original` cru e reconstrói gibberish | **Baixa** (com linter) / Alta (sem linter) | Alta | Linter rule custom + teste unitário que falha se worker_module referencia `node.original` sem decrypt |
| Audit log `crypto_audit.jsonl` cresce sem rotação | Baixa | Baixa | Adicionar rotação anual em Fase 5 (Auto-compaction) |

## Validation

Como saberemos que o mapeamento foi correto:

- **Premissa 1 (Hot inalterado)**: testes de Fase 1 (dirty tracking + compact + FTS dedup) passam com `IMI_CRYPTO=1` E com `IMI_CRYPTO=0` sem mudança no código de `crypto_layer.py`. Como detectar quebra: regressão em `tests/property/test_roundtrip.py` quando rodado com `IMI_CRYPTO=1`.
- **Premissa 2 (Warm preserva ciphertext)**: dump inicial Warm com nós encriptados produz Parquet com bytes byte-equal aos bytes em Hot. Como detectar quebra: differential test `tests/differential/v3_vs_v4_warm.py` mostra mismatch em `node.original` para nós com `[ENC:v1]`.
- **Premissa 3 (Reconstruct não toca ciphertext)**: linter custom (ou unit test reflexivo) na Fase 4 detecta qualquer `node.original` em `reconstruct_worker.py` sem `decrypt_experience` adjacente.
- **Premissa 4 (No-change ADR estável)**: se nenhum dos 4 gatilhos para ADR-005 disparar nas Fases 2-4, esta spec permanece o registro autoritativo. Reavaliar pós-Onda 2.

## Links

- crypto_layer.py: `~/experimentos/tools/imi/imi/integrations/crypto_layer.py`
- ADR governance pai: `~/experimentos/.aiox/stories/ADR-004.adr.md`
- Plano: `~/.claude/plans/gentle-riding-dusk.md` (Fases 2, 4)
- Spec FI reconstruct (T2.2): `~/experimentos/tools/imi/docs/specs/SPEC-fi-format-reconstruct-protocol.md`
- crypto_wrapper / sanitizer_wrapper (Camada 0): `~/.aiox/integrations/`
- Audit JSONL: `~/.imi/crypto_audit.jsonl` (355 eventos como de 2026-05-24, mean risk 0.003, max 0.300)
- Paper IMI Theory §6.3 (Merkle proofs): `~/experimentos/research/papers/paper5-imi-theory/paper.md`
