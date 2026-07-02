# Snapshot Eternal — IMI v3 Pre-v4-Refactor

**Criado em**: 2026-05-24 04:30 UTC-3 (Brasília) / 07:30 UTC
**Autoridade**: ADR-004 (~/experimentos/.aiox/stories/ADR-004.adr.md) — Fase 0 Passo 1
**Playbook de restore**: PB-IMI-ZERO-LOSS-001 (~/.aiox/maintenance/playbooks/imi-zero-loss-failure.md)

## Identificação

- **Path**: `~/experimentos/tools/imi/imi_memory.db.eternal-snapshot-20260524`
- **SHA-256**: `7fcee80b0ba557ff4c4188af5879a893b910836d1a28fce871136f6f87d9b580`
- **SHA-256 file**: `~/experimentos/tools/imi/imi_memory.db.eternal-snapshot-20260524.sha256`
- **Inode**: 176415984 (hard-link com `imi_memory.db` original)
- **Tamanho**: 479.404.032 bytes (479 MB)
- **Mecanismo**: `ln` (hard-link, não cópia — zero-cost no APFS, robusto a edição do original)

## Estado capturado

| Métrica | Valor capturado | Valor do plano (snapshot teórico) | Drift |
|---|---|---|---|
| Linhas físicas (`SELECT COUNT(*) FROM memory_nodes`) | **22.433** | 22.132 | +301 (+1.4%) |
| Nós lógicos (`SELECT COUNT(DISTINCT node_id) FROM memory_nodes`) | **4.044** | 3.994 | +50 (+1.3%) |
| Nós ativos (`WHERE is_deleted=0`) | **17.672** | n/a | n/a |
| Tamanho DB | 479 MB | 459 MB | +20 MB (+4.4%) |
| WAL pós-checkpoint | 0 pages dirty | 4.2 MB | -4.2 MB (truncated) |

**Causa do drift**: ~50 nós lógicos foram gravados na sessão entre o congelamento do plano (24-mai 00:09 UTC-3) e a execução do Passo 1 (24-mai 04:30 UTC-3) — incluindo gravações da própria sessão de planejamento (intent IMI, ClawVault federations, edges de causalidade). **Comportamento esperado**, não indica bug.

## Validação executada

1. **Hard-link** confirmado via `ls -lai`: inode 176415984 compartilhado entre `imi_memory.db` e `imi_memory.db.eternal-snapshot-20260524`, link count = 2
2. **SHA-256** persistido em arquivo separado para detecção de corrupção
3. **Restore-em-clone test**: `cp -al imi_memory.db.eternal-snapshot-20260524 /tmp/imi-restore-test.db` produziu clone byte-equal:
   - Clone count: 22.433 = Snap count: 22.433 ✓
   - Clone SHA-256: `7fcee80b...d9b580` = Snap SHA-256: `7fcee80b...d9b580` ✓
4. **Cleanup**: clone de teste removido

## Validações pendentes (Passos 2-7 da Fase 0)

- [ ] Passo 2 — Audit retroativo baselines → `bench/baseline-v3.json` referenciando este snapshot
- [ ] Passo 3 — Bugs `from_sqlite p99=35s` e `session_1` (raiz comum com discrepância im_sts 3.215 vs DB 4.044)
- [ ] Passo 4 — `tests/invariants/test_zero_loss.py` deve rodar contra ESTE snapshot
- [ ] Passo 5 — `/metrics` Prometheus deve expor `imi_db_size_bytes`, `imi_nodes_logical_total`, etc
- [ ] Passo 6 — RFD-IMI-v4-000 referencia este snapshot como baseline
- [ ] Passo 7 — Gate 0→1 exige este snapshot intacto

## Como restaurar este snapshot (procedimento de emergência)

Conforme PB-IMI-ZERO-LOSS-001 Passo 2:

```bash
cd ~/experimentos/tools/imi

# Verificar SHA-256 ANTES de restaurar (snapshot pode ter corrompido)
EXPECTED_SHA="7fcee80b0ba557ff4c4188af5879a893b910836d1a28fce871136f6f87d9b580"
ACTUAL_SHA=$(shasum -a 256 imi_memory.db.eternal-snapshot-20260524 | awk '{print $1}')
[ "$EXPECTED_SHA" = "$ACTUAL_SHA" ] || { echo "FATAL: snapshot corrompido"; exit 1; }

# Mover DB atual para forensics
mv imi_memory.db imi_memory.db.incident-$(date +%Y%m%d-%H%M%S).forensic

# Restaurar via cp -al
cp -al imi_memory.db.eternal-snapshot-20260524 imi_memory.db

# Verificar restore
ACTUAL_COUNT=$(sqlite3 imi_memory.db "SELECT COUNT(*) FROM memory_nodes;")
[ "$ACTUAL_COUNT" = "22433" ] || { echo "FATAL: count mismatch pós-restore"; exit 1; }
echo "OK: restore verificado ($ACTUAL_COUNT linhas físicas)"
```

## Retention policy

- **Hard-link snapshot**: permanente (mecanismo de hard-link torna o snapshot independente de qualquer modificação do `imi_memory.db` original; ele só é apagado se explicitamente `rm`-ed)
- **SHA-256 file**: permanente (deve acompanhar o snapshot ao longo de sua vida)
- **Snapshots rolling adicionais** (mensais, trimestrais, anuais): tarefa futura — não escopo da Fase 0; pode ser introduzido na Fase 5 ou via tarefa standalone

## Não-objetivos

- **Este snapshot não tem cópia off-machine** — alvo de Onda 2 Fase 5 (Tier-Cold sync externo separado)
- **Este snapshot não tem encryption-at-rest do arquivo todo** — `node.original` individual pode ter `[ENC:v1]` por `crypto_layer.py`, mas o arquivo SQLite em si fica plaintext
- **Este snapshot não substitui backup operacional** — é safety net para Fase 0/1; backup contínuo é tarefa de outra camada

## Links

- ADR autoridade: `~/experimentos/.aiox/stories/ADR-004.adr.md`
- Playbook de restore: `~/.aiox/maintenance/playbooks/imi-zero-loss-failure.md`
- Política estados intermediários: `~/.aiox/maintenance/playbooks/imi-intermediate-states.md`
- Plano: `~/.claude/plans/gentle-riding-dusk.md`
- Próximo passo (Passo 2): `bench/baseline-v3.json` (a criar)
