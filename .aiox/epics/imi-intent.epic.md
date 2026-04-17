# Epic: IMI-INTENT — Memória de Intenções e Foresight Signals

> **ID:** IMI-E04
> **Status:** Ready
> **Prioridade:** P1
> **Depende de:** nenhuma (novo tipo de nó, não modifica lógica existente)
> **Fundamentação científica:** EverMemOS Foresight signals (arXiv:2601.02163) + Springdrift sensorium (arXiv:2604.04660)
> **Problema resolvido:** O IMI sabe o que aconteceu. Não sabe o que você pretendia fazer e não fez.

---

## Objetivo

Criar um novo tipo de nó `intention` no IMI que captura **intenções não cumpridas** com seu contexto de "por que importa" e uma janela temporal (deadline). Intenções pendentes aparecem **sempre** no boot, independente de salience ranking, e são marcadas como `fulfilled` quando concluídas.

Este é o problema mais custoso em sessões multi-dia: você retorna a um projeto e não sabe de onde parou ou por que havia parado naquele ponto específico.

## Definição do nó intention

```json
{
  "id": "intent_xxxx",
  "node_type": "intention",
  "content": "O que pretendia fazer — descrição acionável",
  "context": "Por que importa / o que estava tentando alcançar / blocker que impediu",
  "project": "OXE|KONA|SYMBIONT|RESEARCH|JURIDICO|AIOX",
  "deadline": "2026-05-01T00:00:00Z",
  "confidence": 0.9,
  "status": "pending|fulfilled|abandoned",
  "fulfilled_by": null,
  "blocked_by": null,
  "tags": ["oxe", "sprint-2", "premium-endpoint"],
  "created_at": 1234567890.0,
  "salience": 0.85,
  "source": "session-16abr-imi-melhorias"
}
```

## Stories

### S01: Novo tool im_int() no MCP server
Adicionar tool `im_int` ao `mcp_server.py`:

```python
@mcp.tool()
def im_int(
    content: str,
    context: str,
    project: str = "",
    deadline: str = "",
    confidence: float = 0.85,
    tags: str = "",
    source: str = "",
) -> str:
    """Store pending intention with deadline and context"""
```

- Cria nó com `node_type: "intention"` e `status: "pending"` no campo `data`
- Salience default: 0.85 (intenções são importantes por definição)
- Persiste em `episodic` store (não semântico — são episódios de intenção)
- Retorna `{id, content, deadline, project}` confirmando armazenamento
- **Verify:** `im_int(content="Implementar /premium endpoint", context="Necessário para monetização OXÉ Sprint 2", project="OXE", deadline="2026-05-15")` → retorna node_id, aparece em próximo boot

### S02: Novo tool im_int_fulfill() no MCP server
Tool para marcar intenção como cumprida:

```python
@mcp.tool()
def im_int_fulfill(
    intent_id: str,
    fulfilled_by: str = "",
    notes: str = "",
) -> str:
    """Mark intention as fulfilled, optionally linking to the completing memory node"""
```

- Atualiza `status` para `"fulfilled"`, seta `fulfilled_by` se fornecido
- Se `fulfilled_by` é node_id válido: cria `im_glnk(fulfilled_by, intent_id, "causal", "fulfills")`
- **Verify:** chamar `im_int_fulfill(intent_id, fulfilled_by="abc123")` → nó atualizado + edge criada no grafo

### S03: Novo tool im_int_list() no MCP server
Tool para listar intenções por status/projeto:

```python
@mcp.tool()
def im_int_list(
    project: str = "",
    status: str = "pending",
    top_k: int = 10,
) -> str:
    """List intentions filtered by project and status, ordered by deadline"""
```

- Busca nós com `node_type == "intention"` e filtros
- Ordena por deadline ASC (mais urgente primeiro), depois por salience DESC
- **Verify:** após criar 3 intenções, `im_int_list(status="pending")` retorna as 3 ordenadas por deadline

### S04: Intenções pendentes no boot (slot reservado)
Modificar `build_cache()` em `imi_boot_semantic.py`:

```python
# Após fetch_top_memories(), buscar intenções pendentes separadamente
intentions = fetch_intentions(conn, status="pending", top_k=3)
# Ordenar por deadline ASC
# Adicionar como seção dedicada no bloco boot:
```

```
## Intenções Pendentes (ordenadas por deadline)
  ⚡ [OXE | até 2026-05-15] Implementar /premium endpoint — necessário para monetização Sprint 2
  ⚡ [KONA | até 2026-04-30] Laudo pericial CPC 25 — eleva IC→0.97
  ⚡ [RESEARCH | sem deadline] Submeter paper L2/DS ao ICLR 2026
```

- Intenções aparecem SEMPRE, mesmo que salience < threshold
- Label `⚡` para urgentes (deadline < 7 dias), `📌` para sem deadline
- **Verify:** boot com intenção pendente sempre inclui seção "Intenções Pendentes" com pelo menos 1 item

### S05: Protocolo de intenções no GRAVAR
Adicionar "Canal 0 — Intenções" no `gravar.md`, executado ANTES do Canal 1:

**Canal 0a — Capturar intenções não cumpridas:**
```
1. Listar tasks que foram PLANEJADAS nesta sessão mas NÃO concluídas
2. Para cada uma: im_int(content, context="por que parou + o que ainda falta", project, deadline)
3. Não criar intenção para tasks "adiadas por escolha" — só para "bloqueadas ou incompletas"
```

**Canal 0b — Cumprir intenções anteriores:**
```
1. Verificar im_int_list(status="pending") no início do GRAVAR
2. Para cada intenção que foi cumprida nesta sessão: im_int_fulfill(intent_id, fulfilled_by=encode_node_id)
```

- **Verify:** após sessão onde 1 task bloqueou: boot seguinte mostra intenção pendente com contexto do blocker

### S06: Suggestion de fulfillment em im_enc
Quando `im_enc` é chamado, verificar se há intenções pendentes com tag overlap.

```python
# Após criar nó, buscar intentions pendentes
pending = search_intentions_by_tags(node.tags)
if pending:
    result["possible_fulfillments"] = [
        {"intent_id": i.id, "content": i.content[:80], "similarity": score}
        for i, score in pending[:2]
    ]
```

- Retornar no JSON de `im_enc` como `possible_fulfillments` (sugestão, não automático)
- O agente decide se a intenção foi cumprida e chama `im_int_fulfill`
- **Verify:** `im_enc` com tags `["oxe", "premium-endpoint"]` sugere intenção "Implementar /premium endpoint" como possível fulfillment

## Status de implementação (2026-04-16)

| Story | Status |
|---|---|
| S01 im_int() | ✅ Done — cria nó intention com seed JSON completo |
| S02 im_int_fulfill() | ✅ Done — atualiza status + edge CAUSAL "fulfills" |
| S03 im_int_list() | ✅ Done — filtra por status/projeto, ordena por deadline ASC |
| S04 Boot slot intenções | ✅ Done — seção "Intenções Pendentes" no imi_boot_semantic.py |
| S05 Canal 0 no gravar.md | Backlog — próxima sessão |
| S06 Suggestion em im_enc | Backlog — após S05 |

**Verify smoke test:** PASS — im_int → im_int_list(total=1) → im_int_fulfill → pending=0 ✅
**Boot verify:** seção "Intenções Pendentes" aparece com ⚡ para deadline ≤7 dias ✅

## Métricas de sucesso

- Intenções criadas por sessão: alvo 2-3 (tasks não cumpridas)
- Fulfillment rate: alvo > 60% das intenções marcadas como fulfilled em <= 7 dias
- Boot recall: "sabia de onde parou" sem precisar ler HANDOFF completo

## Referências

- arXiv:2601.02163 — EverMemOS: Foresight signals com janela temporal
- arXiv:2604.04660 — Springdrift: sensorium como self-state injection
