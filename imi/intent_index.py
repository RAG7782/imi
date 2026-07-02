"""IMI — Índice materializado de intenções ativas (link nó↔intenção barato).

Spec: ~/.aiox/stories/ST-IMI-INTENT-INDEX.story.md
Origem: spike T0 da ST-IMI-BUDGET-01 (2026-07-02) — o link nó↔intenção-PENDENTE
não existia barato: o único elo pré-conclusão era `_find_related_intentions`, um
full-scan O(N) + json.loads por nó em `mcp_server.py`, que estoura o budget ~1.5ms
do `im_nav` lexical (base já com ~15k nós).

PORQUÊ este módulo existe:
  1. Habilita ST-IMI-BUDGET-01 SC-3/F1: o trilho reservado de intenção precisa saber,
     em O(1), quais node_ids pertencem a uma intenção ativa — imune ao budget-cap.
  2. Corrige o bug do matcher (SC-4): a `resolution_evidence` era colada em intenções
     díspares porque o umbral de overlap era frouxo (score>=1) e não validava vínculo
     real. Aqui o vínculo é explícito e o overlap exige umbral rígido.

Design: dois dicts em memória, mantidos incrementalmente (não recomputados no caminho
quente). Fonte de verdade continua sendo os nós de intenção no store episódico; este
índice é uma projeção derivada, reconstruível a qualquer momento via `rebuild()`.
"""

from __future__ import annotations

# Umbral de vínculo REAL nó↔intenção (SC-4). Rígido de propósito: o bug antigo usava
# overlap_score>=1, o que deixava evidência genérica ("quer 3 consultas grátis") colar
# em qualquer intenção. Aqui exige co-ocorrência forte de tags OU keywords específicas.
LINK_TAG_OVERLAP_MIN: int = 2
LINK_WORD_OVERLAP_MIN: int = 3


class ActiveIntentIndex:
    """Projeção O(1) do link intenção-ativa ↔ nós de memória.

    Uso:
        idx = ActiveIntentIndex()
        idx.rebuild(pending_intents)          # boot — reaproveita fetch_intentions
        idx.link(intent_id, node_id)          # vínculo real detectado no encode
        idx.on_intent_created(intent_id)      # im_int
        idx.on_intent_fulfilled(intent_id)    # im_int_fulfill / im_enc(resolves_intent)
        node_ids = idx.nodes_for(intent_id)   # O(1)
        is_active = idx.is_node_active(node_id)  # O(1) — usado pelo trilho F1

    Falha graciosa (SC-5): consultas em índice vazio retornam conjunto vazio / False,
    nunca lançam. O caller (retrieval) pode então degradar para o scan antigo.
    """

    def __init__(self) -> None:
        # intent_id -> set(node_id) : nós ligados a cada intenção ativa
        self._intent_to_nodes: dict[str, set[str]] = {}
        # node_id -> set(intent_id) : inverso, para is_node_active em O(1)
        self._node_to_intents: dict[str, set[str]] = {}
        # conjunto de intenções consideradas ativas (pending)
        self._active_intents: set[str] = set()

    # ---- construção / manutenção -------------------------------------------------

    def rebuild(self, pending_intents: list[dict]) -> None:
        """Reconstrói o índice a partir da lista de intenções pendentes do boot.

        `pending_intents` = saída de fetch_intentions(status="pending") — cada item
        tem ao menos {id}. Vínculos nó↔intenção pré-existentes (campo `fulfilled_by`
        NÃO conta — esse é de intenção já cumprida) não são reconstruídos aqui; os
        vínculos ativos nascem via link()/on_intent_created durante a sessão.
        """
        self._intent_to_nodes.clear()
        self._node_to_intents.clear()
        self._active_intents = {
            it["id"] for it in pending_intents if it.get("id")
        }
        for intent_id in self._active_intents:
            self._intent_to_nodes.setdefault(intent_id, set())

    def on_intent_created(self, intent_id: str) -> None:
        """im_int: nova intenção pendente entra no índice ativo."""
        if not intent_id:
            return
        self._active_intents.add(intent_id)
        self._intent_to_nodes.setdefault(intent_id, set())

    def on_intent_fulfilled(self, intent_id: str) -> None:
        """im_int_fulfill / auto-fulfill: intenção deixa de ser ativa.

        Remove a intenção e desliga seus nós do inverso. O(k) no nº de nós ligados.
        """
        if not intent_id:
            return
        self._active_intents.discard(intent_id)
        node_ids = self._intent_to_nodes.pop(intent_id, set())
        for node_id in node_ids:
            intents = self._node_to_intents.get(node_id)
            if intents is not None:
                intents.discard(intent_id)
                if not intents:
                    self._node_to_intents.pop(node_id, None)

    def link(self, intent_id: str, node_id: str) -> None:
        """Registra um vínculo REAL nó→intenção-ativa (detectado no encode).

        Idempotente. Ignora silenciosamente se a intenção não está ativa (não faz
        sentido ligar nó a intenção já cumprida).
        """
        if not intent_id or not node_id or intent_id not in self._active_intents:
            return
        self._intent_to_nodes.setdefault(intent_id, set()).add(node_id)
        self._node_to_intents.setdefault(node_id, set()).add(intent_id)

    # ---- consultas O(1) ----------------------------------------------------------

    def nodes_for(self, intent_id: str) -> set[str]:
        """node_ids ligados a uma intenção ativa. O(1). Vazio se desconhecida."""
        return set(self._intent_to_nodes.get(intent_id, set()))

    def is_node_active(self, node_id: str) -> bool:
        """True se o nó está ligado a >=1 intenção ativa. O(1). Usado pelo trilho F1."""
        return bool(self._node_to_intents.get(node_id))

    def active_node_ids(self) -> set[str]:
        """Todos os node_ids vinculados a alguma intenção ativa. O(nº de nós ligados)."""
        return set(self._node_to_intents.keys())

    @property
    def active_intent_count(self) -> int:
        return len(self._active_intents)


def is_real_link(intent_tags: set[str], intent_words: set[str],
                 node_tags: set[str], node_words: set[str]) -> bool:
    """Vínculo REAL nó↔intenção? (SC-4 — corrige o bug do matcher).

    Rígido de propósito: exige co-ocorrência forte, não o overlap_score>=1 frouxo
    que fazia evidência genérica colar em intenções díspares. Um vínculo é real se
    as TAGS batem o suficiente OU as PALAVRAS específicas batem o suficiente.

    >>> is_real_link({"imi","budget"}, set(), {"imi","budget"}, set())
    True
    >>> is_real_link({"x"}, {"a"}, {"y"}, {"b"})
    False
    """
    tag_overlap = len(intent_tags & node_tags)
    word_overlap = len(intent_words & node_words)
    return tag_overlap >= LINK_TAG_OVERLAP_MIN or word_overlap >= LINK_WORD_OVERLAP_MIN
