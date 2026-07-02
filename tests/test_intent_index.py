"""Tests for ActiveIntentIndex — link nó↔intenção-ativa em O(1).

Spec: ~/.aiox/stories/ST-IMI-INTENT-INDEX.story.md
Estes testes SÃO o Verify executável dos AC da story (não rodam uma vez — são a
rede de não-regressão em CI). Cobrem:
  - AC-1: lookup O(1) (aqui: correção estrutural do índice; a comparação de latência
          vs. full-scan vive no benchmark, não no unit).
  - AC-2: consistência após create / fulfill.
  - AC-3: memória nova vinculada aparece no índice sem full-scan.
  - AC-4: correção do bug — is_real_link é rígido; vínculo espúrio não se forma.
  - AC-5: falha graciosa — índice vazio nunca lança.
"""

import pytest

from imi.intent_index import (
    LINK_TAG_OVERLAP_MIN,
    ActiveIntentIndex,
    is_real_link,
)


class TestActiveIntentIndexConsistency:
    """AC-2 / AC-3 — o índice reflete create / link / fulfill corretamente."""

    def test_rebuild_populates_active_intents(self):
        idx = ActiveIntentIndex()
        idx.rebuild([{"id": "intent_a"}, {"id": "intent_b"}, {"id": ""}])
        assert idx.active_intent_count == 2  # id vazio ignorado
        assert idx.nodes_for("intent_a") == set()  # ativa, mas sem nós ainda

    def test_link_creates_bidirectional_edge(self):
        """AC-3: memória vinculada aparece dos dois lados, O(1)."""
        idx = ActiveIntentIndex()
        idx.on_intent_created("intent_a")
        idx.link("intent_a", "node_1")
        assert idx.nodes_for("intent_a") == {"node_1"}
        assert idx.is_node_active("node_1") is True
        assert idx.active_node_ids() == {"node_1"}

    def test_link_is_idempotent(self):
        idx = ActiveIntentIndex()
        idx.on_intent_created("intent_a")
        idx.link("intent_a", "node_1")
        idx.link("intent_a", "node_1")
        assert idx.nodes_for("intent_a") == {"node_1"}

    def test_link_to_inactive_intent_is_ignored(self):
        """Ligar nó a intenção não-ativa (ex.: já cumprida) é no-op — não ressuscita."""
        idx = ActiveIntentIndex()  # intent_a nunca foi criada/ativada
        idx.link("intent_a", "node_1")
        assert idx.nodes_for("intent_a") == set()
        assert idx.is_node_active("node_1") is False

    def test_fulfill_removes_intent_and_unlinks_nodes(self):
        """AC-2: intenção cumprida sai do índice; nós exclusivos dela são desligados."""
        idx = ActiveIntentIndex()
        idx.on_intent_created("intent_a")
        idx.link("intent_a", "node_1")
        idx.on_intent_fulfilled("intent_a")
        assert idx.nodes_for("intent_a") == set()
        assert idx.is_node_active("node_1") is False
        assert idx.active_intent_count == 0

    def test_node_shared_between_two_intents_survives_partial_fulfill(self):
        """Nó ligado a 2 intenções continua ativo se só uma é cumprida."""
        idx = ActiveIntentIndex()
        idx.on_intent_created("intent_a")
        idx.on_intent_created("intent_b")
        idx.link("intent_a", "node_1")
        idx.link("intent_b", "node_1")
        idx.on_intent_fulfilled("intent_a")
        assert idx.is_node_active("node_1") is True  # ainda ligado a intent_b
        assert idx.nodes_for("intent_b") == {"node_1"}


class TestGracefulDegradation:
    """AC-5 — índice vazio/desconhecido nunca lança."""

    def test_empty_index_queries_return_falsy_not_raise(self):
        idx = ActiveIntentIndex()
        assert idx.nodes_for("unknown") == set()
        assert idx.is_node_active("unknown") is False
        assert idx.active_node_ids() == set()
        assert idx.active_intent_count == 0

    def test_fulfill_unknown_intent_is_noop(self):
        idx = ActiveIntentIndex()
        idx.on_intent_fulfilled("never_existed")  # não deve lançar
        assert idx.active_intent_count == 0


class TestIsRealLinkBugRegression:
    """AC-4 — correção do bug do matcher (resolution_evidence colada em díspares).

    O bug: umbral overlap_score>=1 deixava evidência genérica ("Renato quer 3
    consultas grátis JUDIT") colar em intenções sobre OpenCut, backup, FI-Engine...
    is_real_link é rígido: precisa de tag OU palavra específica em quantidade.
    """

    def test_disparate_intent_and_evidence_do_not_link(self):
        """O cenário EXATO do bug: evidência genérica vs. intenção díspar → sem vínculo."""
        # Intenção sobre backup de 12GB
        intent_tags = {"mac-manutencao"}
        intent_words = {"backup", "google", "drive", "externo", "descomprimida"}
        # "Evidência" genérica sobre JUDIT/Hosp Semiu — nada a ver
        node_tags = {"hospital-semiu", "judit"}
        node_words = {"consultas", "grátis", "demo", "free", "tier", "conta"}
        assert is_real_link(intent_tags, intent_words, node_tags, node_words) is False

    def test_genuinely_related_link_forms(self):
        """Vínculo real (tags fortes) DEVE formar."""
        intent_tags = {"imi", "budget-cap"}
        intent_words = {"retrieval", "tokens", "teto"}
        node_tags = {"imi", "budget-cap"}  # 2 tags batem >= LINK_TAG_OVERLAP_MIN
        node_words = {"implementei", "cap"}
        assert is_real_link(intent_tags, intent_words, node_tags, node_words) is True

    def test_word_overlap_threshold_specific_terms(self):
        """3+ palavras específicas em comum também formam vínculo real."""
        intent_tags = set()
        intent_words = {"active", "intent", "index", "materializar"}
        node_tags = set()
        node_words = {"active", "intent", "index", "pronto"}  # 3 batem
        assert is_real_link(intent_tags, intent_words, node_tags, node_words) is True

    def test_single_common_tag_is_not_enough(self):
        """1 tag em comum (o umbral frouxo antigo) NÃO basta — era a causa do bug."""
        assert LINK_TAG_OVERLAP_MIN >= 2
        assert is_real_link({"aip"}, set(), {"aip"}, set()) is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
