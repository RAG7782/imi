"""Tests for the F1 active-intent rail in im_nav (ST-IMI-BUDGET-01 AC-2).

Verify executável do furo Letal F1 do red-team: um nó vinculado a intenção pendente
ativa NUNCA pode ser silenciosamente cortado do retrieval (senão a Camada A nunca
dispara). Testa `_inject_active_intent_rail` isolando-o do embedder/space pesado.
"""

import pytest

import imi.mcp_server as srv
from imi.intent_index import ActiveIntentIndex


@pytest.fixture
def fresh_index(monkeypatch):
    """Substitui o índice global do módulo por um limpo, restaura ao fim."""
    idx = ActiveIntentIndex()
    monkeypatch.setattr(srv, "_intent_index", idx)
    return idx


def _mem(node_id, score):
    return {"id": node_id, "score": score, "content": f"c-{node_id}", "tags": []}


class TestIntentRailF1:
    """AC-2 — nó de intenção ativa sobrevive ao corte por top_k."""

    def test_active_intent_node_reinjected_when_cut(self, fresh_index):
        """O cenário do furo F1: intenção baixa-salience cortada → re-injetada."""
        fresh_index.on_intent_created("intent_x")
        fresh_index.link("intent_x", "node_intent")  # nó ligado à intenção ativa

        # raw tem 7; top_k=6 corta node_intent (o de menor score, no fim)
        raw = [_mem(f"n{i}", 0.9 - i * 0.05) for i in range(6)] + [_mem("node_intent", 0.4)]
        final = raw[:6]  # node_intent ficou de fora

        result = srv._inject_active_intent_rail(None, raw, final)
        ids = {m["id"] for m in result}
        assert "node_intent" in ids, "nó de intenção ativa foi silenciosamente cortado (F1 falhou)"
        assert len(result) == 7  # os 6 originais + o trilho

    def test_no_duplicate_when_already_present(self, fresh_index):
        """Se o nó de intenção já está no final, não duplica."""
        fresh_index.on_intent_created("intent_x")
        fresh_index.link("intent_x", "n0")
        raw = [_mem(f"n{i}", 0.9 - i * 0.05) for i in range(6)]
        final = raw[:6]  # n0 já está dentro
        result = srv._inject_active_intent_rail(None, raw, final)
        assert [m["id"] for m in result].count("n0") == 1
        assert len(result) == 6

    def test_empty_index_is_noop(self, fresh_index):
        """AC-5/SC-5: sem intenções ativas → final intacto, sem custo."""
        raw = [_mem(f"n{i}", 0.5) for i in range(6)]
        final = raw[:3]
        result = srv._inject_active_intent_rail(None, raw, final)
        assert result is final  # mesmo objeto — no-op puro

    def test_non_intent_nodes_not_injected(self, fresh_index):
        """Nós fora do top_k que NÃO são de intenção ativa permanecem cortados."""
        fresh_index.on_intent_created("intent_x")  # ativa, mas sem nós ligados
        raw = [_mem(f"n{i}", 0.5) for i in range(6)]
        final = raw[:3]
        result = srv._inject_active_intent_rail(None, raw, final)
        assert len(result) == 3  # nada re-injetado — n3/n4/n5 não são de intenção


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
