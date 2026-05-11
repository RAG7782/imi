"""Tests for L0-L3 tiering."""
import numpy as np

from imi.affect import AffectiveTag
from imi.affordance import Affordance
from imi.node import MemoryNode
from imi.tiering import (
    L0Identity,
    apply_tiering,
    compute_tier,
    generate_l1,
    get_tier_stats,
)


def _make_node(
    seed="test memory",
    salience=0.5,
    access_count=0,
    tags=None,
    affordances=None,
    mass=1.0,
) -> MemoryNode:
    node = MemoryNode(
        seed=seed,
        summary_orbital=seed[:30],
        summary_medium=seed[:60],
        affect=AffectiveTag(salience=salience, valence=0.5, arousal=0.5),
        tags=tags or [],
        mass=mass,
        embedding=np.random.randn(384).astype(np.float32),
    )
    node.access_count = access_count
    if affordances:
        node.affordances = affordances
    return node


class TestL0Identity:
    def test_default_render(self):
        l0 = L0Identity()
        text = l0.render()
        assert "Agent:" in text
        assert l0.token_estimate() < 50

    def test_custom_fields(self):
        l0 = L0Identity(
            agent_name="Test",
            domain="tributario",
            user_context="advogado",
        )
        text = l0.render()
        assert "tributario" in text
        assert "advogado" in text

    def test_save_load(self, tmp_path):
        path = tmp_path / "identity.json"
        l0 = L0Identity(agent_name="Custom", domain="juridico")
        l0.save(path)
        loaded = L0Identity.load(path)
        assert loaded.agent_name == "Custom"
        assert loaded.domain == "juridico"


class TestL1Generation:
    def test_basic_generation(self):
        nodes = [
            _make_node("important decision about architecture", salience=0.9, access_count=5),
            _make_node("trivial note", salience=0.2, access_count=0),
            _make_node("medium relevance fact", salience=0.6, access_count=2),
        ]
        l1 = generate_l1(nodes)
        assert len(l1.facts) <= 7
        assert l1.facts[0]["salience"] >= l1.facts[-1]["salience"]  # Sorted by relevance

    def test_domain_filter(self):
        nodes = [
            _make_node("tributario analysis", salience=0.5, tags=["tributario"]),
            _make_node("criminal law note", salience=0.8, tags=["criminal"]),
        ]
        l1 = generate_l1(nodes, domain_filter="tributario")
        # Domain-matching node should be boosted
        assert any("tributario" in f["summary"] for f in l1.facts)

    def test_channel_weights_boost(self):
        nodes = [
            _make_node("channel A memory", salience=0.5, access_count=1, tags=["channel_a"]),
            _make_node("channel B memory", salience=0.5, access_count=1, tags=["channel_b"]),
        ]
        weights = {"channel_a": 0.9, "channel_b": 0.1}
        l1 = generate_l1(nodes, channel_weights=weights)
        # Channel A has higher weight -> should be boosted above B (same base)
        assert len(l1.facts) >= 2
        assert l1.facts[0]["tags"][0] == "channel_a"

    def test_affordances_extraction(self):
        aff = Affordance(
            action="deploy service",
            confidence=0.95,
            conditions="when tests pass",
            domain="ops",
        )
        nodes = [_make_node("deploy decision", salience=0.8, affordances=[aff])]
        l1 = generate_l1(nodes)
        assert len(l1.affordances) >= 1
        assert l1.affordances[0]["action"] == "deploy service"

    def test_token_limit(self):
        nodes = [_make_node(f"fact {i}", salience=0.8, access_count=5) for i in range(50)]
        l1 = generate_l1(nodes)
        assert l1.token_estimate() <= 200  # Should stay under budget


class TestTieringPolicy:
    def test_high_relevance_promotes_to_l1(self):
        node = _make_node("critical", salience=0.9, access_count=5, mass=2.0)
        tier = compute_tier(node)
        assert tier == 1

    def test_low_relevance_stays_l3(self):
        node = _make_node("trivial", salience=0.1, access_count=0, mass=0.1)
        tier = compute_tier(node)
        assert tier == 3

    def test_pattern_defaults_to_l1(self):
        node = _make_node("consolidated pattern", tags=["_pattern"], mass=1.5)
        node.access_count = 3
        tier = compute_tier(node)
        assert tier in (1, 2)  # Pattern should be L1 or L2

    def test_channel_weight_boost(self):
        node = _make_node("boosted", salience=0.6, access_count=2, tags=["hot_channel"])
        tier_without = compute_tier(node)
        tier_with = compute_tier(node, channel_weights={"hot_channel": 0.9})
        assert tier_with <= tier_without  # Should promote (lower tier = higher priority)

    def test_apply_tiering_returns_changes(self):
        nodes = [
            _make_node("high", salience=0.9, access_count=5, mass=2.0),
            _make_node("low", salience=0.1, access_count=0, mass=0.1),
        ]
        # Both default to tier 3
        changes = apply_tiering(nodes)
        assert len(changes) >= 1  # At least the high node should change


class TestTierStats:
    def test_stats_distribution(self):
        nodes = [
            _make_node("a"),
            _make_node("b"),
            _make_node("c"),
        ]
        nodes[0].tier = 1
        nodes[1].tier = 2
        nodes[2].tier = 3
        stats = get_tier_stats(nodes)
        assert stats["l1"] == 1
        assert stats["l2"] == 1
        assert stats["l3"] == 1
        assert stats["total"] == 3
