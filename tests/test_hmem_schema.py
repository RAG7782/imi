"""Tests for the H-MEM cross-layer positional index schema (node.py).

Spec: ~/experimentos/specs/2026-06-14-imi-hmem-positional-index.md (§3.1)
Source: H-MEM, arXiv:2507.22925.

These tests guard the schema-evolution invariants of the H-MEM fields
(layer / parent_id / child_ptrs) added to MemoryNode. They are the
non-regression net (ASSERT-4) for the expand-contract migration — they
must run in CI, not once (pre-mortem cause #10: "ASSERT never became a
continuous test").

Invariants under test:
  1. A legacy node (no H-MEM fields) round-trips unchanged and does NOT
     bloat the serialized dict (forward-compat / expand-contract, Fowler).
  2. A promoted node (layer<3, with child_ptrs) round-trips losslessly.
  3. `invalidated_by` (temporal supersedence) and `child_ptrs` (abstraction
     hierarchy) are distinct fields and never collide — they are ORTHOGONAL,
     mirroring the layer⊥tier rule.
  4. `layer` is orthogonal to `tier`: setting one never moves the other.
"""


from imi.node import MemoryNode


class TestHMemSchemaRoundTrip:
    """Schema-evolution invariants for layer / parent_id / child_ptrs."""

    def test_legacy_node_defaults(self):
        """Legacy node (no H-MEM fields) gets layer=3, empty child_ptrs."""
        legacy = MemoryNode(id="legacy01", seed="memória antiga")
        assert legacy.layer == 3  # Episode = default, not promoted
        assert legacy.parent_id is None
        assert legacy.child_ptrs == []

    def test_legacy_node_does_not_bloat_dict(self):
        """Un-promoted node must NOT serialize H-MEM keys (DB-size discipline)."""
        d = MemoryNode(id="legacy02", seed="x").to_dict()
        assert "layer" not in d, "layer=3 (default) must not serialize"
        assert "parent_id" not in d
        assert "child_ptrs" not in d

    def test_promoted_node_serializes(self):
        """A promoted index node serializes its H-MEM fields."""
        d = MemoryNode(id="dom01", layer=0, child_ptrs=["c1", "c2", "c3"]).to_dict()
        assert d["layer"] == 0
        assert d["child_ptrs"] == ["c1", "c2", "c3"]

    def test_promoted_node_round_trip(self):
        """from_dict(to_dict(node)) preserves H-MEM fields losslessly."""
        node = MemoryNode(id="dom01", layer=0, child_ptrs=["c1", "c2", "c3"])
        node.parent_id = "root0"
        back = MemoryNode.from_dict(node.to_dict())
        assert back.layer == 0
        assert back.parent_id == "root0"
        assert back.child_ptrs == ["c1", "c2", "c3"]

    def test_invalidated_by_and_child_ptrs_coexist(self):
        """Temporal supersedence (invalidated_by) ⊥ abstraction (child_ptrs)."""
        node = MemoryNode(id="x", invalidated_by="y", child_ptrs=["z"])
        d = node.to_dict()
        assert d["invalidated_by"] == "y"
        assert d["child_ptrs"] == ["z"]
        back = MemoryNode.from_dict(d)
        assert back.invalidated_by == "y"
        assert back.child_ptrs == ["z"]

    def test_layer_orthogonal_to_tier(self):
        """Setting layer must not move tier, and vice-versa (the Letal rule)."""
        # A blazing-hot episode: tier=0 (always shown) AND layer=3 (concrete content)
        node = MemoryNode(id="hot1", tier=0, layer=3)
        back = MemoryNode.from_dict(node.to_dict())
        assert back.tier == 0 and back.layer == 3
        # An abstract domain index that is cold: tier=3 AND layer=0
        node2 = MemoryNode(id="dom2", tier=3, layer=0)
        back2 = MemoryNode.from_dict(node2.to_dict())
        assert back2.tier == 3 and back2.layer == 0


class TestHMemForwardCompat:
    """from_dict must tolerate dicts from BEFORE the H-MEM fields existed."""

    def test_from_dict_without_hmem_keys(self):
        """A pre-H-MEM serialized node (no layer key) loads with defaults."""
        pre_hmem = {"id": "old99", "seed": "pré-hmem", "tier": 2}
        node = MemoryNode.from_dict(pre_hmem)
        assert node.layer == 3  # default applied, no crash
        assert node.child_ptrs == []
        assert node.tier == 2  # existing field preserved

    def test_from_dict_ignores_unknown_future_keys(self):
        """M13 filter: unknown keys (future schema) are dropped, not fatal."""
        d = {"id": "fut1", "seed": "y", "layer": 1, "some_future_field": 42}
        node = MemoryNode.from_dict(d)  # must not raise
        assert node.layer == 1
        assert not hasattr(node, "some_future_field")
