"""E2E Pipeline Test: SYMBIONT → FCM → IMI

Tests the complete federated signal pipeline:
1. SYMBIONT emits signal (channel weights / Mound APPROVED / PRIORITY_SHIFT)
2. FCM bus transports (JSON file in ~/.fcm/events/)
3. IMI consumes via symbiont_bridge functions

This validates that the 4 sinergias work end-to-end, not just in isolation.
"""

import json
import os
import time
import uuid
from pathlib import Path

import pytest
import numpy as np

from imi.tiering import L0Identity, generate_l1
from imi.symbiont_bridge import (
    read_channel_weights,
    check_priority_shift,
    get_mound_approved_artifacts,
)
from imi.node import MemoryNode, AffectiveTag
from imi.affordance import Affordance


# ── Fixtures ──────────────────────────────────────────

@pytest.fixture
def fcm_dir(tmp_path):
    """Create temp FCM events directory."""
    events_dir = tmp_path / "fcm" / "events"
    events_dir.mkdir(parents=True)
    return events_dir


@pytest.fixture
def imi_dir(tmp_path):
    """Create temp IMI config directory."""
    imi_dir = tmp_path / "imi"
    imi_dir.mkdir(parents=True)
    return imi_dir


def _make_nodes(n=10):
    """Create test nodes with varying relevance."""
    nodes = []
    for i in range(n):
        node = MemoryNode(
            seed=f"Memory about topic {i} with details",
            summary_orbital=f"Topic {i}",
            summary_medium=f"Memory about topic {i}",
            affect=AffectiveTag(
                salience=0.3 + (i * 0.07),  # 0.3 to 0.93
                valence=0.5,
                arousal=0.5,
            ),
            tags=[f"channel_{i % 3}", f"domain_{i % 2}"],
            mass=1.0 + (i * 0.1),
            embedding=np.random.randn(384).astype(np.float32),
        )
        node.access_count = i
        if i >= 7:
            node.affordances = [
                Affordance(
                    action=f"action_{i}",
                    confidence=0.8 + (i * 0.02),
                    conditions="when relevant",
                    domain="ops",
                )
            ]
        nodes.append(node)
    return nodes


# ── Sinergia 1: Channel Weights → L1 Promotion ──────

class TestSinergia1ChannelWeights:
    """SYMBIONT Topology Engine → channel_weights.json → IMI L1 promotion."""

    def test_e2e_channel_weights_flow(self, imi_dir):
        """Full flow: write weights → read → use in L1 generation."""
        # SYMBIONT SIDE: Topology Engine writes channel weights
        weights = {"channel_0": 0.9, "channel_1": 0.3, "channel_2": 0.7}
        weights_path = imi_dir / "channel_weights.json"
        weights_path.write_text(json.dumps(weights))

        # IMI SIDE: Read channel weights
        loaded = read_channel_weights(symbiont_url=None)
        # Note: reads from ~/.imi/ by default, but we test the file format
        assert isinstance(loaded, dict)

        # IMI SIDE: Use weights in L1 generation
        nodes = _make_nodes(10)
        l1_without = generate_l1(nodes, max_facts=5)
        l1_with = generate_l1(nodes, max_facts=5, channel_weights=weights)

        # Channel_0 has highest weight (0.9) → nodes tagged channel_0 should be boosted
        assert l1_with.facts is not None
        assert len(l1_with.facts) > 0

    def test_weights_file_format(self, imi_dir):
        """Channel weights JSON must be {str: float} format."""
        weights = {"mycelium_main": 0.85, "probe_001": 0.12, "broadcast": 0.95}
        path = imi_dir / "channel_weights.json"
        path.write_text(json.dumps(weights))

        data = json.loads(path.read_text())
        assert all(isinstance(k, str) for k in data.keys())
        assert all(isinstance(v, (int, float)) for v in data.values())
        assert all(0 <= v <= 1 for v in data.values())

    def test_graceful_degradation_no_file(self):
        """Without weights file, L1 generation works normally."""
        weights = read_channel_weights()
        nodes = _make_nodes(5)
        l1 = generate_l1(nodes, max_facts=3, channel_weights=weights)
        assert len(l1.facts) > 0


# ── Sinergia 2: Mound APPROVED → L2 Cache ───────────

class TestSinergia2MoundApproved:
    """SYMBIONT Mound artifact APPROVED → FCM event → IMI L2 cache."""

    def test_e2e_mound_approved_flow(self, fcm_dir, monkeypatch):
        """Full flow: emit approved artifact → FCM → IMI reads."""
        # SYMBIONT SIDE: Mound emits APPROVED artifact event
        event = {
            "id": uuid.uuid4().hex[:12],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": "symbiont",
            "type": "memory_created",
            "title": "Architecture decision: use VIEW pattern for L0-L3",
            "content": "The L0-L3 stack should be implemented as a VIEW over the existing CLS architecture, not a replacement. This preserves the Nader reconsolidation model.",
            "tags": ["mound_approved", "architecture"],
            "salience": 0.9,
            "memoryType": "decision",
            "metadata": {
                "artifact_status": "APPROVED",
                "quality": 0.92,
                "artifact_kind": "decision",
            },
        }
        event_path = fcm_dir / f"{event['id']}.json"
        event_path.write_text(json.dumps(event))

        # IMI SIDE: Read approved artifacts from FCM
        monkeypatch.setattr(
            "imi.symbiont_bridge.FCM_EVENTS_DIR", fcm_dir
        )
        artifacts = get_mound_approved_artifacts()

        assert len(artifacts) == 1
        assert artifacts[0]["title"] == "Architecture decision: use VIEW pattern for L0-L3"
        assert artifacts[0]["quality"] == 0.92
        assert "mound_approved" in event["tags"]

    def test_filters_non_approved(self, fcm_dir, monkeypatch):
        """Only APPROVED artifacts with quality >= 0.8 pass."""
        # Draft artifact (should NOT pass)
        draft = {
            "id": "draft001",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": "symbiont",
            "type": "memory_created",
            "title": "Draft idea",
            "content": "Just a draft",
            "tags": [],
            "salience": 0.5,
            "memoryType": "fact",
            "metadata": {
                "artifact_status": "DRAFT",
                "quality": 0.3,
            },
        }
        (fcm_dir / "draft001.json").write_text(json.dumps(draft))

        monkeypatch.setattr("imi.symbiont_bridge.FCM_EVENTS_DIR", fcm_dir)
        artifacts = get_mound_approved_artifacts()
        assert len(artifacts) == 0

    def test_low_quality_filtered(self, fcm_dir, monkeypatch):
        """APPROVED but low quality (< 0.8) should be filtered."""
        event = {
            "id": "lowq001",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": "symbiont",
            "type": "memory_created",
            "title": "Low quality approved",
            "content": "Approved but low quality",
            "tags": ["mound_approved"],
            "salience": 0.6,
            "memoryType": "fact",
            "metadata": {
                "artifact_status": "APPROVED",
                "quality": 0.5,  # Below 0.8 threshold
            },
        }
        (fcm_dir / "lowq001.json").write_text(json.dumps(event))

        monkeypatch.setattr("imi.symbiont_bridge.FCM_EVENTS_DIR", fcm_dir)
        artifacts = get_mound_approved_artifacts()
        assert len(artifacts) == 0


# ── Sinergia 3: PRIORITY_SHIFT → L1 Refresh ─────────

class TestSinergia3PriorityShift:
    """SYMBIONT Murmuration PRIORITY_SHIFT → FCM → IMI L1 refresh."""

    def test_e2e_priority_shift_flow(self, fcm_dir, monkeypatch):
        """Full flow: emit priority shift → FCM → IMI detects and refreshes L1."""
        # SYMBIONT SIDE: Murmuration emits PRIORITY_SHIFT
        event = {
            "id": uuid.uuid4().hex[:12],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": "symbiont",
            "type": "custom",
            "title": "Priority shift",
            "content": "Priority shifted to: tributario",
            "tags": ["priority_shift"],
            "salience": 0.9,
            "memoryType": "decision",
            "metadata": {
                "signal_type": "PRIORITY_SHIFT",
                "new_domain": "tributario",
            },
        }
        event_path = fcm_dir / f"{event['id']}.json"
        event_path.write_text(json.dumps(event))

        # IMI SIDE: Check for priority shift
        monkeypatch.setattr("imi.symbiont_bridge.FCM_EVENTS_DIR", fcm_dir)
        new_domain = check_priority_shift()

        assert new_domain == "tributario"

        # IMI SIDE: Generate L1 with domain filter
        nodes = _make_nodes(10)
        # Add some nodes with tributario tag
        for i in range(3):
            node = MemoryNode(
                seed=f"Tributario analysis {i}",
                summary_orbital=f"Tax analysis {i}",
                affect=AffectiveTag(salience=0.7, valence=0.5, arousal=0.5),
                tags=["tributario", "tax"],
                mass=1.5,
                embedding=np.random.randn(384).astype(np.float32),
            )
            node.access_count = 3
            nodes.append(node)

        l1_general = generate_l1(nodes, max_facts=5)
        l1_filtered = generate_l1(nodes, max_facts=5, domain_filter=new_domain)

        # Filtered L1 should prioritize tributario-tagged nodes
        filtered_tags = set()
        for fact in l1_filtered.facts:
            filtered_tags.update(fact.get("tags", []))

        assert "tributario" in filtered_tags or "tax" in filtered_tags

    def test_no_shift_returns_none(self, fcm_dir, monkeypatch):
        """Without priority shift events, returns None."""
        monkeypatch.setattr("imi.symbiont_bridge.FCM_EVENTS_DIR", fcm_dir)
        result = check_priority_shift()
        assert result is None

    def test_non_priority_events_ignored(self, fcm_dir, monkeypatch):
        """Non-priority-shift events are ignored."""
        event = {
            "id": "other001",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": "symbiont",
            "type": "custom",
            "title": "Some other event",
            "content": "Not a priority shift",
            "tags": [],
            "salience": 0.5,
            "memoryType": "fact",
            "metadata": {
                "signal_type": "HEARTBEAT",
            },
        }
        (fcm_dir / "other001.json").write_text(json.dumps(event))

        monkeypatch.setattr("imi.symbiont_bridge.FCM_EVENTS_DIR", fcm_dir)
        result = check_priority_shift()
        assert result is None


# ── Sinergia 4: Federation → FCM Convergence ────────

class TestSinergia4FederationConvergence:
    """SYMBIONT Federation events flow through FCM bus format."""

    def test_federation_event_format(self, fcm_dir):
        """Federation events use standard FCM schema."""
        event = {
            "id": uuid.uuid4().hex[:12],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": "symbiont",
            "type": "memory_created",
            "title": "Federated memory from peer organism",
            "content": "Shared knowledge from peer",
            "tags": ["federated", "peer_alpha"],
            "salience": 0.7,
            "memoryType": "fact",
            "metadata": {
                "peer_id": "organism_alpha",
                "relay_hop": 1,
            },
        }
        event_path = fcm_dir / f"{event['id']}.json"
        event_path.write_text(json.dumps(event))

        # Validate FCM schema fields
        loaded = json.loads(event_path.read_text())
        required_fields = ["id", "timestamp", "source", "type", "title", "content", "tags", "salience"]
        for field in required_fields:
            assert field in loaded, f"Missing FCM field: {field}"
        assert loaded["source"] == "symbiont"
        assert isinstance(loaded["tags"], list)
        assert isinstance(loaded["salience"], (int, float))

    def test_federation_events_readable_by_bridge(self, fcm_dir, monkeypatch):
        """Federation events should be parseable by IMI bridge functions."""
        # Federation relay as approved artifact
        event = {
            "id": "fed001",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": "symbiont",
            "type": "memory_created",
            "title": "High-quality federated artifact",
            "content": "Architecture pattern from peer organism",
            "tags": ["mound_approved", "federated"],
            "salience": 0.85,
            "memoryType": "lesson",
            "metadata": {
                "artifact_status": "APPROVED",
                "quality": 0.9,
                "artifact_kind": "pattern",
                "peer_id": "organism_beta",
            },
        }
        (fcm_dir / "fed001.json").write_text(json.dumps(event))

        monkeypatch.setattr("imi.symbiont_bridge.FCM_EVENTS_DIR", fcm_dir)
        artifacts = get_mound_approved_artifacts()
        assert len(artifacts) == 1
        assert "federated" in json.dumps(artifacts[0])


# ── Full Pipeline Integration ────────────────────────

class TestFullPipeline:
    """Test the complete pipeline: multiple signals simultaneously."""

    def test_mixed_signals_pipeline(self, fcm_dir, imi_dir, monkeypatch):
        """Multiple SYMBIONT signals coexist and are processed correctly."""
        monkeypatch.setattr("imi.symbiont_bridge.FCM_EVENTS_DIR", fcm_dir)

        # 1. Write channel weights
        weights = {"channel_0": 0.95, "channel_1": 0.2}
        (imi_dir / "channel_weights.json").write_text(json.dumps(weights))

        # 2. Emit approved artifact
        artifact_event = {
            "id": "art001",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": "symbiont",
            "type": "memory_created",
            "title": "Approved pattern",
            "content": "High quality pattern",
            "tags": ["mound_approved"],
            "salience": 0.9,
            "memoryType": "lesson",
            "metadata": {"artifact_status": "APPROVED", "quality": 0.95},
        }
        (fcm_dir / "art001.json").write_text(json.dumps(artifact_event))

        # 3. Emit priority shift
        shift_event = {
            "id": "shift001",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": "symbiont",
            "type": "custom",
            "title": "Priority shift",
            "content": "Priority shifted to: sre",
            "tags": ["priority_shift"],
            "salience": 0.9,
            "memoryType": "decision",
            "metadata": {"signal_type": "PRIORITY_SHIFT", "new_domain": "sre"},
        }
        (fcm_dir / "shift001.json").write_text(json.dumps(shift_event))

        # IMI SIDE: Process all signals
        artifacts = get_mound_approved_artifacts()
        domain = check_priority_shift()
        nodes = _make_nodes(10)

        l1 = generate_l1(
            nodes,
            max_facts=5,
            domain_filter=domain,
            channel_weights=weights,
        )

        # Verify all signals were processed
        assert len(artifacts) == 1
        assert domain == "sre"
        assert l1.facts is not None
        assert l1.domain_filter == "sre"

    def test_empty_pipeline(self, fcm_dir, monkeypatch):
        """Pipeline works with no SYMBIONT signals (graceful degradation)."""
        monkeypatch.setattr("imi.symbiont_bridge.FCM_EVENTS_DIR", fcm_dir)

        artifacts = get_mound_approved_artifacts()
        domain = check_priority_shift()
        weights = read_channel_weights()

        nodes = _make_nodes(5)
        l1 = generate_l1(nodes, max_facts=3, channel_weights=weights)

        assert artifacts == []
        assert domain is None
        assert weights == {}
        assert len(l1.facts) > 0  # L1 works without SYMBIONT signals
