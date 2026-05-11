"""Tests for FCMBridge — IMI ↔ ClawVault Federation via FCM Bus.

Covers: emit_encode, trust gradient, echo prevention, loop prevention,
mark_consumed, _iso_now, emit_session.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from types import SimpleNamespace

import pytest

# Patch FCM dirs BEFORE importing fcm_bridge so module-level constants use temp dirs
_original_fcm_dir = None
_original_events_dir = None
_original_processed_dir = None


@pytest.fixture(autouse=True)
def _patch_fcm_dirs(tmp_path, monkeypatch):
    """Redirect all FCM I/O to a temp directory for test isolation."""
    import imi.integrations.fcm_bridge as mod

    monkeypatch.setattr(mod, "FCM_DIR", tmp_path / ".fcm")
    monkeypatch.setattr(mod, "FCM_EVENTS_DIR", tmp_path / ".fcm" / "events")
    monkeypatch.setattr(mod, "FCM_PROCESSED_DIR", tmp_path / ".fcm" / "processed")

    # Ensure dirs exist
    (tmp_path / ".fcm" / "events").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".fcm" / "processed").mkdir(parents=True, exist_ok=True)


def _make_node(**kwargs):
    """Create a mock MemoryNode with configurable attributes."""
    defaults = {
        "id": "node-123",
        "original": "Test memory content",
        "seed": "test seed",
        "summary_orbital": "Test summary",
        "tags": ["research"],
        "mass": 7.0,
        "affect": None,
        "source": "",
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


# --- emit_encode tests ---


class TestEmitEncode:
    def test_normal_node_emits_event(self, tmp_path):
        from imi.integrations.fcm_bridge import FCMBridge

        bridge = FCMBridge(source="imi", trust_level="self")
        node = _make_node()

        filepath = bridge.emit_encode(node)

        assert filepath is not None
        assert os.path.exists(filepath)

        with open(filepath) as f:
            event = json.load(f)

        assert event["source"] == "imi"
        assert event["type"] == "memory_created"
        assert event["content"] == "Test memory content"
        assert event["title"] == "Test summary"
        assert "research" in event["tags"]
        assert event["metadata"]["imi_node_id"] == "node-123"

    def test_federated_node_blocked(self, tmp_path):
        """Nodes with federation tags should not be re-emitted (loop prevention)."""
        from imi.integrations.fcm_bridge import FCMBridge

        bridge = FCMBridge()
        node = _make_node(tags=["federated", "architecture"])

        result = bridge.emit_encode(node)
        assert result is None

    def test_clawvault_source_blocked(self, tmp_path):
        """Nodes originating from clawvault should not be emitted back."""
        from imi.integrations.fcm_bridge import FCMBridge

        bridge = FCMBridge()
        node = _make_node(source="clawvault")

        result = bridge.emit_encode(node)
        assert result is None

    def test_external_source_blocked(self, tmp_path):
        """Nodes originating from external should not be emitted back."""
        from imi.integrations.fcm_bridge import FCMBridge

        bridge = FCMBridge()
        node = _make_node(source="external")

        result = bridge.emit_encode(node)
        assert result is None

    def test_empty_content_skipped(self, tmp_path):
        """Nodes with no content should be skipped."""
        from imi.integrations.fcm_bridge import FCMBridge

        bridge = FCMBridge()
        node = _make_node(original="", seed="")

        result = bridge.emit_encode(node)
        assert result is None

    def test_from_imi_tag_blocked(self, tmp_path):
        """Nodes tagged 'from-imi' should be blocked (loop prevention)."""
        from imi.integrations.fcm_bridge import FCMBridge

        bridge = FCMBridge()
        node = _make_node(tags=["from-imi"])

        result = bridge.emit_encode(node)
        assert result is None

    def test_from_clawvault_tag_blocked(self, tmp_path):
        """Nodes tagged 'from-clawvault' should be blocked."""
        from imi.integrations.fcm_bridge import FCMBridge

        bridge = FCMBridge()
        node = _make_node(tags=["from-clawvault"])

        result = bridge.emit_encode(node)
        assert result is None

    def test_extra_tags_appended(self, tmp_path):
        from imi.integrations.fcm_bridge import FCMBridge

        bridge = FCMBridge()
        node = _make_node()

        filepath = bridge.emit_encode(node, extra_tags=["extra1", "extra2"])
        assert filepath is not None

        with open(filepath) as f:
            event = json.load(f)

        assert "extra1" in event["tags"]
        assert "extra2" in event["tags"]
        assert "research" in event["tags"]

    def test_salience_override(self, tmp_path):
        from imi.integrations.fcm_bridge import FCMBridge

        bridge = FCMBridge(trust_level="external")
        node = _make_node()

        filepath = bridge.emit_encode(node, salience=0.1)
        assert filepath is not None

        with open(filepath) as f:
            event = json.load(f)

        # salience=0.1 but floor for external is 0.3, so should be 0.3
        assert event["salience"] == 0.3

    def test_salience_from_mass(self, tmp_path):
        from imi.integrations.fcm_bridge import FCMBridge

        bridge = FCMBridge(trust_level="peer")
        node = _make_node(mass=3.0)  # mass/10 = 0.3, floor for peer = 0.5

        filepath = bridge.emit_encode(node)
        with open(filepath) as f:
            event = json.load(f)

        assert event["salience"] == 0.5  # floor applied


# --- Trust gradient tests ---


class TestTrustGradient:
    def test_self_floor_0_9(self, tmp_path):
        from imi.integrations.fcm_bridge import FCMBridge

        bridge = FCMBridge(trust_level="self")
        node = _make_node(mass=1.0)  # mass/10 = 0.1

        filepath = bridge.emit_encode(node)
        with open(filepath) as f:
            event = json.load(f)

        assert event["salience"] == 0.9

    def test_peer_floor_0_5(self, tmp_path):
        from imi.integrations.fcm_bridge import FCMBridge

        bridge = FCMBridge(trust_level="peer")
        node = _make_node(mass=1.0)  # mass/10 = 0.1

        filepath = bridge.emit_encode(node)
        with open(filepath) as f:
            event = json.load(f)

        assert event["salience"] == 0.5

    def test_external_floor_0_3(self, tmp_path):
        from imi.integrations.fcm_bridge import FCMBridge

        bridge = FCMBridge(trust_level="external")
        node = _make_node(mass=1.0)  # mass/10 = 0.1

        filepath = bridge.emit_encode(node)
        with open(filepath) as f:
            event = json.load(f)

        assert event["salience"] == 0.3

    def test_trusted_floor_0_7(self, tmp_path):
        from imi.integrations.fcm_bridge import FCMBridge

        bridge = FCMBridge(trust_level="trusted")
        node = _make_node(mass=1.0)

        filepath = bridge.emit_encode(node)
        with open(filepath) as f:
            event = json.load(f)

        assert event["salience"] == 0.7

    def test_unknown_trust_defaults_to_0_3(self, tmp_path):
        from imi.integrations.fcm_bridge import FCMBridge

        bridge = FCMBridge(trust_level="unknown-level")
        node = _make_node(mass=1.0)

        filepath = bridge.emit_encode(node)
        with open(filepath) as f:
            event = json.load(f)

        assert event["salience"] == 0.3


# --- Echo prevention tests ---


class TestEchoPrevention:
    def test_poll_ignores_own_source_events(self, tmp_path):
        """poll_clawvault_events should skip events from the bridge's own source."""
        from imi.integrations.fcm_bridge import FCM_EVENTS_DIR, FCMBridge

        bridge = FCMBridge(source="imi")

        # Write an event with source=imi (same as bridge)
        event = {
            "id": "echo-test-1",
            "timestamp": "2026-04-06T12:00:00.000Z",
            "source": "imi",
            "type": "memory_created",
            "title": "Echo test",
            "content": "Should be ignored",
            "tags": ["test"],
            "salience": 0.8,
        }
        filepath = FCM_EVENTS_DIR / "echo_test.json"
        filepath.write_text(json.dumps(event), "utf-8")

        events = bridge.poll_clawvault_events()
        assert len(events) == 0

    def test_poll_accepts_clawvault_source(self, tmp_path):
        """poll_clawvault_events should accept events from source=clawvault."""
        from imi.integrations.fcm_bridge import FCM_EVENTS_DIR, FCMBridge

        bridge = FCMBridge(source="imi")

        event = {
            "id": "cv-test-1",
            "timestamp": "2026-04-06T12:00:00.000Z",
            "source": "clawvault",
            "type": "memory_created",
            "title": "CV test",
            "content": "Should be accepted",
            "tags": ["test"],
            "salience": 0.8,
        }
        filepath = FCM_EVENTS_DIR / "cv_test.json"
        filepath.write_text(json.dumps(event), "utf-8")

        events = bridge.poll_clawvault_events()
        assert len(events) == 1
        assert events[0]["id"] == "cv-test-1"

    def test_poll_rejects_non_clawvault_non_self(self, tmp_path):
        """poll_clawvault_events only accepts source=clawvault."""
        from imi.integrations.fcm_bridge import FCM_EVENTS_DIR, FCMBridge

        bridge = FCMBridge(source="imi")

        event = {
            "id": "ext-test-1",
            "timestamp": "2026-04-06T12:00:00.000Z",
            "source": "external",
            "type": "memory_created",
            "title": "Ext test",
            "content": "Should be rejected",
            "tags": ["test"],
            "salience": 0.8,
        }
        filepath = FCM_EVENTS_DIR / "ext_test.json"
        filepath.write_text(json.dumps(event), "utf-8")

        events = bridge.poll_clawvault_events()
        assert len(events) == 0


# --- Loop prevention tests ---


class TestLoopPrevention:
    def test_poll_skips_events_with_federated_tag(self, tmp_path):
        from imi.integrations.fcm_bridge import FCM_EVENTS_DIR, FCMBridge

        bridge = FCMBridge(source="imi")

        event = {
            "id": "loop-test-1",
            "timestamp": "2026-04-06T12:00:00.000Z",
            "source": "clawvault",
            "type": "memory_created",
            "title": "Loop test",
            "content": "Has federated tag",
            "tags": ["federated", "test"],
            "salience": 0.8,
        }
        filepath = FCM_EVENTS_DIR / "loop_test.json"
        filepath.write_text(json.dumps(event), "utf-8")

        events = bridge.poll_clawvault_events()
        assert len(events) == 0

    def test_poll_skips_events_with_from_imi_tag(self, tmp_path):
        from imi.integrations.fcm_bridge import FCM_EVENTS_DIR, FCMBridge

        bridge = FCMBridge(source="imi")

        event = {
            "id": "loop-test-2",
            "timestamp": "2026-04-06T12:00:00.000Z",
            "source": "clawvault",
            "type": "memory_created",
            "title": "Loop test 2",
            "content": "Has from-imi tag",
            "tags": ["from-imi"],
            "salience": 0.8,
        }
        filepath = FCM_EVENTS_DIR / "loop_test2.json"
        filepath.write_text(json.dumps(event), "utf-8")

        events = bridge.poll_clawvault_events()
        assert len(events) == 0

    def test_poll_skips_from_clawvault_tag(self, tmp_path):
        from imi.integrations.fcm_bridge import FCM_EVENTS_DIR, FCMBridge

        bridge = FCMBridge(source="imi")

        event = {
            "id": "loop-test-3",
            "timestamp": "2026-04-06T12:00:00.000Z",
            "source": "clawvault",
            "type": "memory_created",
            "title": "Loop test 3",
            "content": "Has from-clawvault tag",
            "tags": ["from-clawvault"],
            "salience": 0.8,
        }
        filepath = FCM_EVENTS_DIR / "loop_test3.json"
        filepath.write_text(json.dumps(event), "utf-8")

        events = bridge.poll_clawvault_events()
        assert len(events) == 0

    def test_poll_skips_from_external_tag(self, tmp_path):
        from imi.integrations.fcm_bridge import FCM_EVENTS_DIR, FCMBridge

        bridge = FCMBridge(source="imi")

        event = {
            "id": "loop-test-4",
            "timestamp": "2026-04-06T12:00:00.000Z",
            "source": "clawvault",
            "type": "memory_created",
            "title": "Loop test 4",
            "content": "Has from-external tag",
            "tags": ["from-external"],
            "salience": 0.8,
        }
        filepath = FCM_EVENTS_DIR / "loop_test4.json"
        filepath.write_text(json.dumps(event), "utf-8")

        events = bridge.poll_clawvault_events()
        assert len(events) == 0


# --- mark_consumed tests ---


class TestMarkConsumed:
    def test_moves_file_to_processed(self, tmp_path):
        from imi.integrations.fcm_bridge import FCM_EVENTS_DIR, FCM_PROCESSED_DIR, FCMBridge

        bridge = FCMBridge(source="imi")

        event = {
            "id": "consume-test-1",
            "content": "test",
        }
        filepath = FCM_EVENTS_DIR / "consume_test.json"
        filepath.write_text(json.dumps(event), "utf-8")

        bridge.mark_consumed(str(filepath))

        assert not filepath.exists()
        assert (FCM_PROCESSED_DIR / "consume_test.json").exists()

    def test_tracks_consumed_id(self, tmp_path):
        from imi.integrations.fcm_bridge import FCM_EVENTS_DIR, FCMBridge

        bridge = FCMBridge(source="imi")

        event = {
            "id": "consume-test-2",
            "content": "test",
        }
        filepath = FCM_EVENTS_DIR / "consume_test2.json"
        filepath.write_text(json.dumps(event), "utf-8")

        bridge.mark_consumed(str(filepath))

        assert "consume-test-2" in bridge._consumed_ids

    def test_consumed_events_skipped_in_poll(self, tmp_path):
        from imi.integrations.fcm_bridge import FCM_EVENTS_DIR, FCMBridge

        bridge = FCMBridge(source="imi")

        # Write two events
        for i in range(2):
            event = {
                "id": f"dedup-test-{i}",
                "timestamp": "2026-04-06T12:00:00.000Z",
                "source": "clawvault",
                "type": "memory_created",
                "title": f"Dedup {i}",
                "content": f"Content {i}",
                "tags": ["test"],
                "salience": 0.8,
            }
            (FCM_EVENTS_DIR / f"dedup_{i}.json").write_text(json.dumps(event), "utf-8")

        # Consume the first one
        bridge.mark_consumed(str(FCM_EVENTS_DIR / "dedup_0.json"))

        # Only second should appear (first was consumed and moved)
        events = bridge.poll_clawvault_events()
        assert len(events) == 1
        assert events[0]["id"] == "dedup-test-1"

    def test_mark_consumed_nonexistent_file(self, tmp_path):
        """Consuming a nonexistent file should not raise."""
        from imi.integrations.fcm_bridge import FCMBridge

        bridge = FCMBridge()
        bridge.mark_consumed("/nonexistent/path.json")  # Should not raise


# --- _iso_now tests ---


class TestIsoNow:
    def test_returns_valid_utc_iso_format(self):
        from imi.integrations.fcm_bridge import _iso_now

        result = _iso_now()

        # Should be parseable
        parsed = datetime.fromisoformat(result)
        assert parsed.tzinfo is not None

        # Should contain 'T' separator
        assert "T" in result

        # Should have millisecond precision
        assert "." in result

    def test_returns_utc_timezone(self):
        from imi.integrations.fcm_bridge import _iso_now

        result = _iso_now()
        parsed = datetime.fromisoformat(result)
        assert parsed.utcoffset().total_seconds() == 0


# --- emit_session tests ---


class TestEmitSession:
    def test_writes_session_event(self, tmp_path):
        from imi.integrations.fcm_bridge import FCMBridge

        bridge = FCMBridge(source="imi")

        filepath = bridge.emit_session(
            "session_start",
            "Starting work on FCM bridge",
            tags=["session", "fcm"],
            salience=0.9,
        )

        assert filepath is not None
        assert os.path.exists(filepath)

        with open(filepath) as f:
            event = json.load(f)

        assert event["type"] == "session_start"
        assert event["source"] == "imi"
        assert "session" in event["tags"]
        assert "fcm" in event["tags"]
        assert event["salience"] == 0.9
        assert "Starting work on FCM bridge" in event["content"]

    def test_session_end_event(self, tmp_path):
        from imi.integrations.fcm_bridge import FCMBridge

        bridge = FCMBridge(source="imi")

        filepath = bridge.emit_session(
            "session_end",
            "Finished FCM bridge work",
            metadata={"decisions": ["use filesystem bus"]},
        )

        assert filepath is not None
        with open(filepath) as f:
            event = json.load(f)

        assert event["type"] == "session_end"
        assert event["metadata"]["decisions"] == ["use filesystem bus"]

    def test_default_tags_and_salience(self, tmp_path):
        from imi.integrations.fcm_bridge import FCMBridge

        bridge = FCMBridge()

        filepath = bridge.emit_session("session_checkpoint", "Mid-session save")
        with open(filepath) as f:
            event = json.load(f)

        assert event["tags"] == ["session"]
        assert event["salience"] == 0.8

    def test_title_includes_type_prefix(self, tmp_path):
        from imi.integrations.fcm_bridge import FCMBridge

        bridge = FCMBridge()

        filepath = bridge.emit_session("session_start", "My session summary")
        with open(filepath) as f:
            event = json.load(f)

        assert event["title"].startswith("session_start:")

    def test_long_summary_truncated_in_title(self, tmp_path):
        from imi.integrations.fcm_bridge import FCMBridge

        bridge = FCMBridge()
        long_summary = "x" * 200

        filepath = bridge.emit_session("session_start", long_summary)
        with open(filepath) as f:
            event = json.load(f)

        # Title should be truncated (type prefix + 60 chars of summary)
        assert len(event["title"]) < len(long_summary)
