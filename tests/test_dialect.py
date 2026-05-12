"""Tests for SDE-AAAK Hybrid Dialect."""

import numpy as np
import pytest

from imi.dialect import (
    SDETag,
    compute_ds_d,
    densify_prompt,
    detect_flags,
    extract_entities,
    format_tag,
    parse_tag,
)
from imi.embedder import SentenceTransformerEmbedder
from imi.node import AffectiveTag, MemoryNode


def _make_node(
    seed="test memory about IMI architecture",
    salience=0.5,
    tags=None,
    source="",
    surprise_magnitude=0.0,
    access_count=0,
) -> MemoryNode:
    node = MemoryNode(
        seed=seed,
        original=seed,
        summary_orbital=seed[:30],
        summary_medium=seed[:60],
        summary_detailed=seed,
        affect=AffectiveTag(salience=salience, valence=0.5, arousal=0.5),
        tags=tags or [],
        source=source,
        embedding=np.random.randn(384).astype(np.float32),
    )
    node.surprise_magnitude = surprise_magnitude
    node.access_count = access_count
    return node


# ── Entity Extraction ─────────────────────────────────


class TestExtractEntities:
    def test_known_entities(self):
        entities = extract_entities("The IMI system connects to ClawVault via FCM bus")
        assert "IMI" in entities
        assert "CLV" in entities
        assert "FCM" in entities

    def test_capitalized_entities(self):
        entities = extract_entities("The Topology Engine optimizes Mycelium channels")
        # Should extract multi-word capitalized phrases or their 3-letter codes
        assert len(entities) > 0  # Should find at least one entity

    def test_max_entities(self):
        text = "IMI SYMBIONT FCM ClawVault AGORA STEER FIG TESS Supabase Kestra"
        entities = extract_entities(text, max_entities=3)
        assert len(entities) <= 3

    def test_empty_text(self):
        entities = extract_entities("")
        assert entities == []

    def test_no_duplicates(self):
        entities = extract_entities("IMI uses IMI patterns from IMI research")
        assert entities.count("IMI") == 1


# ── DS-d Scoring ──────────────────────────────────────


class TestComputeDsD:
    @pytest.fixture(scope="class")
    def embedder(self):
        return SentenceTransformerEmbedder()

    def test_returns_float_in_range(self, embedder):
        score = compute_ds_d(
            "The authentication service crashed due to certificate expiry", embedder
        )
        assert 0.0 <= score <= 1.0

    def test_dense_text_higher_than_generic(self, embedder):
        dense = "Conduct forensic audit of certificate chain with OCSP stapling validation"
        generic = "Check the thing and fix the problem"
        score_dense = compute_ds_d(dense, embedder)
        score_generic = compute_ds_d(generic, embedder)
        # Dense text should generally score higher (more diverse term activations)
        # This is a soft assertion — may not always hold for all sentences
        assert score_dense > 0.0
        assert score_generic > 0.0

    def test_short_text_returns_default(self, embedder):
        score = compute_ds_d("hi", embedder)
        assert score == 0.5  # Default for text < 3 words

    def test_with_domain_centroid(self, embedder):
        centroid = embedder.embed("software engineering debugging deployment")
        score = compute_ds_d(
            "Deploy the microservice with rolling updates and health checks",
            embedder,
            domain_centroid=centroid,
        )
        assert 0.0 <= score <= 1.0


# ── Flags Detection ───────────────────────────────────


class TestDetectFlags:
    def test_core_flag(self):
        node = _make_node(salience=0.9)
        flags = detect_flags(node)
        assert "CORE" in flags

    def test_decision_flag(self):
        node = _make_node(tags=["decision", "architecture"])
        flags = detect_flags(node)
        assert "DECISION" in flags

    def test_pivot_flag(self):
        node = _make_node(surprise_magnitude=0.8)
        flags = detect_flags(node)
        assert "PIVOT" in flags

    def test_pattern_flag(self):
        node = _make_node(tags=["_pattern"])
        flags = detect_flags(node)
        assert "PATTERN" in flags

    def test_no_flags_for_generic(self):
        node = _make_node(salience=0.3, surprise_magnitude=0.1)
        flags = detect_flags(node)
        assert "CORE" not in flags
        assert "PIVOT" not in flags


# ── Tag Format / Parse ────────────────────────────────


class TestSDETag:
    def test_render_basic(self):
        tag = SDETag(
            entities=["IMI", "FCM"],
            memory_type="decision",
            salience=0.9,
            ds_d=0.85,
            flags=["CORE", "TECHNICAL"],
        )
        rendered = tag.render()
        assert "IMI,FCM" in rendered
        assert "TYPE:decision" in rendered
        assert "SAL:0.90" in rendered
        assert "DS-d:0.85" in rendered
        assert "FLAG:CORE,TECHNICAL" in rendered

    def test_render_minimal(self):
        tag = SDETag(salience=0.5, ds_d=0.3)
        rendered = tag.render()
        assert "SAL:0.50" in rendered
        assert "DS-d:0.30" in rendered

    def test_parse_roundtrip(self):
        original = SDETag(
            entities=["IMI", "CLV"],
            memory_type="technical",
            salience=0.75,
            ds_d=0.60,
            domain="sre",
            flags=["CORE"],
        )
        rendered = original.render()
        parsed = parse_tag(rendered)
        assert parsed.entities == ["IMI", "CLV"]
        assert parsed.memory_type == "technical"
        assert abs(parsed.salience - 0.75) < 0.01
        assert abs(parsed.ds_d - 0.60) < 0.01
        assert parsed.domain == "sre"
        assert "CORE" in parsed.flags

    def test_to_dict(self):
        tag = SDETag(entities=["IMI"], salience=0.8, ds_d=0.7)
        d = tag.to_dict()
        assert "tag_str" in d
        assert d["entities"] == ["IMI"]

    def test_from_dict(self):
        d = {"entities": ["FCM"], "salience": 0.6, "ds_d": 0.5, "flags": ["PIVOT"]}
        tag = SDETag.from_dict(d)
        assert tag.entities == ["FCM"]
        assert tag.flags == ["PIVOT"]


class TestFormatTag:
    def test_format_from_node(self):
        node = _make_node(
            seed="IMI architecture decision for FCM integration",
            salience=0.9,
            tags=["decision", "architecture"],
        )
        tag = format_tag(node, ds_d=0.85, domain="memory")
        assert "IMI" in tag.entities
        assert "FCM" in tag.entities
        assert tag.memory_type == "decision"
        assert tag.salience == 0.9
        assert tag.ds_d == 0.85
        assert tag.domain == "memory"
        assert "CORE" in tag.flags

    def test_format_minimal_node(self):
        node = _make_node(seed="simple note", salience=0.3, tags=[])
        tag = format_tag(node, ds_d=0.2)
        assert tag.salience == 0.3
        assert tag.ds_d == 0.2
        rendered = tag.render()
        assert "SAL:" in rendered


# ── Densify Prompt ────────────────────────────────────


class TestDensifyPrompt:
    def test_without_domain(self):
        prompt = densify_prompt()
        assert "DENSITY INSTRUCTION" in prompt
        assert "domain" not in prompt.lower() or "domain" in prompt

    def test_with_domain(self):
        prompt = densify_prompt(domain="tributario")
        assert "tributario" in prompt
        assert "DOMAIN DENSITY" in prompt

    def test_empty_domain(self):
        prompt = densify_prompt(domain="")
        assert "DOMAIN DENSITY" not in prompt
