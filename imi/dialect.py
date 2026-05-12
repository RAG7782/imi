"""SDE-AAAK Hybrid Dialect — structured compression with semiotic density.

Combines IMI's functional seed reconstruction with structured metadata tags
inspired by MemPalace's AAAK format, enhanced with Semiotic Density scoring.

The dialect operates as ADDITIVE METADATA on MemoryNodes — it never replaces
the seed. Tags are compact structured strings that any LLM can parse without
a decoder.

Architecture layers:
  - Seed (IMI original): ~80 tokens, functional reconstruction key
  - SDE-AAAK tag: ~20-30 tokens, structured metadata (entities, flags, DS-d)
  - DS-d score: distributional density metric via embedding centroid distance

Theoretical grounding:
  - AAAK dialect (MemPalace): entity codes, flags, pipe-separated fields
  - Semiotic Density (Paper 9): DS-d (distributional) vs DS-c (cultural)
  - Densify/Rarefy operations from SDE framework

Author: Renato Aparecido Gomes (ORCID: 0009-0005-7380-9876)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .embedder import Embedder
    from .node import MemoryNode


# ── Entity Extraction ─────────────────────────────────────

# Common 3-letter entity codes for the AIP ecosystem
KNOWN_ENTITIES: dict[str, str] = {
    "imi": "IMI",
    "fcm": "FCM",
    "clawvault": "CLV",
    "symbiont": "SYM",
    "agora": "AGO",
    "steer": "STR",
    "fig": "FIG",
    "tess": "TSS",
    "supabase": "SPB",
    "kestra": "KST",
}

# Regex for extracting capitalized entities or known proper nouns
_ENTITY_PATTERN = re.compile(r"\b([A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]+)*)\b")


def extract_entities(text: str, max_entities: int = 5) -> list[str]:
    """Extract named entities as 3-letter codes from text.

    Uses pattern matching (no LLM call) for speed. Recognizes:
    - Known AIP ecosystem terms → mapped codes
    - Capitalized proper nouns → first 3 letters uppercased

    Returns:
        List of 3-letter entity codes, deduplicated, max N.
    """
    codes: list[str] = []
    seen: set[str] = set()

    # Check known entities first
    text_lower = text.lower()
    for term, code in KNOWN_ENTITIES.items():
        if term in text_lower and code not in seen:
            codes.append(code)
            seen.add(code)

    # Extract capitalized entities
    for match in _ENTITY_PATTERN.finditer(text):
        entity = match.group(1)
        code = entity[:3].upper()
        if code not in seen and len(code) == 3:
            codes.append(code)
            seen.add(code)

    return codes[:max_entities]


# ── DS-d Scoring ──────────────────────────────────────────


def compute_ds_d(
    text: str,
    embedder: "Embedder",
    *,
    domain_centroid: np.ndarray | None = None,
    reference_texts: list[str] | None = None,
) -> float:
    """Compute distributional semiotic density (DS-d) score.

    DS-d measures how much semantic information is packed into the text
    relative to a domain centroid. Higher DS-d = more semantically loaded
    terms (high distributional density).

    Method:
    1. Embed the text
    2. Embed individual terms (words/phrases)
    3. Measure average cosine distance of terms from the text centroid
    4. Higher variance in term embeddings = higher density
       (terms activate diverse semantic fields)

    If domain_centroid provided: also measures alignment with domain.
    If reference_texts provided: computes centroid from them.

    Returns:
        float: DS-d score in [0, 1]. Higher = denser.
    """
    # Embed full text
    text_emb = embedder.embed(text)

    # Split into meaningful chunks (not single words — bigrams work better)
    words = text.split()
    if len(words) < 3:
        return 0.5  # Too short to measure density

    # Create overlapping bigrams for richer term analysis
    terms = []
    for i in range(len(words)):
        terms.append(words[i])
        if i < len(words) - 1:
            terms.append(f"{words[i]} {words[i + 1]}")

    # Sample if too many terms
    if len(terms) > 20:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(terms), 20, replace=False)
        terms = [terms[i] for i in indices]

    # Embed terms
    term_embs = embedder.embed_batch(terms)

    # Measure 1: Semantic spread (variance of term embeddings)
    # Higher spread = terms activate diverse fields = higher density
    centroid = term_embs.mean(axis=0)
    distances = 1.0 - (term_embs @ centroid)  # cosine distance from centroid
    spread = float(np.std(distances))

    # Measure 2: Domain alignment (if centroid provided)
    alignment = 0.5  # neutral default
    if domain_centroid is not None:
        alignment = float(text_emb @ domain_centroid)  # cosine similarity
    elif reference_texts:
        ref_embs = embedder.embed_batch(reference_texts)
        domain_centroid = ref_embs.mean(axis=0)
        domain_centroid /= np.linalg.norm(domain_centroid) + 1e-10
        alignment = float(text_emb @ domain_centroid)

    # Combine: spread (0-0.5 typical) + alignment (0-1)
    # Normalize spread to 0-1 range (typical std is 0.05-0.3)
    spread_normalized = min(1.0, spread / 0.3)

    # Weighted combination: 60% spread + 40% alignment
    ds_d = 0.6 * spread_normalized + 0.4 * max(0, alignment)

    return min(1.0, max(0.0, ds_d))


# ── Flags ─────────────────────────────────────────────────

# Memory significance flags (inspired by AAAK)
FLAGS = {
    "CORE": lambda n: n.affect.salience >= 0.8 if n.affect else False,
    "DECISION": lambda n: "decision" in " ".join(n.tags).lower(),
    "TECHNICAL": lambda n: "technical" in " ".join(n.tags).lower() or n.source == "code",
    "PATTERN": lambda n: "_pattern" in n.tags,
    "PIVOT": lambda n: n.surprise_magnitude >= 0.7,
    "ORIGIN": lambda n: n.access_count == 0 and n.surprise_magnitude >= 0.5,
}


def detect_flags(node: "MemoryNode") -> list[str]:
    """Detect applicable flags for a memory node."""
    return [name for name, check in FLAGS.items() if check(node)]


# ── SDE-AAAK Tag Format ──────────────────────────────────


@dataclass
class SDETag:
    """Structured SDE-AAAK metadata tag for a memory node.

    Format: ENT1,ENT2|TYPE:decision|SAL:0.9|DS-d:0.85|FLAG:CORE,PIVOT
    """

    entities: list[str] = field(default_factory=list)
    memory_type: str = ""
    salience: float = 0.0
    ds_d: float = 0.0
    ds_c_note: str = ""  # Cultural density annotation (optional, human-provided)
    flags: list[str] = field(default_factory=list)
    domain: str = ""

    def render(self) -> str:
        """Render tag as compact pipe-separated string."""
        parts = []

        if self.entities:
            parts.append(",".join(self.entities))

        if self.memory_type:
            parts.append(f"TYPE:{self.memory_type}")

        parts.append(f"SAL:{self.salience:.2f}")
        parts.append(f"DS-d:{self.ds_d:.2f}")

        if self.ds_c_note:
            parts.append(f"DS-c:{self.ds_c_note}")

        if self.domain:
            parts.append(f"DOM:{self.domain}")

        if self.flags:
            parts.append(f"FLAG:{','.join(self.flags)}")

        return "|".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities": self.entities,
            "memory_type": self.memory_type,
            "salience": self.salience,
            "ds_d": self.ds_d,
            "ds_c_note": self.ds_c_note,
            "flags": self.flags,
            "domain": self.domain,
            "tag_str": self.render(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SDETag":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def parse_tag(tag_str: str) -> SDETag:
    """Parse an SDE-AAAK tag string back into structured data.

    Handles format: ENT1,ENT2|TYPE:decision|SAL:0.9|DS-d:0.85|FLAG:CORE,PIVOT
    """
    tag = SDETag()
    parts = tag_str.split("|")

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if part.startswith("TYPE:"):
            tag.memory_type = part[5:]
        elif part.startswith("SAL:"):
            try:
                tag.salience = float(part[4:])
            except ValueError:
                pass
        elif part.startswith("DS-d:"):
            try:
                tag.ds_d = float(part[5:])
            except ValueError:
                pass
        elif part.startswith("DS-c:"):
            tag.ds_c_note = part[5:]
        elif part.startswith("DOM:"):
            tag.domain = part[4:]
        elif part.startswith("FLAG:"):
            tag.flags = [f.strip() for f in part[5:].split(",") if f.strip()]
        elif ":" not in part:
            # First part without colon = entity codes
            tag.entities = [e.strip() for e in part.split(",") if e.strip()]

    return tag


def format_tag(
    node: "MemoryNode",
    *,
    ds_d: float = 0.0,
    domain: str = "",
) -> SDETag:
    """Generate SDE-AAAK tag for a memory node.

    Args:
        node: The memory node to tag
        ds_d: Pre-computed DS-d score (0 if not computed)
        domain: Domain context for the tag

    Returns:
        SDETag with all fields populated from node metadata.
    """
    # Extract entities from seed or original text
    text = node.original or node.seed or node.summary_detailed or ""
    entities = extract_entities(text)

    # Detect memory type from tags
    memory_type = ""
    type_keywords = {
        "decision": "decision",
        "architecture": "technical",
        "bug": "incident",
        "fix": "incident",
        "_pattern": "pattern",
        "lesson": "lesson",
        "preference": "preference",
    }
    for tag in node.tags:
        tag_lower = tag.lower()
        for keyword, mtype in type_keywords.items():
            if keyword in tag_lower:
                memory_type = mtype
                break
        if memory_type:
            break

    # Detect flags
    flags = detect_flags(node)

    return SDETag(
        entities=entities,
        memory_type=memory_type,
        salience=node.affect.salience if node.affect else 0.5,
        ds_d=ds_d,
        flags=flags,
        domain=domain,
    )


# ── Densify Prompt Enhancement ────────────────────────────

DENSIFY_INSTRUCTION = """\

DENSITY INSTRUCTION: When compressing, prefer high-density domain terms over \
generic ones. For example: "analyze" → "audit", "improve" → "optimize", \
"check" → "validate". Each term should activate a specific professional \
frame, not a generic one. Preserve technical jargon — it carries semantic \
payload."""

DENSIFY_DOMAIN_TEMPLATE = """\

DOMAIN DENSITY: This experience is in the "{domain}" domain. Use terminology \
native to this domain — these terms carry implicit frames that generic \
language does not. Domain-specific terms are denser (higher DS-d) than \
their generic equivalents."""


def densify_prompt(domain: str = "") -> str:
    """Generate Densify instruction to append to compress_seed system prompt.

    This implements the SDE Densify operation: replace generic terms with
    high-density domain terms in the seed compression.

    Args:
        domain: Optional domain context (e.g., "tributario", "sre", "legal")

    Returns:
        Instruction string to append to COMPRESS_SYSTEM prompt.
    """
    parts = [DENSIFY_INSTRUCTION]
    if domain:
        parts.append(DENSIFY_DOMAIN_TEMPLATE.format(domain=domain))
    return "\n".join(parts)
