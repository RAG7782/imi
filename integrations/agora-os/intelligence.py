"""AGORA Intelligence ↔ IMI — Historical pattern analysis for AGORA workflows.

AGORA Intelligence (the --agora-full flag in /ax skills) performs strategic analysis.
This module gives it historical memory:

1. Sprint Analytics: "auth features always take 2x estimated time"
2. Decision Patterns: "@po rejected 3x stories without Spec Constraints"
3. Risk Signals: "Friday deploys have 2x rollback rate"
4. Team Insights: "database stories need @dev + DBA review"

Usage in AGORA skills:
    from integrations.agora_os.intelligence import AgoraIntelligence

    intel = AgoraIntelligence()

    # Before @pm creates a PRD
    context = intel.inform_pm("auth service refactor")
    # → "Past auth changes: 3 incidents, avg 2x timeline, key risk: cascade failures"

    # Before @qa reviews
    context = intel.inform_qa("story-AUTH-005")
    # → "Similar stories had: missing error handling (2x), no rollback plan (1x)"

    # Before @devops deploys
    context = intel.inform_devops("auth-service v2.3")
    # → "Last 3 auth deploys: 1 rollback (Friday), 2 smooth (Tuesday/Wednesday)"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class IntelligenceReport:
    """Structured intelligence report for AGORA skills."""
    context: str  # Human-readable summary
    memories: list[dict[str, Any]] = field(default_factory=list)
    patterns: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    affordances: list[str] = field(default_factory=list)
    confidence: float = 0.0  # 0-1, based on number of relevant memories

    def to_agora_context(self) -> str:
        """Format as AGORA Intelligence context block."""
        lines = [
            "## MEMORY INTELLIGENCE (from IMI episodic memory)",
            "",
            f"**Context:** {self.context}",
            f"**Confidence:** {self.confidence:.0%} (based on {len(self.memories)} relevant memories)",
        ]

        if self.patterns:
            lines.append("")
            lines.append("**Detected Patterns:**")
            for p in self.patterns:
                lines.append(f"- {p}")

        if self.risks:
            lines.append("")
            lines.append("**Risk Signals:**")
            for r in self.risks:
                lines.append(f"- {r}")

        if self.affordances:
            lines.append("")
            lines.append("**Recommended Actions (from past experience):**")
            for a in self.affordances:
                lines.append(f"- {a}")

        if self.memories:
            lines.append("")
            lines.append("**Supporting Memories:**")
            for m in self.memories[:5]:
                score = m.get("score", 0)
                content = m.get("content", "")[:100]
                lines.append(f"  [{score:.2f}] {content}")

        return "\n".join(lines)


class AgoraIntelligence:
    """Historical intelligence layer for AGORA workflows.

    Queries IMI memory to provide context-aware insights
    to each AGORA skill before execution.
    """

    def __init__(self, imi_db: str = "~/.claude/plugins/data/imi/agora-memory.db"):
        self._db_path = imi_db
        self._space = None

    def _get_space(self):
        if self._space is None:
            from pathlib import Path
            from imi.space import IMISpace
            db = str(Path(self._db_path).expanduser())
            self._space = IMISpace.from_sqlite(db)
        return self._space

    def inform_pm(self, topic: str) -> IntelligenceReport:
        """Intelligence for @pm — before creating PRD/Epic.

        Queries: past decisions, timeline estimates, stakeholder feedback.
        """
        space = self._get_space()

        # Search for past PRD/planning decisions
        nav = space.navigate(
            f"product decisions about {topic}",
            top_k=10,
        )

        # Search for past timeline data
        timeline_nav = space.navigate(
            f"timeline estimation {topic}",
            top_k=5,
        )

        # Search for past incidents related to this area
        incidents = space.navigate(
            f"incidents failures related to {topic}",
            top_k=5,
        )

        all_memories = nav.memories + timeline_nav.memories + incidents.memories
        # Deduplicate by ID
        seen = set()
        unique = []
        for m in all_memories:
            mid = m.get("id", "")
            if mid not in seen:
                seen.add(mid)
                unique.append(m)

        patterns = self._extract_patterns(unique, "pm")
        risks = self._extract_risks(unique)
        affordances = self._get_affordances(space, f"plan {topic}")

        confidence = min(1.0, len(unique) / 10.0)

        return IntelligenceReport(
            context=f"Historical intelligence for '{topic}': {len(unique)} relevant memories found",
            memories=unique[:10],
            patterns=patterns,
            risks=risks,
            affordances=affordances,
            confidence=confidence,
        )

    def inform_sm(self, story_context: str) -> IntelligenceReport:
        """Intelligence for @sm — before creating stories/tasks.

        Queries: past story estimates, blockers, dependencies.
        """
        space = self._get_space()

        nav = space.navigate(f"story estimation {story_context}", top_k=10)
        blockers = space.navigate(f"blocked by dependencies {story_context}", top_k=5)

        all_memories = nav.memories + blockers.memories
        seen = set()
        unique = [m for m in all_memories if m.get("id", "") not in seen and not seen.add(m.get("id", ""))]

        patterns = self._extract_patterns(unique, "sm")
        risks = self._extract_risks(unique)

        return IntelligenceReport(
            context=f"Story planning intelligence: {len(unique)} relevant memories",
            memories=unique[:10],
            patterns=patterns,
            risks=risks,
            confidence=min(1.0, len(unique) / 10.0),
        )

    def inform_dev(self, implementation_context: str) -> IntelligenceReport:
        """Intelligence for @dev — before implementing.

        Queries: past implementation issues, code patterns, tech debt.
        """
        space = self._get_space()

        nav = space.navigate(f"implementation issues {implementation_context}", top_k=10)
        tech_debt = space.navigate(f"tech debt workaround {implementation_context}", top_k=5)

        all_memories = nav.memories + tech_debt.memories
        seen = set()
        unique = [m for m in all_memories if m.get("id", "") not in seen and not seen.add(m.get("id", ""))]

        affordances = self._get_affordances(space, f"implement {implementation_context}")

        return IntelligenceReport(
            context=f"Implementation intelligence: {len(unique)} relevant memories",
            memories=unique[:10],
            patterns=self._extract_patterns(unique, "dev"),
            risks=self._extract_risks(unique),
            affordances=affordances,
            confidence=min(1.0, len(unique) / 10.0),
        )

    def inform_qa(self, review_context: str) -> IntelligenceReport:
        """Intelligence for @qa — before reviewing.

        Queries: past quality issues, common bugs, regression areas.
        """
        space = self._get_space()

        quality = space.navigate(f"quality issues bugs {review_context}", top_k=10)
        regressions = space.navigate(f"regression test failure {review_context}", top_k=5)

        all_memories = quality.memories + regressions.memories
        seen = set()
        unique = [m for m in all_memories if m.get("id", "") not in seen and not seen.add(m.get("id", ""))]

        return IntelligenceReport(
            context=f"QA intelligence: {len(unique)} relevant memories",
            memories=unique[:10],
            patterns=self._extract_patterns(unique, "qa"),
            risks=self._extract_risks(unique),
            confidence=min(1.0, len(unique) / 10.0),
        )

    def inform_devops(self, deploy_context: str) -> IntelligenceReport:
        """Intelligence for @devops — before deploying.

        Queries: past deploy incidents, rollbacks, environment issues.
        """
        space = self._get_space()

        deploys = space.navigate(f"deploy rollback incident {deploy_context}", top_k=10)
        env_issues = space.navigate(f"environment configuration {deploy_context}", top_k=5)

        all_memories = deploys.memories + env_issues.memories
        seen = set()
        unique = [m for m in all_memories if m.get("id", "") not in seen and not seen.add(m.get("id", ""))]

        affordances = self._get_affordances(space, f"deploy {deploy_context}")

        return IntelligenceReport(
            context=f"Deploy intelligence: {len(unique)} relevant memories",
            memories=unique[:10],
            patterns=self._extract_patterns(unique, "devops"),
            risks=self._extract_risks(unique),
            affordances=affordances,
            confidence=min(1.0, len(unique) / 10.0),
        )

    def inform_po(self, validation_context: str) -> IntelligenceReport:
        """Intelligence for @po — before validating GO/NO-GO.

        Queries: past validation rejections, common issues.
        """
        space = self._get_space()

        validations = space.navigate(f"validation rejected NO-GO {validation_context}", top_k=10)

        return IntelligenceReport(
            context=f"Validation intelligence: {len(validations.memories)} relevant memories",
            memories=validations.memories[:10],
            patterns=self._extract_patterns(validations.memories, "po"),
            risks=self._extract_risks(validations.memories),
            confidence=min(1.0, len(validations.memories) / 10.0),
        )

    def full_analysis(self, topic: str) -> str:
        """Full AGORA Intelligence analysis — used with --agora-full flag.

        Combines all role-specific intelligence into one comprehensive report.
        """
        reports = {
            "Product (@pm)": self.inform_pm(topic),
            "Planning (@sm)": self.inform_sm(topic),
            "Implementation (@dev)": self.inform_dev(topic),
            "Quality (@qa)": self.inform_qa(topic),
            "Deployment (@devops)": self.inform_devops(topic),
            "Validation (@po)": self.inform_po(topic),
        }

        lines = [
            "# AGORA INTELLIGENCE — Full Memory Analysis",
            f"## Topic: {topic}",
            "",
        ]

        total_memories = 0
        all_risks = []
        all_patterns = []

        for role, report in reports.items():
            if report.memories:
                lines.append(f"### {role}")
                lines.append(report.to_agora_context())
                lines.append("")
                total_memories += len(report.memories)
                all_risks.extend(report.risks)
                all_patterns.extend(report.patterns)

        # Cross-cutting synthesis
        lines.append("## Cross-Cutting Synthesis")
        lines.append(f"Total relevant memories: {total_memories}")

        if all_risks:
            unique_risks = list(dict.fromkeys(all_risks))  # deduplicate preserving order
            lines.append(f"\n**Top risks ({len(unique_risks)}):**")
            for r in unique_risks[:5]:
                lines.append(f"- {r}")

        if all_patterns:
            unique_patterns = list(dict.fromkeys(all_patterns))
            lines.append(f"\n**Recurring patterns ({len(unique_patterns)}):**")
            for p in unique_patterns[:5]:
                lines.append(f"- {p}")

        avg_confidence = sum(r.confidence for r in reports.values()) / len(reports)
        lines.append(f"\n**Overall confidence:** {avg_confidence:.0%}")

        if avg_confidence < 0.3:
            lines.append("\n> Low confidence: limited historical data. Consider encoding more workflow outcomes.")

        return "\n".join(lines)

    def _extract_patterns(self, memories: list[dict], role: str) -> list[str]:
        """Extract patterns from memory content (heuristic)."""
        patterns = []
        tags_count: dict[str, int] = {}
        for m in memories:
            for tag in m.get("tags", []):
                if not tag.startswith("agent:") and not tag.startswith("nao:"):
                    tags_count[tag] = tags_count.get(tag, 0) + 1

        for tag, count in sorted(tags_count.items(), key=lambda x: -x[1]):
            if count >= 2:
                patterns.append(f"'{tag}' appears in {count} related memories")
            if len(patterns) >= 3:
                break

        return patterns

    def _extract_risks(self, memories: list[dict]) -> list[str]:
        """Extract risk signals from memory content."""
        risks = []
        risk_keywords = ["fail", "error", "incident", "rollback", "outage", "timeout", "rejected"]
        for m in memories:
            content = m.get("content", "").lower()
            for kw in risk_keywords:
                if kw in content:
                    risks.append(f"Past {kw}: {m.get('content', '')[:80]}")
                    break
            if len(risks) >= 3:
                break
        return risks

    def _get_affordances(self, space, query: str) -> list[str]:
        """Get affordances from IMI."""
        try:
            results = space.search_affordances(query, top_k=3)
            return [f"[{r['confidence']:.0%}] {r['action']}" for r in results]
        except Exception:
            return []
