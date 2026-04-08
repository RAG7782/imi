# Construction Grammar × AdaptiveRW: Theoretical Connection

> The 4 query intents of AdaptiveRW are constructions in the Goldbergian sense.

## Background

**Construction Grammar (CxG)** — Goldberg (1995, 2006) defines constructions as learned
form-meaning pairings that exist at all levels of grammatical description. A construction
is not just a syntactic pattern — it carries its own meaning independent of the words
that fill it.

**AdaptiveRW** — IMI's adaptive relevance weighting classifies queries into 4 intents
(TEMPORAL, EXPLORATORY, ACTION, DEFAULT) and adjusts the retrieval weight accordingly.
Each intent has a characteristic linguistic signature (keyword patterns) and an optimal
retrieval configuration.

## The Connection

Each AdaptiveRW intent is a **construction** — a form-meaning pairing where:
- **Form** = the keyword/structural pattern that triggers classification
- **Meaning** = the retrieval strategy (rw value) that the pattern activates

| Intent | Form (Pattern) | Meaning (rw) | CxG Analysis |
|---|---|---|---|
| TEMPORAL | "recent X", "last week", "today" | rw=0.15 (recency boost) | Temporal deictic construction: the deictic frame ("recent", "last") COERCES the retrieval into recency-weighted mode |
| EXPLORATORY | "find all X", "list every X" | rw=0.00 (pure cosine) | Exhaustive quantifier construction: "all/every" COERCES completeness — no relevance filtering |
| ACTION | "how to X", "fix X", "prevent X" | rw=0.05 (slight relevance) | Purposive construction: "how to" frame COERCES action-oriented retrieval |
| DEFAULT | (no pattern match) | rw=0.10 (balanced) | Unmarked construction: absence of marking = default interpretation |

## Coercion in CxG and AdaptiveRW

Goldberg's key insight: constructions can **coerce** their arguments into new interpretations.
The ditransitive construction "She baked him a cake" coerces "bake" into a transfer meaning
it doesn't inherently have.

Similarly, AdaptiveRW constructions coerce the retrieval system:

- **Query:** "find all auth failures" → The EXPLORATORY construction coerces the retrieval
  into exhaustive mode (rw=0.00), even if the system's default would prioritize recency.
  The construction overrides the system's preference.

- **Query:** "recent DNS issues" → The TEMPORAL construction coerces recency weighting
  (rw=0.15), even for a topic where comprehensive search might be more useful.

This is exactly **P3 (coercion hypothesis)** from our semiotic density research,
validated empirically with 15% displacement in STEER experiments.

## Non-Synonymy Principle

Goldberg's **Principle of No Synonymy**: if two constructions are syntactically distinct,
they must be semantically/pragmatically distinct.

Applied to AdaptiveRW: the 4 intents are NOT interchangeable. They produce measurably
different retrieval results (WS-A/WS-B experiments confirmed this). The form distinction
(keyword patterns) maps to a real meaning distinction (rw values).

| Evidence | Source |
|---|---|
| rw=0.10 optimal for DEFAULT | WS-A parameter sweep |
| rw=0.15 optimal for TEMPORAL | WS-B temporal experiments |
| rw=0.00 optimal for EXPLORATORY | WS-A comprehensive search test |
| rw=0.05 optimal for ACTION | WS-B action-oriented queries |

## Inheritance Hierarchy

CxG constructions form inheritance hierarchies. AdaptiveRW intents do too:

```
QUERY (abstract construction)
├── DEFAULT (rw=0.10) — unmarked, most general
├── TEMPORAL (rw=0.15) — inherits from DEFAULT + adds recency
├── EXPLORATORY (rw=0.00) — inherits from DEFAULT + removes relevance
└── ACTION (rw=0.05) — inherits from DEFAULT + reduces relevance
```

The DEFAULT construction is the "base" — the others are specializations that
override the relevance weight.

## Implications

1. **Theoretical**: AdaptiveRW is an empirical implementation of CxG coercion in the
   retrieval domain. This connects IMI to 30 years of Construction Grammar research.

2. **Practical**: New intents can be added as new constructions (form + meaning pairs)
   without changing the architecture. The system is extensible by adding constructions.

3. **Research**: This bridges Paper 9 (Semiotic Density, P3 coercion) with IMI's
   retrieval mechanism. The same coercion principle operates at both levels:
   - In FI: framework terms coerce LLM reasoning
   - In IMI: query constructions coerce retrieval strategy

## References

- Goldberg, A. E. (1995). *Constructions: A Construction Grammar Approach to Argument Structure*. University of Chicago Press.
- Goldberg, A. E. (2006). *Constructions at Work: The Nature of Generalization in Language*. Oxford University Press.
- Weissweiler, L., et al. (2023). "Construction Grammar and Language Models." *arXiv:2308.02490*.
- Liu et al. (2023). "Lost in the Middle: How Language Models Use Long Contexts." *arXiv:2307.03172*.
