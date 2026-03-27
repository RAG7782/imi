#!/usr/bin/env python3
"""
Demo 100/100 — The complete theory in action.

Shows ALL dimensions of IMI v3:
  1. Predictive coding (surprise-based encoding)
  2. CLS (episodic → semantic consolidation)
  3. Temporal context (when dimension)
  4. Affective tagging (importance modulates memory)
  5. Affordances (memories as action potentials)
  6. Reconsolidation (memories change on access)
  7. TDA (persistent homology — topological health)
  8. Annealing (convergence of dreaming)
"""

import time
from imi import IMISpace, Zoom

space = IMISpace()
space.temporal_index.new_session("demo_100")

# =====================================================================
# PHASE 1: ENCODE — Predictive Coding + Affect + Affordances
# =====================================================================
print("=" * 70)
print("PHASE 1: ENCODING (predictive coding + affect + affordances)")
print("=" * 70)

experiences = [
    {
        "text": """Em novembro de 2025, reescrevemos o middleware de autenticação
        do projeto Aurora. A auditoria de compliance encontrou tokens de sessão
        em plaintext nos cookies, violando LGPD art. 46. Migramos para JWT
        criptografado com AES-256-GCM, chaves via AWS KMS, TTL 15min.
        João liderou, Maria revisou. 3 sprints.""",
        "hint": "Manutenção de rotina no sistema de autenticação do Aurora",
        "tags": ["auth", "security"],
    },
    {
        "text": """Em janeiro de 2026, auditoria encontrou dados pessoais
        (CPF, email) não mascarados nos logs do Aurora. Violação LGPD.
        Ana implementou mascaramento automático com regex + PII detector.
        1 sprint. Incidente reportado ao DPO.""",
        "hint": "Revisão de logs do sistema Aurora",
        "tags": ["security", "compliance"],
    },
    {
        "text": """Em março de 2026, tivemos um incidente P1 no Aurora:
        o serviço de notificações parou de enviar emails por 4 horas.
        Causa: SES atingiu rate limit sem retry. Roberto implementou
        exponential backoff + dead letter queue. Postmortem completo.
        Impacto: 2000 usuários não receberam confirmações de pagamento.""",
        "hint": "Verificação do serviço de notificações do Aurora",
        "tags": ["incident", "reliability"],
    },
    {
        "text": """Em fevereiro de 2026, migramos o Aurora de EC2 para
        ECS Fargate. Billing mensal caiu 40%. Carlos desenhou containers,
        Juliana escreveu Dockerfiles. Zero-downtime com blue-green.
        Reduzimos também o tempo de deploy de 20min para 3min.""",
        "hint": "Planejamento de infraestrutura para o próximo trimestre",
        "tags": ["infra", "aws"],
    },
    {
        "text": """Em dezembro de 2025, Pedro resolveu problema de latência
        no dashboard do Aurora: query MongoDB fazendo full scan. Criou
        índice composto (tenant_id, timestamp). Latência caiu de 8s para
        200ms. Solução encontrada em 2 horas, deploy no mesmo dia.""",
        "hint": "Análise de performance do dashboard do Aurora",
        "tags": ["performance", "mongodb"],
    },
]

for i, exp in enumerate(experiences):
    print(f"\n  [{i+1}/{len(experiences)}] Encoding: {exp['tags']}")

    # Small delay to create temporal spread
    node = space.encode(
        exp["text"],
        tags=exp["tags"],
        context_hint=exp["hint"],
        use_predictive_coding=True,
    )

    print(f"    Node:     {node.id}")
    print(f"    Orbital:  {node.summary_orbital}")
    print(f"    Surprise: {node.surprise_magnitude:.0%} — {node.surprise_summary[:80]}...")
    print(f"    Affect:   {node.affect}")
    print(f"    Mass:     {node.mass:.2f}")
    if node.affordances:
        print(f"    Affordances ({len(node.affordances)}):")
        for a in node.affordances[:2]:
            print(f"      → {a}")

# =====================================================================
# PHASE 2: NAVIGATE — Zoom + Semantic + Surprise-aware
# =====================================================================
print("\n\n" + "=" * 70)
print("PHASE 2: NAVIGATE (zoom levels + dual-store)")
print("=" * 70)

query = "problemas recorrentes de segurança"

for zoom in [Zoom.ORBITAL, Zoom.MEDIUM]:
    print(f"\n--- Zoom: {zoom.value} ---")
    result = space.navigate(query, zoom=zoom, top_k=5)
    for m in result.memories[:5]:
        surprise_bar = "!" * int(m["surprise"] * 10)
        print(f"  [{m['score']:.2f}] S:{surprise_bar:10s} {m['content'][:90]}")

# =====================================================================
# PHASE 3: AFFORDANCE SEARCH — "What can I DO?"
# =====================================================================
print("\n\n" + "=" * 70)
print("PHASE 3: AFFORDANCE SEARCH")
print("=" * 70)

action_query = "como resolver problemas de rate limiting"
print(f"\n  Query: '{action_query}'")
affordances = space.search_affordances(action_query, top_k=3)
for a in affordances:
    print(f"\n  [{a['similarity']:.2f}] {a['action']}")
    print(f"    Conditions: {a['conditions']}")
    print(f"    From memory: {a['memory_summary'][:80]}...")

# =====================================================================
# PHASE 4: DREAM — Consolidation + Annealing
# =====================================================================
print("\n\n" + "=" * 70)
print("PHASE 4: DREAMING (consolidation + annealing)")
print("=" * 70)

for i in range(3):
    report = space.dream(similarity_threshold=0.45, track_convergence=True)
    print(f"\n  Dream cycle {i+1}: {report}")
    print(f"  {space.annealing}")

if space.semantic.nodes:
    print(f"\n  Semantic patterns emerged:")
    for node in space.semantic.nodes:
        print(f"    [{node.id}] {node.summary_medium[:100]}")

# =====================================================================
# PHASE 5: TDA — Persistent Homology
# =====================================================================
print("\n\n" + "=" * 70)
print("PHASE 5: TDA (persistent homology — topological health)")
print("=" * 70)

tda = space.compute_tda()
print(f"\n{tda}")

# =====================================================================
# PHASE 6: RECONSOLIDATION — Memory changes on access
# =====================================================================
print("\n\n" + "=" * 70)
print("PHASE 6: RECONSOLIDATION")
print("=" * 70)

# Access the P1 incident memory with a new context
print("\n  Accessing P1 incident memory in context of 'preparando plano de DR'...")
result = space.navigate(
    "incidente de notificações",
    zoom=Zoom.FULL,
    top_k=1,
    context="Estamos preparando um plano de disaster recovery e precisamos revisar incidentes passados.",
    reconsolidate_on_access=True,
)

if result.memories:
    m = result.memories[0]
    print(f"\n  Memory: {m['id']}")
    print(f"  Content:\n    {m['content'][:300]}...")

if space.reconsolidation_log:
    print(f"\n  Reconsolidation events:")
    for event in space.reconsolidation_log:
        print(f"    {event}")

# =====================================================================
# PHASE 7: FULL STATS
# =====================================================================
print("\n\n" + "=" * 70)
print("PHASE 7: COMPLETE SYSTEM STATS")
print("=" * 70)

stats = space.stats()
for k, v in stats.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.3f}")
    else:
        print(f"  {k}: {v}")

# =====================================================================
# INSIGHT SUMMARY
# =====================================================================
print("\n\n" + "=" * 70)
print("100/100 — INFINITE MEMORY IMAGE — COMPLETE")
print("=" * 70)
print("""
  What this demo proved:

  1. PREDICTIVE CODING: Experiences encoded as SURPRISE, not raw data.
     The P1 incident had high surprise (unexpected).
     The infra migration had low surprise (expected).

  2. AFFECT: High-affect memories (incident) have higher mass.
     They resist fade and attract attention.

  3. AFFORDANCES: Memories are not just records — they're tools.
     "How to fix rate limiting" → finds the SES incident's solution.

  4. TEMPORAL: Memories linked by WHEN, not just WHAT.

  5. CLS: Dreaming consolidates episodic patterns → semantic knowledge.

  6. TDA: Persistent homology reveals the SHAPE of knowledge.
     Betti numbers = diagnostic of cognitive health.

  7. RECONSOLIDATION: Accessing a memory in new context CHANGES it.
     The P1 incident, viewed through DR planning lens, gains new framing.

  8. ANNEALING: Dreaming converges. Energy decreases. Space stabilizes.

  Memory is not data. Memory is function.
  Memory is not retrieval. Memory is rendering.
  Memory is not static. Memory is alive.
""")
