#!/usr/bin/env python3
"""
Demo 20/80 — The complete IMI system.

Shows: CLS (episodic→semantic), anchors, spatial topology, dreaming.
"""

from imi import IMISpace, Zoom

# --- Create space ---
space = IMISpace()

# --- Encode experiences (goes to EPISODIC store) ---
experiences = [
    {
        "text": """Em novembro de 2025, reescrevemos o middleware de autenticação
        do projeto Aurora. A auditoria de compliance encontrou tokens de sessão
        em plaintext nos cookies, violando LGPD art. 46. Migramos para JWT
        criptografado com AES-256-GCM, chaves via AWS KMS, TTL 15min.
        João liderou, Maria revisou. 3 sprints.""",
        "tags": ["auth", "security", "compliance"],
    },
    {
        "text": """Em janeiro de 2026, houve outra auditoria de segurança no Aurora.
        Desta vez encontraram que os logs de acesso continham dados pessoais (CPF,
        email) sem mascaramento. Violação da LGPD art. 46 novamente. Ana implementou
        mascaramento automático nos logs usando regex + PII detector. 1 sprint.""",
        "tags": ["security", "compliance", "logs"],
    },
    {
        "text": """Em fevereiro de 2026, terceira auditoria de segurança no Aurora.
        Encontraram que as API keys de terceiros estavam hardcoded no código fonte.
        Migração para AWS Secrets Manager. Carlos implementou, 2 sprints.
        Motivação: compliance e segurança.""",
        "tags": ["security", "compliance", "secrets"],
    },
    {
        "text": """Em dezembro de 2025, o dashboard de métricas do Aurora apresentou
        latência de 8 segundos. Query MongoDB fazendo full collection scan.
        Pedro criou índice composto (tenant_id, timestamp), latência caiu para 200ms.""",
        "tags": ["performance", "mongodb"],
    },
    {
        "text": """Em março de 2026, migramos o Aurora de EC2 para ECS Fargate.
        Billing mensal caiu 40%. Carlos desenhou containers, Juliana escreveu
        Dockerfiles. Migration zero-downtime com blue-green deployment.""",
        "tags": ["infra", "aws", "containers"],
    },
]

print("=" * 70)
print("FASE 1: ENCODING (→ episodic store)")
print("=" * 70)

for i, exp in enumerate(experiences):
    print(f"\n  Encoding [{i+1}/{len(experiences)}]: {exp['tags']}")
    node = space.encode(exp["text"], tags=exp["tags"])
    print(f"    → {node.id}: {node.summary_orbital}")

# --- Stats after encoding ---
print("\n" + "=" * 70)
print("STATS APÓS ENCODING")
print("=" * 70)
stats = space.stats()
for k, v in stats.items():
    print(f"  {k}: {v}")

# --- Dreaming: consolidate episodic → semantic ---
print("\n" + "=" * 70)
print("FASE 2: DREAMING (consolidação episodic → semantic)")
print("=" * 70)

# The 3 security/compliance memories should cluster and produce a pattern
report = space.dream(similarity_threshold=0.45)
print(f"\n  {report}")

# Check semantic store
print(f"\n  Semantic store:")
for node in space.semantic.nodes:
    print(f"    [{node.id}] {node.summary_medium}")

# --- Stats after dreaming ---
print("\n" + "=" * 70)
print("STATS APÓS DREAMING")
print("=" * 70)
stats = space.stats()
for k, v in stats.items():
    print(f"  {k}: {v}")

# --- Navigate with semantic patterns ---
print("\n" + "=" * 70)
print("FASE 3: NAVIGATE (busca em episodic + semantic)")
print("=" * 70)

query = "problemas recorrentes de compliance"
print(f"\n  Query: '{query}' (zoom medium)")
result = space.navigate(query, zoom=Zoom.MEDIUM, top_k=5, include_semantic=True)
for m in result.memories:
    store_tag = f"[{m['store'].upper()}]"
    print(f"\n  [{m['score']:.2f}] {store_tag} {m['content'][:150]}")

# --- Topology ---
print("\n" + "=" * 70)
print("FASE 4: TOPOLOGY (metacognição espacial)")
print("=" * 70)

topo = space.compute_topology()
print(f"\n  {topo}")

# --- Confidence / Anchors ---
print("\n" + "=" * 70)
print("FASE 5: CONFIDENCE (anchors anti-confabulação)")
print("=" * 70)

# Pick first node and verify
first_node = space.episodic.nodes[0]
print(f"\n  Verificando memória: {first_node.id}")
print(f"  Seed: {first_node.seed[:80]}...")

anchors = space._anchors.get(first_node.id, [])
print(f"\n  Âncoras extraídas ({len(anchors)}):")
for a in anchors:
    print(f"    [{a.type.value}] {a.reference}")

conf = space.verify_memory(first_node.id)
if conf:
    print(f"\n  {conf}")

# --- Final summary ---
print("\n" + "=" * 70)
print("20/80 COMPLETO — Sistema IMI funcional:")
print("  1. CLS: episodic (eventos) + semantic (padrões consolidados)")
print("  2. Dreaming: episodic clusters → semantic patterns")
print("  3. Navigate: busca em ambos stores, zoom multi-resolução")
print("  4. Topology: clusters, bridges, metacognição espacial")
print("  5. Anchors: âncoras factuais + scoring de confiança")
print("=" * 70)
