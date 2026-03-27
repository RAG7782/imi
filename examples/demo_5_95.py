#!/usr/bin/env python3
"""
Demo 5/95 — The viable core of IMI.

Shows: encode multiple experiences, navigate with zoom levels,
observe how the same query returns different resolutions.
"""

from imi import IMISpace, Zoom

# --- Create space ---
space = IMISpace()

# --- Encode experiences ---
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
        "text": """Em dezembro de 2025, o dashboard de métricas do Aurora começou
        a apresentar latência de 8 segundos. Investigamos e descobrimos que a
        query de aggregation no MongoDB estava fazendo full collection scan.
        Pedro criou um índice composto em (tenant_id, timestamp) e a latência
        caiu para 200ms. Fix em 2 dias.""",
        "tags": ["performance", "mongodb", "dashboard"],
    },
    {
        "text": """Em janeiro de 2026, implantamos feature flags no Aurora usando
        LaunchDarkly. Isso permitiu deploy contínuo sem risco. A primeira feature
        flagged foi o novo fluxo de onboarding. Ana configurou os segmentos
        por tenant e Lucas integrou o SDK no frontend React.""",
        "tags": ["feature-flags", "deploy", "frontend"],
    },
    {
        "text": """Em fevereiro de 2026, migramos o Aurora de EC2 para ECS Fargate.
        Motivação: custos e escalabilidade. O billing mensal caiu 40%.
        Carlos desenhou a arquitetura de containers, Juliana escreveu os
        Dockerfiles e task definitions. Migration zero-downtime com blue-green.""",
        "tags": ["infra", "aws", "containers"],
    },
    {
        "text": """Em março de 2026, tivemos um incidente P1: o serviço de
        notificações do Aurora parou de enviar emails por 4 horas. Causa raiz:
        o SES atingiu o rate limit e não tínhamos retry com backoff. Roberto
        implementou exponential backoff + dead letter queue. Post-mortem feito.""",
        "tags": ["incident", "email", "reliability"],
    },
]

print("=" * 70)
print("ENCODING 5 EXPERIÊNCIAS")
print("=" * 70)

for i, exp in enumerate(experiences):
    print(f"\n  Encoding [{i+1}/5]: {exp['tags']}")
    node = space.encode(exp["text"], tags=exp["tags"])
    print(f"    → Node {node.id}")
    print(f"    → Orbital: {node.summary_orbital}")
    print(f"    → Seed: {node.seed[:80]}...")

# --- Navigate with different zoom levels ---
query = "problemas de segurança e compliance"

print("\n\n" + "=" * 70)
print(f"QUERY: '{query}'")
print("=" * 70)

for zoom in [Zoom.ORBITAL, Zoom.MEDIUM, Zoom.DETAILED]:
    print(f"\n--- Zoom: {zoom.value} ---")
    result = space.navigate(query, zoom=zoom, top_k=5)
    print(f"  ~{result.total_tokens_approx} tokens para {len(result.memories)} memórias\n")
    for m in result.memories:
        print(f"  [{m['score']:.2f}] {m['content']}")
        print()

# --- Full zoom (reconstruction) ---
print(f"\n--- Zoom: FULL (com reconstrução LLM) ---")
result_full = space.navigate(
    query,
    zoom=Zoom.FULL,
    top_k=3,
    context="Estou preparando relatório de segurança para o board.",
)
print(f"  ~{result_full.total_tokens_approx} tokens\n")
for m in result_full.memories[:3]:
    print(f"  [{m['score']:.2f}] {m['content'][:300]}...")
    print()

# --- Different query, same memories ---
query2 = "performance e escalabilidade"
print("\n" + "=" * 70)
print(f"QUERY: '{query2}' (zoom medium)")
print("=" * 70)
result2 = space.navigate(query2, zoom=Zoom.MEDIUM, top_k=3)
for m in result2.memories:
    print(f"\n  [{m['score']:.2f}] {m['content']}")

# --- Stats ---
print("\n\n" + "=" * 70)
print("STATS DO ESPAÇO")
print("=" * 70)
stats = space.stats()
for k, v in stats.items():
    print(f"  {k}: {v}")

print("\n" + "=" * 70)
print("5/95 COMPLETO: encode + zoom + vector search + fade")
print("  - 5 experiências codificadas com seeds + 3 zoom levels")
print("  - Busca vetorial com ponderação por relevância")
print("  - Zoom orbital (~50 tokens total) até full (~600 tokens)")
print("  - Fade natural: memórias antigas perdem relevância")
print("=" * 70)
