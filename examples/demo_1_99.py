#!/usr/bin/env python3
"""
Demo 1/99 — The irreducible atom of IMI.

Shows: compress an experience into a seed, then reconstruct it
in different contexts. Same seed, different reconstructions.
"""

from imi.core import compress_seed, remember

# --- A raw experience to encode ---
experience = """
Em novembro de 2025, reescrevemos completamente o middleware de autenticação
do projeto Aurora. A motivação foi uma auditoria de compliance que identificou
que tokens de sessão estavam sendo armazenados em plaintext nos cookies.
Isso violava o artigo 46 da LGPD. A solução foi migrar para JWT encriptado
usando AES-256-GCM, com chaves rotacionadas via AWS KMS e TTL de 15 minutos.
O João liderou a implementação e a Maria fez o code review. Levou 3 sprints.
"""

print("=" * 60)
print("EXPERIÊNCIA ORIGINAL")
print("=" * 60)
print(experience.strip())

# --- Step 1: Compress into seed ---
print("\n" + "=" * 60)
print("SEED (compressão)")
print("=" * 60)
seed = compress_seed(experience)
print(seed)

# --- Step 2: Reconstruct in different contexts ---

# Context A: Working on security
print("\n" + "=" * 60)
print("RECONSTRUÇÃO — contexto: trabalhando em segurança")
print("=" * 60)
recon_a = remember(seed, context="Estou revisando a postura de segurança do projeto Aurora.")
print(recon_a)

# Context B: Onboarding a new team member
print("\n" + "=" * 60)
print("RECONSTRUÇÃO — contexto: onboarding de novo membro")
print("=" * 60)
recon_b = remember(seed, context="Estou explicando o histórico do projeto para um dev novo no time.")
print(recon_b)

# Context C: No context (pure reconstruction)
print("\n" + "=" * 60)
print("RECONSTRUÇÃO — sem contexto")
print("=" * 60)
recon_c = remember(seed)
print(recon_c)

print("\n" + "=" * 60)
print("INSIGHT: Mesma seed, 3 reconstruções diferentes.")
print("A memória se adapta ao presente. Memória é função, não dado.")
print("=" * 60)
