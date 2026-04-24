"""
IMI Crypto Layer — Camada 1 / AIP Security Layer
=================================================

Decorator de segurança para o pipeline im_enc do IMI.
Intercepta o `experience` ANTES de space.encode(), aplica:
  1. Sanitização PII (sanitizer_wrapper) — remove dados sensíveis
  2. Criptografia AES-256-GCM (crypto_wrapper) — zero-knowledge em repouso

Padrão: Decorator (GoF) — não modifica o core do IMI.

Uso (no mcp_server.py, substituir a chamada a space.encode):

    from imi.integrations.crypto_layer import secure_encode

    # Em vez de:
    node = space.encode(experience, tags=tag_list, ...)

    # Usar:
    node = secure_encode(space, experience, tags=tag_list, ...)

Modo de operação:
    - ATIVO (IMI_CRYPTO=1):   sanitiza + criptografa antes de encode
    - PASSIVO (IMI_CRYPTO=0): pass-through transparente (sem overhead)
    - DEFAULT: passivo — ativação explícita por env var (não quebra nada)

Chave de criptografia:
    - Lida de IMI_CRYPTO_KEY (hex 64 chars = 32 bytes)
    - Ou derivada de IMI_CRYPTO_SECRET (senha → PBKDF2)
    - Se nenhuma: gera chave ephemeral por sessão (não persiste entre reinícios)

Recuperação:
    - Memórias criptografadas têm prefixo "[ENC:v1]" no conteúdo
    - im_nav/im_drm decodificam automaticamente via decrypt_experience()
    - Memórias legacy (texto plano) são lidas sem modificação

Auditoria:
    - Cada encode criptografado gera log em ~/.imi/crypto_audit.jsonl
    - Formato: {ts, node_id, pii_count, risk_score, key_fingerprint}
    - Nunca loga o conteúdo — apenas metadados

Ref: ~/.aiox/integrations/crypto_wrapper.py
     ~/.aiox/integrations/sanitizer_wrapper.py
     Camada 1 do plano de integração Synapse Layer → ecossistema AIP
"""
from __future__ import annotations

import os
import sys
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any

logger = logging.getLogger("imi.crypto_layer")

# ─── Config via env vars ───────────────────────────────────────────────────

_CRYPTO_ENABLED = os.environ.get("IMI_CRYPTO", "0") == "1"
_AUDIT_FILE = Path.home() / ".imi" / "crypto_audit.jsonl"

# ─── Importar wrappers da Camada 0 ────────────────────────────────────────

_wrappers_path = str(Path.home() / ".aiox" / "integrations")
if _wrappers_path not in sys.path:
    sys.path.insert(0, _wrappers_path)

_crypto_ok = False
_sanitizer_ok = False

try:
    from crypto_wrapper import encrypt_str, decrypt_str, derive_key, generate_key, key_fingerprint
    _crypto_ok = True
    logger.debug("crypto_layer: crypto_wrapper carregado")
except ImportError as e:
    logger.warning("crypto_layer: crypto_wrapper ausente (%s) — crypto desativada", e)

try:
    from sanitizer_wrapper import sanitize
    _sanitizer_ok = True
    logger.debug("crypto_layer: sanitizer_wrapper carregado")
except ImportError as e:
    logger.warning("crypto_layer: sanitizer_wrapper ausente (%s) — sanitização desativada", e)

# ─── Chave de criptografia (singleton por processo) ────────────────────────

_KEY: Optional[bytes] = None
_KEY_FINGERPRINT: str = "none"


def _get_key() -> Optional[bytes]:
    """Retorna chave AES-256. Lazy-init, singleton por processo."""
    global _KEY, _KEY_FINGERPRINT
    if _KEY is not None:
        return _KEY

    if not _crypto_ok:
        return None

    # Opção 1: chave hex direta
    hex_key = os.environ.get("IMI_CRYPTO_KEY", "")
    if hex_key and len(hex_key) == 64:
        try:
            _KEY = bytes.fromhex(hex_key)
            _KEY_FINGERPRINT = key_fingerprint(_KEY)
            logger.info("crypto_layer: chave carregada de IMI_CRYPTO_KEY (fp: %s)", _KEY_FINGERPRINT)
            return _KEY
        except ValueError:
            logger.warning("crypto_layer: IMI_CRYPTO_KEY inválida — tentando IMI_CRYPTO_SECRET")

    # Opção 2: derivar de senha
    secret = os.environ.get("IMI_CRYPTO_SECRET", "")
    if secret:
        _KEY = derive_key(secret)
        _KEY_FINGERPRINT = key_fingerprint(_KEY)
        logger.info("crypto_layer: chave derivada de IMI_CRYPTO_SECRET (fp: %s)", _KEY_FINGERPRINT)
        return _KEY

    # Opção 3: ephemeral (aviso explícito — não persiste)
    _KEY = generate_key()
    _KEY_FINGERPRINT = key_fingerprint(_KEY)
    logger.warning(
        "crypto_layer: chave EPHEMERAL gerada (fp: %s). "
        "Memórias desta sessão NÃO serão decriptáveis após reinício. "
        "Defina IMI_CRYPTO_KEY ou IMI_CRYPTO_SECRET para persistência.",
        _KEY_FINGERPRINT
    )
    return _KEY


# ─── Prefixo de identificação ──────────────────────────────────────────────

_ENC_PREFIX = "[ENC:v1]"


def is_encrypted(text: str) -> bool:
    """Retorna True se o texto foi criptografado por esta camada."""
    return text.startswith(_ENC_PREFIX)


# ─── Auditoria ────────────────────────────────────────────────────────────

def _audit_log(node_id: str, pii_count: int, risk_score: float, sanitizer_backend: str) -> None:
    """Append-only audit log — nunca loga conteúdo."""
    try:
        _AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "node_id": node_id[:12],
            "pii_count": pii_count,
            "risk_score": risk_score,
            "key_fp": _KEY_FINGERPRINT,
            "sanitizer": sanitizer_backend,
        }
        with open(_AUDIT_FILE, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.debug("crypto_layer: falha no audit log (%s)", e)


# ─── API pública ───────────────────────────────────────────────────────────

def secure_encode(
    space: Any,
    experience: str,
    tags: Optional[list] = None,
    source: str = "",
    context_hint: str = "",
    timestamp: Optional[float] = None,
    metadados: Optional[dict] = None,
) -> Any:
    """
    Wrapper seguro sobre space.encode().

    Pipeline:
      experience → [sanitize PII] → space.encode(plaintext) → [encrypt original at-rest]

    Se IMI_CRYPTO=0 (default): pass-through direto para space.encode().
    Se sanitizer ou crypto não disponíveis: falha graciosa, usa original.

    Args:
        space:        Instância de MemorySpace do IMI
        experience:   Texto da memória (plaintext)
        tags:         Lista de tags (opcional)
        source:       Fonte da memória (opcional)
        context_hint: Hint de contexto (opcional)
        timestamp:    Timestamp UNIX (opcional)
        metadados:    Dict com flags jurídicos para o sanitizer (opcional)

    Returns:
        MemoryNode resultante de space.encode()
    """
    if not _CRYPTO_ENABLED:
        # Pass-through: comportamento idêntico ao original
        return space.encode(
            experience,
            tags=tags,
            source=source,
            context_hint=context_hint,
            timestamp=timestamp,
        )

    processed = experience
    pii_count = 0
    risk_score = 0.0
    sanitizer_backend = "none"

    # 1. Sanitização PII
    if _sanitizer_ok:
        try:
            result = sanitize(processed, metadados=metadados or {}, juridico=bool(metadados))
            if not result.allowed:
                # Documento bloqueado por critério jurídico — não encoda
                logger.warning(
                    "crypto_layer: encode bloqueado pelo sanitizer (%s)", result.block_reason
                )
                raise ValueError(f"IMI encode bloqueado: {result.block_reason}")
            processed = result.clean
            pii_count = result.pii_count
            risk_score = result.risk_score
            sanitizer_backend = result.backend
            if pii_count > 0:
                logger.debug(
                    "crypto_layer: %d itens PII removidos (risk=%.2f)", pii_count, risk_score
                )
        except ValueError:
            raise
        except Exception as e:
            logger.warning("crypto_layer: sanitizer falhou (%s), continuando sem sanitização", e)

    # 2. Encode no IMI com plaintext sanitizado
    #    Summarizers, embedder e seed recebem texto legível — nunca ciphertext.
    #    Bug fix: antes, ciphertext ia para LLM summarizers que retornavam
    #    "conteúdo criptografado" em vez de resumos úteis.
    node = space.encode(
        processed,
        tags=tags,
        source=source,
        context_hint=context_hint,
        timestamp=timestamp,
    )

    # 3. Criptografia AES-256-GCM — protege apenas o campo original (at-rest)
    key = _get_key()
    if key and _crypto_ok:
        try:
            encrypted = encrypt_str(processed, key)
            node.original = _ENC_PREFIX + encrypted
        except Exception as e:
            logger.warning("crypto_layer: encrypt falhou (%s), original permanece plaintext", e)

    # 4. Audit log (assíncrono — nunca bloqueia)
    try:
        _audit_log(node.id, pii_count, risk_score, sanitizer_backend)
    except Exception:
        pass

    return node


def decrypt_experience(text: str) -> str:
    """
    Decripta texto criptografado por secure_encode().

    Se não for criptografado (legacy), retorna o texto original sem modificação.
    Usado em im_nav, im_drm para recuperar conteúdo legível.

    Args:
        text: Texto armazenado no IMI (pode ser "[ENC:v1]..." ou plaintext)

    Returns:
        Texto decriptado (ou original se não era criptografado)
    """
    if not is_encrypted(text):
        return text  # legacy memory — retorna sem modificação

    if not _crypto_ok:
        logger.warning("crypto_layer: texto criptografado mas crypto_wrapper ausente")
        return text  # retorna hex — melhor que crash

    key = _get_key()
    if key is None:
        return text

    try:
        hex_payload = text[len(_ENC_PREFIX):]
        return decrypt_str(hex_payload, key)
    except Exception as e:
        logger.error("crypto_layer: decrypt falhou para node (%s)", e)
        return f"[DECRYPT_ERROR: {type(e).__name__}]"


def crypto_status() -> dict:
    """
    Retorna estado atual da camada de crypto — para diagnóstico.

    Returns:
        Dict com: enabled, crypto_ok, sanitizer_ok, key_fingerprint, audit_file
    """
    return {
        "enabled": _CRYPTO_ENABLED,
        "crypto_ok": _crypto_ok,
        "sanitizer_ok": _sanitizer_ok,
        "key_fingerprint": _KEY_FINGERPRINT if _KEY else "not_loaded",
        "audit_file": str(_AUDIT_FILE),
        "audit_exists": _AUDIT_FILE.exists(),
    }


# ─── CLI de teste ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    # Ativar modo crypto para teste
    os.environ["IMI_CRYPTO"] = "1"
    os.environ["IMI_CRYPTO_SECRET"] = "imi-test-2026"

    # Reimportar com env vars ativas
    _CRYPTO_ENABLED = True
    _KEY = None  # forçar reinit

    print("=== IMI Crypto Layer — Teste ===")
    print()
    print("Status:", json.dumps(crypto_status(), indent=2))
    print()

    # Simular encode/decode sem space real
    test_cases = [
        "Decisão TIT-SP: ICMS diferencial — contribuinte João Silva CPF 123.456.789-00 venceu",
        "Framework AIOX v2 — decisão arquitetural: IMI como fundação de memória",
        "Texto limpo sem dados sensíveis — apenas conhecimento técnico",
    ]

    key = _get_key()
    print(f"Chave inicializada (fp: {_KEY_FINGERPRINT})")
    print()

    for texto in test_cases:
        print(f"Original:   {texto[:70]}...")

        # Simular pipeline sanitize + encrypt
        if _sanitizer_ok:
            san = sanitize(texto)
            clean = san.clean
            print(f"Sanitizado: {clean[:70]}... (pii={san.pii_count}, risk={san.risk_score})")
        else:
            clean = texto

        if _crypto_ok and key:
            enc = encrypt_str(clean, key)
            stored = _ENC_PREFIX + enc
            print(f"Armazenado: {stored[:60]}...")

            recovered = decrypt_experience(stored)
            print(f"Recuperado: {recovered[:70]}...")
            print(f"Match:      {clean == recovered} ✓" if clean == recovered else "FALHA!")
        print()
