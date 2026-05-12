"""
FCM Security Layer — Camada 3 / AIP Security Layer
===================================================

Extensão da FCMBridge com:
  1. Deduplicação semântica por SHA-256 (conteúdo normalizado)
     Previne eventos duplicados num janela de tempo configurável.
  2. TriggerDetector — classifica importância de eventos (1-5)
     antes de federar: alert=5, milestone=4, decision=3, normal≤2
  3. Filtro de importância mínima — eventos abaixo do threshold não são emitidos

Uso (drop-in):

    from imi.integrations.fcm_security import SecureFCMBridge

    bridge = SecureFCMBridge(min_importance=3)  # só federa relevantes
    bridge.emit_encode(node)  # detecta importância + dedup automáticos

Ref: autosave/policy.py + autosave/triggers.py (Synapse Layer — padrão adaptado)
     Camada 3 do plano de integração Synapse Layer → ecossistema AIP
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from typing import Any

from .fcm_bridge import FCMBridge

logger = logging.getLogger("imi.fcm_security")


# ─── TriggerDetector (padrão Synapse autosave/triggers.py) ────────────────

_ALERT_PATTERN = re.compile(
    r"\b(crash(?:ed)?|error|breach|falhou|crítico|urgent[e]?|emergência|exception|"
    r"traceback|timeout|offline|down|security|broke|failed|CRITICAL|crashed)\b",
    re.IGNORECASE,
)

_MILESTONE_PATTERN = re.compile(
    r"\b(deployed|launched|publicado|lançado|entregue|concluí|"
    r"first.user|go.live|em.produção|milestone|100\%|complete[d]?)\b",
    re.IGNORECASE,
)

_DECISION_PATTERN = re.compile(
    r"\b(decidiu|decisão|decided|pivot|commit|arquitetura|"
    r"vamos.usar|adotar|escolhemos|migrando.para|deprecated|ADR)\b",
    re.IGNORECASE,
)


def _classify_importance(content: str) -> int:
    """
    Classifica importância 1-5 pelo conteúdo.

    5 = alert/segurança
    4 = milestone
    3 = decisão
    2 = informação relevante
    1 = ruído / baixa relevância
    """
    if _ALERT_PATTERN.search(content):
        return 5
    if _MILESTONE_PATTERN.search(content):
        return 4
    if _DECISION_PATTERN.search(content):
        return 3
    # Heurística de relevância: conteúdo com ≥5 palavras é relevante
    word_count = len(content.split())
    if word_count >= 5:
        return 2
    return 1


def _content_hash(content: str, source: str = "") -> str:
    """SHA-256 do conteúdo normalizado (lowercase, whitespace colapsado)."""
    normalized = re.sub(r"\s+", " ", content.lower().strip())
    payload = f"{source}:{normalized}"
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


# ─── SecureFCMBridge ──────────────────────────────────────────────────────


class SecureFCMBridge(FCMBridge):
    """
    FCMBridge com deduplicação SHA-256 + classificação de importância.

    Args:
        min_importance: Importância mínima para federar (1-5, default=2)
        dedup_window_s: Janela de dedup em segundos (default=60)
        source:         Identificador desta fonte
        trust_level:    Nível de confiança ('self', 'trusted', 'peer', 'external')
    """

    def __init__(
        self,
        min_importance: int = 2,
        dedup_window_s: float = 60.0,
        source: str = "imi",
        trust_level: str = "self",
    ):
        super().__init__(source=source, trust_level=trust_level)
        self.min_importance = min_importance
        self.dedup_window_s = dedup_window_s
        # {content_hash: timestamp} — LRU simples
        self._dedup_cache: dict[str, float] = {}

    def _is_duplicate(self, content: str) -> bool:
        """Retorna True se conteúdo idêntico foi emitido na janela de dedup."""
        h = _content_hash(content, self.source)
        # M5 fix: use time.time() instead of time.monotonic() so dedup
        # window survives across restarts (monotonic resets on restart)
        now = time.time()

        # Limpar entradas expiradas (máx 500 itens)
        if len(self._dedup_cache) > 500:
            expired = [k for k, ts in self._dedup_cache.items() if now - ts > self.dedup_window_s]
            for k in expired:
                del self._dedup_cache[k]

        if h in self._dedup_cache:
            age = now - self._dedup_cache[h]
            if age < self.dedup_window_s:
                logger.debug("fcm_security: dedup bloqueou evento (age=%.1fs)", age)
                return True

        self._dedup_cache[h] = now
        return False

    def emit_encode(
        self,
        node: Any,
        *,
        salience: float | None = None,
        extra_tags: list[str] | None = None,
    ) -> str | None:
        """
        emit_encode com pipeline de segurança:
          1. Extrai conteúdo do nó
          2. Classifica importância (1-5)
          3. Filtra por min_importance
          4. Verifica dedup SHA-256
          5. Delega para FCMBridge.emit_encode() com tags enriquecidas
        """
        # M2 fix: use seed (always plaintext) instead of original (may be ciphertext)
        content = getattr(node, "seed", "") or getattr(node, "summary_medium", "")
        if not content:
            return None

        # 1. Classificar importância
        importance = _classify_importance(content)

        # 2. Filtrar por importância mínima
        if importance < self.min_importance:
            logger.debug(
                "fcm_security: evento bloqueado por importância (%d < %d)",
                importance,
                self.min_importance,
            )
            return None

        # 3. Dedup SHA-256
        if self._is_duplicate(content):
            return None

        # 4. Enriquecer tags com categoria
        category_tag = {5: "auto-alert", 4: "auto-milestone", 3: "auto-decision"}.get(importance)
        if category_tag:
            extra_tags = list(extra_tags or []) + [category_tag]

        # 5. Delegar para bridge base
        result = super().emit_encode(node, salience=salience, extra_tags=extra_tags)
        if result:
            logger.debug(
                "fcm_security: evento federado (importance=%d, categoria=%s)",
                importance,
                category_tag or "normal",
            )
        return result
