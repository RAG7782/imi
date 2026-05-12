"""LRU Embedding Cache — wraps any Embedder to cache repeated queries.

Avoids redundant embed() calls within a session. Particularly useful
when im_nav is called multiple times with similar or identical queries.

Usage (activated in IMISpace.__post_init__ when IMI_EMBED_CACHE=1)
------
    embedder = LRUEmbedderCache(create_embedder_from_env(), maxsize=256)

Measuring hit rate
------------------
    print(f"Cache hit rate: {embedder.cache_hit_rate:.1%}")

Env vars
--------
    IMI_EMBED_CACHE=1          enable (default: off for backward compat)
    IMI_EMBED_CACHE_SIZE=256   LRU capacity (number of distinct queries)
"""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from imi.embedder import Embedder

_CACHE_ENABLED: bool = os.getenv("IMI_EMBED_CACHE", "0") == "1"
_CACHE_SIZE: int = int(os.getenv("IMI_EMBED_CACHE_SIZE", "256"))


class LRUEmbedderCache:
    """LRU-evicting wrapper around any Embedder.

    Thread-safety: not thread-safe by design (MCP server is single-threaded
    per request). If multi-threading is needed, add threading.Lock.
    """

    def __init__(self, embedder: "Embedder", maxsize: int = _CACHE_SIZE) -> None:
        self._embedder = embedder
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._maxsize = maxsize
        self._hits = 0
        self._misses = 0

    # --- Embedder Protocol ---

    @property
    def dimensions(self) -> int:
        return self._embedder.dimensions

    def embed(self, text: str) -> np.ndarray:
        key = hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()
        if key in self._cache:
            self._cache.move_to_end(key)  # LRU: mark as recently used
            self._hits += 1
            return self._cache[key]
        emb = self._embedder.embed(text)
        self._cache[key] = emb
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)  # evict least-recently-used
        self._misses += 1
        return emb

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self.embed(t) for t in texts])

    # --- Diagnostics ---

    @property
    def cache_hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def stats(self) -> dict:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.cache_hit_rate, 3),
            "size": len(self._cache),
            "maxsize": self._maxsize,
        }


def wrap_with_cache(embedder: "Embedder") -> "Embedder | LRUEmbedderCache":
    """Return a cached wrapper if IMI_EMBED_CACHE=1, otherwise pass through."""
    if _CACHE_ENABLED:
        return LRUEmbedderCache(embedder, maxsize=_CACHE_SIZE)
    return embedder
