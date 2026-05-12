"""Topological Data Analysis — rigorous shape analysis of memory space.

Uses persistent homology to compute:
  - H0 (connected components): how fragmented is knowledge?
  - H1 (loops): circular reasoning / rumination?
  - Persistence: which features are structural vs noise?

Persistence diagram = diagnostic of cognitive health.

Also implements dreaming-as-annealing with convergence monitoring.

Based on: Edelsbrunner & Harer (2010), ripser (Bauer, 2021).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PersistenceFeature:
    """A topological feature with birth and death times."""

    dimension: int  # 0 = component, 1 = loop, 2 = void
    birth: float
    death: float

    @property
    def persistence(self) -> float:
        """How long this feature survives = how 'real' it is."""
        if np.isinf(self.death):
            return float("inf")
        return self.death - self.birth

    @property
    def is_significant(self, threshold: float = 0.1) -> bool:
        return self.persistence > threshold


@dataclass
class TDAReport:
    """Full topological analysis of the memory space."""

    # Betti numbers (counts of features per dimension)
    betti_0: int = 0  # connected components
    betti_1: int = 0  # loops
    betti_2: int = 0  # voids/cavities

    # Persistence features
    features: list[PersistenceFeature] = field(default_factory=list)

    # Diagnostic interpretation
    fragmentation: float = 0.0  # 0 = fully connected, 1 = fully fragmented
    rumination_risk: float = 0.0  # 0 = no loops, 1 = many persistent loops
    structural_stability: float = 0.0  # avg persistence of significant features

    # Raw persistence diagrams (for visualization)
    diagrams: list[np.ndarray] = field(default_factory=list, repr=False)

    def __str__(self) -> str:
        lines = [
            "TDA Report:",
            f"  Betti numbers: H0={self.betti_0} (components), "
            f"H1={self.betti_1} (loops), H2={self.betti_2} (voids)",
            f"  Fragmentation:  {self.fragmentation:.0%}",
            f"  Rumination risk: {self.rumination_risk:.0%}",
            f"  Stability:       {self.structural_stability:.2f}",
        ]
        # Interpretation
        if self.fragmentation > 0.7:
            lines.append("  [!] Knowledge is highly fragmented — consider bridging topics")
        if self.rumination_risk > 0.5:
            lines.append("  [!] Persistent loops detected — possible circular reasoning")
        if self.betti_0 == 1:
            lines.append("  [ok] Knowledge is fully connected")
        return "\n".join(lines)


def compute_persistent_homology(
    embeddings: np.ndarray,
    max_dimension: int = 2,
) -> TDAReport:
    """Compute persistent homology of the memory space.

    Uses ripser for Vietoris-Rips persistent homology.
    Falls back to distance-based heuristics if ripser unavailable.
    """
    if len(embeddings) < 3:
        return TDAReport(betti_0=len(embeddings))

    try:
        from ripser import ripser

        result = ripser(
            embeddings,
            maxdim=min(max_dimension, 2),
            metric="cosine",
        )
        diagrams = result["dgms"]

        features = []
        betti = [0, 0, 0]

        for dim, dgm in enumerate(diagrams):
            if dim > 2:
                break
            for birth, death in dgm:
                feat = PersistenceFeature(
                    dimension=dim,
                    birth=float(birth),
                    death=float(death),
                )
                features.append(feat)
                if feat.persistence > 0.05 or np.isinf(death):
                    betti[dim] += 1

        # Compute diagnostics
        n_points = len(embeddings)
        fragmentation = max(0.0, (betti[0] - 1) / max(1, n_points - 1))

        # Rumination: ratio of persistent H1 features
        h1_features = [f for f in features if f.dimension == 1 and f.persistence > 0.1]
        rumination_risk = min(1.0, len(h1_features) / max(1, n_points // 3))

        # Stability: average persistence of significant features
        significant = [f for f in features if f.persistence > 0.05 and not np.isinf(f.persistence)]
        structural_stability = np.mean([f.persistence for f in significant]) if significant else 0.0

        return TDAReport(
            betti_0=betti[0],
            betti_1=betti[1],
            betti_2=betti[2],
            features=features,
            fragmentation=fragmentation,
            rumination_risk=rumination_risk,
            structural_stability=float(structural_stability),
            diagrams=[dgm.copy() for dgm in diagrams],
        )

    except ImportError:
        # Fallback: basic distance analysis
        from sklearn.metrics import pairwise_distances

        dists = pairwise_distances(embeddings, metric="cosine")
        # Approximate H0 by counting connected components at threshold
        threshold = np.median(dists)
        adjacency = dists < threshold
        visited = set()
        components = 0
        for i in range(len(embeddings)):
            if i not in visited:
                components += 1
                stack = [i]
                while stack:
                    node = stack.pop()
                    if node in visited:
                        continue
                    visited.add(node)
                    for j in range(len(embeddings)):
                        if adjacency[i, j] and j not in visited:
                            stack.append(j)

        return TDAReport(
            betti_0=components,
            fragmentation=max(0.0, (components - 1) / max(1, len(embeddings) - 1)),
        )


@dataclass
class AnnealingState:
    """Tracks convergence of dreaming-as-simulated-annealing."""

    iteration: int = 0
    temperature: float = 1.0
    energy_history: list[float] = field(default_factory=list)
    converged: bool = False

    @property
    def energy(self) -> float:
        return self.energy_history[-1] if self.energy_history else float("inf")

    @property
    def energy_delta(self) -> float:
        if len(self.energy_history) < 2:
            return float("inf")
        return abs(self.energy_history[-1] - self.energy_history[-2])

    def step(self, new_energy: float, cooling_rate: float = 0.95) -> None:
        """Record one annealing step."""
        self.iteration += 1
        self.energy_history.append(new_energy)
        self.temperature *= cooling_rate

        # Convergence: energy stable for 5 iterations
        if len(self.energy_history) >= 5:
            recent = self.energy_history[-5:]
            variation = max(recent) - min(recent)
            if variation < 0.01:
                self.converged = True

    def __str__(self) -> str:
        status = "CONVERGED" if self.converged else "running"
        return (
            f"Annealing [{status}]: iter={self.iteration}, "
            f"T={self.temperature:.4f}, E={self.energy:.4f}, "
            f"dE={self.energy_delta:.4f}"
        )


def compute_space_energy(
    embeddings: np.ndarray,
    masses: np.ndarray,
) -> float:
    """Compute the 'energy' of the memory space.

    Energy = sum of weighted pairwise distances.
    Lower energy = more organized space.
    Used as the objective for simulated annealing.
    """
    if len(embeddings) < 2:
        return 0.0

    from sklearn.metrics import pairwise_distances

    dists = pairwise_distances(embeddings, metric="cosine")

    # Weighted by mass product (massive nodes closer together = lower energy)
    mass_product = np.outer(masses, masses)
    energy = np.sum(dists * mass_product) / 2  # symmetric, avoid double counting
    return float(energy)
