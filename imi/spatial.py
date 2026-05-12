"""Spatial projection and topology analysis.

Projects high-dimensional embeddings into navigable 3D space.
Analyzes topology: clusters, bridges, voids, anomalies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TopologyReport:
    """Metacognitive snapshot of the memory space."""

    n_clusters: int = 0
    cluster_labels: list[int] = field(default_factory=list)
    cluster_sizes: dict[int, int] = field(default_factory=dict)
    isolated_count: int = 0  # nodes not in any cluster
    density_scores: list[float] = field(default_factory=list)
    bridges: list[tuple[str, str]] = field(default_factory=list)  # node ids

    def __str__(self) -> str:
        lines = [
            f"Topology: {self.n_clusters} clusters, {self.isolated_count} isolated nodes",
        ]
        for cid, size in sorted(self.cluster_sizes.items()):
            if cid == -1:
                continue
            lines.append(f"  Cluster {cid}: {size} nodes")
        if self.bridges:
            lines.append(f"  Bridges: {len(self.bridges)} connections between clusters")
        return "\n".join(lines)


@dataclass
class SpatialIndex:
    """Projects embeddings into low-dimensional navigable space.

    Uses UMAP for dimensionality reduction and HDBSCAN for clustering.
    Falls back to PCA + KMeans if spatial dependencies not installed.
    """

    positions: np.ndarray | None = field(default=None, repr=False)
    labels: np.ndarray | None = field(default=None, repr=False)
    node_ids: list[str] = field(default_factory=list)

    def project(
        self,
        embeddings: np.ndarray,
        node_ids: list[str],
        n_components: int = 3,
    ) -> np.ndarray:
        """Project high-dim embeddings to low-dim navigable space."""
        self.node_ids = node_ids

        if len(embeddings) < 4:
            # Too few points for UMAP — use PCA fallback
            from sklearn.decomposition import PCA

            n_comp = min(n_components, len(embeddings), embeddings.shape[1])
            pca = PCA(n_components=n_comp)
            self.positions = pca.fit_transform(embeddings)
            return self.positions

        try:
            import umap

            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=min(15, len(embeddings) - 1),
                min_dist=0.1,
                metric="cosine",
            )
            self.positions = reducer.fit_transform(embeddings)
        except ImportError:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=n_components)
            self.positions = pca.fit_transform(embeddings)

        return self.positions

    def cluster(self, min_cluster_size: int = 2) -> np.ndarray:
        """Detect clusters in the projected space."""
        if self.positions is None:
            return np.array([])

        if len(self.positions) < 4:
            self.labels = np.zeros(len(self.positions), dtype=int)
            return self.labels

        try:
            from sklearn.cluster import HDBSCAN

            clusterer = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=1,
            )
            self.labels = clusterer.fit_predict(self.positions)
        except ImportError:
            from sklearn.cluster import KMeans

            k = min(3, len(self.positions))
            km = KMeans(n_clusters=k, n_init="auto", random_state=42)
            self.labels = km.fit_predict(self.positions)

        return self.labels

    def topology(self) -> TopologyReport:
        """Analyze the topology of the memory space."""
        if self.labels is None:
            self.cluster()
        if self.labels is None or len(self.labels) == 0:
            return TopologyReport()

        unique_labels = set(self.labels)
        cluster_sizes = {}
        for lab in unique_labels:
            cluster_sizes[int(lab)] = int(np.sum(self.labels == lab))

        isolated = cluster_sizes.pop(-1, 0)
        n_clusters = len(cluster_sizes)

        # Detect bridges: nodes closest to another cluster's centroid
        bridges = []
        if n_clusters >= 2 and self.positions is not None:
            centroids = {}
            for lab in cluster_sizes:
                mask = self.labels == lab
                centroids[lab] = self.positions[mask].mean(axis=0)

            for i, (pos, lab) in enumerate(zip(self.positions, self.labels)):
                if lab == -1:
                    continue
                # Distance to own centroid
                own_dist = np.linalg.norm(pos - centroids[lab])
                # Distance to nearest other centroid
                for other_lab, other_cent in centroids.items():
                    if other_lab == lab:
                        continue
                    other_dist = np.linalg.norm(pos - other_cent)
                    if other_dist < own_dist * 1.5:  # close to another cluster
                        bridges.append((self.node_ids[i], f"cluster_{other_lab}"))

        return TopologyReport(
            n_clusters=n_clusters,
            cluster_labels=[int(x) for x in self.labels],
            cluster_sizes=cluster_sizes,
            isolated_count=isolated,
            bridges=bridges,
        )

    def get_position(self, node_id: str) -> np.ndarray | None:
        """Get the 3D position of a specific node."""
        if node_id in self.node_ids and self.positions is not None:
            idx = self.node_ids.index(node_id)
            return self.positions[idx]
        return None
