"""Diversity filtering using embeddings and k-center selection."""
from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import faiss


class DiversityFilter:
    """Filter seeds for maximum diversity using embeddings."""

    def __init__(
        self,
        near_dup_threshold: float = 0.92,
        target_total_seeds: int = 3000,
        per_domain_tolerance: float = 0.02
    ):
        """
        Initialize diversity filter.

        Args:
            near_dup_threshold: Cosine similarity threshold for near-duplicates
            target_total_seeds: Target number of final seeds
            per_domain_tolerance: Tolerance for domain balance (±2%)
        """
        self.near_dup_threshold = near_dup_threshold
        self.target_total_seeds = target_total_seeds
        self.per_domain_tolerance = per_domain_tolerance

    def compute_embeddings(self, seeds: List[str]) -> np.ndarray:
        """
        Compute sentence embeddings for seeds.

        Uses simple averaged word embeddings as placeholder.
        For production, use sentence-transformers or similar.

        Args:
            seeds: List of seed strings

        Returns:
            Embedding matrix (N x D)
        """
        # Placeholder: Use hash-based random embeddings
        # In production, replace with actual sentence-transformers
        embeddings = []
        np.random.seed(42)

        for seed in seeds:
            # Simple hash-based embedding (deterministic)
            seed_hash = hash(seed) % (2**31)
            np.random.seed(seed_hash)
            embedding = np.random.randn(384)  # 384-dim like MiniLM
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)

        return np.array(embeddings)

    def remove_near_duplicates(
        self,
        seeds: List[Dict],
        embeddings: np.ndarray
    ) -> Tuple[List[Dict], np.ndarray]:
        """
        Remove near-duplicate seeds based on cosine similarity.

        Args:
            seeds: List of seed dictionaries
            embeddings: Embedding matrix

        Returns:
            Filtered seeds and embeddings
        """
        n = len(seeds)
        keep_mask = np.ones(n, dtype=bool)

        # Compute pairwise similarities
        similarities = cosine_similarity(embeddings)

        # Remove near-duplicates (keep first occurrence)
        for i in range(n):
            if not keep_mask[i]:
                continue

            for j in range(i + 1, n):
                if not keep_mask[j]:
                    continue

                if similarities[i, j] >= self.near_dup_threshold:
                    keep_mask[j] = False

        filtered_seeds = [seeds[i] for i in range(n) if keep_mask[i]]
        filtered_embeddings = embeddings[keep_mask]

        return filtered_seeds, filtered_embeddings

    def k_center_selection(
        self,
        seeds: List[Dict],
        embeddings: np.ndarray,
        k: int
    ) -> Tuple[List[Dict], np.ndarray]:
        """
        Select k diverse seeds using k-center greedy algorithm.

        Args:
            seeds: Candidate seeds
            embeddings: Embedding matrix
            k: Number of seeds to select

        Returns:
            Selected seeds and embeddings
        """
        if len(seeds) <= k:
            return seeds, embeddings

        n = len(seeds)
        selected_indices = []

        # Start with a random seed
        current_idx = np.random.randint(0, n)
        selected_indices.append(current_idx)

        # Compute distances to selected set
        min_distances = np.full(n, np.inf)

        for _ in range(k - 1):
            # Update minimum distances
            current_emb = embeddings[current_idx:current_idx+1]
            distances = 1 - cosine_similarity(embeddings, current_emb).flatten()
            min_distances = np.minimum(min_distances, distances)

            # Select point with maximum distance to selected set
            current_idx = np.argmax(min_distances)
            selected_indices.append(current_idx)

        selected_seeds = [seeds[i] for i in selected_indices]
        selected_embeddings = embeddings[selected_indices]

        return selected_seeds, selected_embeddings

    def balance_domains(
        self,
        seeds: List[Dict],
        embeddings: np.ndarray,
        num_domains: int = 12
    ) -> Tuple[List[Dict], np.ndarray]:
        """
        Balance seeds across domains.

        Args:
            seeds: Candidate seeds
            embeddings: Embedding matrix
            num_domains: Number of domains

        Returns:
            Balanced seeds and embeddings
        """
        target_per_domain = self.target_total_seeds // num_domains
        tolerance = int(target_per_domain * self.per_domain_tolerance)

        # Group by domain
        domain_seeds = {}
        domain_embeddings = {}

        for i, seed in enumerate(seeds):
            domain = seed['domain']
            if domain not in domain_seeds:
                domain_seeds[domain] = []
                domain_embeddings[domain] = []

            domain_seeds[domain].append(seed)
            domain_embeddings[domain].append(embeddings[i])

        # Balance each domain
        balanced_seeds = []
        balanced_embeddings = []

        for domain in sorted(domain_seeds.keys()):
            domain_list = domain_seeds[domain]
            domain_embs = np.array(domain_embeddings[domain])

            # Sample to target (with tolerance)
            n_domain = len(domain_list)
            target = min(target_per_domain + tolerance, n_domain)

            if n_domain > target:
                # Use k-center to select diverse subset
                selected, selected_embs = self.k_center_selection(
                    domain_list, domain_embs, target
                )
            else:
                selected = domain_list
                selected_embs = domain_embs

            balanced_seeds.extend(selected)
            balanced_embeddings.append(selected_embs)

        balanced_embeddings = np.vstack(balanced_embeddings)

        return balanced_seeds, balanced_embeddings

    def compute_diagnostics(
        self,
        seeds: List[Dict],
        embeddings: np.ndarray
    ) -> Dict:
        """
        Compute diversity diagnostics.

        Args:
            seeds: Final seeds
            embeddings: Embedding matrix

        Returns:
            Dictionary of diagnostic metrics
        """
        # Domain distribution
        domain_counts = {}
        for seed in seeds:
            domain = seed['domain']
            domain_counts[domain] = domain_counts.get(domain, 0) + 1

        # Pairwise similarities
        similarities = cosine_similarity(embeddings)
        # Get upper triangle (excluding diagonal)
        triu_indices = np.triu_indices_from(similarities, k=1)
        pairwise_sims = similarities[triu_indices]

        # Similarity percentiles
        percentiles = {
            "50th": float(np.percentile(pairwise_sims, 50)),
            "75th": float(np.percentile(pairwise_sims, 75)),
            "90th": float(np.percentile(pairwise_sims, 90)),
            "95th": float(np.percentile(pairwise_sims, 95)),
            "99th": float(np.percentile(pairwise_sims, 99))
        }

        # KMeans utilization
        n_clusters = min(256, len(seeds) // 10)
        if len(seeds) >= n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            unique_labels, counts = np.unique(labels, return_counts=True)
            clusters_with_5plus = int(np.sum(counts >= 5))
        else:
            clusters_with_5plus = 0

        # Top-100 nearest neighbor average distance
        if len(seeds) >= 100:
            # For each point, find distance to 100th nearest neighbor
            distances = 1 - similarities  # Convert similarity to distance
            sorted_distances = np.sort(distances, axis=1)
            # 100th nearest (index 100, since index 0 is self)
            top100_distances = sorted_distances[:, min(100, len(seeds) - 1)]
            avg_top100_dist = float(np.mean(top100_distances))
        else:
            avg_top100_dist = 0.0

        return {
            "total_seeds": len(seeds),
            "domain_counts": domain_counts,
            "pairwise_similarity_percentiles": percentiles,
            "kmeans_clusters_with_5plus": clusters_with_5plus,
            "kmeans_total_clusters": n_clusters,
            "avg_top100_neighbor_distance": avg_top100_dist
        }
