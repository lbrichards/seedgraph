"""KL divergence and coverage-based node selection."""
from typing import List, Optional
import numpy as np
import faiss
from sklearn.decomposition import PCA
from loguru import logger


def compute_softmax(logits: np.ndarray) -> np.ndarray:
    """
    Compute softmax probabilities from logits.

    Args:
        logits: Logit array

    Returns:
        Probability distribution
    """
    logits_shifted = logits - logits.max()
    exp_logits = np.exp(logits_shifted)
    return exp_logits / exp_logits.sum()


def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute KL divergence: D_KL(P || Q).

    Args:
        p: Probability distribution P
        q: Probability distribution Q
        epsilon: Small value to avoid log(0)

    Returns:
        KL divergence value (always >= 0)
    """
    # Clip to avoid log(0)
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)

    # KL divergence formula: sum(p * log(p / q))
    kl = np.sum(p * np.log(p / q))
    return float(max(0.0, kl))  # Ensure non-negative


class CoverageSelector:
    """
    FAISS-based coverage selector for node expansion.

    Uses hybrid approach:
    - L2 distance in PCA-reduced logit space (FAISS)
    - KL divergence from centroids
    """

    def __init__(
        self,
        vocab_size: int,
        use_pca: bool = True,
        pca_dims: int = 256,
        update_interval: int = 50
    ):
        """
        Initialize coverage selector.

        Args:
            vocab_size: Size of vocabulary (dimension of logit vectors)
            use_pca: Whether to use PCA for dimensionality reduction
            pca_dims: Number of PCA dimensions
            update_interval: How often to update centroids
        """
        self.vocab_size = vocab_size
        self.use_pca = use_pca
        self.pca_dims = min(pca_dims, vocab_size)  # Can't exceed vocab size
        self.update_interval = update_interval

        # PCA for dimensionality reduction
        self.pca: Optional[PCA] = PCA(n_components=self.pca_dims) if use_pca else None
        self.pca_fitted = False

        # FAISS index for L2 distance
        self.index: Optional[faiss.IndexFlatL2] = None
        self.indexed_embeddings: List[np.ndarray] = []

        # Centroids for KL divergence
        self.centroids: List[np.ndarray] = []

        self.num_updates = 0
        logger.info(f"CoverageSelector initialized: PCA={use_pca}, dims={self.pca_dims}")

    def _prepare_embedding(self, probs: np.ndarray) -> np.ndarray:
        """
        Prepare embedding for FAISS (apply PCA if enabled).

        Args:
            probs: Probability distribution

        Returns:
            Embedding vector
        """
        if not self.use_pca:
            return probs.astype('float32')

        # If PCA not fitted yet, return truncated probs matching index dimension
        if not self.pca_fitted:
            # Match the dimension used by the index
            if self.index is not None:
                target_dim = self.index.d
                return probs.astype('float32')[:target_dim]
            else:
                # No index yet, use pca_dims
                return probs.astype('float32')[:self.pca_dims]

        # Transform using fitted PCA
        embedding = self.pca.transform(probs.reshape(1, -1)).flatten()
        return embedding.astype('float32')

    def update_index(self, embeddings: List[np.ndarray]) -> None:
        """
        Update FAISS index with new embeddings.

        Args:
            embeddings: List of probability distributions
        """
        if not embeddings:
            return

        embeddings_array = np.array(embeddings).astype('float32')
        n_samples = embeddings_array.shape[0]

        # Fit PCA if using and not yet fitted
        # Need at least pca_dims samples to fit PCA
        if self.use_pca and not self.pca_fitted:
            if n_samples >= self.pca_dims:
                self.pca.fit(embeddings_array)
                self.pca_fitted = True
                logger.info(f"PCA fitted with {n_samples} samples to {self.pca_dims} dimensions")
            else:
                # Not enough samples yet, accumulate for later
                logger.debug(f"Accumulating samples for PCA fit: have {n_samples}, need {self.pca_dims}")
                self.indexed_embeddings.extend(embeddings)
                return

        # Transform embeddings
        if self.use_pca and self.pca_fitted:
            embeddings_array = self.pca.transform(embeddings_array).astype('float32')
        elif not self.use_pca:
            # No PCA, use full vectors
            pass

        # Create FAISS index on first call after PCA is ready
        if self.index is None:
            dim = embeddings_array.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            logger.info(f"Created FAISS index with dimension {dim}")

            # Add any previously accumulated embeddings
            if self.indexed_embeddings:
                logger.info(f"Adding {len(self.indexed_embeddings)} accumulated samples to FAISS index")
                accumulated_array = np.array(self.indexed_embeddings).astype('float32')
                if self.use_pca and self.pca_fitted:
                    accumulated_array = self.pca.transform(accumulated_array).astype('float32')
                self.index.add(accumulated_array)

        self.index.add(embeddings_array)
        self.indexed_embeddings.extend(embeddings)
        self.num_updates += 1

        logger.debug(f"Updated FAISS index: {self.index.ntotal} vectors")

    def update_centroids(self, embeddings: List[np.ndarray], k: int = 5) -> None:
        """
        Update centroids from embeddings.

        Args:
            embeddings: List of probability distributions
            k: Number of centroids (clusters)
        """
        if len(embeddings) < k:
            # Not enough data, use all as centroids
            self.centroids = [np.array(e) for e in embeddings]
        else:
            # Use k-means-like approach: sample k points uniformly
            indices = np.linspace(0, len(embeddings) - 1, k, dtype=int)
            self.centroids = [np.array(embeddings[i]) for i in indices]

        logger.debug(f"Updated {len(self.centroids)} centroids")

    def priority(self, probs: np.ndarray) -> float:
        """
        Compute expansion priority for a node.

        Higher priority = farther from existing coverage.

        Args:
            probs: Probability distribution for node

        Returns:
            Priority score (higher = expand first)
        """
        priority_score = 0.0

        # Component 1: FAISS-based L2 distance
        if self.index is not None and self.index.ntotal > 0:
            embedding = self._prepare_embedding(probs)
            D, I = self.index.search(embedding.reshape(1, -1), k=1)
            l2_dist = float(D[0][0])
            priority_score += l2_dist

        # Component 2: KL divergence from centroids
        if self.centroids:
            min_kl = min(
                kl_divergence(probs, centroid)
                for centroid in self.centroids
            )
            priority_score += min_kl

        return priority_score

    def select_next_node(
        self,
        candidate_ids: List[int],
        candidate_probs: List[np.ndarray]
    ) -> int:
        """
        Select which node to expand next.

        Args:
            candidate_ids: List of candidate node IDs
            candidate_probs: Corresponding probability distributions

        Returns:
            Node ID with highest priority
        """
        if not candidate_ids:
            raise ValueError("No candidate nodes to select from")

        if len(candidate_ids) == 1:
            return candidate_ids[0]

        # Compute priorities
        priorities = [self.priority(probs) for probs in candidate_probs]

        # Select node with highest priority
        best_idx = np.argmax(priorities)
        best_id = candidate_ids[best_idx]

        logger.debug(f"Selected node {best_id} with priority {priorities[best_idx]:.4f}")
        return best_id
