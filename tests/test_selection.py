"""Tests for selection logic."""
import pytest
import numpy as np
from seedgraph.core.selection import compute_softmax, kl_divergence, CoverageSelector


def test_compute_softmax():
    """Test softmax computation."""
    logits = np.array([1.0, 2.0, 3.0])
    probs = compute_softmax(logits)

    # Should sum to 1
    assert np.isclose(probs.sum(), 1.0)

    # Should be in (0, 1)
    assert all(0 < p < 1 for p in probs)

    # Higher logit should give higher prob
    assert probs[2] > probs[1] > probs[0]


def test_kl_divergence_properties():
    """Test KL divergence properties."""
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.4, 0.4, 0.2])

    # KL divergence should be non-negative
    kl = kl_divergence(p, q)
    assert kl >= 0

    # KL(P || P) should be 0
    kl_self = kl_divergence(p, p)
    assert kl_self < 1e-6  # Nearly zero

    # KL divergence is not symmetric
    kl_pq = kl_divergence(p, q)
    kl_qp = kl_divergence(q, p)
    # These will generally be different
    assert isinstance(kl_pq, float)
    assert isinstance(kl_qp, float)


def test_kl_divergence_zero_handling():
    """Test KL divergence with near-zero probabilities."""
    p = np.array([0.9, 0.09, 0.01])
    q = np.array([0.5, 0.5, 0.0])  # Has a zero

    # Should not crash with log(0)
    kl = kl_divergence(p, q)
    assert np.isfinite(kl)
    assert kl >= 0


def test_coverage_selector_init():
    """Test CoverageSelector initialization."""
    vocab_size = 1000
    selector = CoverageSelector(vocab_size=vocab_size, use_pca=True, pca_dims=256)

    assert selector.vocab_size == vocab_size
    assert selector.pca_dims == 256
    assert selector.use_pca is True
    assert selector.index is None  # Not initialized yet
    assert len(selector.centroids) == 0


def test_coverage_selector_update_index():
    """Test updating FAISS index."""
    vocab_size = 100
    selector = CoverageSelector(vocab_size=vocab_size, use_pca=False)

    # Create some random probability distributions
    embeddings = [np.random.dirichlet(np.ones(vocab_size)) for _ in range(10)]

    selector.update_index(embeddings)

    # Index should be created
    assert selector.index is not None
    assert selector.index.ntotal == 10


def test_coverage_selector_priority():
    """Test priority computation."""
    vocab_size = 50
    selector = CoverageSelector(vocab_size=vocab_size, use_pca=False)

    # Add some embeddings
    embeddings = [np.random.dirichlet(np.ones(vocab_size)) for _ in range(5)]
    selector.update_index(embeddings)
    selector.update_centroids(embeddings)

    # Compute priority for a new distribution
    new_dist = np.random.dirichlet(np.ones(vocab_size))
    priority = selector.priority(new_dist)

    # Should return a finite number
    assert np.isfinite(priority)
    assert priority >= 0


def test_coverage_selector_select_next_node():
    """Test node selection."""
    vocab_size = 50
    selector = CoverageSelector(vocab_size=vocab_size, use_pca=False)

    # Add some embeddings
    embeddings = [np.random.dirichlet(np.ones(vocab_size)) for _ in range(5)]
    selector.update_index(embeddings)
    selector.update_centroids(embeddings)

    # Create candidate nodes
    candidate_ids = [0, 1, 2]
    candidate_probs = [np.random.dirichlet(np.ones(vocab_size)) for _ in range(3)]

    # Select best node
    best_id = selector.select_next_node(candidate_ids, candidate_probs)

    # Should return one of the candidates
    assert best_id in candidate_ids


def test_coverage_selector_single_candidate():
    """Test selection with single candidate."""
    vocab_size = 50
    selector = CoverageSelector(vocab_size=vocab_size)

    candidate_ids = [42]
    candidate_probs = [np.random.dirichlet(np.ones(vocab_size))]

    # Should return the only candidate
    best_id = selector.select_next_node(candidate_ids, candidate_probs)
    assert best_id == 42


def test_coverage_selector_with_pca():
    """Test CoverageSelector with PCA enabled."""
    vocab_size = 200
    # PCA dims must be <= number of samples, so use fewer dims
    selector = CoverageSelector(vocab_size=vocab_size, use_pca=True, pca_dims=5)

    # Need at least 2 samples to fit PCA, use more than pca_dims
    embeddings = [np.random.dirichlet(np.ones(vocab_size)) for _ in range(10)]
    selector.update_index(embeddings)

    # PCA should be fitted now
    assert selector.pca_fitted is True

    # Priority computation should work
    new_dist = np.random.dirichlet(np.ones(vocab_size))
    priority = selector.priority(new_dist)
    assert np.isfinite(priority)
