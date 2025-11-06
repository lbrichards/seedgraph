"""Tests for Qwen logit introspection."""
import pytest
import numpy as np
from seedgraph.llm.qwen import QwenGenerator


# Mark as requiring model download
@pytest.mark.slow
def test_qwen_generator_init():
    """Test QwenGenerator initialization."""
    gen = QwenGenerator()
    assert gen.model is not None
    assert gen.tokenizer is not None


@pytest.mark.slow
def test_next_token_distribution():
    """Test next-token distribution extraction."""
    gen = QwenGenerator()
    prompt = "AI connects biology and language"

    result = gen.next_token_distribution(prompt, top_k=10)

    # Verify structure
    assert "topk" in result
    assert "probs" in result
    assert "logits" in result

    # Check top-k results
    assert len(result["topk"]) == 10

    # Each top-k entry should have required fields
    for item in result["topk"]:
        assert "token" in item
        assert "id" in item
        assert "prob" in item
        assert "logit" in item
        assert isinstance(item["prob"], float)
        assert isinstance(item["logit"], float)
        assert 0.0 <= item["prob"] <= 1.0

    # Probabilities should sum to approximately 1.0
    total_prob = np.sum(result["probs"])
    assert np.isclose(total_prob, 1.0, rtol=1e-5), f"Probabilities sum to {total_prob}, not 1.0"

    # Top-k probs should be sorted descending
    topk_probs = [item["prob"] for item in result["topk"]]
    assert topk_probs == sorted(topk_probs, reverse=True)


@pytest.mark.slow
def test_append_token():
    """Test token appending to prompt."""
    gen = QwenGenerator()
    prompt = "AI connects biology"

    # Get next token distribution
    result = gen.next_token_distribution(prompt, top_k=5)
    first_token_id = result["topk"][0]["id"]

    # Append the top token
    new_prompt = gen.append_token(prompt, first_token_id)

    # New prompt should be different and longer (or same length if stripped)
    assert isinstance(new_prompt, str)
    assert len(new_prompt) >= len(prompt)


@pytest.mark.slow
def test_distribution_consistency():
    """Test that same prompt gives consistent distributions."""
    gen = QwenGenerator()
    prompt = "SeedGraph builds graphs from logits"

    # Run twice
    result1 = gen.next_token_distribution(prompt, top_k=10)
    result2 = gen.next_token_distribution(prompt, top_k=10)

    # Top tokens should be identical (deterministic)
    ids1 = [item["id"] for item in result1["topk"]]
    ids2 = [item["id"] for item in result2["topk"]]

    assert ids1 == ids2, "Distribution should be deterministic"


# Fast test that doesn't require model loading
def test_qwen_import():
    """Test that QwenGenerator can be imported."""
    from seedgraph.llm.qwen import QwenGenerator
    assert QwenGenerator is not None
