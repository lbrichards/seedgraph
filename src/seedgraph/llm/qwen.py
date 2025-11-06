"""Qwen-2.5-0.5B logit introspection."""
from typing import Dict, List, Any
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger


class QwenGenerator:
    """
    Qwen-2.5-0.5B logit introspection.

    Extracts next-token distributions from the model for branching logic.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B",
        device: str = None,
        dtype: torch.dtype = torch.float16
    ):
        """
        Initialize Qwen generator.

        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on (None = auto)
            dtype: Data type for model weights
        """
        logger.info(f"Loading model: {model_name}")

        # Load model with automatic device mapping
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map="auto" if device is None else device,
            trust_remote_code=True
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        self.model.eval()  # Set to evaluation mode
        logger.info(f"Model loaded on device: {self.model.device}")

    def next_token_distribution(
        self,
        prompt: str,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Compute next-token distribution for a given prompt.

        Args:
            prompt: Input text context
            top_k: Number of top candidates to return

        Returns:
            Dictionary with:
                - topk: List of dicts with token, id, prob, logit
                - probs: Full probability distribution (numpy array)
                - logits: Full logit distribution (numpy array)
        """
        # Tokenize input
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).input_ids.to(self.model.device)

        # Get logits (no gradient needed)
        with torch.no_grad():
            outputs = self.model(input_ids)
            # Extract logits for next token (last position)
            logits = outputs.logits[:, -1, :].float().cpu().numpy().flatten()

        # Compute softmax probabilities
        # Subtract max for numerical stability
        logits_shifted = logits - logits.max()
        probs = np.exp(logits_shifted)
        probs = probs / probs.sum()

        # Get top-k tokens
        top_indices = np.argsort(probs)[::-1][:top_k]

        topk_results = []
        for idx in top_indices:
            token_text = self.tokenizer.decode([idx])
            topk_results.append({
                "token": token_text,
                "id": int(idx),
                "prob": float(probs[idx]),
                "logit": float(logits[idx])
            })

        return {
            "topk": topk_results,
            "probs": probs,
            "logits": logits
        }

    def append_token(self, prompt: str, token_id: int) -> str:
        """
        Append a token to a prompt string.

        Args:
            prompt: Current prompt text
            token_id: Token ID to append

        Returns:
            New prompt with token appended
        """
        token_text = self.tokenizer.decode([token_id])
        # Don't strip! Whitespace tokens (spaces, newlines) are meaningful
        return prompt + token_text
