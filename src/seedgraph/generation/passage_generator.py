"""Generate long-form passages from diverse seed prompts."""
from typing import Dict, List, Optional
import torch
from loguru import logger

from seedgraph.llm.qwen import QwenGenerator


class PassageGenerator:
    """Generates long-form text passages from seed prompts."""

    def __init__(
        self,
        generator: QwenGenerator,
        target_words: int = 1000,
        min_words: int = 500,
        max_tokens: int = 5000,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50
    ):
        """
        Initialize passage generator.

        Args:
            generator: Qwen generator instance
            target_words: Target word count for passages
            min_words: Minimum acceptable word count
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
        """
        self.generator = generator
        self.target_words = target_words
        self.min_words = min_words
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    def generate_passage(self, prompt: str) -> Dict:
        """
        Generate a long passage from a prompt.

        Args:
            prompt: Seed prompt to expand

        Returns:
            Dictionary with generated text and metadata
        """
        logger.debug(f"Generating passage from: '{prompt[:50]}...'")

        # Tokenize input
        input_ids = self.generator.tokenizer(
            prompt,
            return_tensors="pt"
        ).input_ids.to(self.generator.model.device)

        # Generate with sampling
        with torch.no_grad():
            output_ids = self.generator.model.generate(
                input_ids,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                do_sample=True,
                pad_token_id=self.generator.tokenizer.eos_token_id,
                eos_token_id=self.generator.tokenizer.eos_token_id,
                repetition_penalty=1.1  # Reduce repetition
            )

        # Decode output
        generated_text = self.generator.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )

        # Count words
        word_count = len(generated_text.split())
        token_count = len(output_ids[0])

        # Compute average log probability (approximate)
        # For actual implementation, would need to compute during generation
        avg_logp = -2.0  # Placeholder

        result = {
            "prompt": prompt,
            "generated_text": generated_text,
            "word_count": word_count,
            "token_count": token_count,
            "avg_logp": avg_logp,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k
        }

        logger.debug(f"Generated {word_count} words ({token_count} tokens)")

        return result

    def generate_multiple_temperatures(
        self,
        prompt: str,
        temperatures: List[float] = [1.0, 2.0, 3.0, 4.0]
    ) -> List[Dict]:
        """
        Generate multiple passages from same prompt with different temperatures.

        Args:
            prompt: Seed prompt
            temperatures: List of temperature values to use

        Returns:
            List of generation results (one per temperature)
        """
        results = []

        original_temp = self.temperature

        for temp in temperatures:
            self.temperature = temp
            result = self.generate_passage(prompt)
            result['temperature'] = temp  # Update to actual used temperature
            results.append(result)

        # Restore original temperature
        self.temperature = original_temp

        return results

    def generate_batch(
        self,
        prompts: List[str],
        progress_callback: Optional[callable] = None
    ) -> List[Dict]:
        """
        Generate passages for a batch of prompts.

        Args:
            prompts: List of seed prompts
            progress_callback: Optional callback for progress updates

        Returns:
            List of generation results
        """
        results = []

        for i, prompt in enumerate(prompts):
            result = self.generate_passage(prompt)
            results.append(result)

            if progress_callback:
                progress_callback(i + 1, len(prompts), result)

        return results
