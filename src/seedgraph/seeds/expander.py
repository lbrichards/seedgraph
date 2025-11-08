"""Expand Tier-1 seeds into Tier-2 variants with diversity filtering."""
from typing import List, Dict, Set, Tuple
import re


class SeedExpander:
    """Expands base topics into refined variants."""

    # Expansion patterns per domain type
    SUBFIELD_TEMPLATES = [
        "{topic} methods",
        "{topic} techniques",
        "{topic} algorithms",
        "{topic} theory",
        "{topic} applications",
        "advanced {topic}",
        "computational {topic}",
        "experimental {topic}"
    ]

    PROCESS_TEMPLATES = [
        "{topic} processes",
        "{topic} mechanisms",
        "{topic} pathways",
        "{topic} systems",
        "{topic} frameworks",
        "{topic} architectures"
    ]

    APPLICATION_TEMPLATES = [
        "{topic} for real-world problems",
        "{topic} in practice",
        "{topic} implementation",
        "{topic} engineering",
        "practical {topic}"
    ]

    def __init__(self, min_tokens: int = 2, max_tokens: int = 8):
        """
        Initialize seed expander.

        Args:
            min_tokens: Minimum token count
            max_tokens: Maximum token count
        """
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.seen_seeds: Set[str] = set()

    def expand_tier1_to_tier2(self, tier1_topic: str, domain: str, variants: int = 4) -> List[Dict]:
        """
        Expand a Tier-1 topic into Tier-2 variants.

        Args:
            tier1_topic: Base topic
            domain: Domain name
            variants: Number of variants to generate

        Returns:
            List of seed dictionaries
        """
        expansions = []

        # Generate subfield variants
        for template in self.SUBFIELD_TEMPLATES[:variants]:
            candidate = template.format(topic=tier1_topic)
            if self._is_valid_seed(candidate):
                expansions.append({
                    "topic": candidate,
                    "domain": domain,
                    "tier": 2
                })

        # Generate process variants for science/medicine domains
        if domain in ["Science & Math", "Medicine & Biology", "Environment & Energy"]:
            for template in self.PROCESS_TEMPLATES[:2]:
                candidate = template.format(topic=tier1_topic)
                if self._is_valid_seed(candidate):
                    expansions.append({
                        "topic": candidate,
                        "domain": domain,
                        "tier": 2
                    })

        # Generate application variants for engineering/business
        if domain in ["Engineering & Tech", "Business & Finance"]:
            for template in self.APPLICATION_TEMPLATES[:2]:
                candidate = template.format(topic=tier1_topic)
                if self._is_valid_seed(candidate):
                    expansions.append({
                        "topic": candidate,
                        "domain": domain,
                        "tier": 2
                    })

        # Limit to requested number of variants
        return expansions[:variants]

    def generate_tier3_prompts(self, topic: str, domain: str) -> List[Dict]:
        """
        Generate Tier-3 prompt wrappers for a topic.

        Args:
            topic: Base topic
            domain: Domain name

        Returns:
            List of seed dictionaries with prompt wrappers
        """
        prompts = [
            f"Explain the core concepts of {topic} to an advanced student.",
            f"List key trade-offs in {topic}.",
            f"Outline a concise study plan for {topic}."
        ]

        tier3_seeds = []
        for prompt in prompts:
            if self._is_valid_seed(prompt):
                tier3_seeds.append({
                    "topic": prompt,
                    "domain": domain,
                    "tier": 3
                })

        return tier3_seeds

    def _is_valid_seed(self, seed: str) -> bool:
        """
        Validate seed string.

        Args:
            seed: Candidate seed string

        Returns:
            True if valid
        """
        # Remove trailing punctuation
        seed = seed.rstrip('.,!?;:')

        # Check if already seen
        if seed.lower() in self.seen_seeds:
            return False

        # Token count check
        tokens = seed.split()
        if len(tokens) < self.min_tokens or len(tokens) > self.max_tokens:
            return False

        # Reject ultra-general single words
        if len(tokens) == 1 and seed.lower() in ['technology', 'science', 'math', 'business']:
            return False

        # Check for unsafe content (basic filter)
        if self._contains_unsafe_content(seed):
            return False

        self.seen_seeds.add(seed.lower())
        return True

    def _contains_unsafe_content(self, seed: str) -> bool:
        """
        Check for unsafe/sensitive content.

        Args:
            seed: Candidate seed

        Returns:
            True if unsafe
        """
        # Basic safety filter (can be expanded)
        unsafe_patterns = [
            r'\b(password|secret|private|ssn|credit\s*card)\b',
            r'\b(hack|exploit|attack|weapon)\b',
            r'\b(porn|sexual|explicit)\b'
        ]

        for pattern in unsafe_patterns:
            if re.search(pattern, seed.lower()):
                return True

        return False

    def clean_seed(self, seed: str) -> str:
        """
        Clean and normalize seed string.

        Args:
            seed: Raw seed string

        Returns:
            Cleaned seed
        """
        # Remove trailing punctuation
        seed = seed.rstrip('.,!?;:')

        # Normalize whitespace
        seed = ' '.join(seed.split())

        return seed
