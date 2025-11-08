"""Main seed list builder orchestrating all steps."""
from typing import List, Dict, Optional
from pathlib import Path
import json
import random
import numpy as np
from loguru import logger

from seedgraph.seeds.taxonomy import DOMAINS, TIER1_SEEDS, validate_tier1_coverage
from seedgraph.seeds.expander import SeedExpander
from seedgraph.seeds.diversity import DiversityFilter
from seedgraph.utils.io import ensure_dir


class SeedListBuilder:
    """Orchestrates diverse seed list generation."""

    def __init__(
        self,
        target_total_seeds: int = 3000,
        near_dup_threshold: float = 0.92,
        tier3_fraction: float = 0.2,
        k_center_candidates: int = 12000,
        output_dir: Path = Path("data/seeds")
    ):
        """
        Initialize seed list builder.

        Args:
            target_total_seeds: Target number of final seeds
            near_dup_threshold: Cosine similarity threshold for deduplication
            tier3_fraction: Fraction of seeds to wrap with prompts
            k_center_candidates: Number of candidates before k-center selection
            output_dir: Output directory for seed files
        """
        self.target_total_seeds = target_total_seeds
        self.near_dup_threshold = near_dup_threshold
        self.tier3_fraction = tier3_fraction
        self.k_center_candidates = k_center_candidates
        self.output_dir = Path(output_dir)

        self.expander = SeedExpander()
        self.diversity_filter = DiversityFilter(
            near_dup_threshold=near_dup_threshold,
            target_total_seeds=target_total_seeds
        )

        # Set random seeds for reproducibility
        random.seed(42)
        np.random.seed(42)

    def build(self) -> Dict:
        """
        Build diverse seed list through all steps.

        Returns:
            Statistics dictionary
        """
        logger.info("Starting seed list generation...")

        # Step 1: Validate taxonomy
        logger.info("Step 1: Validating taxonomy...")
        validate_tier1_coverage()
        logger.info(f"Validated {len(DOMAINS)} domains")

        # Step 2: Bootstrap Tier-1 seeds
        logger.info("Step 2: Bootstrapping Tier-1 seeds...")
        tier1_seeds = self._bootstrap_tier1()
        logger.info(f"Generated {len(tier1_seeds)} Tier-1 seeds")

        # Step 3: Expand to Tier-2
        logger.info("Step 3: Expanding to Tier-2...")
        tier2_seeds = self._expand_tier2(tier1_seeds)
        logger.info(f"Generated {len(tier2_seeds)} Tier-2 seeds")

        # Step 4: Add Tier-3 prompts
        logger.info("Step 4: Adding Tier-3 prompt wrappers...")
        tier3_seeds = self._generate_tier3(tier1_seeds + tier2_seeds)
        logger.info(f"Generated {len(tier3_seeds)} Tier-3 seeds")

        # Combine all tiers
        all_seeds = tier1_seeds + tier2_seeds + tier3_seeds
        logger.info(f"Total candidate seeds: {len(all_seeds)}")

        # Step 5: Diversity filter
        logger.info("Step 5: Computing embeddings...")
        seed_texts = [seed['topic'] for seed in all_seeds]
        embeddings = self.diversity_filter.compute_embeddings(seed_texts)
        logger.info(f"Computed embeddings: {embeddings.shape}")

        logger.info("Step 5a: Removing near-duplicates...")
        filtered_seeds, filtered_embs = self.diversity_filter.remove_near_duplicates(
            all_seeds, embeddings
        )
        logger.info(f"After dedup: {len(filtered_seeds)} seeds")

        logger.info("Step 5b: K-center selection...")
        if len(filtered_seeds) > self.target_total_seeds:
            selected_seeds, selected_embs = self.diversity_filter.k_center_selection(
                filtered_seeds, filtered_embs, self.target_total_seeds
            )
        else:
            selected_seeds, selected_embs = filtered_seeds, filtered_embs
        logger.info(f"After k-center: {len(selected_seeds)} seeds")

        logger.info("Step 5c: Balancing domains...")
        balanced_seeds, balanced_embs = self.diversity_filter.balance_domains(
            selected_seeds, selected_embs, len(DOMAINS)
        )
        logger.info(f"Final balanced seeds: {len(balanced_seeds)}")

        # Step 6: Sanity & hygiene (already done in expander validation)

        # Compute diagnostics
        logger.info("Computing diagnostics...")
        diagnostics = self.diversity_filter.compute_diagnostics(
            balanced_seeds, balanced_embs
        )

        # Save outputs
        logger.info("Saving outputs...")
        self._save_seeds(balanced_seeds)
        self._save_stats(diagnostics)

        logger.info("Seed list generation complete!")

        return diagnostics

    def _bootstrap_tier1(self) -> List[Dict]:
        """Bootstrap Tier-1 canonical seeds."""
        tier1_seeds = []

        for domain, topics in TIER1_SEEDS.items():
            for topic in topics:
                clean_topic = self.expander.clean_seed(topic)
                tier1_seeds.append({
                    "topic": clean_topic,
                    "domain": domain,
                    "tier": 1
                })

        return tier1_seeds

    def _expand_tier2(self, tier1_seeds: List[Dict]) -> List[Dict]:
        """Expand Tier-1 to Tier-2 variants."""
        tier2_seeds = []

        for seed in tier1_seeds:
            variants = self.expander.expand_tier1_to_tier2(
                seed['topic'],
                seed['domain'],
                variants=4
            )
            tier2_seeds.extend(variants)

        return tier2_seeds

    def _generate_tier3(self, base_seeds: List[Dict]) -> List[Dict]:
        """Generate Tier-3 prompt wrappers."""
        tier3_seeds = []

        # Sample 20% of base seeds for tier-3 expansion
        n_tier3 = int(len(base_seeds) * self.tier3_fraction)
        sampled_seeds = random.sample(base_seeds, n_tier3)

        for seed in sampled_seeds:
            prompts = self.expander.generate_tier3_prompts(
                seed['topic'],
                seed['domain']
            )
            tier3_seeds.extend(prompts)

        return tier3_seeds

    def _save_seeds(self, seeds: List[Dict]) -> None:
        """Save seeds to JSONL file."""
        ensure_dir(self.output_dir)
        output_file = self.output_dir / "seeds_v1.jsonl"

        with open(output_file, 'w') as f:
            for seed in seeds:
                f.write(json.dumps(seed) + '\n')

        logger.info(f"Saved {len(seeds)} seeds to {output_file}")

    def _save_stats(self, diagnostics: Dict) -> None:
        """Save statistics and diagnostics."""
        ensure_dir(self.output_dir)
        stats_file = self.output_dir / "seeds_stats.json"

        with open(stats_file, 'w') as f:
            json.dump(diagnostics, f, indent=2)

        logger.info(f"Saved diagnostics to {stats_file}")

        # Print summary
        print("\n" + "="*80)
        print("SEED LIST STATISTICS")
        print("="*80)
        print(f"\nTotal seeds: {diagnostics['total_seeds']}")
        print(f"\nDomain distribution:")
        for domain, count in sorted(diagnostics['domain_counts'].items()):
            pct = (count / diagnostics['total_seeds']) * 100
            print(f"  {domain:30s}: {count:4d} ({pct:5.1f}%)")

        print(f"\nPairwise similarity percentiles:")
        for pct, value in diagnostics['pairwise_similarity_percentiles'].items():
            print(f"  {pct:5s}: {value:.4f}")

        print(f"\nKMeans utilization:")
        print(f"  Clusters with ≥5 seeds: {diagnostics['kmeans_clusters_with_5plus']}/{diagnostics['kmeans_total_clusters']}")

        print(f"\nAvg top-100 neighbor distance: {diagnostics['avg_top100_neighbor_distance']:.4f}")
        print("="*80)
