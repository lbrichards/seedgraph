#!/usr/bin/env python3
"""Two-stage experiment: SeedGraph diversity → Long passage generation."""
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from seedgraph.seeds.taxonomy import DOMAINS


def run_two_stage_domain_experiment(
    domain: str,
    seed_topic: str,
    diversity_nodes: int = 1000,
    target_words: int = 1000,
    temperatures: str = "1.0,2.0,3.0,4.0",
    model: str = "Qwen/Qwen2.5-7B",
    output_dir: Path = Path("results/two_stage_experiments")
):
    """
    Run two-stage experiment for a domain.

    Stage 1: Generate diverse seeds with SeedGraph branching
    Stage 2: Expand each seed into long passages at multiple temperatures

    Args:
        domain: Domain name
        seed_topic: Representative seed topic
        diversity_nodes: Nodes for diversity exploration
        target_words: Target words per passage
        temperatures: Comma-separated temperatures
        model: Model to use
        output_dir: Output directory
    """
    domain_slug = domain.lower().replace(" ", "_").replace("&", "and")
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_id = f"domain_{domain_slug}_{timestamp_str}"

    print(f"\n{'='*80}")
    print(f"TWO-STAGE EXPERIMENT: {domain}")
    print(f"{'='*80}")
    print(f"Seed topic: {seed_topic}")
    print(f"Stage 1: Generate {diversity_nodes} diverse seeds")
    print(f"Stage 2: Expand to {target_words}-word passages at temps {temperatures}")
    print(f"Run ID: {run_id}")
    print(f"{'='*80}\n")

    stage1_start = time.time()

    # STAGE 1: Generate diverse seeds with SeedGraph
    print(f"[STAGE 1] Generating diverse seeds...")
    stage1_cmd = [
        "poetry", "run", "seedgraph", "grow",
        "--prompt", seed_topic,
        "--max-nodes", str(diversity_nodes),
        "--max-depth", "20",
        "--top-k", "5",
        "--checkpoint-interval", "100",
        "--slim-output",
        "--compression", "zstd",
        "--leaves-only",
        "--model", model,
        "--run-id", run_id
    ]

    result1 = subprocess.run(stage1_cmd, capture_output=True, text=True)
    stage1_elapsed = time.time() - stage1_start

    if result1.returncode != 0:
        print(f"✗ Stage 1 FAILED after {stage1_elapsed:.1f}s")
        print(f"Error: {result1.stderr[-500:]}")
        return {
            "domain": domain,
            "success": False,
            "stage": 1,
            "error": result1.stderr[-500:]
        }

    print(f"✓ Stage 1 completed in {stage1_elapsed:.1f}s")

    # Find checkpoint file
    checkpoint_path = Path(f"checkpoints/ckpt_{run_id}.jsonl.zst")
    if not checkpoint_path.exists():
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        return {"domain": domain, "success": False, "stage": 1, "error": "Checkpoint not found"}

    # STAGE 2: Expand leaves into long passages
    print(f"\n[STAGE 2] Expanding {diversity_nodes} seeds into passages...")
    stage2_start = time.time()

    output_dir.mkdir(parents=True, exist_ok=True)
    passages_file = output_dir / f"{domain_slug}_passages.jsonl"

    stage2_cmd = [
        "poetry", "run", "seedgraph", "expand-leaves",
        "--checkpoint", str(checkpoint_path),
        "--output", str(passages_file),
        "--target-words", str(target_words),
        "--temperatures", temperatures,
        "--model", model
    ]

    result2 = subprocess.run(stage2_cmd, capture_output=True, text=True)
    stage2_elapsed = time.time() - stage2_start

    if result2.returncode != 0:
        print(f"✗ Stage 2 FAILED after {stage2_elapsed:.1f}s")
        print(f"Error: {result2.stderr[-500:]}")
        return {
            "domain": domain,
            "success": False,
            "stage": 2,
            "error": result2.stderr[-500:]
        }

    print(f"✓ Stage 2 completed in {stage2_elapsed:.1f}s")

    total_elapsed = stage1_elapsed + stage2_elapsed

    # Parse results
    stats = {
        "domain": domain,
        "seed_topic": seed_topic,
        "run_id": run_id,
        "diversity_nodes": diversity_nodes,
        "target_words": target_words,
        "temperatures": temperatures,
        "model": model,
        "stage1_elapsed_seconds": stage1_elapsed,
        "stage2_elapsed_seconds": stage2_elapsed,
        "total_elapsed_seconds": total_elapsed,
        "success": True,
        "checkpoint_file": str(checkpoint_path),
        "passages_file": str(passages_file)
    }

    # Count passages generated
    if passages_file.exists():
        with open(passages_file) as f:
            passage_count = sum(1 for _ in f)
        stats["passages_generated"] = passage_count

    # Save domain result
    result_file = output_dir / f"{domain_slug}_result.json"
    with open(result_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n✓ TWO-STAGE COMPLETE:")
    print(f"  Stage 1: {stage1_elapsed/60:.1f} min")
    print(f"  Stage 2: {stage2_elapsed/60:.1f} min")
    print(f"  Total: {total_elapsed/60:.1f} min")
    print(f"  Passages: {stats.get('passages_generated', 'unknown')}")
    print(f"  Output: {passages_file}")

    return stats


def main():
    """Run two-stage experiments for all 12 domains."""
    # Sample representative seed topic for each domain
    domain_seeds = {
        "Science & Math": "quantum mechanics and statistical physics",
        "Engineering & Tech": "distributed systems architecture",
        "Medicine & Biology": "cardiovascular physiology and molecular pathways",
        "Social Science & Economics": "behavioral economics and game theory",
        "Law & Policy": "constitutional law and regulatory policy",
        "History & Geography": "ancient civilizations and cultural history",
        "Arts & Literature": "modernist literature and narrative theory",
        "Business & Finance": "corporate finance and investment analysis",
        "Education & Study Skills": "pedagogical theory and learning strategies",
        "Daily Life & Hobbies": "nutrition science and fitness training",
        "Sports & Games": "sports physiology and competitive strategy",
        "Environment & Energy": "climate science and renewable energy systems"
    }

    output_dir = Path("results/two_stage_experiments")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []
    total_start = time.time()

    print("\n" + "="*80)
    print("TWO-STAGE BATCH EXPERIMENTS")
    print("="*80)
    print(f"Total domains: {len(DOMAINS)}")
    print(f"Stage 1: {10} diverse seeds per domain")
    print(f"Stage 2: 1000-word passages × 4 temperatures = 4 passages per seed")
    print(f"Expected passages per domain: ~8-10 seeds × 4 temps = ~32-40")
    print(f"Total passages: ~384-480 across all domains")
    print(f"Model: Qwen/Qwen2.5-7B")
    print(f"Output: {output_dir}")
    print("="*80)

    for i, domain in enumerate(DOMAINS, 1):
        print(f"\n\n[{i}/{len(DOMAINS)}] Processing domain: {domain}")

        seed_topic = domain_seeds.get(domain, domain)

        try:
            stats = run_two_stage_domain_experiment(
                domain=domain,
                seed_topic=seed_topic,
                diversity_nodes=10,  # Scaled back by 100x
                target_words=1000,
                temperatures="1.0,2.0,3.0,4.0",
                model="Qwen/Qwen2.5-7B",
                output_dir=output_dir
            )
            all_stats.append(stats)

        except Exception as e:
            print(f"✗ Error processing {domain}: {e}")
            all_stats.append({
                "domain": domain,
                "success": False,
                "error": str(e)
            })

    total_elapsed = time.time() - total_start

    # Save summary
    summary = {
        "total_domains": len(DOMAINS),
        "completed_domains": sum(1 for s in all_stats if s.get("success", False)),
        "total_elapsed_seconds": total_elapsed,
        "total_elapsed_hours": total_elapsed / 3600,
        "avg_seconds_per_domain": total_elapsed / len(DOMAINS),
        "total_passages": sum(s.get("passages_generated", 0) for s in all_stats),
        "domain_stats": all_stats
    }

    summary_file = output_dir / "experiment_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Total elapsed time: {total_elapsed / 3600:.2f} hours ({total_elapsed/60:.1f} min)")
    print(f"Domains completed: {summary['completed_domains']}/{len(DOMAINS)}")
    print(f"Avg time per domain: {summary['avg_seconds_per_domain']/60:.1f} min")
    print(f"Total passages generated: {summary['total_passages']:,}")
    print(f"\nSummary saved to: {summary_file}")
    print("="*80)


if __name__ == "__main__":
    main()
