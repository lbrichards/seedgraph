#!/usr/bin/env python3
"""Run batch experiments across all 12 domains."""
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from seedgraph.seeds.taxonomy import DOMAINS


def run_domain_experiment(
    domain: str,
    seed_topic: str,
    max_nodes: int = 1000,
    model: str = "Qwen/Qwen2.5-7B",
    output_dir: Path = Path("results/domain_experiments")
):
    """
    Run experiment for a single domain.

    Args:
        domain: Domain name
        seed_topic: Representative seed topic for domain
        max_nodes: Number of nodes to generate
        model: Model to use
        output_dir: Output directory for results
    """
    domain_slug = domain.lower().replace(" ", "_").replace("&", "and")
    run_id = f"domain_{domain_slug}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"\n{'='*80}")
    print(f"DOMAIN: {domain}")
    print(f"Seed: {seed_topic}")
    print(f"Target nodes: {max_nodes}")
    print(f"Run ID: {run_id}")
    print(f"{'='*80}\n")

    start_time = time.time()

    # Run seedgraph
    cmd = [
        "poetry", "run", "seedgraph", "grow",
        "--prompt", seed_topic,
        "--max-nodes", str(max_nodes),
        "--max-depth", "20",
        "--top-k", "5",
        "--checkpoint-interval", "100",
        "--slim-output",
        "--compression", "zstd",
        "--leaves-only",
        "--model", model,
        "--run-id", run_id
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    elapsed = time.time() - start_time

    # Parse output for stats
    stats = {
        "domain": domain,
        "seed_topic": seed_topic,
        "run_id": run_id,
        "max_nodes": max_nodes,
        "model": model,
        "elapsed_seconds": elapsed,
        "success": result.returncode == 0,
        "stdout": result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout,  # Last 1000 chars
        "stderr": result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr
    }

    # Save domain result
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / f"{domain_slug}_result.json"
    with open(result_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n✓ Completed in {elapsed:.1f}s")
    print(f"  Status: {'SUCCESS' if stats['success'] else 'FAILED'}")
    print(f"  Result saved to: {result_file}")

    return stats


def main():
    """Run experiments for all 12 domains."""
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

    output_dir = Path("results/domain_experiments")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = []
    total_start = time.time()

    print("\n" + "="*80)
    print("BATCH DOMAIN EXPERIMENTS")
    print("="*80)
    print(f"Total domains: {len(DOMAINS)}")
    print(f"Nodes per domain: 1000")
    print(f"Total nodes: {len(DOMAINS) * 1000}")
    print(f"Model: Qwen/Qwen2.5-7B")
    print(f"Output: {output_dir}")
    print("="*80)

    for i, domain in enumerate(DOMAINS, 1):
        print(f"\n\n[{i}/{len(DOMAINS)}] Processing domain: {domain}")

        seed_topic = domain_seeds.get(domain, domain)

        try:
            stats = run_domain_experiment(
                domain=domain,
                seed_topic=seed_topic,
                max_nodes=1000,
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
        "domain_stats": all_stats
    }

    summary_file = output_dir / "experiment_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"Total elapsed time: {total_elapsed / 3600:.2f} hours ({total_elapsed:.1f}s)")
    print(f"Domains completed: {summary['completed_domains']}/{len(DOMAINS)}")
    print(f"Avg time per domain: {summary['avg_seconds_per_domain']:.1f}s")
    print(f"\nSummary saved to: {summary_file}")
    print("="*80)


if __name__ == "__main__":
    main()
