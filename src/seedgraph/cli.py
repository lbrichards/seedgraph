"""Command-line interface for SeedGraph."""
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from loguru import logger
import sys

from seedgraph.llm.qwen import QwenGenerator
from seedgraph.core.graph_store import GraphStore
from seedgraph.core.selection import CoverageSelector
from seedgraph.core.brancher import Brancher
from seedgraph.core.slim_brancher import SlimBrancher
from seedgraph.core.slim_checkpoint import SlimCheckpointLoader

app = typer.Typer(
    name="seedgraph",
    help="Recursive knowledge-graph generator using Qwen-2.5-0.5B logit introspection",
    add_completion=False
)
console = Console()


@app.command()
def grow(
    prompt: str = typer.Option(
        ...,
        "--prompt",
        "-p",
        help="Seed prompt to start graph growth"
    ),
    top_k: int = typer.Option(
        10,
        "--top-k",
        "-k",
        help="Number of top-k tokens to branch on per node"
    ),
    max_nodes: int = typer.Option(
        1000,
        "--max-nodes",
        "-n",
        help="Maximum number of nodes to generate"
    ),
    max_depth: int = typer.Option(
        10,
        "--max-depth",
        "-d",
        help="Maximum depth of the tree"
    ),
    checkpoint_interval: int = typer.Option(
        50,
        "--checkpoint-interval",
        "-c",
        help="Save checkpoint every N nodes"
    ),
    checkpoint_dir: Path = typer.Option(
        Path("checkpoints"),
        "--checkpoint-dir",
        help="Directory for saving checkpoints"
    ),
    model_name: str = typer.Option(
        "Qwen/Qwen2.5-0.5B",
        "--model",
        "-m",
        help="HuggingFace model name"
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        help="Device to run on (None = auto)"
    ),
    use_pca: bool = typer.Option(
        True,
        "--use-pca/--no-pca",
        help="Use PCA for dimensionality reduction in FAISS"
    ),
    pca_dims: int = typer.Option(
        256,
        "--pca-dims",
        help="Number of PCA dimensions"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging"
    ),
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        help="Unique identifier for this run"
    ),
    # Slim output options
    slim_output: bool = typer.Option(
        False,
        "--slim-output",
        help="Enable slim output mode (minimal disk usage)"
    ),
    leaves_only: bool = typer.Option(
        True,
        "--leaves-only/--all-nodes",
        help="Save only leaf nodes to corpus (default: True)"
    ),
    save_topk: bool = typer.Option(
        False,
        "--save-topk/--no-topk",
        help="Include top-k tokens in checkpoint"
    ),
    save_logits: bool = typer.Option(
        False,
        "--save-logits/--no-logits",
        help="Include logits/probs in checkpoint"
    ),
    compression: str = typer.Option(
        "zstd",
        "--compression",
        help="Compression type (zstd or gzip)"
    ),
    include_token_ids: bool = typer.Option(
        False,
        "--include-token-ids",
        help="Include token IDs in corpus for exact reconstruction"
    ),
    resume_from: Optional[Path] = typer.Option(
        None,
        "--resume-from",
        help="Resume from checkpoint file"
    )
):
    """
    Grow a knowledge graph from a seed prompt.

    Example:
        seedgraph grow --prompt "SeedGraph builds graphs from logits" --max-nodes 100
    """
    # Configure logging
    logger.remove()
    log_level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=log_level, format="<level>{level: <8}</level> | {message}")

    # Display banner
    console.print(Panel.fit(
        "[bold cyan]SeedGraph[/bold cyan]\n"
        "Recursive knowledge-graph generator\n"
        f"Seed: [italic]{prompt[:60]}{'...' if len(prompt) > 60 else ''}[/italic]",
        border_style="cyan"
    ))

    try:
        # Initialize components
        console.print("\n[yellow]Initializing components...[/yellow]")

        logger.info("Loading Qwen generator...")
        generator = QwenGenerator(model_name=model_name, device=device)

        logger.info("Initializing coverage selector...")
        # Get vocab size from tokenizer
        vocab_size = len(generator.tokenizer)
        selector = CoverageSelector(
            vocab_size=vocab_size,
            use_pca=use_pca,
            pca_dims=pca_dims
        )

        # Handle resume if specified
        resume_data = None
        if resume_from:
            logger.info(f"Loading checkpoint from {resume_from}")
            loader = SlimCheckpointLoader(resume_from)
            resume_data = loader.load()
            console.print(f"[yellow]Resuming from checkpoint with {len(resume_data['frontier'])} frontier nodes[/yellow]")

        # Choose brancher based on slim_output flag
        if slim_output:
            logger.info("Initializing slim brancher...")
            brancher = SlimBrancher(
                generator=generator,
                selector=selector,
                top_k=top_k,
                max_nodes=max_nodes,
                max_depth=max_depth,
                checkpoint_interval=checkpoint_interval,
                checkpoint_dir=checkpoint_dir,
                compression=compression,
                leaves_only=leaves_only,
                save_topk=save_topk,
                save_logits=save_logits,
                include_token_ids=include_token_ids
            )

            # Start growth
            console.print("\n[green]Starting graph growth (SLIM MODE)...[/green]\n")
            result = brancher.grow(seed_prompt=prompt, run_id=run_id, resume_data=resume_data)

            # Display final statistics
            stats = {
                "total_nodes": result["total_nodes"],
                "total_edges": result.get("total_edges", result["total_nodes"] - 1),
                "unexpanded_nodes": result.get("frontier_nodes", 0),
                "expanded_nodes": result["expanded_nodes"],
                "max_depth": result["max_depth"]
            }

            # Display file paths and sizes
            import os
            ckpt_size = os.path.getsize(result["checkpoint_path"]) / (1024 * 1024)
            corpus_size = os.path.getsize(result["corpus_path"]) / (1024 * 1024) if result["corpus_path"].exists() else 0

            console.print(f"\n[cyan]Checkpoint:[/cyan] {result['checkpoint_path']} ({ckpt_size:.2f} MB)")
            console.print(f"[cyan]Corpus:[/cyan] {result['corpus_path']} ({corpus_size:.2f} MB)")
            console.print(f"[green]Total size:[/green] {ckpt_size + corpus_size:.2f} MB")

        else:
            logger.info("Initializing graph store...")
            store = GraphStore()

            logger.info("Initializing brancher...")
            brancher = Brancher(
                generator=generator,
                store=store,
                selector=selector,
                top_k=top_k,
                max_nodes=max_nodes,
                max_depth=max_depth,
                checkpoint_interval=checkpoint_interval,
                checkpoint_dir=checkpoint_dir
            )

            # Start growth
            console.print("\n[green]Starting graph growth...[/green]\n")
            brancher.grow(seed_prompt=prompt, run_id=run_id)

            # Display final statistics
            stats = store.get_stats()
        console.print(Panel(
            f"[bold green]Graph Growth Complete![/bold green]\n\n"
            f"Total Nodes: {stats['total_nodes']}\n"
            f"Total Edges: {stats['total_edges']}\n"
            f"Max Depth: {stats['max_depth']}\n"
            f"Expanded: {stats['expanded_nodes']}\n"
            f"Unexpanded: {stats['unexpanded_nodes']}",
            border_style="green",
            title="Results"
        ))

        console.print(f"\n[cyan]Checkpoints saved to:[/cyan] {checkpoint_dir}")

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        logger.exception("Error during graph growth")
        raise typer.Exit(code=1)


@app.command()
def build_seeds(
    target_seeds: int = typer.Option(
        3000,
        "--target-seeds",
        "-n",
        help="Target number of seeds to generate"
    ),
    near_dup_threshold: float = typer.Option(
        0.92,
        "--near-dup-threshold",
        help="Cosine similarity threshold for near-duplicates"
    ),
    tier3_fraction: float = typer.Option(
        0.2,
        "--tier3-fraction",
        help="Fraction of seeds to wrap with prompts (Tier-3)"
    ),
    output_dir: Path = typer.Option(
        Path("data/seeds"),
        "--output-dir",
        "-o",
        help="Output directory for seed files"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging"
    )
):
    """
    Build diverse seed topic list for LM-head generation.

    Generates seeds_v1.jsonl and seeds_stats.json with high-diversity
    topics across 12 domains.
    """
    # Configure logging
    logger.remove()
    log_level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=log_level, format="<level>{level: <8}</level> | {message}")

    # Display banner
    console.print(Panel.fit(
        "[bold cyan]SeedGraph - Seed List Builder[/bold cyan]\n"
        "Generating diverse seed topics for LM-head exploration",
        border_style="cyan"
    ))

    try:
        from seedgraph.seeds.builder import SeedListBuilder

        # Initialize builder
        builder = SeedListBuilder(
            target_total_seeds=target_seeds,
            near_dup_threshold=near_dup_threshold,
            tier3_fraction=tier3_fraction,
            output_dir=output_dir
        )

        # Build seed list
        console.print("\n[yellow]Building seed list...[/yellow]\n")
        stats = builder.build()

        # Display success
        console.print(Panel(
            f"[bold green]Seed List Generation Complete![/bold green]\n\n"
            f"Total Seeds: {stats['total_seeds']}\n"
            f"95th Percentile Similarity: {stats['pairwise_similarity_percentiles']['95th']:.4f}\n"
            f"KMeans Utilization: {stats['kmeans_clusters_with_5plus']}/{stats['kmeans_total_clusters']}\n"
            f"Output: {output_dir}/seeds_v1.jsonl",
            border_style="green",
            title="Results"
        ))

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        logger.exception("Error during seed list generation")
        raise typer.Exit(code=1)


@app.command()
def expand_leaves(
    checkpoint: Path = typer.Option(
        ...,
        "--checkpoint",
        "-c",
        help="Input checkpoint file from SeedGraph grow"
    ),
    output: Path = typer.Option(
        Path("data/passages.jsonl"),
        "--output",
        "-o",
        help="Output file for generated passages"
    ),
    target_words: int = typer.Option(
        1000,
        "--target-words",
        "-w",
        help="Target word count per passage"
    ),
    model_name: str = typer.Option(
        "Qwen/Qwen2.5-7B",
        "--model",
        "-m",
        help="HuggingFace model name"
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        help="Device to run on (None = auto)"
    ),
    temperatures: str = typer.Option(
        "1.0,2.0,3.0,4.0",
        "--temperatures",
        help="Comma-separated temperature values (generates one passage per temperature)"
    ),
    top_p: float = typer.Option(
        0.9,
        "--top-p",
        help="Nucleus sampling parameter"
    ),
    max_passages: Optional[int] = typer.Option(
        None,
        "--max-passages",
        help="Maximum number of passages to generate (None = all leaves)"
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging"
    )
):
    """
    Expand diverse leaves from SeedGraph into long-form passages.

    Takes a checkpoint from 'seedgraph grow' and generates 1000-word
    passages from each diverse leaf node.

    Example:
        seedgraph expand-leaves --checkpoint checkpoints/ckpt_*.jsonl.zst --target-words 1000
    """
    # Configure logging
    logger.remove()
    log_level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=log_level, format="<level>{level: <8}</level> | {message}")

    # Display banner
    console.print(Panel.fit(
        "[bold cyan]SeedGraph - Passage Expander[/bold cyan]\n"
        f"Generating {target_words}-word passages from diverse seeds",
        border_style="cyan"
    ))

    try:
        from seedgraph.core.slim_checkpoint import SlimCheckpointLoader
        from seedgraph.generation.passage_generator import PassageGenerator
        from seedgraph.llm.qwen import QwenGenerator
        from tqdm import tqdm
        import json

        # Load checkpoint
        console.print(f"\n[yellow]Loading checkpoint: {checkpoint}[/yellow]")
        loader = SlimCheckpointLoader(checkpoint)
        ckpt_data = loader.load()

        # Get unexpanded leaves
        frontier = ckpt_data['frontier']
        leaves = [node for node in frontier]  # All frontier nodes are potential seeds

        if max_passages:
            leaves = leaves[:max_passages]

        console.print(f"[green]Found {len(leaves)} diverse leaf nodes[/green]")

        # Initialize generator
        console.print(f"\n[yellow]Loading model: {model_name}[/yellow]")
        generator = QwenGenerator(model_name=model_name, device=device)

        # Parse temperatures
        temp_list = [float(t.strip()) for t in temperatures.split(',')]
        console.print(f"[cyan]Temperatures: {temp_list}[/cyan]")
        console.print(f"[cyan]Passages per seed: {len(temp_list)}[/cyan]")

        # Initialize passage generator
        passage_gen = PassageGenerator(
            generator=generator,
            target_words=target_words,
            temperature=temp_list[0],  # Default, will be overridden
            top_p=top_p
        )

        # Reconstruct prompts from token_ids and generate passages
        console.print(f"\n[green]Generating {target_words}-word passages with {len(temp_list)} temperatures...[/green]\n")

        seed_prompt = ckpt_data['run_meta']['seed']
        generated_count = 0
        total_words = 0

        # Ensure output directory exists
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, 'w') as f:
            for leaf in tqdm(leaves, desc="Generating passages", unit="seed"):
                # Reconstruct prompt
                if leaf.token_ids:
                    leaf_text = seed_prompt + generator.tokenizer.decode(leaf.token_ids)
                else:
                    leaf_text = seed_prompt

                # Generate multiple passages with different temperatures
                multi_results = passage_gen.generate_multiple_temperatures(leaf_text, temp_list)

                # Write each result
                for result in multi_results:
                    record = {
                        "seed_prompt": leaf_text,
                        "generated_text": result['generated_text'],
                        "word_count": result['word_count'],
                        "token_count": result['token_count'],
                        "temperature": result['temperature'],
                        "depth": leaf.depth,
                        "parent_id": leaf.parent_id
                    }

                    f.write(json.dumps(record) + '\n')

                    generated_count += 1
                    total_words += result['word_count']

        avg_words = total_words / generated_count if generated_count > 0 else 0
        seeds_processed = len(leaves)
        passages_per_seed = len(temp_list)

        # Display results
        console.print(Panel(
            f"[bold green]Passage Generation Complete![/bold green]\n\n"
            f"Seeds processed: {seeds_processed}\n"
            f"Passages per seed: {passages_per_seed} (temps: {temp_list})\n"
            f"Total passages: {generated_count}\n"
            f"Total words: {total_words:,}\n"
            f"Avg words per passage: {avg_words:.0f}\n"
            f"Output: {output}",
            border_style="green",
            title="Results"
        ))

    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        logger.exception("Error during passage generation")
        raise typer.Exit(code=1)


@app.command()
def info():
    """
    Display information about SeedGraph.
    """
    console.print(Panel(
        "[bold cyan]SeedGraph v0.1.0[/bold cyan]\n\n"
        "A recursive knowledge-graph generator that introspects\n"
        "Qwen-2.5-0.5B logits, expands branches from top-k candidates,\n"
        "and uses KL divergence + FAISS-based distance to guide\n"
        "manifold coverage.\n\n"
        "[bold]Key Features:[/bold]\n"
        "  • Logit introspection via Qwen-2.5-0.5B\n"
        "  • KL divergence-based node selection\n"
        "  • FAISS-accelerated coverage computation\n"
        "  • Automatic checkpointing\n"
        "  • Recursive graph branching",
        border_style="cyan"
    ))


if __name__ == "__main__":
    app()
