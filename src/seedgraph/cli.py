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
