"""Recursive graph branching orchestrator."""
from pathlib import Path
from typing import Optional
import numpy as np
from tqdm import tqdm
from loguru import logger

from seedgraph.llm.qwen import QwenGenerator
from seedgraph.core.graph_store import GraphStore
from seedgraph.core.selection import CoverageSelector
from seedgraph.utils.io import timestamp


class Brancher:
    """
    Orchestrates recursive graph growth.

    Manages the expansion loop, selection strategy, and checkpointing.
    """

    def __init__(
        self,
        generator: QwenGenerator,
        store: GraphStore,
        selector: CoverageSelector,
        top_k: int = 10,
        max_nodes: int = 1000,
        max_depth: int = 10,
        checkpoint_interval: int = 50,
        checkpoint_dir: Path = Path("checkpoints")
    ):
        """
        Initialize brancher.

        Args:
            generator: Qwen generator for logit extraction
            store: Graph storage
            selector: Coverage-based selector
            top_k: Number of top-k tokens to branch on
            max_nodes: Maximum number of nodes to generate
            max_depth: Maximum depth of tree
            checkpoint_interval: Save checkpoint every N nodes
            checkpoint_dir: Directory for checkpoints
        """
        self.generator = generator
        self.store = store
        self.selector = selector
        self.top_k = top_k
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = Path(checkpoint_dir)

        self.run_id: Optional[str] = None
        self.step_count = 0

        logger.info(
            f"Brancher initialized: top_k={top_k}, max_nodes={max_nodes}, "
            f"max_depth={max_depth}, checkpoint_interval={checkpoint_interval}"
        )

    def grow(self, seed_prompt: str, run_id: Optional[str] = None) -> None:
        """
        Grow knowledge graph from seed prompt.

        Args:
            seed_prompt: Initial prompt to start from
            run_id: Unique identifier for this run
        """
        # Generate run ID if not provided
        if run_id is None:
            run_id = f"run_{timestamp().replace(':', '-')}"
        self.run_id = run_id

        logger.info(f"Starting graph growth from seed: '{seed_prompt[:50]}...'")
        logger.info(f"Run ID: {run_id}")

        # Create root node
        root_dist = self.generator.next_token_distribution(seed_prompt, self.top_k)
        root_id = self.store.add_node(
            prompt=seed_prompt,
            parent_id=None,
            depth=0,
            topk=root_dist["topk"],
            probs=root_dist["probs"],
            logits=root_dist["logits"],
            created_at=timestamp()
        )

        # Initialize selector with root
        self.selector.update_index([root_dist["probs"]])
        self.selector.update_centroids([root_dist["probs"]])

        logger.info(f"Created root node {root_id}")

        # Main expansion loop with progress bar
        with tqdm(total=self.max_nodes, desc="Growing graph", unit="nodes") as pbar:
            pbar.update(1)  # Count root node

            while len(self.store.nodes) < self.max_nodes:
                # Get unexpanded nodes
                candidates = self.store.get_unexpanded_nodes(max_depth=self.max_depth)

                if not candidates:
                    logger.info("No more unexpanded nodes within depth limit")
                    break

                # Select next node to expand
                candidate_probs = []
                for node_id in candidates:
                    node = self.store.get_node(node_id)
                    probs = np.array(node.probs) if node.probs else np.zeros(self.selector.vocab_size)
                    candidate_probs.append(probs)

                best_node_id = self.selector.select_next_node(candidates, candidate_probs)
                parent = self.store.get_node(best_node_id)

                # Expand this node (branch on top-k tokens)
                new_nodes = []
                for tok_info in parent.topk[:self.top_k]:
                    # Check if we've hit max nodes
                    if len(self.store.nodes) >= self.max_nodes:
                        break

                    # Generate child prompt
                    child_prompt = self.generator.append_token(parent.prompt, tok_info["id"])

                    # Get distribution for child
                    child_dist = self.generator.next_token_distribution(child_prompt, self.top_k)

                    # Add child node
                    child_id = self.store.add_node(
                        prompt=child_prompt,
                        parent_id=parent.id,
                        depth=parent.depth + 1,
                        topk=child_dist["topk"],
                        probs=child_dist["probs"],
                        logits=child_dist["logits"],
                        created_at=timestamp()
                    )

                    # Add edge
                    self.store.add_edge(parent.id, child_id, tok_info["id"])

                    new_nodes.append(child_dist["probs"])
                    pbar.update(1)

                # Mark parent as expanded
                self.store.mark_expanded(best_node_id)

                # Update selector periodically
                if new_nodes:
                    self.selector.update_index(new_nodes)

                    # Update centroids at interval
                    if len(self.store.nodes) % self.selector.update_interval == 0:
                        all_probs = [
                            np.array(node.probs)
                            for node in self.store.nodes.values()
                            if node.probs is not None
                        ]
                        self.selector.update_centroids(all_probs)

                # Checkpoint at interval
                self.step_count += 1
                if self.step_count % self.checkpoint_interval == 0:
                    self._save_checkpoint()

        # Final checkpoint
        self._save_checkpoint()

        # Print final stats
        stats = self.store.get_stats()
        logger.info("Graph growth complete!")
        logger.info(f"Final stats: {stats}")

    def _save_checkpoint(self) -> None:
        """Save checkpoint of current graph state."""
        if self.run_id is None:
            logger.warning("Cannot save checkpoint: no run_id set")
            return

        checkpoint_path = self.store.save_checkpoint(self.checkpoint_dir, self.run_id)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
