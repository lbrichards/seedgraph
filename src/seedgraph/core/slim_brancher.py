"""Slim brancher with minimal disk usage."""
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
from tqdm import tqdm
from loguru import logger

from seedgraph.llm.qwen import QwenGenerator
from seedgraph.core.graph_store import GraphStore, Node
from seedgraph.core.selection import CoverageSelector
from seedgraph.core.slim_checkpoint import SlimCheckpointWriter, SlimCheckpointLoader, FrontierNode, compute_node_hash
from seedgraph.utils.io import timestamp


class SlimBrancher:
    """
    Orchestrates recursive graph growth with slim outputs.

    Manages the expansion loop, selection strategy, and minimal checkpointing.
    """

    def __init__(
        self,
        generator: QwenGenerator,
        selector: CoverageSelector,
        top_k: int = 10,
        max_nodes: int = 1000,
        max_depth: int = 10,
        checkpoint_interval: int = 100,
        checkpoint_dir: Path = Path("checkpoints"),
        compression: str = "zstd",
        leaves_only: bool = True,
        save_topk: bool = False,
        save_logits: bool = False,
        include_token_ids: bool = False
    ):
        """
        Initialize slim brancher.

        Args:
            generator: Qwen generator for logit extraction
            selector: Coverage-based selector
            top_k: Number of top-k tokens to branch on
            max_nodes: Maximum number of nodes to generate
            max_depth: Maximum depth of tree
            checkpoint_interval: Save checkpoint every N nodes
            checkpoint_dir: Directory for checkpoints
            compression: Compression type ("zstd" or "gzip")
            leaves_only: Only save leaf nodes to corpus
            save_topk: Include top-k tokens in checkpoint
            save_logits: Include logits/probs in checkpoint
            include_token_ids: Include token IDs in corpus
        """
        self.generator = generator
        self.selector = selector
        self.top_k = top_k
        self.max_nodes = max_nodes
        self.max_depth = max_depth
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.compression = compression
        self.leaves_only = leaves_only
        self.save_topk = save_topk
        self.save_logits = save_logits
        self.include_token_ids = include_token_ids

        # Lightweight in-memory storage
        self.frontier: Dict[int, FrontierNode] = {}  # node_id -> FrontierNode
        self.expanded_ids: set = set()
        self.next_id: int = 0
        self.run_id: Optional[str] = None
        self.step_count = 0
        self.checkpoint_writer: Optional[SlimCheckpointWriter] = None
        self.seed_prompt: str = ""  # Store seed prompt for root reconstruction

        # Track paths for corpus generation
        self.node_paths: Dict[int, List[int]] = {}  # node_id -> token_ids from root
        self.node_logps: Dict[int, List[float]] = {}  # node_id -> logps for averaging
        self.node_prompts: Dict[int, str] = {}  # Cache reconstructed prompts

        logger.info(
            f"SlimBrancher initialized: top_k={top_k}, max_nodes={max_nodes}, "
            f"max_depth={max_depth}, checkpoint_interval={checkpoint_interval}, "
            f"compression={compression}, leaves_only={leaves_only}"
        )

    def grow(self, seed_prompt: str, run_id: Optional[str] = None, resume_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Grow knowledge graph from seed prompt.

        Args:
            seed_prompt: Initial prompt to start from
            run_id: Unique identifier for this run
            resume_data: Optional data for resuming from checkpoint

        Returns:
            Statistics and file paths
        """
        # Generate run ID if not provided
        if run_id is None:
            run_id = f"run_{timestamp().replace(':', '-')}"
        self.run_id = run_id

        # Initialize checkpoint writer
        self.checkpoint_writer = SlimCheckpointWriter(
            checkpoint_dir=self.checkpoint_dir,
            run_id=run_id,
            compression=self.compression,
            leaves_only=self.leaves_only,
            save_topk=self.save_topk,
            save_logits=self.save_logits
        )

        # Store seed prompt
        self.seed_prompt = seed_prompt

        logger.info(f"Starting graph growth from seed: '{seed_prompt[:50]}...'")
        logger.info(f"Run ID: {run_id}")

        # Write run metadata
        gen_params = {
            "top_k": self.top_k,
            "max_nodes": self.max_nodes,
            "max_depth": self.max_depth
        }
        self.checkpoint_writer.write_run_meta(
            model_id=self.generator.tokenizer.name_or_path,
            gen_params=gen_params,
            seed=seed_prompt
        )

        # Resume from checkpoint if provided
        if resume_data:
            self._resume_from_checkpoint(resume_data)
            start_nodes = len(self.frontier) + len(self.expanded_ids)
        else:
            # Create root node
            root_dist = self.generator.next_token_distribution(seed_prompt, self.top_k)
            root_node = FrontierNode(
                node_id=0,
                parent_id=None,
                token_ids=[],
                cum_logp=0.0,
                priority=1.0,
                depth=0
            )
            self.frontier[0] = root_node
            self.node_paths[0] = []
            self.node_logps[0] = []
            self.node_prompts[0] = seed_prompt
            self.next_id = 1

            # Initialize selector with root
            self.selector.update_index([root_dist["probs"]])
            self.selector.update_centroids([root_dist["probs"]])

            logger.info(f"Created root node 0")
            start_nodes = 1

        total_nodes = start_nodes

        # Main expansion loop with progress bar
        with tqdm(total=self.max_nodes, initial=start_nodes, desc="Growing graph", unit="nodes") as pbar:
            while total_nodes < self.max_nodes:
                # Get unexpanded frontier nodes
                candidates = [
                    (node_id, node) for node_id, node in self.frontier.items()
                    if node_id not in self.expanded_ids and node.depth < self.max_depth
                ]

                if not candidates:
                    logger.info("No more unexpanded nodes within depth limit")
                    break

                # Select next node to expand based on priority
                best_node_id, best_node = max(candidates, key=lambda x: x[1].priority)

                # Reconstruct prompt from token path
                parent_prompt = self._reconstruct_prompt(best_node_id)

                # Expand this node (branch on top-k tokens)
                child_probs = []
                for tok_info in self._get_topk_for_node(parent_prompt)[:self.top_k]:
                    # Check if we've hit max nodes
                    if total_nodes >= self.max_nodes:
                        break

                    # Generate child prompt
                    child_prompt = self.generator.append_token(parent_prompt, tok_info["id"])

                    # Skip if duplicate
                    child_token_ids = best_node.token_ids + [tok_info["id"]]
                    if not self.checkpoint_writer.add_visited_hash(str(child_token_ids)):
                        continue

                    # Get distribution for child
                    child_dist = self.generator.next_token_distribution(child_prompt, self.top_k)

                    # Create child frontier node
                    child_logp = best_node.cum_logp + np.log(tok_info["prob"] + 1e-10)
                    child_node = FrontierNode(
                        node_id=self.next_id,
                        parent_id=best_node_id,
                        token_ids=child_token_ids,
                        cum_logp=child_logp,
                        priority=0.0,  # Will be updated
                        depth=best_node.depth + 1
                    )

                    # Store path, logps, and prompt
                    self.node_paths[self.next_id] = child_token_ids
                    self.node_logps[self.next_id] = self.node_logps.get(best_node_id, []) + [np.log(tok_info["prob"] + 1e-10)]
                    self.node_prompts[self.next_id] = child_prompt

                    self.frontier[self.next_id] = child_node
                    self.next_id += 1
                    total_nodes += 1

                    child_probs.append(child_dist["probs"])
                    pbar.update(1)

                # Mark parent as expanded
                self.expanded_ids.add(best_node_id)

                # If parent is now a non-leaf that won't be expanded further, write to corpus
                if not self.leaves_only or len(child_probs) == 0:
                    self._write_to_corpus(best_node_id, parent_prompt)

                # Update selector periodically
                if child_probs:
                    self.selector.update_index(child_probs)

                    # Update priorities for frontier
                    self._update_frontier_priorities()

                    # Update centroids at interval
                    if total_nodes % self.selector.update_interval == 0:
                        all_probs = self._get_all_frontier_probs()
                        if all_probs:
                            self.selector.update_centroids(all_probs)

                # Checkpoint at interval
                self.step_count += 1
                if self.step_count % self.checkpoint_interval == 0:
                    self._save_checkpoint()

        # Final checkpoint
        self._save_checkpoint()

        # Write remaining leaf nodes to corpus
        if self.leaves_only:
            for node_id, node in self.frontier.items():
                if node_id not in self.expanded_ids:
                    prompt = self._reconstruct_prompt(node_id)
                    self._write_to_corpus(node_id, prompt)

        # Finalize and get file paths
        file_paths = self.checkpoint_writer.finalize()

        # Print final stats
        stats = self._get_stats()
        logger.info("Graph growth complete!")
        logger.info(f"Final stats: {stats}")

        return {
            **stats,
            "checkpoint_path": file_paths["checkpoint"],
            "corpus_path": file_paths["corpus"]
        }

    def _get_topk_for_node(self, prompt: str) -> List[Dict[str, Any]]:
        """Get top-k tokens for a prompt."""
        dist = self.generator.next_token_distribution(prompt, self.top_k)
        return dist["topk"]

    def _reconstruct_prompt(self, node_id: int) -> str:
        """Reconstruct prompt from node ID using cache."""
        # Use cached prompt if available
        if node_id in self.node_prompts:
            return self.node_prompts[node_id]

        # Otherwise reconstruct from token IDs
        token_ids = self.node_paths.get(node_id, [])
        if not token_ids:
            # This is the root
            return self.seed_prompt

        # Reconstruct: prepend seed to token sequence
        full_prompt = self.seed_prompt + self.generator.tokenizer.decode(token_ids)
        self.node_prompts[node_id] = full_prompt
        return full_prompt

    def _write_to_corpus(self, node_id: int, prompt: str) -> None:
        """Write a node to the corpus shard."""
        if node_id not in self.node_logps:
            avg_logp = 0.0
        else:
            logps = self.node_logps[node_id]
            avg_logp = np.mean(logps) if logps else 0.0

        node = self.frontier.get(node_id)
        depth = node.depth if node else 0

        token_ids = self.node_paths.get(node_id) if self.include_token_ids else None

        self.checkpoint_writer.write_corpus_record(
            text=prompt,
            topic=None,  # Could extract from metadata
            avg_logp=float(avg_logp),
            depth=depth,
            token_ids=token_ids
        )

    def _update_frontier_priorities(self) -> None:
        """Update priority scores for all frontier nodes."""
        for node_id, node in self.frontier.items():
            if node_id in self.expanded_ids:
                continue

            # Reconstruct prompt to get distribution
            prompt = self._reconstruct_prompt(node_id)
            dist = self.generator.next_token_distribution(prompt, self.top_k)
            probs = dist["probs"]

            # Compute coverage priority using selector
            candidates = [node_id]
            candidate_probs = [np.array(probs)]
            selected_id = self.selector.select_next_node(candidates, candidate_probs)

            # Use depth-based priority as fallback (deeper = lower priority)
            node.priority = 1.0 / (1.0 + node.depth)

    def _get_all_frontier_probs(self) -> List[np.ndarray]:
        """Get probability distributions for all frontier nodes."""
        # This would require storing or recomputing
        # For slim mode, we skip this or use cached values
        return []

    def _save_checkpoint(self) -> None:
        """Save checkpoint of current frontier state."""
        if self.checkpoint_writer is None:
            return

        frontier_list = [node for node in self.frontier.values() if node.node_id not in self.expanded_ids]
        self.checkpoint_writer.write_frontier(frontier_list)
        logger.info(f"Checkpoint saved: {len(frontier_list)} frontier nodes")

    def _resume_from_checkpoint(self, resume_data: Dict[str, Any]) -> None:
        """Resume from checkpoint data."""
        run_meta = resume_data["run_meta"]
        frontier = resume_data["frontier"]
        visited_keys = resume_data["visited_keys"]

        logger.info(f"Resuming from checkpoint: {len(frontier)} frontier nodes")

        # Restore frontier
        for node in frontier:
            self.frontier[node.node_id] = node
            self.node_paths[node.node_id] = node.token_ids
            self.next_id = max(self.next_id, node.node_id + 1)

        # Restore visited hashes
        self.checkpoint_writer.visited_hashes = set(visited_keys)

    def _get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            "total_nodes": len(self.frontier) + len(self.expanded_ids),
            "frontier_nodes": len(self.frontier) - len(self.expanded_ids),
            "expanded_nodes": len(self.expanded_ids),
            "max_depth": max((n.depth for n in self.frontier.values()), default=0)
        }
