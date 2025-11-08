"""Slim checkpoint save/load for minimal disk usage."""
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import hashlib
from loguru import logger

from seedgraph.utils.io import write_jsonl_compressed, read_jsonl_compressed, ensure_dir, timestamp


@dataclass
class FrontierNode:
    """Minimal frontier node for checkpoint."""
    node_id: int
    parent_id: Optional[int]
    token_ids: List[int]  # Path from root as token IDs
    cum_logp: float  # Cumulative log probability
    priority: float  # Coverage priority score
    depth: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict with native Python types."""
        return {
            "node_id": int(self.node_id),
            "parent_id": int(self.parent_id) if self.parent_id is not None else None,
            "token_ids": [int(x) for x in self.token_ids],
            "cum_logp": float(self.cum_logp),
            "priority": float(self.priority),
            "depth": int(self.depth)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrontierNode":
        return cls(**data)


class SlimCheckpointWriter:
    """Writes minimal checkpoints with compression."""

    def __init__(
        self,
        checkpoint_dir: Path,
        run_id: str,
        compression: str = "zstd",
        leaves_only: bool = True,
        save_topk: bool = False,
        save_logits: bool = False
    ):
        """
        Initialize slim checkpoint writer.

        Args:
            checkpoint_dir: Directory for checkpoints
            run_id: Unique run identifier
            compression: Compression type ("zstd" or "gzip")
            leaves_only: Only save leaf nodes to corpus
            save_topk: Include top-k tokens in checkpoint
            save_logits: Include logits/probs in checkpoint
        """
        self.checkpoint_dir = ensure_dir(checkpoint_dir)
        self.run_id = run_id
        self.compression = compression
        self.leaves_only = leaves_only
        self.save_topk = save_topk
        self.save_logits = save_logits

        # File paths
        ext = ".zst" if compression == "zstd" else ".gz"
        self.checkpoint_path = self.checkpoint_dir / f"ckpt_{run_id}.jsonl{ext}"
        self.corpus_path = self.checkpoint_dir / f"corpus_{run_id}.jsonl.gz"

        self.visited_hashes: set = set()

    def write_run_meta(self, model_id: str, gen_params: Dict[str, Any], seed: str) -> None:
        """Write run metadata to checkpoint."""
        metadata = {
            "type": "run_meta",
            "run_id": self.run_id,
            "model_id": model_id,
            "gen_params": gen_params,
            "seed": seed,
            "timestamp": timestamp()
        }
        write_jsonl_compressed(self.checkpoint_path, metadata, self.compression)
        logger.info(f"Wrote run metadata to {self.checkpoint_path}")

    def write_frontier(self, frontier: List[FrontierNode]) -> None:
        """Write frontier nodes to checkpoint."""
        for node in frontier:
            entry = {"type": "frontier", **node.to_dict()}
            write_jsonl_compressed(self.checkpoint_path, entry, self.compression)

    def write_visited_keys(self, keys: List[str]) -> None:
        """Write visited dedup hashes to checkpoint."""
        entry = {"type": "visited_keys", "keys": keys}
        write_jsonl_compressed(self.checkpoint_path, entry, self.compression)

    def write_corpus_record(
        self,
        text: str,
        topic: Optional[str],
        avg_logp: float,
        depth: int,
        token_ids: Optional[List[int]] = None
    ) -> None:
        """
        Write a leaf node to the corpus shard.

        Args:
            text: Generated text
            topic: Optional topic/category
            avg_logp: Average log probability
            depth: Depth in tree
            token_ids: Optional token IDs for exact reconstruction
        """
        record = {
            "text": text,
            "topic": topic,
            "avg_logp": avg_logp,
            "depth": depth
        }
        if token_ids is not None:
            record["token_ids"] = token_ids

        write_jsonl_compressed(self.corpus_path, record, "gzip")

    def add_visited_hash(self, text: str) -> bool:
        """
        Add text hash to visited set for deduplication.

        Returns:
            True if new, False if duplicate
        """
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.visited_hashes:
            return False
        self.visited_hashes.add(text_hash)
        return True

    def finalize(self) -> Dict[str, Path]:
        """
        Finalize checkpoint and return file paths.

        Returns:
            Dictionary with checkpoint and corpus paths
        """
        # Write all visited hashes
        if self.visited_hashes:
            self.write_visited_keys(list(self.visited_hashes))

        return {
            "checkpoint": self.checkpoint_path,
            "corpus": self.corpus_path
        }


class SlimCheckpointLoader:
    """Loads minimal checkpoints."""

    def __init__(self, checkpoint_path: Path):
        """
        Initialize loader.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        self.checkpoint_path = checkpoint_path

    def load(self) -> Dict[str, Any]:
        """
        Load checkpoint data.

        Returns:
            Dictionary with:
                - run_meta: Run metadata
                - frontier: List of FrontierNode objects
                - visited_keys: List of visited hashes
        """
        run_meta = None
        frontier = []
        visited_keys = []

        for entry in read_jsonl_compressed(self.checkpoint_path):
            entry_type = entry.pop("type")

            if entry_type == "run_meta":
                run_meta = entry
            elif entry_type == "frontier":
                frontier.append(FrontierNode.from_dict(entry))
            elif entry_type == "visited_keys":
                visited_keys.extend(entry["keys"])

        logger.info(
            f"Loaded checkpoint: {len(frontier)} frontier nodes, "
            f"{len(visited_keys)} visited keys"
        )

        return {
            "run_meta": run_meta,
            "frontier": frontier,
            "visited_keys": visited_keys
        }


def compute_node_hash(token_ids: List[int]) -> str:
    """
    Compute dedup hash for a token sequence.

    Args:
        token_ids: List of token IDs

    Returns:
        MD5 hash string
    """
    return hashlib.md5(str(token_ids).encode()).hexdigest()
