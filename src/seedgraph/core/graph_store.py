"""Graph storage and node/edge management."""
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict
import numpy as np
import networkx as nx
from pathlib import Path
from loguru import logger

from seedgraph.utils.io import write_jsonl, read_jsonl, ensure_dir


@dataclass
class Node:
    """
    A node in the knowledge graph.

    Attributes:
        id: Unique node identifier
        prompt: Text prompt for this node
        parent_id: ID of parent node (None for root)
        depth: Depth in the tree
        expanded: Whether this node has been expanded
        topk: Top-k next tokens from distribution
        probs: Full probability distribution (as list for serialization)
        logits: Full logit distribution (as list for serialization)
        created_at: Timestamp of creation
    """
    id: int
    prompt: str
    parent_id: Optional[int]
    depth: int
    expanded: bool = False
    topk: List[Dict[str, Any]] = field(default_factory=list)
    probs: Optional[List[float]] = None
    logits: Optional[List[float]] = None
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Node":
        """Create node from dictionary."""
        # Convert lists back to numpy arrays if needed
        if data.get("probs") is not None:
            data["probs"] = data["probs"]  # Keep as list for now
        if data.get("logits") is not None:
            data["logits"] = data["logits"]
        return cls(**data)


class GraphStore:
    """
    Storage and management for knowledge graph.

    Maintains nodes, edges, and expansion state.
    """

    def __init__(self):
        """Initialize empty graph store."""
        self.nodes: Dict[int, Node] = {}
        self.edges: List[tuple] = []  # (parent_id, child_id, token_id)
        self.next_id: int = 0
        self.graph: nx.DiGraph = nx.DiGraph()

    def add_node(
        self,
        prompt: str,
        parent_id: Optional[int],
        depth: int,
        topk: List[Dict[str, Any]],
        probs: np.ndarray,
        logits: np.ndarray,
        created_at: str
    ) -> int:
        """
        Add a new node to the graph.

        Args:
            prompt: Text prompt
            parent_id: Parent node ID (None for root)
            depth: Depth in tree
            topk: Top-k token candidates
            probs: Probability distribution
            logits: Logit distribution
            created_at: Creation timestamp

        Returns:
            Node ID
        """
        node_id = self.next_id
        self.next_id += 1

        # Convert numpy arrays to lists for serialization
        probs_list = probs.tolist() if isinstance(probs, np.ndarray) else probs
        logits_list = logits.tolist() if isinstance(logits, np.ndarray) else logits

        node = Node(
            id=node_id,
            prompt=prompt,
            parent_id=parent_id,
            depth=depth,
            topk=topk,
            probs=probs_list,
            logits=logits_list,
            created_at=created_at
        )

        self.nodes[node_id] = node
        self.graph.add_node(node_id, **node.to_dict())

        logger.debug(f"Added node {node_id} at depth {depth}")
        return node_id

    def add_edge(self, parent_id: int, child_id: int, token_id: int) -> None:
        """
        Add an edge between nodes.

        Args:
            parent_id: Parent node ID
            child_id: Child node ID
            token_id: Token that generated this branch
        """
        self.edges.append((parent_id, child_id, token_id))
        self.graph.add_edge(parent_id, child_id, token_id=token_id)

    def get_node(self, node_id: int) -> Optional[Node]:
        """Get node by ID."""
        return self.nodes.get(node_id)

    def mark_expanded(self, node_id: int) -> None:
        """Mark a node as expanded."""
        if node_id in self.nodes:
            self.nodes[node_id].expanded = True

    def get_unexpanded_nodes(self, max_depth: Optional[int] = None) -> List[int]:
        """
        Get list of unexpanded node IDs.

        Args:
            max_depth: Optional maximum depth to consider

        Returns:
            List of node IDs
        """
        candidates = []
        for node_id, node in self.nodes.items():
            if not node.expanded:
                if max_depth is None or node.depth < max_depth:
                    candidates.append(node_id)
        return candidates

    def save_checkpoint(self, checkpoint_dir: Path, run_id: str) -> Path:
        """
        Save graph to checkpoint file.

        Args:
            checkpoint_dir: Directory for checkpoints
            run_id: Unique run identifier

        Returns:
            Path to checkpoint file
        """
        ensure_dir(checkpoint_dir)
        checkpoint_path = checkpoint_dir / f"{run_id}_checkpoint.jsonl"

        # Write metadata
        metadata = {
            "type": "metadata",
            "run_id": run_id,
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges)
        }
        write_jsonl(checkpoint_path, metadata)

        # Write all nodes
        for node in self.nodes.values():
            write_jsonl(checkpoint_path, {"type": "node", **node.to_dict()})

        # Write all edges
        for parent_id, child_id, token_id in self.edges:
            write_jsonl(checkpoint_path, {
                "type": "edge",
                "parent_id": parent_id,
                "child_id": child_id,
                "token_id": token_id
            })

        logger.info(f"Saved checkpoint to {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """
        Load graph from checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        self.nodes.clear()
        self.edges.clear()
        self.graph.clear()
        self.next_id = 0

        for entry in read_jsonl(checkpoint_path):
            entry_type = entry.pop("type")

            if entry_type == "node":
                node = Node.from_dict(entry)
                self.nodes[node.id] = node
                self.graph.add_node(node.id, **node.to_dict())
                self.next_id = max(self.next_id, node.id + 1)

            elif entry_type == "edge":
                parent_id = entry["parent_id"]
                child_id = entry["child_id"]
                token_id = entry["token_id"]
                self.edges.append((parent_id, child_id, token_id))
                self.graph.add_edge(parent_id, child_id, token_id=token_id)

        logger.info(f"Loaded checkpoint from {checkpoint_path}: {len(self.nodes)} nodes, {len(self.edges)} edges")

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        unexpanded = len(self.get_unexpanded_nodes())
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "unexpanded_nodes": unexpanded,
            "expanded_nodes": len(self.nodes) - unexpanded,
            "max_depth": max((n.depth for n in self.nodes.values()), default=0)
        }
