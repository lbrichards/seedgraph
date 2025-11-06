"""Tests for brancher functionality."""
import pytest
import tempfile
from pathlib import Path
import numpy as np
from seedgraph.core.graph_store import GraphStore, Node
from seedgraph.utils.io import timestamp


def test_node_creation():
    """Test Node dataclass."""
    node = Node(
        id=0,
        prompt="test prompt",
        parent_id=None,
        depth=0,
        topk=[{"token": "the", "id": 1, "prob": 0.5, "logit": 1.0}],
        probs=[0.5, 0.3, 0.2],
        logits=[1.0, 0.5, 0.2],
        created_at=timestamp()
    )

    assert node.id == 0
    assert node.prompt == "test prompt"
    assert not node.expanded
    assert node.depth == 0


def test_node_serialization():
    """Test Node to/from dict."""
    node = Node(
        id=0,
        prompt="test",
        parent_id=None,
        depth=0,
        topk=[],
        probs=[0.5, 0.5],
        logits=[1.0, 1.0]
    )

    # Convert to dict
    node_dict = node.to_dict()
    assert isinstance(node_dict, dict)
    assert node_dict["id"] == 0

    # Convert back
    node2 = Node.from_dict(node_dict)
    assert node2.id == node.id
    assert node2.prompt == node.prompt


def test_graph_store_add_node():
    """Test adding nodes to graph store."""
    store = GraphStore()

    probs = np.array([0.5, 0.3, 0.2])
    logits = np.array([1.0, 0.5, 0.2])

    node_id = store.add_node(
        prompt="test",
        parent_id=None,
        depth=0,
        topk=[],
        probs=probs,
        logits=logits,
        created_at=timestamp()
    )

    assert node_id == 0
    assert len(store.nodes) == 1
    assert store.get_node(0) is not None


def test_graph_store_add_edge():
    """Test adding edges to graph store."""
    store = GraphStore()

    # Add two nodes
    probs = np.array([0.5, 0.5])
    logits = np.array([1.0, 1.0])

    node0 = store.add_node("parent", None, 0, [], probs, logits, timestamp())
    node1 = store.add_node("child", node0, 1, [], probs, logits, timestamp())

    # Add edge
    store.add_edge(node0, node1, token_id=42)

    assert len(store.edges) == 1
    assert store.edges[0] == (node0, node1, 42)


def test_graph_store_expansion():
    """Test marking nodes as expanded."""
    store = GraphStore()

    probs = np.array([0.5, 0.5])
    logits = np.array([1.0, 1.0])

    node_id = store.add_node("test", None, 0, [], probs, logits, timestamp())

    # Initially not expanded
    assert not store.get_node(node_id).expanded

    # Mark as expanded
    store.mark_expanded(node_id)
    assert store.get_node(node_id).expanded


def test_graph_store_unexpanded_nodes():
    """Test getting unexpanded nodes."""
    store = GraphStore()

    probs = np.array([0.5, 0.5])
    logits = np.array([1.0, 1.0])

    # Add 3 nodes
    id0 = store.add_node("node0", None, 0, [], probs, logits, timestamp())
    id1 = store.add_node("node1", None, 1, [], probs, logits, timestamp())
    id2 = store.add_node("node2", None, 2, [], probs, logits, timestamp())

    # All should be unexpanded
    unexpanded = store.get_unexpanded_nodes()
    assert len(unexpanded) == 3

    # Mark one as expanded
    store.mark_expanded(id1)
    unexpanded = store.get_unexpanded_nodes()
    assert len(unexpanded) == 2
    assert id1 not in unexpanded


def test_graph_store_max_depth():
    """Test max depth filtering in unexpanded nodes."""
    store = GraphStore()

    probs = np.array([0.5, 0.5])
    logits = np.array([1.0, 1.0])

    # Add nodes at different depths
    id0 = store.add_node("d0", None, 0, [], probs, logits, timestamp())
    id1 = store.add_node("d1", None, 1, [], probs, logits, timestamp())
    id2 = store.add_node("d2", None, 2, [], probs, logits, timestamp())
    id3 = store.add_node("d3", None, 3, [], probs, logits, timestamp())

    # Get unexpanded with max depth 2
    unexpanded = store.get_unexpanded_nodes(max_depth=2)
    assert len(unexpanded) == 2  # Should only get depths 0 and 1
    assert id0 in unexpanded
    assert id1 in unexpanded
    assert id2 not in unexpanded
    assert id3 not in unexpanded


def test_graph_store_checkpoint():
    """Test checkpoint save/load."""
    store = GraphStore()

    probs = np.array([0.5, 0.3, 0.2])
    logits = np.array([1.0, 0.5, 0.2])

    # Add some nodes
    id0 = store.add_node("node0", None, 0, [], probs, logits, timestamp())
    id1 = store.add_node("node1", id0, 1, [], probs, logits, timestamp())
    store.add_edge(id0, id1, token_id=42)

    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir)
        checkpoint_path = store.save_checkpoint(checkpoint_dir, "test_run")

        # Verify file exists
        assert checkpoint_path.exists()

        # Load into new store
        store2 = GraphStore()
        store2.load_checkpoint(checkpoint_path)

        # Verify content
        assert len(store2.nodes) == 2
        assert len(store2.edges) == 1
        assert store2.get_node(id0) is not None
        assert store2.get_node(id1) is not None


def test_graph_store_stats():
    """Test graph statistics."""
    store = GraphStore()

    probs = np.array([0.5, 0.5])
    logits = np.array([1.0, 1.0])

    # Add nodes
    id0 = store.add_node("node0", None, 0, [], probs, logits, timestamp())
    id1 = store.add_node("node1", id0, 1, [], probs, logits, timestamp())
    id2 = store.add_node("node2", id1, 2, [], probs, logits, timestamp())
    store.add_edge(id0, id1, 1)
    store.add_edge(id1, id2, 2)

    # Mark one as expanded
    store.mark_expanded(id0)

    stats = store.get_stats()

    assert stats["total_nodes"] == 3
    assert stats["total_edges"] == 2
    assert stats["expanded_nodes"] == 1
    assert stats["unexpanded_nodes"] == 2
    assert stats["max_depth"] == 2
