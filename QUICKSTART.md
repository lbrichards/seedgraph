# SeedGraph Quick Start Guide

## Installation

```bash
# Clone or navigate to the repo
cd /Users/macmini/projects/seedgraph

# Install dependencies
poetry install

# Verify installation
poetry run seedgraph --help
```

## First Run (Quick Test)

Generate a small 50-node graph:

```bash
poetry run seedgraph grow \
  --prompt "SeedGraph builds graphs from logits" \
  --max-nodes 50 \
  --max-depth 3 \
  --checkpoint-interval 25
```

**What happens:**
1. Downloads Qwen-2.5-0.5B model (~500MB, one-time)
2. Generates recursive graph from seed prompt
3. Saves checkpoints to `./checkpoints/`
4. Shows progress bar and final statistics

## Example Output

```
╭──────────────────────────────────────────────────────╮
│ SeedGraph                                            │
│ Recursive knowledge-graph generator                  │
│ Seed: SeedGraph builds graphs from logits           │
╰──────────────────────────────────────────────────────╯

Initializing components...
Loading Qwen generator...
INFO     | Loading model: Qwen/Qwen2.5-0.5B
INFO     | Model loaded on device: cpu

Growing graph: 100%|████████████| 50/50 [00:45<00:00]

╭─ Results ─────────────────────────────────────────────╮
│ Graph Growth Complete!                                │
│                                                       │
│ Total Nodes: 50                                       │
│ Total Edges: 49                                       │
│ Max Depth: 3                                          │
│ Expanded: 5                                           │
│ Unexpanded: 45                                        │
╰───────────────────────────────────────────────────────╯

Checkpoints saved to: checkpoints
```

## Testing

```bash
# Run fast unit tests (no model loading)
poetry run pytest tests/test_selection.py tests/test_brancher.py -v

# Run all tests including model tests (slow, requires download)
poetry run pytest -v

# Run with coverage
poetry run pytest --cov=seedgraph
```

## Common Options

### Control Graph Size
```bash
--max-nodes 100        # Generate 100 nodes
--max-depth 5          # Limit tree depth to 5
--top-k 5              # Branch on top-5 tokens (less branching)
```

### Performance Options
```bash
--no-pca               # Disable PCA (faster, more memory)
--pca-dims 128         # Use 128 PCA dimensions (vs 256 default)
--checkpoint-interval 100  # Checkpoint less frequently
```

### Debugging
```bash
--verbose              # Enable debug logging
--run-id my_experiment # Custom run identifier
```

## Advanced: Inspecting Checkpoints

```python
from pathlib import Path
from seedgraph.utils.io import read_jsonl
from seedgraph.core.graph_store import GraphStore

# Load checkpoint
store = GraphStore()
checkpoint_path = Path("checkpoints/run_TIMESTAMP_checkpoint.jsonl")
store.load_checkpoint(checkpoint_path)

# Analyze graph
stats = store.get_stats()
print(f"Loaded {stats['total_nodes']} nodes")

# Explore nodes
for node_id, node in list(store.nodes.items())[:5]:
    print(f"Node {node_id} (depth {node.depth}): {node.prompt[:60]}")
```

## Example: Generate Multiple Graphs

```bash
# Experiment 1: Broad exploration
poetry run seedgraph grow \
  --prompt "Neural networks learn representations" \
  --max-nodes 200 \
  --top-k 10 \
  --max-depth 4 \
  --run-id broad_exploration

# Experiment 2: Deep focused exploration
poetry run seedgraph grow \
  --prompt "Neural networks learn representations" \
  --max-nodes 200 \
  --top-k 3 \
  --max-depth 8 \
  --run-id deep_exploration
```

## Troubleshooting

### "Model not found"
- First run downloads Qwen-2.5-0.5B (~500MB)
- Requires internet connection
- Model cached in `~/.cache/huggingface/`

### "Out of memory"
- Use `--no-pca` for less memory overhead
- Reduce `--max-nodes`
- Reduce `--top-k` (less branching)

### "Too slow"
- Reduce `--max-nodes` and `--max-depth`
- Use GPU: model will auto-detect and use if available
- Consider reducing `--pca-dims`

## Next Steps

1. **Experiment with prompts**: Try different seed texts
2. **Tune parameters**: Adjust top-k, depth, PCA dims
3. **Analyze results**: Load checkpoints and visualize with NetworkX
4. **Compare runs**: Generate graphs with different settings

Enjoy exploring the logit manifold!
