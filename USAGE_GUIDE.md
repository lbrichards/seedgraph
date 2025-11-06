# SeedGraph Usage Guide

## 🎯 Understanding How SeedGraph Works

### Key Principle: **Novelty-Driven, Not Exhaustive**

SeedGraph doesn't explore every possible branch. Instead, it:
1. Generates nodes by following top-k tokens from Qwen
2. **Selects** which nodes to expand based on novelty (KL divergence + FAISS distance)
3. Stops when it hits `max_nodes` or `max_depth`

This means:
- ✅ You get EXACTLY the number of nodes you request
- ✅ Depth is controlled and won't explode
- ✅ Most nodes are NOT expanded (they're leaves)
- ✅ Expansion follows interesting/novel directions only

## 🛡️ Safety Features

### 1. Hard Limits (Always Enforced)
```bash
--max-nodes N      # Stop after exactly N nodes (default: 1000)
--max-depth D      # Never go deeper than D levels (default: 10)
--top-k K          # Each node branches to at most K children (default: 10)
```

### 2. Selective Expansion
- Only ~1-10% of nodes get expanded (varies by prompt and novelty)
- Remaining nodes are unexpanded leaves
- System prioritizes novel/interesting branches

### 3. Checkpoint Safety
```bash
--checkpoint-interval 50   # Save progress every 50 nodes
```
- If run is interrupted, you can resume from checkpoint
- Prevents losing progress on long runs

## 📊 What Happens with Different Settings

### Example 1: Quick Exploration (Recommended for Testing)
```bash
seedgraph grow \
  --prompt "Neural networks learn representations" \
  --max-nodes 100 \
  --max-depth 4 \
  --top-k 5 \
  --no-pca
```

**Expected:**
- Runtime: ~30 seconds (CPU), ~10 seconds (GPU)
- Nodes expanded: ~10-20 (10-20%)
- Max depth: 4 (enforced)
- Disk: ~50 MB checkpoint
- Result: Sparse tree focusing on novel directions

### Example 2: Medium Exploration
```bash
seedgraph grow \
  --prompt "Large language models generate text" \
  --max-nodes 1000 \
  --max-depth 6 \
  --top-k 8 \
  --pca-dims 64
```

**Expected:**
- Runtime: ~5 minutes (CPU), ~2 minutes (GPU)
- Nodes expanded: ~50-100 (5-10%)
- Max depth: 6 (enforced)
- Disk: ~500 MB checkpoint
- Result: Deeper exploration with PCA-enhanced coverage

### Example 3: Large Exploration (Multi-Hour)
```bash
seedgraph grow \
  --prompt "Artificial intelligence systems" \
  --max-nodes 10000 \
  --max-depth 8 \
  --top-k 10 \
  --pca-dims 128 \
  --checkpoint-interval 500
```

**Expected:**
- Runtime: ~50 minutes (CPU), ~20 minutes (GPU)
- Nodes expanded: ~500-1000 (5-10%)
- Max depth: 8 (enforced)
- Disk: ~5 GB checkpoint
- Result: Comprehensive manifold coverage

### Example 4: VERY Large Exploration (What You Asked About)
```bash
seedgraph grow \
  --prompt "Knowledge representation in AI" \
  --max-nodes 1000000 \
  --max-depth 10 \
  --top-k 10 \
  --pca-dims 256 \
  --checkpoint-interval 5000
```

**Expected:**
- Runtime: ~80 hours (CPU), ~30 hours (GPU)
- Nodes expanded: ~10,000-50,000 (1-5%)
- Max depth: 10 (enforced)
- Disk: ~500 GB checkpoint
- Memory: ~8 GB
- Result: Massive sparse graph with 990,000+ leaf nodes

**⚠️ Considerations for 1M nodes:**
1. **Disk Space**: You need ~500 GB free
2. **Time**: Plan for 1-3 days of runtime
3. **Interruptions**: Use checkpoints! Can resume if interrupted
4. **Analysis**: Consider breaking into multiple smaller runs

## 🎮 Parameter Tuning Guide

### `--max-nodes` (How many nodes total)
- **Small (10-100)**: Quick tests, understanding behavior
- **Medium (100-10K)**: Typical research use
- **Large (10K-100K)**: Comprehensive exploration
- **Very Large (100K-1M+)**: Extended experiments (plan accordingly)

### `--max-depth` (How deep to go)
- **Shallow (2-4)**: Broad exploration, many branches
- **Medium (5-8)**: Balanced depth and breadth
- **Deep (9-15)**: Follow specific paths deeply
- **Very Deep (16+)**: Extreme focus (rare)

### `--top-k` (Branching factor)
- **Low (2-4)**: Focused exploration, fewer branches
- **Medium (5-10)**: Balanced (default: 10)
- **High (11-20)**: More branches, broader coverage

### `--pca-dims` (Coverage precision)
- **Low (16-64)**: Faster, less precise coverage
- **Medium (64-256)**: Balanced (default: 256)
- **High (256-512)**: More precise, slower
- **None (`--no-pca`)**: Fastest, no dim reduction

## 🔍 Understanding Your Results

After running, check the statistics:

```bash
# The CLI shows this automatically
Total Nodes: 1000
Total Edges: 999
Max Depth: 6
Expanded: 87 (8.7%)
Unexpanded: 913 (91.3%)
```

**Interpretation:**
- **87 expanded nodes**: These were selected as "novel" and branched
- **913 unexpanded**: These are leaves (not interesting enough to expand)
- **Max depth 6**: Hit the depth limit (or no novel paths beyond)

### Good Expansion Rates
- **1-5%**: Very selective (high novelty threshold)
- **5-15%**: Normal (balanced exploration)
- **15-30%**: Broad (lower novelty threshold)
- **>30%**: Potentially exploring too much (consider lowering top-k)

## 💡 Practical Tips

### 1. Start Small, Scale Up
```bash
# First: Test with 50 nodes
seedgraph grow --prompt "Your topic" --max-nodes 50 --no-pca

# If interesting: Scale to 500
seedgraph grow --prompt "Your topic" --max-nodes 500 --pca-dims 64

# Then: Go larger if needed
seedgraph grow --prompt "Your topic" --max-nodes 5000 --pca-dims 128
```

### 2. Use Checkpoints for Long Runs
```bash
# For runs >1 hour, checkpoint frequently
seedgraph grow \
  --prompt "Your topic" \
  --max-nodes 10000 \
  --checkpoint-interval 100  # Save every 100 nodes
```

### 3. Monitor Resource Usage
```bash
# Check disk space before large runs
df -h checkpoints/

# Monitor memory during run
top -p $(pgrep -f seedgraph)
```

### 4. Compare Different Prompts
```bash
# Generate graphs for different starting points
seedgraph grow --prompt "Neural networks" --max-nodes 1000 --run-id neural
seedgraph grow --prompt "Transformers" --max-nodes 1000 --run-id transformers
seedgraph grow --prompt "Diffusion models" --max-nodes 1000 --run-id diffusion

# Compare coverage patterns
```

## 🚨 Common Pitfalls to Avoid

❌ **Don't**: Request 1M nodes without testing first
✅ **Do**: Start with 100, then 1000, then scale

❌ **Don't**: Use very high `--top-k` (>20) with deep trees
✅ **Do**: Use moderate top-k (5-10) for balanced exploration

❌ **Don't**: Forget to check disk space for large runs
✅ **Do**: Estimate ~5KB per node for checkpoint size

❌ **Don't**: Run without checkpoints for long experiments
✅ **Do**: Set `--checkpoint-interval` for runs >10 minutes

## 📈 Example Workflows

### Research: Explore Semantic Space
```bash
# Phase 1: Quick survey (5 minutes)
seedgraph grow --prompt "Machine learning methods" --max-nodes 500

# Phase 2: Deep dive on interesting areas (30 minutes)
seedgraph grow --prompt "Reinforcement learning algorithms" --max-nodes 5000

# Phase 3: Comprehensive coverage (few hours)
seedgraph grow --prompt "Deep reinforcement learning" --max-nodes 50000
```

### Analysis: Compare Topics
```bash
# Generate comparable graphs
for topic in "supervised learning" "unsupervised learning" "reinforcement learning"; do
  seedgraph grow \
    --prompt "$topic" \
    --max-nodes 1000 \
    --max-depth 5 \
    --run-id "${topic// /_}"
done
```

### Production: Large-Scale Mapping
```bash
# Multi-day comprehensive exploration
seedgraph grow \
  --prompt "Artificial intelligence landscape" \
  --max-nodes 100000 \
  --max-depth 10 \
  --top-k 8 \
  --checkpoint-interval 1000 \
  --run-id ai_landscape_2025 \
  --verbose
```

## 🎓 Advanced: Understanding the Algorithm

The selection algorithm works as follows:

```python
# Pseudocode for node selection
for step in range(max_nodes):
    # Get unexpanded nodes within depth limit
    candidates = get_unexpanded_nodes(max_depth)

    # Compute novelty score for each
    scores = []
    for node in candidates:
        # FAISS L2 distance (spatial coverage)
        spatial_novelty = distance_to_nearest_in_faiss(node.probs)

        # KL divergence (information coverage)
        info_novelty = min_kl_to_centroids(node.probs)

        # Combined score
        scores.append(spatial_novelty + info_novelty)

    # Select highest novelty
    best_node = candidates[argmax(scores)]

    # Expand it (create top-k children)
    for token in best_node.top_k_tokens:
        create_child_node(best_node, token)
```

This ensures:
- ✅ Nodes with unique distributions get expanded
- ✅ Similar nodes are left as leaves
- ✅ Coverage is maximized, not redundancy
