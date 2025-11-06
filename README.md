# SeedGraph

Recursive knowledge-graph generator using Qwen-2.5-0.5B logit introspection.

## Overview

SeedGraph builds recursive knowledge graphs by:
- Introspecting Qwen-2.5-0.5B next-token logits
- Branching on top-k candidate tokens
- Using KL divergence + FAISS-based coverage to guide manifold exploration

## Installation

```bash
poetry install
```

## Usage

```bash
# Grow a knowledge graph from a seed prompt
seedgraph grow --prompt "SeedGraph builds graphs from logits" --max-nodes 100

# Show help
seedgraph --help

# Show info
seedgraph info
```

## Architecture

- **LLM Layer** (`llm/qwen.py`): Logit introspection via Qwen-2.5-0.5B
- **Graph Layer** (`core/graph_store.py`): Node/edge management
- **Selection Layer** (`core/selection.py`): KL divergence + FAISS coverage
- **Orchestration** (`core/brancher.py`): Recursive graph growth
- **CLI** (`cli.py`): Command-line interface

## Testing

```bash
# Run all tests
make test

# Run with pytest directly
poetry run pytest
```

## Features

- ✅ Qwen-2.5-0.5B logit introspection
- ✅ KL divergence-based node selection
- ✅ FAISS-accelerated coverage computation
- ✅ Automatic checkpointing (JSONL format)
- ✅ Recursive graph branching with depth control
- ✅ Rich CLI with progress tracking

## License

See LICENSE file.
