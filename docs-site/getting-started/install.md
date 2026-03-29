# Installation

IMI requires Python 3.11 or higher.

## Basic install

```bash
pip install imi-memory
```

This installs the core package with numpy, sentence-transformers, and chromadb.

## Install with extras

IMI ships four optional extras:

| Extra | What it adds | Install command |
|-------|-------------|-----------------|
| `llm` | Anthropic SDK for LLM-powered encoding, affordances, consolidation | `pip install "imi-memory[llm]"` |
| `mcp` | MCP server to expose IMI as tools for Claude Code or any MCP client | `pip install "imi-memory[mcp]"` |
| `api` | FastAPI + Uvicorn REST server | `pip install "imi-memory[api]"` |
| `spatial` | UMAP + HDBSCAN for topology visualization | `pip install "imi-memory[spatial]"` |

Install everything at once:

```bash
pip install "imi-memory[all]"
```

## Development install

```bash
git clone https://github.com/RAG7782/imi
cd imi
pip install -e ".[all,dev]"
```

Run the test suite:

```bash
pytest tests/
```

## Verify the install

```python
from imi.space import IMISpace
space = IMISpace.from_sqlite("test.db")
print(space.stats())
```

Expected output:

```
{'episodic_total': 0, 'semantic_total': 0, ...}
```

## Dependencies

| Package | Purpose | Required |
|---------|---------|----------|
| `numpy >= 1.26` | Embeddings and scoring | Yes |
| `sentence-transformers >= 3.0` | Default embedder (all-MiniLM-L6-v2) | Yes |
| `chromadb >= 0.5` | ChromaDB backend (optional storage) | Yes |
| `anthropic >= 0.80` | LLM adapter for encoding | With `[llm]` |
| `mcp[cli] >= 1.0` | MCP server transport | With `[mcp]` |
| `fastapi >= 0.115` | REST API | With `[api]` |
| `uvicorn >= 0.34` | ASGI server | With `[api]` |
| `umap-learn >= 0.5` | Dimensionality reduction | With `[spatial]` |
| `hdbscan >= 0.8` | Clustering | With `[spatial]` |
