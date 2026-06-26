# BiomedAgent

A single LangChain/LangGraph ReAct agent that answers biomedical research
questions by calling read-only tools over public bioinformatics services
(MyGene.info, Open Targets, GTEx, Human Protein Atlas, STRING, KEGG, g:Profiler,
AtlasApprox, AlphaGenome). The LLM runs **locally via [Ollama](https://ollama.com)** —
no cloud inference.

## Prerequisites

- [Ollama](https://ollama.com) installed and running
- [Pixi](https://pixi.sh) for dependency management

## Installation

1. **Pull the model** (hardcoded as `LLM_MODEL` in `unified_agent.py`):

   ```bash
   ollama pull qwen3.5:9b
   ```

2. **Install the Python environment** with Pixi:

   ```bash
   cd dependencies
   pixi install
   ```

3. **Configure secrets** — create a `.env` file in the project root:

   ```ini
   # Required only for predict_genomics (AlphaGenome AI predictions)
   ALPHAGENOME_API_KEY=your_key_here

   # Optional — raises NCBI E-utilities rate limit from 3 to 10 req/s
   PUBMED_API_KEY=your_key_here
   ```

   All other data sources (MyGene, Open Targets, GTEx, HPA, STRING, KEGG,
   g:Profiler, AtlasApprox) require **no key**.

## Usage

Run the interactive REPL:

```bash
pixi run --manifest-path dependencies/pixi.toml python unified_agent.py
```

Then type a question, e.g.:

```
>>> What diseases are associated with TP53?
>>> Which cell types express CD8A in humans?
```

Commands:

- `exit` / `quit` — stop the model in Ollama and quit
- `unload` — drop the model from VRAM without quitting

Conversation state is kept within a session, so follow-up questions share context.

## Notes

- `alphagenome` and `atlasapprox` are optional; if either is unavailable the
  agent still runs, just without AlphaGenome predictions or AtlasApprox
  single-cell tools.
- HTTP responses are cached in a local SQLite database (`http_cache.sqlite`) for
  24h, which speeds up repeated lookups and reduces upstream load.
