# Continue.dev Configuration

This directory contains the [Continue.dev](https://continue.dev) configuration for local AI-assisted coding in the chatty-buoy project.

## Prerequisites

Before using Continue, ensure the local infrastructure is running:

1. **vLLM** (LLM inference): `kanoa mlops serve vllm olmo3`
2. **Ollama** (embeddings): `kanoa mlops serve ollama`
3. **Embedding model**: `docker exec -it kanoa-ollama ollama pull nomic-embed-text`

Verify services are running:
```bash
curl http://localhost:8000/v1/models  # vLLM
curl http://localhost:11434/api/tags  # Ollama
```

## Configuration

`config.json` points to:
- **Chat/Autocomplete**: Local vLLM server (Olmo 3 7B) on port 8000
- **Embeddings**: Local Ollama (nomic-embed-text) on port 11434
- **Privacy**: Telemetry disabled, no external API calls

## Usage

After services are running:
1. Reload VSCode window
2. **Autocomplete**: Type code, wait for gray suggestions
3. **Chat**: Press `Ctrl+L`, ask questions with `@codebase` for context

See [kanoa-mlops local AI coding setup docs](https://github.com/lhzn-io/kanoa-mlops) for detailed setup instructions.
