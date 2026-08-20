# Ollama shell with llama.cpp MTP

The `ollama-mtp` command runs the real Ollama CLI against a local compatibility bridge. The bridge starts Qwen3.8-27B UD-Q4_K_XL in llama.cpp with its embedded MTP head and stops it when the CLI exits.

## Interactive shell

```bash
ollama
# or
ollama run
# or explicitly
ollama run qwen3.8-MTP:27b
```

`qwen3.8-MTP:27b` is the default model for the local `ollama` wrapper. Both Q4 and the explicit Q6 launcher use MTP depth 10 by default.

The Q6 variant is available separately and does not replace the Q4 default:

```bash
ollama run qwen3.8-MTP:27b-q6
```

The virtual MTP model is also included in the normal list:

```bash
ollama list
```

The explicit `ollama-mtp` command remains available as a shortcut.

Normal Ollama shell commands such as `/clear`, `/set`, `/?`, and `/bye` remain available.

## One-shot prompt

```bash
ollama run qwen3.8-MTP:27b "Briefly explain what MTP is."
```

Verbose timing:

```bash
ollama-mtp run qwen3.8-MTP:27b --verbose "Return exactly one word: READY"
```

## Implementation

- Ollama-compatible API: `127.0.0.1:11435`
- llama.cpp backend: `127.0.0.1:18080`
- Context: 65,536 tokens
- MTP: `--spec-type draft-mtp --spec-draft-n-max 10` (operational compromise from the depth sweep: 37.01 tok/s, 4.03x baseline, 87.98% acceptance, a 25.62 tok/s minimum case, and 9/9 objective and semantic checks)
- Q6 bridge/API: `127.0.0.1:11436`, llama.cpp backend: `127.0.0.1:18081`
- Clean content parsing: `--reasoning-format deepseek`
- Both services bind only to localhost and automatically stop after `/bye`.

All model names except `qwen3.8-MTP:27b` continue to use the system Ollama backend. The former lowercase alias `qwen3.8-mtp:27b` is still accepted for compatibility. This bridge currently targets that single text model; vision and Ollama model-management operations are not proxied.
