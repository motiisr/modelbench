# ModelbBench

Benchmark LLMs on your local hardware. Get TTFT, tokens/sec, VRAM usage, and estimated self-hosting cost — in one command.

```
pip install modelbench
modelbench run llama3.2:3b
```

## Output

```
ModelbBench v0.1.0 — llama3.2:3b on ollama (rtx4080super)
──────────────────────────────────────────────────────────
  Model load time     8.2s
  Prompt suite        v1 (5 prompts, temp=0.0)
  Warm-up             5 requests (not counted)

┌─────────────────────┬──────────┬──────────┬──────────┐
│ Metric              │   p50    │   p90    │   p99    │
├─────────────────────┼──────────┼──────────┼──────────┤
│ Time to first token │  180ms   │  210ms   │  280ms   │
│ End-to-end latency  │  4200ms  │  5100ms  │  6800ms  │
│ Tokens / sec        │  45.2    │    —     │    —     │
│ VRAM (peak)         │  3840 MB │    —     │    —     │
└─────────────────────┴──────────┴──────────┴──────────┘
```

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- NVIDIA GPU (VRAM stats require `nvidia-smi`)

## Commands

```bash
# Run benchmark
modelbench run llama3.2:3b

# View stored results
modelbench show
modelbench show llama3.2:3b
```

Results are saved to `~/.modelbench/benchmarks.json`.

## Stack

- Backend: Ollama (V1)
- Platform: WSL2 / Linux + NVIDIA GPU

## License

MIT
