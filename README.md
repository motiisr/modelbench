# ModelBench

Benchmark LLMs on your local hardware, or replay real traffic against candidate models to verify quality and cost before you switch. Published on PyPI as `tokmark`.

```
pip install tokmark
modelbench run llama3.2:3b
```

## Benchmark: local hardware

Get TTFT, tokens/sec, VRAM usage, and estimated self-hosting cost — in one command.

```
ModelBench v0.3.0 — llama3.2:3b on ollama (rtx4080super)
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

```bash
modelbench run llama3.2:3b
modelbench show                # view stored results
modelbench show llama3.2:3b
```

Results are saved to `~/.modelbench/benchmarks.json`.

## Replay: verify a candidate model against your own traffic

Given a JSONL file of prompts (one JSON object per line, `{"prompt": "...", "expected": "..."}` — `expected` is optional), replay them against one or more candidate models and compare cost, latency, and measured quality:

```bash
modelbench replay samples.jsonl --candidates llama3.2:3b,haiku,gpt-4o-mini
```

```
Candidate       Cost/1M tokens          p50 latency   Tokens/sec   Quality
─────────────────────────────────────────────────────────────────────────
llama3.2:3b     $0.04 (self-hosted)     320ms         45.2         0.91 (92% conf, exact)
haiku           cloud (see provider pricing)  280ms   38.1         0.94 (60% conf, llm_judge)
gpt-4o-mini     cloud (see provider pricing)  310ms   41.5         0.96 (60% conf, llm_judge)
```

- **Candidates**: a known alias (`haiku`, `sonnet`, `gpt-4o`, `gpt-4o-mini`) or an Ollama model id containing `:` (e.g. `llama3.3:70b`). Cloud candidates read `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` from your environment.
- **Quality**: exact/fuzzy string match against `expected` when present (confidence always 100%); an LLM judge (`--judge <candidate>`) when `expected` is omitted — judge scores always carry a confidence label below 100%, never presented as ground truth.
- **`--repeats N`** (default 3): each sample is run N times per candidate; the table and `--json` output report the full score distribution, not just a mean.
- **`--json`**: prints the full `ReplayReport` as JSON instead of the table — used to paste a result into [Tokensor's recommendation endpoint](https://github.com/motiisr/tokensor/blob/main/docs/integrations/modelbench-replay.md) to verify a cost-savings Finding.

```bash
modelbench replay samples.jsonl \
  --candidates llama3.2:3b,haiku,gpt-4o-mini \
  --judge haiku \
  --repeats 3 \
  --hardware-cost 1.50 \
  --json
```

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) installed and running, for local/self-hosted candidates
- NVIDIA GPU (VRAM stats require `nvidia-smi`) — benchmark mode only
- `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` env vars for cloud replay candidates

## Stack

- Backends: Ollama, OpenAI, Anthropic
- Platform: WSL2 / Linux + NVIDIA GPU (benchmark mode); any platform for replay against cloud candidates

## License

MIT
