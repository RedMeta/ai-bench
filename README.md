# Local Ollama Benchmark

A small, repeatable benchmark for locally installed Ollama models. The script
runs coding, debugging, architecture, tool-call, and long-context prompts and
records model responses, Ollama timing/token metrics, and system telemetry.

## Requirements

- Python 3.10 or newer
- Ollama running locally at `http://localhost:11434`
- The models being benchmarked already installed in Ollama
- `nvidia-smi` is optional; without it, GPU telemetry is omitted

The script uses only the Python standard library.

## Run

```bash
python3 ollama_bench.py
```

By default, this runs `qwen3.5:4b` and `bonsai-2:9b`. To select models or
adjust generation settings:

```bash
python3 ollama_bench.py --models qwen3.5:4b bonsai-2:9b --ctx 4096 --predict 600
```

Run `python3 ollama_bench.py --help` for all options. Each run creates a
timestamped directory under `llm-bench-results/`, containing a summary CSV,
per-test response JSON and telemetry CSV files, run metadata, and combined
results JSON.

The tool-call test records whether a model emits a tool call; it does not
execute tools. Results are hardware- and configuration-dependent and are not
intended as a general model ranking.

## Included results

`llm-bench-results/20261120-131836/` contains one captured run for
`qwen3.5:4b` and `bonsai-2:9b`. The response files and telemetry are included
to make that run inspectable and reproducible; they may contain machine
configuration and generated model output.
