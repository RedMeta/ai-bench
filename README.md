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

By default, this runs every model currently installed in Ollama, so newly
installed models are included automatically. To select models or adjust
generation settings:

```bash
python3 ollama_bench.py --models qwen3.5:4b bonsai-2:9b --ctx 8192 --repeat 3 --vary-seed
```

The benchmark warms each model before measuring it, uses a fixed seed and
temperature, and records one measured run per test by default. `--repeat N`
repeats each test to show run-to-run variation; `--no-warmup` disables warmup.
By default, every repeat uses seed `42`; use `--seed N` to select another seed
or `--vary-seed` to increment it for each repetition and compare different
outputs in one run.
`--predict N` overrides every test's generation cap, which can truncate longer
answers. Run `python3 ollama_bench.py --help` for all options.

Each run creates a timestamped directory under the ignored
`llm-bench-results/` directory, with raw responses, telemetry, run metadata,
and CSV/JSON summaries. The tool-use test executes only deterministic local
mock tools and checks the full lookup → calculation → answer sequence. The
long-context test checks both an exact JSON fact object and required checklist
items. Coding and architecture answers remain available for qualitative review.
The debugging task has a machine-checked rubric for two reproducible defects
and two deliberate non-bugs, with separate credit for identifying each bug,
naming its exception, and proposing a correction. Coding and architecture
receive static heuristic scores; these identify obvious omissions and common
errors but do not execute or prove generated code correct. Review those answers
manually. A nonzero exit indicates an API, telemetry, completion, or strict
tool/context check failure; a partial debugging score is reported without
failing the whole run.

Generation tasks request answer-only output so a visible reasoning trace does
not consume the budget needed for the answer. The prompts still ask for
explanations and design justification where relevant.

Run the local unit tests with:

```bash
python3 -m unittest discover -s tests -v
```
