#!/usr/bin/env python3
"""
Local Ollama coding-model benchmark.
- Runs a repeatable test suite against each configured model.
- Records raw responses, Ollama timing/token metrics, GPU samples, RAM and swap.
- Uses only Python's standard library and the local Ollama HTTP API.

Usage:
  python3 ollama_bench.py
  python3 ollama_bench.py --models qwen3.5:4b bonsai-2:9b
  python3 ollama_bench.py --ctx 4096 --predict 600
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

OLLAMA_URL = "http://localhost:11434"

TESTS = [
    {
        "id": "coding",
        "name": "Basic coding",
        "kind": "generate",
        "predict": 650,
        "prompt": """Write a production-conscious Python function that finds duplicate files under a directory.
Requirements:
- Use SHA-256, but first group files by size to avoid hashing files that cannot be duplicates.
- Read files in chunks; never load an entire file into memory.
- Handle permission errors and files disappearing during the scan without crashing.
- Return groups of duplicate paths (only groups containing at least two files).
- Include type hints, a short usage example, and a few important limitations.
Return code and concise explanation."""
    },
    {
        "id": "debugging",
        "name": "Debugging",
        "kind": "generate",
        "predict": 650,
        "prompt": """Review this Python code. Identify every meaningful bug, explain why it occurs, and provide a corrected implementation.

```python
import threading

class Cache:
    def __init__(self):
        self.data = {}
        self.lock = threading.Lock()

    def get_or_load(self, key, loader):
        if key in self.data:
            return self.data[key]
        with self.lock:
            if key not in self.data:
                value = loader()
                self.data[key] = value
        return value

def read_config(path):
    f = open(path)
    try:
        return f.read()
    finally:
        f.close()

def average(values):
    return sum(values) / len(values)

def save_items(items, path):
    with open(path, "w") as f:
        for item in items:
            f.write(item + "\\n")
```

Consider concurrency, exceptions, resource handling, empty input, and data types. Do not invent problems that are not present."""
    },
    {
        "id": "architecture",
        "name": "Reasoning and architecture",
        "kind": "generate",
        "predict": 800,
        "prompt": """Design a transaction-safe SQLite repository layer for a small multi-user CRM.
Requirements:
- Python, SQLite, multiple application threads, and a connection-per-unit-of-work approach.
- Create/update a Deal record with required fields cliente, pfpg (PF or PG), competenza, referente; optional fields ingresso, fattibilita, istituto, dataInvio, stato, esito, note.
- Validate input, use parameterized SQL, enforce transaction boundaries, rollback on failure, and avoid lost updates.
- Explain SQLite locking/WAL/busy_timeout trade-offs.
- Show concise but coherent schema, repository methods, and a service-layer transaction example.
State assumptions and distinguish what SQLite can and cannot guarantee."""
    },
    {
        "id": "tool_call",
        "name": "Tool calling",
        "kind": "chat",
        "predict": 350,
        "prompt": """You have access to a calculator and a lookup_customer tool.
The user asks: "Look up customer C-104, then calculate the customer's annual spend if their monthly spend is 187.50 euros. Do not guess the customer data."
You must call lookup_customer first. If the tool returns a monthly spend, use the calculator to multiply it by 12. If the lookup returns no customer, explain that you cannot calculate it.
Call the appropriate tool(s); do not merely describe what you would do.""",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup_customer",
                    "description": "Look up a customer record by customer ID.",
                    "parameters": {
                        "type": "object",
                        "properties": {"customer_id": {"type": "string"}},
                        "required": ["customer_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Multiply two numbers.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"}
                        },
                        "required": ["a", "b"]
                    }
                }
            }
        ]
    },
    {
        "id": "long_context",
        "name": "Context retention",
        "kind": "generate",
        "predict": 450,
        "prompt": """You are reviewing a small application. Treat the following project notes as authoritative. Some details are deliberately easy to confuse.

PROJECT NOTES
- Framework: Nuxt 4.
- Database: SQLite using better-sqlite3 and Prisma.
- Prisma generated client output: ../app/generated/prisma.
- Authentication: cookie-based JWT.
- Roles: ADMIN and CLIENT.
- Admin can CRUD all deals.
- A CLIENT can read only deals assigned to that client.
- Deal required fields: cliente, pfpg (enum PF/PG), competenza, referente.
- Optional fields: ingresso, fattibilita, istituto, dataInvio, stato, esito, note.
- The application is a client portal, not a public API.
- Do not recommend replacing SQLite unless a concrete requirement makes it necessary.

TASK
Produce a concise implementation checklist for adding an endpoint that lets a CLIENT read their own deals. Include authorization checks, query scoping, validation, error handling, and tests.

At the end, state exactly:
1. The database engine.
2. The Prisma client output path.
3. The two role names.
4. The four required Deal fields.
Do not substitute similar-looking field names."""
    },
]

def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")

def post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))

def get_json(url: str, timeout: int = 10) -> dict[str, Any]:
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))

def read_memory() -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            match = re.match(r"(MemTotal|MemAvailable|SwapTotal|SwapFree):\s+(\d+)", line)
            if match:
                result[match.group(1)] = int(match.group(2)) * 1024
    except OSError:
        pass
    return result

def gpu_snapshot() -> dict[str, Any]:
    query = "timestamp,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu"
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True
        )
        fields = [x.strip() for x in completed.stdout.strip().split(",")]
        if len(fields) >= 6:
            return {
                "gpu_timestamp": fields[0],
                "gpu_util_percent": to_num(fields[1]),
                "gpu_memory_used_mib": to_num(fields[2]),
                "gpu_memory_total_mib": to_num(fields[3]),
                "gpu_power_w": to_num(fields[4]),
                "gpu_temp_c": to_num(fields[5]),
            }
    except Exception:
        pass
    return {}

def to_num(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

class Monitor:
    def __init__(self, output: Path, interval: float):
        self.output = output
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3)

    def _run(self) -> None:
        fields = [
            "sample_time", "gpu_timestamp", "gpu_util_percent",
            "gpu_memory_used_mib", "gpu_memory_total_mib",
            "gpu_power_w", "gpu_temp_c",
            "ram_total_gib", "ram_available_gib",
            "swap_total_gib", "swap_free_gib"
        ]
        with self.output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            while not self.stop_event.is_set():
                mem = read_memory()
                row: dict[str, Any] = {"sample_time": now_iso()}
                row.update(gpu_snapshot())
                for source, dest in [
                    ("MemTotal", "ram_total_gib"),
                    ("MemAvailable", "ram_available_gib"),
                    ("SwapTotal", "swap_total_gib"),
                    ("SwapFree", "swap_free_gib"),
                ]:
                    row[dest] = round(mem[source] / (1024**3), 3) if source in mem else None
                writer.writerow(row)
                f.flush()
                self.stop_event.wait(self.interval)

def ollama_request(model: str, test: dict[str, Any], ctx: int, timeout: int) -> dict[str, Any]:
    if test["kind"] == "chat":
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": test["prompt"]}],
            "tools": test["tools"],
            "stream": False,
            "options": {"num_ctx": ctx, "num_predict": test["predict"], "temperature": 0}
        }
        return post_json(f"{OLLAMA_URL}/api/chat", payload, timeout)
    payload = {
        "model": model,
        "prompt": test["prompt"],
        "stream": False,
        "options": {"num_ctx": ctx, "num_predict": test["predict"], "temperature": 0}
    }
    return post_json(f"{OLLAMA_URL}/api/generate", payload, timeout)

def extract_metrics(response: dict[str, Any]) -> dict[str, Any]:
    # Ollama durations are nanoseconds.
    eval_ns = response.get("eval_duration") or 0
    prompt_ns = response.get("prompt_eval_duration") or 0
    eval_count = response.get("eval_count") or 0
    return {
        "prompt_tokens": response.get("prompt_eval_count"),
        "generated_tokens": eval_count,
        "generation_seconds": round(eval_ns / 1e9, 3) if eval_ns else None,
        "tokens_per_second": round(eval_count / (eval_ns / 1e9), 2) if eval_ns and eval_count else None,
        "prompt_eval_seconds": round(prompt_ns / 1e9, 3) if prompt_ns else None,
        "load_seconds": round((response.get("load_duration") or 0) / 1e9, 3),
        "total_seconds": round((response.get("total_duration") or 0) / 1e9, 3),
        "done_reason": response.get("done_reason"),
    }

def response_text(response: dict[str, Any]) -> str:
    if "response" in response:
        return response.get("response") or ""
    message = response.get("message") or {}
    return message.get("content") or ""

def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark locally installed Ollama models.")
    parser.add_argument("--models", nargs="+", default=["qwen3.5:4b", "bonsai-2:9b"])
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--predict", type=int, default=None,
                        help="Override per-test generation limit.")
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--out", default="llm-bench-results")
    parser.add_argument("--warmup", action="store_true",
                        help="Send a short warm-up request before measured tests.")
    args = parser.parse_args()

    outdir = Path(args.out).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = outdir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        tags = get_json(f"{OLLAMA_URL}/api/tags")
    except Exception as exc:
        print(f"ERROR: Cannot reach Ollama at {OLLAMA_URL}: {exc}", file=sys.stderr)
        return 2

    installed = {m.get("name") for m in tags.get("models", [])}
    missing = [m for m in args.models if m not in installed]
    if missing:
        print("ERROR: These models are not installed: " + ", ".join(missing), file=sys.stderr)
        print("Installed: " + ", ".join(sorted(x for x in installed if x)), file=sys.stderr)
        return 2

    metadata = {
        "run_id": run_id,
        "started_at": now_iso(),
        "host": platform.platform(),
        "python": sys.version,
        "ollama_url": OLLAMA_URL,
        "models": args.models,
        "context": args.ctx,
        "tests": [t["id"] for t in TESTS],
        "note": "Tool-call test measures whether the model emits a tool call; tools are not executed."
    }
    (run_dir / "run-info.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    all_results: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    total = len(args.models) * (len(TESTS) + (1 if args.warmup else 0))
    step = 0

    for model in args.models:
        safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model)
        print(f"\n=== MODEL: {model} ===", flush=True)

        if args.warmup:
            step += 1
            print(f"[{step}/{total}] Warm-up", flush=True)
            try:
                post_json(f"{OLLAMA_URL}/api/generate", {
                    "model": model, "prompt": "Reply with: ready",
                    "stream": False,
                    "options": {"num_ctx": args.ctx, "num_predict": 16, "temperature": 0}
                }, args.timeout)
            except Exception as exc:
                print(f"Warm-up failed: {exc}", flush=True)

        for test in TESTS:
            step += 1
            print(f"[{step}/{total}] {model} — {test['name']}", flush=True)
            test_copy = dict(test)
            if args.predict is not None:
                test_copy["predict"] = args.predict

            monitor_path = run_dir / f"{safe_model}-{test['id']}-telemetry.csv"
            response_path = run_dir / f"{safe_model}-{test['id']}-response.json"
            monitor = Monitor(monitor_path, args.interval)
            started = time.time()
            monitor.start()
            try:
                response = ollama_request(model, test_copy, args.ctx, args.timeout)
                status = "ok"
                error = None
            except Exception as exc:
                response = {}
                status = "error"
                error = f"{type(exc).__name__}: {exc}"
            finally:
                elapsed = time.time() - started
                monitor.stop()

            response_path.write_text(
                json.dumps(response if response else {"error": error}, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            metrics = extract_metrics(response) if response else {}
            row = {
                "model": model,
                "test_id": test["id"],
                "test_name": test["name"],
                "status": status,
                "wall_seconds": round(elapsed, 2),
                **metrics,
                "response_file": response_path.name,
                "telemetry_file": monitor_path.name,
                "error": error,
            }
            all_results.append(row)
            summary_rows.append(row)
            print(
                f"  status={status} | wall={elapsed:.1f}s | "
                f"tokens/s={metrics.get('tokens_per_second', 'n/a')} | "
                f"generated={metrics.get('generated_tokens', 'n/a')}",
                flush=True
            )
            if error:
                print(f"  ERROR: {error}", flush=True)
            elif test["id"] == "tool_call":
                msg = response.get("message", {})
                calls = msg.get("tool_calls") or []
                print(f"  tool calls emitted: {len(calls)}", flush=True)

    (run_dir / "results.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    csv_path = run_dir / "summary.csv"
    columns = [
        "model", "test_id", "test_name", "status", "wall_seconds",
        "prompt_tokens", "generated_tokens", "generation_seconds",
        "tokens_per_second", "prompt_eval_seconds", "load_seconds",
        "total_seconds", "done_reason", "response_file", "telemetry_file", "error"
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summary_rows)

    metadata["finished_at"] = now_iso()
    (run_dir / "run-info.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n=== COMPLETE ===")
    print(f"Results folder: {run_dir}")
    print(f"Summary:        {csv_path}")
    print("Send me the entire results folder as a ZIP, or at minimum summary.csv + telemetry CSVs + response JSONs.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
