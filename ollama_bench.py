#!/usr/bin/env python3
"""
Local Ollama model benchmark.
- Runs a repeatable test suite against each configured model.
- Records raw responses, Ollama timing/token metrics, GPU samples, RAM and swap.
- Uses only Python's standard library and the local Ollama HTTP API.

Usage:
  python3 ollama_bench.py
  python3 ollama_bench.py --models qwen3.5:4b bonsai-2:9b
  python3 ollama_bench.py --ctx 8192 --repeat 3
"""
from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

OLLAMA_URL = "http://localhost:11434"

TESTS = [
    {
        "id": "coding",
        "name": "Basic coding",
        "kind": "generate",
        "predict": 2000,
        "think": False,
        "prompt": """Write a production-conscious Python function that finds duplicate files under a directory.
Requirements:
- Use SHA-256, but first group files by size to avoid hashing files that cannot be duplicates.
- Read files in chunks; never load an entire file into memory.
- Handle permission errors and files disappearing during the scan without crashing.
- Return groups of duplicate paths (only groups containing at least two files).
- Include type hints, a short usage example, and a few important limitations.
Return the implementation and concise explanation only; do not include analysis or a plan."""
    },
    {
        "id": "debugging",
        "name": "Debugging",
        "kind": "generate",
        "predict": 2000,
        "think": False,
        "prompt": """Review this Python code against the contracts below. Return ONLY one valid JSON object, with no Markdown fences or prose outside it.

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

def save_items(items: list[str], path):
    with open(path, "w") as f:
        for item in items:
            f.write(item + "\\n")
```

Contracts and required cases:
- `Cache.get_or_load` must return the cached value for every caller. In the race, both callers miss the outer check. Caller A acquires the lock, enters the inner `if`, assigns the local variable `value`, stores the value, and releases the lock. Caller B then acquires the lock; the inner `if` is false, so B does not assign its local variable `value`. B leaves the lock and executes `return value`. State the exact exception B raises; do not claim duplicate cache entries or a second call to `loader()`.
- `average` receives a sequence of numbers and must raise `ValueError` for an empty sequence.
- `save_items` accepts strings only. Passing a non-string violates its input contract and is not a bug to report.
- For the cache fix, return `self.data[key]` from inside the lock after the optional load.

`read_config` is a deliberate non-bug: `finally` closes the file even if `read()` raises. Do not report missing type hints or missing exception handling as bugs.

Use this exact JSON shape:
{
  "bugs": [
    {"id": "cache_missing_value", "failure": "exact exception class", "explanation": "why the specified interleaving fails", "fix": "how to fix it"},
    {"id": "average_empty_sequence", "failure": "exact exception class", "explanation": "why empty input fails", "fix": "how to fix it"}
  ],
  "not_bugs": ["read_config_resource_handling", "save_items_string_contract"],
  "corrected_code": "corrected Cache.get_or_load and average implementations only"
}"""
    },
    {
        "id": "architecture",
        "name": "Reasoning and architecture",
        "kind": "generate",
        "predict": 2500,
        "think": False,
        "prompt": """Design a transaction-safe SQLite repository layer for a small multi-user CRM.
Requirements:
- Python, SQLite, multiple application threads, and a connection-per-unit-of-work approach.
- Create/update a Deal record with required fields cliente, pfpg (PF or PG), competenza, referente; optional fields ingresso, fattibilita, istituto, dataInvio, stato, esito, note.
- Validate input, use parameterized SQL, enforce transaction boundaries, rollback on failure, and avoid lost updates.
- Explain SQLite locking/WAL/busy_timeout trade-offs.
- Include a compact schema, create/update repository examples, and a service-layer transaction example.
- State assumptions and distinguish what SQLite guarantees from what it cannot guarantee.
Use these five short headings: Schema and assumptions; Repository; Service transaction; Concurrency; Guarantees and limits. Keep code illustrative and concise, avoid repeating the prompt, and return the final design only."""
    },
    {
        "id": "tool_call",
        "name": "Tool calling",
        "kind": "chat",
        "predict": 512,
        "prompt": """The user asks: "Look up customer C-104 and calculate their annual spend. Do not guess any customer data."
You must first call lookup_customer with the requested ID. After receiving its result, call calculator with the returned monthly_spend and 12. Then report the calculator result in euros. Never calculate from information that was not returned by the lookup.""",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup_customer",
                    "description": "Look up a customer record by ID. The returned record contains monthly_spend.",
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
                    "description": "Multiply two numbers and return the product.",
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
        "predict": 1500,
        "think": False,
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
Return ONLY one valid JSON object, with no Markdown fences or prose outside it. Use this exact shape:
{
  "checklist": {
    "authorization": "concise check",
    "query_scoping": "concise check",
    "validation": "concise check",
    "error_handling": "concise check",
    "tests": "concise check"
  },
  "facts": {
    "database_engine": "exact value",
    "prisma_client_output": "exact path",
    "roles": ["role one", "role two"],
    "required_deal_fields": ["field one", "field two", "field three", "field four"]
  }
}
Use only the project notes. Preserve the exact role names, field names, field order, and path."""
    },
]

LONG_CONTEXT_EXPECTED = {
    "database_engine": "SQLite",
    "prisma_client_output": "../app/generated/prisma",
    "roles": ["ADMIN", "CLIENT"],
    "required_deal_fields": ["cliente", "pfpg", "competenza", "referente"],
}

LONG_CONTEXT_CHECKLIST_KEYS = (
    "authorization",
    "query_scoping",
    "validation",
    "error_handling",
    "tests",
)

DEBUGGING_EXPECTED_BUG_IDS = {"cache_missing_value", "average_empty_sequence"}
DEBUGGING_EXPECTED_NOT_BUGS = {
    "read_config_resource_handling",
    "save_items_string_contract",
}
DEBUGGING_EXPECTED_FAILURES = {
    "cache_missing_value": "UnboundLocalError",
    "average_empty_sequence": "ZeroDivisionError",
}

TOOL_MAX_ROUNDS = 4
TOOL_LOOKUP_RESULT = {"customer_id": "C-104", "monthly_spend": 187.50}
BENCHMARK_SEED = 42


def tool_call_specs(test: dict[str, Any]) -> list[dict[str, Any]]:
    return test["tools"]


def resolve_models(requested: list[str] | None, tags: dict[str, Any]) -> list[str]:
    installed = sorted({
        model["name"]
        for model in tags.get("models", [])
        if isinstance(model, dict) and isinstance(model.get("name"), str)
    })
    if not installed:
        raise ValueError("No Ollama models are installed.")
    models = requested or installed
    if len(set(models)) != len(models):
        raise ValueError("A model was selected more than once.")
    missing = [model for model in models if model not in installed]
    if missing:
        raise ValueError(
            "These models are not installed: " + ", ".join(missing)
            + "\nInstalled: " + ", ".join(installed)
        )
    return models


def execute_tool(name: str, arguments: Any) -> Any:
    if not isinstance(arguments, dict):
        return {"error": "Tool arguments must be a JSON object."}
    if name == "lookup_customer":
        if arguments.get("customer_id") != TOOL_LOOKUP_RESULT["customer_id"]:
            return {"error": "Customer not found."}
        return TOOL_LOOKUP_RESULT.copy()
    if name == "calculator":
        try:
            a = float(arguments["a"])
            b = float(arguments["b"])
        except (KeyError, TypeError, ValueError):
            return {"error": "Calculator requires numeric a and b arguments."}
        if not math.isfinite(a) or not math.isfinite(b):
            return {"error": "Calculator arguments must be finite numbers."}
        return {"result": a * b}
    return {"error": f"Unknown tool: {name}"}


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return parsed


def model_file_prefixes(models: list[str]) -> dict[str, str]:
    groups: dict[str, list[str]] = {}
    for model in models:
        prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", model)
        groups.setdefault(prefix, []).append(model)
    return {
        model: (
            prefix
            if len(group) == 1
            else f"{prefix}-{hashlib.sha256(model.encode('utf-8')).hexdigest()[:8]}"
        )
        for prefix, group in groups.items()
        for model in group
    }


def output_status(test: dict[str, Any], response: dict[str, Any]) -> str:
    if response.get("done_reason") == "length":
        return "truncated"
    if response.get("done") is False:
        return "incomplete"
    if test["kind"] == "chat":
        message = response.get("message") or {}
        if message.get("tool_calls"):
            return "complete"
    if not response_text(response).strip():
        return "empty"
    if response.get("done") is True:
        return "complete"
    return "unknown"


def evaluation(
    status: str, score: float | None, maximum: float | None, details: str
) -> dict[str, Any]:
    return {
        "status": status,
        "score": score,
        "max_score": maximum,
        "details": details,
    }


def python_code_blocks(text: str) -> list[str]:
    blocks = re.findall(r"```(?:python|py)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    return blocks or [text]


def _is_cached_value_return(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Return)
        and isinstance(node.value, ast.Subscript)
        and isinstance(node.value.value, ast.Attribute)
        and isinstance(node.value.value.value, ast.Name)
        and node.value.value.value.id == "self"
        and node.value.value.attr == "data"
        and isinstance(node.value.slice, ast.Name)
        and node.value.slice.id == "key"
    )


def debugging_fix_checks(answer: dict[str, Any]) -> dict[str, bool]:
    code = answer.get("corrected_code")
    if not isinstance(code, str):
        return {"cache_fix": False, "average_fix": False}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"cache_fix": False, "average_fix": False}

    cache_fix = False
    average_fix = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name == "get_or_load":
            cache_fix = any(
                _is_cached_value_return(statement)
                for item in ast.walk(node)
                if isinstance(item, (ast.With, ast.AsyncWith))
                for statement in ast.walk(item)
            )
        elif node.name == "average":
            average_fix = any(
                isinstance(statement, ast.Raise)
                and isinstance(statement.exc, ast.Call)
                and isinstance(statement.exc.func, ast.Name)
                and statement.exc.func.id == "ValueError"
                for conditional in ast.walk(node)
                if isinstance(conditional, ast.If)
                for statement in ast.walk(conditional)
            )
    return {"cache_fix": cache_fix, "average_fix": average_fix}


def evaluate_debugging(text: str) -> dict[str, Any]:
    maximum = 8.0
    try:
        answer = json.loads(text)
    except json.JSONDecodeError as exc:
        return evaluation("failed", 0, maximum, f"Invalid JSON: {exc.msg}.")
    if not isinstance(answer, dict):
        return evaluation("failed", 0, maximum, "Response JSON must be an object.")

    bugs = answer.get("bugs")
    bug_items = {
        item.get("id"): item
        for item in bugs
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    } if isinstance(bugs, list) else {}
    score = 0.0
    notes: list[str] = []
    for bug_id, expected_failure in DEBUGGING_EXPECTED_FAILURES.items():
        item = bug_items.get(bug_id)
        if item is None:
            notes.append(f"{bug_id}: not identified")
            continue
        score += 1
        actual_failure = item.get("failure")
        if actual_failure == expected_failure:
            score += 1
        elif (
            expected_failure == "UnboundLocalError"
            and actual_failure == "NameError"
        ):
            score += 0.5
            notes.append("cache exception: NameError is the base class; expected UnboundLocalError")
        else:
            notes.append(f"{bug_id}: incorrect exception {actual_failure!r}")

    fixes = debugging_fix_checks(answer)
    for bug_id, key in (
        ("cache_missing_value", "cache_fix"),
        ("average_empty_sequence", "average_fix"),
    ):
        if fixes[key]:
            score += 1
        else:
            notes.append(f"{bug_id}: correction not verified")

    not_bugs = answer.get("not_bugs")
    if isinstance(not_bugs, list) and all(isinstance(item, str) for item in not_bugs):
        not_bug_set = set(not_bugs)
        score += len(not_bug_set & DEBUGGING_EXPECTED_NOT_BUGS)
        unexpected = not_bug_set - DEBUGGING_EXPECTED_NOT_BUGS
        score = max(0, score - 0.5 * len(unexpected))
        if unexpected:
            notes.append(
                "unexpected non-bug classifications: "
                + ", ".join(sorted(unexpected, key=str))
            )
        missing = DEBUGGING_EXPECTED_NOT_BUGS - not_bug_set
        if missing:
            notes.append("missing non-bug classifications: " + ", ".join(sorted(missing)))
    else:
        notes.append("non-bug classifications missing or malformed")

    if isinstance(bugs, list):
        unexpected_bugs = {
            item.get("id") for item in bugs
            if isinstance(item, dict) and item.get("id") not in DEBUGGING_EXPECTED_BUG_IDS
        }
        score = max(0, score - 0.5 * len(unexpected_bugs))
        if unexpected_bugs:
            notes.append(
                "unexpected bug classifications: "
                + ", ".join(sorted(unexpected_bugs, key=str))
            )

    status = "passed" if score == maximum else "partial" if score > 0 else "failed"
    details = f"{score:g}/{maximum:g}"
    if notes:
        details += "; " + "; ".join(notes)
    return evaluation(status, score, maximum, details)


def evaluate_coding(text: str) -> dict[str, Any]:
    code = "\n".join(python_code_blocks(text))
    parsed_blocks: list[ast.AST] = []
    for block in python_code_blocks(text):
        try:
            parsed_blocks.append(ast.parse(block))
        except SyntaxError:
            continue

    valid_walk = False
    for tree in parsed_blocks:
        for node in ast.walk(tree):
            if not isinstance(node, (ast.For, ast.AsyncFor)) or not isinstance(node.iter, ast.Call):
                continue
            call = node.iter
            func = call.func
            if isinstance(func, ast.Attribute) and func.attr in {"rglob", "glob", "iterdir", "scandir"}:
                valid_walk = valid_walk or not isinstance(node.target, (ast.Tuple, ast.List))
            elif (
                isinstance(func, ast.Attribute)
                and func.attr == "walk"
                and isinstance(func.value, ast.Name)
                and func.value.id == "os"
            ):
                valid_walk = valid_walk or isinstance(node.target, (ast.Tuple, ast.List))

    checks = {
        "Python syntax parses": bool(parsed_blocks)
        and len(parsed_blocks) == len(python_code_blocks(text)),
        "valid directory traversal": valid_walk,
        "grouping by file size": bool(
            re.search(r"\b(?:st_size|file_size)\b", code)
            and re.search(r"\b(?:size_groups|files_by_size|size_to_\w+)\b", code)
        ),
        "chunked file reads": bool(
            re.search(r"\.read\(\s*(?:chunk_size|\d+)", code)
            and re.search(r"\b(?:while|for)\b", code)
        ),
        "SHA-256 digest retrieval": bool(
            re.search(r"hashlib\.sha256\s*\(", code)
            and re.search(r"\.(?:digest|hexdigest)\s*\(", code)
        ),
        "permission and disappearing-file handling": bool(
            "PermissionError" in code
            and ("FileNotFoundError" in code or re.search(r"\bOSError\b", code))
        ),
        "duplicate groups require at least two files": bool(
            re.search(r"len\([^)]*\)\s*(?:>=\s*2|>\s*1)", code)
        ),
    }
    score = sum(checks.values())
    missing = [name for name, passed in checks.items() if not passed]
    details = f"Static rubric {score}/{len(checks)}; manual review still required."
    if missing:
        details += " Missing or unverified: " + ", ".join(missing)
    return evaluation("heuristic", score, len(checks), details)


def evaluate_architecture(text: str) -> dict[str, Any]:
    code = "\n".join(python_code_blocks(text))
    required_fields = {"cliente", "pfpg", "competenza", "referente"}
    schema = (
        "create table" in code.casefold()
        and all(re.search(rf"\b{field}\b", code, re.IGNORECASE) for field in required_fields)
    )
    parameterized = bool(
        re.search(r"\.execute(?:many)?\s*\(", code)
        and re.search(r"(?:\?|:\w+)", code)
    )
    rollback = bool(
        re.search(r"\.rollback\s*\(", code)
        and (re.search(r"\.commit\s*\(", code) or re.search(r"with\s+\w+(?:\.\w+)?\s*:", code))
    )
    connection_per_unit = bool(
        re.search(
            r"def\s+(?:transaction|unit_of_work)\b[\s\S]{0,800}"
            r"sqlite3\.connect\s*\([\s\S]{0,1200}"
            r"(?:\.close\s*\(|finally)",
            code,
            re.IGNORECASE,
        )
    )
    update_statements = re.findall(
        r"UPDATE\b[\s\S]{0,300}?(?=(?:'''|\"\"\"|$))", code, flags=re.IGNORECASE
    )
    valid_update = bool(update_statements) and all(
        re.search(r"\bSET\b[\s\S]*\bWHERE\b", statement, re.IGNORECASE)
        and not re.search(r",\s*WHERE\b", statement, re.IGNORECASE)
        and not re.search(r"\bSET\s+\{[^}]+\}\s*=\s*\?", statement, re.IGNORECASE)
        for statement in update_statements
    )
    lost_update_strategy = bool(
        re.search(r"\bBEGIN\s+IMMEDIATE\b", code, re.IGNORECASE)
        or (
            re.search(r"\bversion\b", code, re.IGNORECASE)
            and re.search(r"\bWHERE\b[\s\S]{0,100}\bversion\b", code, re.IGNORECASE)
        )
    )
    service_transaction = bool(
        re.search(r"with\s+[\w.]+\.transaction\s*\(", code)
        and re.search(r"\b(?:repo|repository)\.\w+\s*\(", code)
    )
    checks = {
        "schema declares required Deal fields": schema,
        "SQL uses bound parameters": parameterized,
        "transaction commits and rolls back": rollback,
        "connection is opened/closed per unit of work": connection_per_unit,
        "create/update includes valid UPDATE SQL": valid_update,
        "lost-update prevention is explicit": lost_update_strategy,
        "service uses a transaction boundary": service_transaction,
    }
    score = sum(checks.values())
    missing = [name for name, passed in checks.items() if not passed]
    details = f"Static rubric {score}/{len(checks)}; manual review still required."
    if missing:
        details += " Missing or unverified: " + ", ".join(missing)
    return evaluation("heuristic", score, len(checks), details)


def evaluate_test(test_id: str, response: dict[str, Any]) -> dict[str, Any]:
    if test_id == "tool_call":
        trace = response.get("tool_trace") or []
        expected_names = ["lookup_customer", "calculator"]
        actual_names = [call.get("name") for call in trace]
        if actual_names != expected_names:
            return evaluation("failed", 0, 1, f"Expected tool sequence {expected_names}, got {actual_names}.")
        lookup, calculation = trace
        if lookup.get("arguments", {}).get("customer_id") != "C-104":
            return evaluation("failed", 0, 1, "Lookup did not use customer ID C-104.")
        if lookup.get("result") != TOOL_LOOKUP_RESULT:
            return evaluation("failed", 0, 1, "Lookup did not use the benchmark's known customer record.")
        arguments = calculation.get("arguments") or {}
        try:
            correct_arguments = (
                math.isclose(float(arguments["a"]), TOOL_LOOKUP_RESULT["monthly_spend"])
                and math.isclose(float(arguments["b"]), 12)
            )
        except (KeyError, TypeError, ValueError):
            correct_arguments = False
        if not correct_arguments:
            return evaluation("failed", 0, 1, "Calculator did not multiply the looked-up monthly spend by 12.")
        if calculation.get("result") != {"result": 2250.0}:
            return evaluation("failed", 0, 1, "Calculator result was not 2250.")
        if re.search(r"(?<!\d)(?:2250|2[,. ]250)(?:[,.]00)?(?!\d)", response_text(response)) is None:
            return evaluation("failed", 0, 1, "Final answer did not report the calculated annual spend of 2250 euros.")
        return evaluation("passed", 1, 1, "Looked up the synthetic customer, calculated annual spend, and reported it.")

    text = response_text(response).casefold()
    if test_id == "debugging":
        return evaluate_debugging(response_text(response))

    if test_id == "coding":
        return evaluate_coding(response_text(response))

    if test_id == "architecture":
        return evaluate_architecture(response_text(response))

    if test_id == "long_context":
        try:
            answer = json.loads(response_text(response))
        except json.JSONDecodeError as exc:
            return evaluation("failed", 0, 1, f"Response was not the required JSON object: {exc.msg}.")
        if not isinstance(answer, dict):
            return evaluation("failed", 0, 1, "Response JSON must be an object.")
        facts = answer.get("facts")
        if not isinstance(facts, dict):
            return evaluation("failed", 0, 1, "Response is missing the facts object.")
        if facts != LONG_CONTEXT_EXPECTED:
            return evaluation("failed", 0, 1, "Facts object did not exactly match the authoritative project notes.")
        checklist = answer.get("checklist")
        if not isinstance(checklist, dict):
            return evaluation("failed", 0, 1, "Response is missing the checklist object.")
        missing_checks = [
            key for key in LONG_CONTEXT_CHECKLIST_KEYS
            if not isinstance(checklist.get(key), str) or not checklist[key].strip()
        ]
        if missing_checks:
            return evaluation("failed", 0, 1, "Missing checklist items: " + ", ".join(missing_checks))
        return evaluation("passed", 1, 1, "Returned all checklist items and exact project facts in JSON.")

    if not text.strip():
        return evaluation("failed", 0, 1, "The model returned no user-visible answer.")
    return evaluation("not_scored", None, None, "No automatic semantic rubric is defined for this test.")


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
        self.error: Exception | None = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=6)
            if self.thread.is_alive():
                raise TimeoutError("Telemetry monitor did not stop within 6 seconds.")
        if self.error is not None:
            raise RuntimeError(f"Telemetry monitor failed: {self.error}") from self.error

    def _run(self) -> None:
        try:
            self._write_samples()
        except Exception as exc:
            self.error = exc

    def _write_samples(self) -> None:
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

def ollama_request(
    model: str,
    test: dict[str, Any],
    ctx: int,
    timeout: int,
    seed: int = BENCHMARK_SEED,
) -> dict[str, Any]:
    if test["kind"] == "chat":
        if test["id"] == "tool_call":
            return run_tool_call_test(model, test, ctx, timeout, seed)
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": test["prompt"]}],
            "tools": tool_call_specs(test),
            "stream": False,
            "options": {
                "num_ctx": ctx,
                "num_predict": test["predict"],
                "temperature": 0,
                "seed": seed,
            }
        }
        return post_json(f"{OLLAMA_URL}/api/chat", payload, timeout)
    payload = {
        "model": model,
        "prompt": test["prompt"],
        "stream": False,
        "options": {
            "num_ctx": ctx,
            "num_predict": test["predict"],
            "temperature": 0,
            "seed": seed,
        }
    }
    if "think" in test:
        payload["think"] = test["think"]
    return post_json(f"{OLLAMA_URL}/api/generate", payload, timeout)


def run_tool_call_test(
    model: str,
    test: dict[str, Any],
    ctx: int,
    timeout: int,
    seed: int = BENCHMARK_SEED,
) -> dict[str, Any]:
    messages = [{"role": "user", "content": test["prompt"]}]
    trace: list[dict[str, Any]] = []
    responses: list[dict[str, Any]] = []
    for _ in range(TOOL_MAX_ROUNDS):
        response = post_json(
            f"{OLLAMA_URL}/api/chat",
            {
                "model": model,
                "messages": messages,
                "tools": tool_call_specs(test),
                "stream": False,
                "options": {
                    "num_ctx": ctx,
                    "num_predict": test["predict"],
                    "temperature": 0,
                    "seed": seed,
                },
            },
            timeout,
        )
        responses.append(response)
        message = response.get("message") or {}
        calls = message.get("tool_calls") or []
        if not calls:
            break
        messages.append(message)
        for call in calls:
            function = call.get("function") or {}
            name = function.get("name")
            arguments = function.get("arguments") or {}
            result = execute_tool(name, arguments)
            trace.append({
                "name": name,
                "arguments": arguments,
                "result": result,
            })
            messages.append({
                "role": "tool",
                "tool_name": name,
                "content": json.dumps(result, ensure_ascii=False),
            })
    else:
        response = responses[-1]
        response["done"] = False
        response["done_reason"] = "tool_round_limit"

    totals = {
        "prompt_eval_count": sum(item.get("prompt_eval_count") or 0 for item in responses),
        "eval_count": sum(item.get("eval_count") or 0 for item in responses),
        "prompt_eval_duration": sum(item.get("prompt_eval_duration") or 0 for item in responses),
        "eval_duration": sum(item.get("eval_duration") or 0 for item in responses),
        "load_duration": sum(item.get("load_duration") or 0 for item in responses),
        "total_duration": sum(item.get("total_duration") or 0 for item in responses),
    }
    return {
        **response,
        **totals,
        "tool_trace": trace,
        "interaction_count": len(responses),
        "interaction_responses": responses,
    }


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
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to run (default: every model installed in Ollama).",
    )
    parser.add_argument("--ctx", type=positive_int, default=4096)
    parser.add_argument("--predict", type=positive_int, default=None,
                        help="Override per-test generation limit.")
    parser.add_argument("--repeat", type=positive_int, default=1,
                        help="Measured repetitions of each test per model.")
    parser.add_argument("--seed", type=int, default=BENCHMARK_SEED,
                        help=f"Base generation seed (default: {BENCHMARK_SEED}).")
    parser.add_argument("--vary-seed", action="store_true",
                        help="Use seed + repetition - 1 for each repeated test.")
    parser.add_argument("--interval", type=positive_float, default=0.5)
    parser.add_argument("--timeout", type=positive_int, default=900)
    parser.add_argument("--out", default="llm-bench-results")
    warmup_group = parser.add_mutually_exclusive_group()
    warmup_group.add_argument("--warmup", dest="warmup", action="store_true",
                              help="Run a warm-up request before measured tests (default).")
    warmup_group.add_argument("--no-warmup", dest="warmup", action="store_false",
                              help="Disable the default per-model warm-up.")
    parser.set_defaults(warmup=True)
    args = parser.parse_args()

    try:
        tags = get_json(f"{OLLAMA_URL}/api/tags")
    except Exception as exc:
        print(f"ERROR: Cannot reach Ollama at {OLLAMA_URL}: {exc}", file=sys.stderr)
        return 2

    try:
        args.models = resolve_models(args.models, tags)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    outdir = Path(args.out).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = outdir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    metadata = {
        "run_id": run_id,
        "started_at": now_iso(),
        "host": platform.platform(),
        "python": sys.version,
        "ollama_url": OLLAMA_URL,
        "models": args.models,
        "model_info": {
            model["name"]: {
                "digest": model.get("digest"),
                "size_bytes": model.get("size"),
                "details": model.get("details", {}),
            }
            for model in tags.get("models", [])
            if isinstance(model, dict) and model.get("name") in args.models
        },
        "context": args.ctx,
        "temperature": 0,
        "max_tokens": {
            test["id"]: args.predict if args.predict is not None else test["predict"]
            for test in TESTS
        },
        "thinking": {
            test["id"]: test.get("think", "model_default")
            for test in TESTS
        },
        "repeat": args.repeat,
        "seed": args.seed,
        "vary_seed": args.vary_seed,
        "warmup": args.warmup,
        "tests": [t["id"] for t in TESTS],
        "note": "Tool-call test uses deterministic local lookup and calculator tools."
    }
    (run_dir / "run-info.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    all_results: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    warmup_results: list[dict[str, Any]] = []
    total = len(args.models) * (len(TESTS) * args.repeat + (1 if args.warmup else 0))
    step = 0
    model_prefixes = model_file_prefixes(args.models)

    for model in args.models:
        safe_model = model_prefixes[model]
        print(f"\n=== MODEL: {model} ===", flush=True)

        if args.warmup:
            step += 1
            print(f"[{step}/{total}] Warm-up", flush=True)
            warmup_started = time.time()
            warmup_error = None
            try:
                post_json(f"{OLLAMA_URL}/api/generate", {
                    "model": model, "prompt": "Reply with: ready",
                    "stream": False,
                    "options": {
                        "num_ctx": args.ctx,
                        "num_predict": 16,
                        "temperature": 0,
                        "seed": args.seed,
                    }
                }, args.timeout)
            except Exception as exc:
                warmup_error = f"{type(exc).__name__}: {exc}"
                print(f"Warm-up failed: {warmup_error}", flush=True)
            warmup_results.append({
                "model": model,
                "status": "error" if warmup_error else "ok",
                "wall_seconds": round(time.time() - warmup_started, 2),
                "error": warmup_error,
            })

        for test in TESTS:
            for repetition in range(1, args.repeat + 1):
                step += 1
                print(
                    f"[{step}/{total}] {model} — {test['name']} "
                    f"(run {repetition}/{args.repeat})",
                    flush=True,
                )
                test_copy = dict(test)
                if args.predict is not None:
                    test_copy["predict"] = args.predict
                test_seed = args.seed + repetition - 1 if args.vary_seed else args.seed

                repeat_suffix = f"-r{repetition:02d}" if args.repeat > 1 else ""
                monitor_path = run_dir / (
                    f"{safe_model}-{test['id']}{repeat_suffix}-telemetry.csv"
                )
                response_path = run_dir / (
                    f"{safe_model}-{test['id']}{repeat_suffix}-response.json"
                )
                monitor = Monitor(monitor_path, args.interval)
                telemetry_error = None
                monitor_started = False
                try:
                    monitor.start()
                    monitor_started = True
                except Exception as exc:
                    telemetry_error = f"{type(exc).__name__}: {exc}"
                started = time.time()
                try:
                    response = ollama_request(
                        model, test_copy, args.ctx, args.timeout, test_seed
                    )
                    status = "ok"
                    error = None
                except Exception as exc:
                    response = {}
                    status = "error"
                    error = f"{type(exc).__name__}: {exc}"
                finally:
                    elapsed = time.time() - started
                    if monitor_started:
                        try:
                            monitor.stop()
                        except Exception as exc:
                            telemetry_error = f"{type(exc).__name__}: {exc}"

                response_path.write_text(
                    json.dumps(
                        response if error is None else {"error": error},
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                metrics = extract_metrics(response) if error is None else {}
                check = (
                    evaluate_test(test["id"], response)
                    if error is None
                    else evaluation("not_run", None, None, error)
                )
                check_status = check["status"]
                result_output_status = (
                    output_status(test, response) if error is None else "error"
                )
                row = {
                    "model": model,
                    "test_id": test["id"],
                    "test_name": test["name"],
                    "repeat": repetition,
                    "seed": test_seed,
                    "status": status,
                    "output_status": result_output_status,
                    "check_status": check_status,
                    "check_score": check["score"],
                    "check_max_score": check["max_score"],
                    "check_details": check["details"],
                    "telemetry_error": telemetry_error,
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
                    f"output={result_output_status} | "
                    f"check={check_status}"
                    f"({check['score']}/{check['max_score']}) | "
                    f"tokens/s={metrics.get('tokens_per_second', 'n/a')} | "
                    f"generated={metrics.get('generated_tokens', 'n/a')}",
                    flush=True,
                )
                if error:
                    print(f"  ERROR: {error}", flush=True)
                if telemetry_error:
                    print(f"  TELEMETRY ERROR: {telemetry_error}", flush=True)
                if error is None and test["id"] == "tool_call":
                    print(
                        f"  tool calls: "
                        f"{[call['name'] for call in response.get('tool_trace', [])]}",
                        flush=True,
                    )

    (run_dir / "results.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    csv_path = run_dir / "summary.csv"
    columns = [
        "model", "test_id", "test_name", "repeat", "seed", "status", "output_status",
        "check_status", "check_score", "check_max_score", "check_details",
        "telemetry_error", "wall_seconds",
        "prompt_tokens", "generated_tokens", "generation_seconds",
        "tokens_per_second", "prompt_eval_seconds", "load_seconds",
        "total_seconds", "done_reason", "response_file", "telemetry_file", "error"
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summary_rows)

    metadata["finished_at"] = now_iso()
    metadata["warmup_results"] = warmup_results
    (run_dir / "run-info.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\n=== COMPLETE ===")
    print(f"Results folder: {run_dir}")
    print(f"Summary:        {csv_path}")
    failed = any(
        row["status"] == "error"
        or row["telemetry_error"] is not None
        or row["output_status"] in {"truncated", "empty", "incomplete", "unknown"}
        or row["check_status"] == "failed"
        for row in all_results
    ) or any(row["status"] == "error" for row in warmup_results)
    if failed:
        print("One or more tests failed, were incomplete, or were truncated.", file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
