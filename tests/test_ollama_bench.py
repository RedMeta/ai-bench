import argparse
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import ollama_bench


class ArgumentValidationTests(unittest.TestCase):
    def test_positive_int_rejects_non_positive_values(self) -> None:
        self.assertEqual(ollama_bench.positive_int("16"), 16)
        for value in ("0", "-1", "1.5"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                ollama_bench.positive_int(value)

    def test_positive_float_rejects_non_finite_and_non_positive_values(self) -> None:
        self.assertEqual(ollama_bench.positive_float("0.5"), 0.5)
        for value in ("0", "-0.1", "nan", "inf"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                ollama_bench.positive_float(value)

    def test_model_names_that_share_a_file_prefix_get_distinct_filenames(self) -> None:
        prefixes = ollama_bench.model_file_prefixes(["model:a", "model_a"])
        self.assertEqual(len(set(prefixes.values())), 2)

    def test_default_model_selection_includes_all_installed_models(self) -> None:
        tags = {"models": [{"name": "z-model"}, {"name": "a-model"}]}
        self.assertEqual(ollama_bench.resolve_models(None, tags), ["a-model", "z-model"])
        self.assertEqual(ollama_bench.resolve_models(["z-model"], tags), ["z-model"])

    def test_model_selection_reports_unknown_models(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing"):
            ollama_bench.resolve_models(["missing"], {"models": [{"name": "installed"}]})

    def test_model_selection_rejects_duplicate_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "more than once"):
            ollama_bench.resolve_models(
                ["installed", "installed"], {"models": [{"name": "installed"}]}
            )


class ResultEvaluationTests(unittest.TestCase):
    def test_output_status_distinguishes_truncation_empty_and_tool_calls(self) -> None:
        generate_test = {"kind": "generate"}
        chat_test = {"kind": "chat"}
        self.assertEqual(
            ollama_bench.output_status(
                generate_test, {"response": "partial", "done": True, "done_reason": "length"}
            ),
            "truncated",
        )
        self.assertEqual(
            ollama_bench.output_status(generate_test, {"response": "", "done": True}),
            "empty",
        )
        self.assertEqual(
            ollama_bench.output_status(
                chat_test, {"message": {"content": "", "tool_calls": [{"function": {}}]}}
            ),
            "complete",
        )

    def test_long_context_check_requires_each_authoritative_fact(self) -> None:
        response = {"response": json.dumps({
            "checklist": {key: "check" for key in ollama_bench.LONG_CONTEXT_CHECKLIST_KEYS},
            "facts": ollama_bench.LONG_CONTEXT_EXPECTED,
        })}
        self.assertEqual(ollama_bench.evaluate_test("long_context", response)["status"], "passed")
        incorrect_facts = ollama_bench.LONG_CONTEXT_EXPECTED | {
            "required_deal_fields": ["cliente", "pfpg", "competenza", "client"]
        }
        result = ollama_bench.evaluate_test("long_context", {"response": json.dumps({
            "checklist": {key: "check" for key in ollama_bench.LONG_CONTEXT_CHECKLIST_KEYS},
            "facts": incorrect_facts,
        })})
        self.assertEqual(result["status"], "failed")
        self.assertIn("exactly match", result["details"])

    def test_debugging_check_requires_only_expected_bugs_and_non_bugs(self) -> None:
        response = {"response": json.dumps({
            "bugs": [
                {
                    "id": bug_id,
                    "failure": {
                        "cache_missing_value": "UnboundLocalError",
                        "average_empty_sequence": "ZeroDivisionError",
                    }[bug_id],
                    "explanation": "reproduced",
                    "fix": "corrected",
                }
                for bug_id in sorted(ollama_bench.DEBUGGING_EXPECTED_BUG_IDS)
            ],
            "not_bugs": sorted(ollama_bench.DEBUGGING_EXPECTED_NOT_BUGS),
            "corrected_code": (
                "def get_or_load(self, key, loader):\n"
                "    with self.lock:\n"
                "        if key not in self.data:\n"
                "            self.data[key] = loader()\n"
                "        return self.data[key]\n"
                "def average(values):\n"
                "    if not values: raise ValueError('empty')\n"
                "    return sum(values) / len(values)"
            ),
        })}
        result = ollama_bench.evaluate_test("debugging", response)
        self.assertEqual(result["status"], "passed")
        self.assertEqual((result["score"], result["max_score"]), (8, 8))
        answer = json.loads(response["response"])
        answer["bugs"].append({
            "id": "read_config_resource_handling",
            "failure": "OSError",
            "explanation": "invented",
            "fix": "unneeded",
        })
        self.assertEqual(
            ollama_bench.evaluate_test("debugging", {"response": json.dumps(answer)})["status"],
            "partial",
        )
        answer = json.loads(response["response"])
        next(
            item for item in answer["bugs"]
            if item["id"] == "cache_missing_value"
        )["failure"] = "NameError"
        result = ollama_bench.evaluate_test("debugging", {"response": json.dumps(answer)})
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["score"], 7.5)
        self.assertIn("base class", result["details"])
        answer = json.loads(response["response"])
        answer["corrected_code"] = answer["corrected_code"].replace(
            "return self.data[key]", "return value"
        )
        self.assertEqual(
            ollama_bench.evaluate_test("debugging", {"response": json.dumps(answer)})["status"],
            "partial",
        )

    def test_coding_rubric_catches_known_digest_and_traversal_errors(self) -> None:
        bonsai_code = """```python
from pathlib import Path
import hashlib
def find(root, file, digest, paths):
    size_groups = {}
    try:
        for path in Path(root).rglob('*'):
            size_groups.setdefault(path.stat().st_size, []).append(path)
        digest = hashlib.sha256()
        try:
            while chunk := file.read(8192):
                digest.update(chunk)
        except PermissionError:
            pass
        except FileNotFoundError:
            pass
    except OSError:
        pass
    if len(paths) >= 2:
        return digest.hash
```"""
        result = ollama_bench.evaluate_test("coding", {"response": bonsai_code})
        self.assertEqual(result["status"], "heuristic")
        self.assertIn("SHA-256 digest retrieval", result["details"])
        self.assertEqual(result["score"], 6)

        qwen_code = """```python
import hashlib
def hash_file(path):
    return hashlib.sha256().digest()
def find(directory, file, paths):
    size_to_hashes = {}
    file_size = Path(directory).stat().st_size
    size_to_hashes.setdefault(file_size, {})
    try:
        for root, _, files in Path(directory).rglob('*'):
            while chunk := file.read(8192):
                pass
    except PermissionError:
        pass
    except FileNotFoundError:
        pass
    if len(paths) >= 2:
        return paths
```"""
        result = ollama_bench.evaluate_test("coding", {"response": qwen_code})
        self.assertEqual(result["status"], "heuristic")
        self.assertIn("valid directory traversal", result["details"])
        self.assertEqual(result["score"], 6)

    def test_architecture_rubric_flags_malformed_update_and_shared_connection(self) -> None:
        answer = """### Schema
```python
CREATE_TABLE = \"CREATE TABLE deals (cliente TEXT, pfpg TEXT, competenza TEXT, referente TEXT)\"
class Repo:
    def __init__(self, path):
        self.conn = sqlite3.connect(path)
    def create_or_update(self, deal):
        with self.conn:
            self.conn.execute(\"INSERT INTO deals VALUES (?, ?, ?, ?)\", values)
            self.conn.execute(\"UPDATE deals SET {cols} = ?, WHERE id = ?\", values)
```
"""
        result = ollama_bench.evaluate_test("architecture", {"response": answer})
        self.assertEqual(result["status"], "heuristic")
        self.assertIn("valid UPDATE SQL", result["details"])
        self.assertIn("connection is opened/closed per unit of work", result["details"])

    def test_tool_execution_is_deterministic_and_rejects_bad_arguments(self) -> None:
        lookup = ollama_bench.execute_tool("lookup_customer", {"customer_id": "C-104"})
        self.assertEqual(lookup, ollama_bench.TOOL_LOOKUP_RESULT)
        self.assertEqual(
            ollama_bench.execute_tool("calculator", {"a": lookup["monthly_spend"], "b": 12}),
            {"result": 2250.0},
        )
        self.assertIn("error", ollama_bench.execute_tool("lookup_customer", {}))
        self.assertIn("error", ollama_bench.execute_tool("calculator", {"a": "x", "b": 12}))

    def test_tool_check_requires_ordered_calls_and_correct_final_answer(self) -> None:
        trace = [
            {
                "name": "lookup_customer",
                "arguments": {"customer_id": "C-104"},
                "result": ollama_bench.TOOL_LOOKUP_RESULT,
            },
            {
                "name": "calculator",
                "arguments": {"a": 187.5, "b": 12},
                "result": {"result": 2250.0},
            },
        ]
        valid = {"tool_trace": trace, "message": {"content": "Annual spend: €2,250.00"}}
        self.assertEqual(ollama_bench.evaluate_test("tool_call", valid)["status"], "passed")
        wrong_order = {"tool_trace": list(reversed(trace)), "message": {"content": "2250"}}
        self.assertEqual(ollama_bench.evaluate_test("tool_call", wrong_order)["status"], "failed")
        wrong_answer = {"tool_trace": trace, "message": {"content": "Annual spend: €2,000"}}
        self.assertEqual(ollama_bench.evaluate_test("tool_call", wrong_answer)["status"], "failed")


class ToolCallFlowTests(unittest.TestCase):
    def test_model_completes_lookup_calculation_and_final_answer_turns(self) -> None:
        test = next(test for test in ollama_bench.TESTS if test["id"] == "tool_call")
        responses = [
            {"message": {"role": "assistant", "tool_calls": [{
                "function": {"name": "lookup_customer", "arguments": {"customer_id": "C-104"}}
            }]}, "eval_count": 5, "eval_duration": 10},
            {"message": {"role": "assistant", "tool_calls": [{
                "function": {"name": "calculator", "arguments": {"a": 187.5, "b": 12}}
            }]}, "eval_count": 5, "eval_duration": 10},
            {"message": {"role": "assistant", "content": "Annual spend: €2,250.00"},
             "done": True, "done_reason": "stop", "eval_count": 5, "eval_duration": 10},
        ]
        requests = []

        def fake_post_json(*request_args: Any) -> dict[str, Any]:
            payload = request_args[1]
            requests.append(json.loads(json.dumps(payload)))
            return responses[len(requests) - 1]

        with patch.object(ollama_bench, "post_json", side_effect=fake_post_json):
            response = ollama_bench.run_tool_call_test("test", test, 4096, 10)

        self.assertEqual(response["interaction_count"], 3)
        self.assertEqual(response["eval_count"], 15)
        self.assertEqual(ollama_bench.evaluate_test("tool_call", response)["status"], "passed")
        second_messages = requests[1]["messages"]
        self.assertEqual(second_messages[-1]["role"], "tool")
        self.assertEqual(second_messages[-1]["tool_name"], "lookup_customer")
        third_messages = requests[2]["messages"]
        self.assertEqual(third_messages[-1]["tool_name"], "calculator")


class RequestConfigurationTests(unittest.TestCase):
    def test_generation_tests_disable_thinking_and_use_fixed_seed(self) -> None:
        generation_tests = [test for test in ollama_bench.TESTS if test["kind"] == "generate"]
        for test in generation_tests:
            with self.subTest(test=test["id"]):
                with patch.object(ollama_bench, "post_json", return_value={}) as post:
                    ollama_bench.ollama_request("test", test, 4096, 10)
                payload = post.call_args.args[1]
                self.assertIs(payload["think"], False)
                self.assertEqual(payload["options"]["seed"], ollama_bench.BENCHMARK_SEED)

    def test_seed_override_reaches_generation_request(self) -> None:
        test = next(test for test in ollama_bench.TESTS if test["id"] == "coding")
        with patch.object(ollama_bench, "post_json", return_value={}) as post:
            ollama_bench.ollama_request("test", test, 4096, 10, seed=99)
        self.assertEqual(post.call_args.args[1]["options"]["seed"], 99)

    def test_seed_override_reaches_tool_call_rounds(self) -> None:
        test = next(test for test in ollama_bench.TESTS if test["id"] == "tool_call")
        with patch.object(
            ollama_bench, "post_json",
            return_value={"message": {"content": "done"}, "done": True},
        ) as post:
            ollama_bench.run_tool_call_test("test", test, 4096, 10, seed=99)
        self.assertEqual(post.call_args.args[1]["options"]["seed"], 99)


class MonitorTests(unittest.TestCase):
    def test_monitor_reports_file_creation_errors_to_caller(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            monitor = ollama_bench.Monitor(
                Path(output_dir) / "missing" / "telemetry.csv", interval=0.1
            )
            monitor.start()
            with self.assertRaisesRegex(RuntimeError, "Telemetry monitor failed"):
                monitor.stop()


class MainResultTests(unittest.TestCase):
    def _run_single_test(
        self,
        output_dir: str,
        response: dict[str, object] | Exception,
        extra_args: tuple[str, ...] = (),
        request_calls: list[Any] | None = None,
    ) -> int:
        tests = [{
            "id": "coding",
            "name": "Test",
            "kind": "generate",
            "predict": 10,
            "prompt": "test",
        }]
        side_effect = response if isinstance(response, Exception) else None
        return_value = None if side_effect else response
        with (
            patch.object(ollama_bench, "TESTS", tests),
            patch.object(ollama_bench, "get_json", return_value={"models": [{"name": "test"}]}),
            patch.object(
                ollama_bench, "ollama_request",
                return_value=return_value, side_effect=side_effect,
            ) as request,
            patch.object(ollama_bench, "Monitor"),
            patch.object(sys, "argv", [
                "ollama_bench.py", "--models", "test", "--out", output_dir,
                "--no-warmup", *extra_args,
            ]),
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            result = ollama_bench.main()
            if request_calls is not None:
                request_calls.extend(request.call_args_list)
            return result

    def test_request_failure_returns_nonzero_and_writes_error(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            self.assertEqual(self._run_single_test(output_dir, OSError("offline")), 1)
            run_dir = next(Path(output_dir).iterdir())
            results = json.loads((run_dir / "results.json").read_text())
            self.assertEqual(results[0]["status"], "error")
            self.assertEqual(results[0]["check_status"], "not_run")

    def test_empty_success_response_is_saved_as_empty_object_and_fails(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            self.assertEqual(self._run_single_test(output_dir, {}), 1)
            run_dir = next(Path(output_dir).iterdir())
            results = json.loads((run_dir / "results.json").read_text())
            response = json.loads((run_dir / results[0]["response_file"]).read_text())
            self.assertEqual(response, {})
            self.assertEqual(results[0]["output_status"], "empty")

    def test_complete_response_returns_success(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            response = {"response": "answer", "done": True, "done_reason": "stop"}
            self.assertEqual(self._run_single_test(output_dir, response), 0)
            run_dir = next(Path(output_dir).iterdir())
            results = json.loads((run_dir / "results.json").read_text())
            self.assertEqual(results[0]["output_status"], "complete")
            self.assertEqual(results[0]["check_status"], "heuristic")
            self.assertEqual(results[0]["check_score"], 1)

    def test_repeat_runs_create_separate_result_rows_and_files(self) -> None:
        with tempfile.TemporaryDirectory() as output_dir:
            response = {"response": "answer", "done": True, "done_reason": "stop"}
            calls: list[Any] = []
            self.assertEqual(
                self._run_single_test(
                    output_dir,
                    response,
                    ("--repeat", "2", "--vary-seed", "--seed", "10"),
                    calls,
                ),
                0,
            )
            run_dir = next(Path(output_dir).iterdir())
            results = json.loads((run_dir / "results.json").read_text())
            self.assertEqual([row["repeat"] for row in results], [1, 2])
            self.assertEqual([row["seed"] for row in results], [10, 11])
            self.assertNotEqual(results[0]["response_file"], results[1]["response_file"])
            self.assertEqual([call.args[-1] for call in calls], [10, 11])


if __name__ == "__main__":
    unittest.main()
