"""Test the one-request pipeline with a fake model and a mocked evaluator."""

import contextlib
import io
import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError, URLError

import main
from evaluation.evaluate import Feedback


VALID_CODE = "def solve(**kwargs):\n    return {'schedule': {}}\n"


def model_response(content, finish_reason="stop"):
    payload = {
        "choices": [{"finish_reason": finish_reason, "message": {"role": "assistant", "content": content}}]
    }
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=finish_reason, message=SimpleNamespace(content=content))],
        model_dump=lambda **kwargs: payload,
    )


class PipelineTests(unittest.TestCase):
    def run_pipeline(self, response=None, exception=None, instance_error=False):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        data = SimpleNamespace(problem_description="Aircraft scheduling test", test_cases=["fake.txt"])
        results = {"fake.txt": (["Exception: fake constraint failure"] if instance_error else [1.0], None)}
        feedback = Feedback(0.0 if instance_error else 1.0, 1.0, 1.0, "all", "dev", "test", results)
        with (
            patch("main.load_task", return_value=data),
            patch("main.ensure_server"),
            patch("main.Evaluator") as evaluator,
            patch("litellm.completion", return_value=response, side_effect=exception) as completion,
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            evaluator.return_value.evaluate.return_value = feedback
            exit_code = main.main(["--output-dir", str(self.output)])
        run_dir, = self.output.iterdir()
        self.run_dir = run_dir
        self.state = json.loads((run_dir / "run.json").read_text())
        self.completion = completion
        self.evaluator = evaluator
        return exit_code

    def test_success_uses_one_request_and_one_evaluation(self):
        exit_code = self.run_pipeline(model_response(f"```python\n{VALID_CODE}```"))
        self.assertEqual(exit_code, 0)
        self.completion.assert_called_once()
        self.evaluator.return_value.evaluate.assert_called_once()
        self.assertEqual(self.state["generation_requests"], 1)
        self.assertEqual(self.state["refinement_iterations"], 0)
        self.assertEqual(self.state["status"], "completed")
        self.assertEqual((self.run_dir / "solution.py").read_text().strip(), VALID_CODE.strip())
        self.assertTrue((self.run_dir / "feedback.json").is_file())
        request = json.loads((self.run_dir / "request.json").read_text())
        self.assertNotIn("api_key", request)
        self.assertEqual(request["max_retries"], 0)
        self.assertEqual(request["num_retries"], 0)

    def test_api_failure_does_not_retry_or_evaluate(self):
        self.assertEqual(self.run_pipeline(exception=ConnectionError("fake API failure")), 1)
        self.completion.assert_called_once()
        self.evaluator.assert_not_called()
        self.assertEqual(self.state["status"], "generating_failed")
        self.assertTrue((self.run_dir / "request.json").is_file())
        self.assertIn("fake API failure", (self.run_dir / "error.txt").read_text())

    def test_missing_code_keeps_response_without_evaluation(self):
        self.assertEqual(self.run_pipeline(model_response("I did not produce code.")), 1)
        self.completion.assert_called_once()
        self.evaluator.assert_not_called()
        self.assertEqual((self.run_dir / "response.txt").read_text(), "I did not produce code.")

    def test_syntax_error_keeps_code_without_evaluation(self):
        self.assertEqual(self.run_pipeline(model_response("```python\ndef solve(:\n```")), 1)
        self.completion.assert_called_once()
        self.evaluator.assert_not_called()
        self.assertEqual(self.state["status"], "validating_failed")
        self.assertTrue((self.run_dir / "solution.py").is_file())

    def test_token_limit_is_not_retried(self):
        self.assertEqual(self.run_pipeline(model_response("<think>still thinking", "length")), 1)
        self.completion.assert_called_once()
        self.evaluator.assert_not_called()
        self.assertTrue((self.run_dir / "response.json").is_file())

    def test_instance_failure_is_reported_as_exit_two(self):
        self.assertEqual(self.run_pipeline(model_response(f"```python\n{VALID_CODE}```"), instance_error=True), 2)
        self.assertEqual(self.state["status"], "completed_with_errors")
        self.assertEqual(self.state["failed"], 1)

    def test_saved_solver_never_contacts_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            code_path = Path(tmp) / "saved.py"
            code_path.write_text(VALID_CODE)
            data = SimpleNamespace(problem_description="unused", test_cases=["fake.txt"])
            feedback = Feedback(1, 1, 1, "all", "dev", "test", {"fake.txt": ([1], None)})
            with (
                patch("main.load_task", return_value=data),
                patch("main.ensure_server") as server,
                patch("litellm.completion") as completion,
                patch("main.Evaluator") as evaluator,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                evaluator.return_value.evaluate.return_value = feedback
                self.assertEqual(main.main(["--code", str(code_path), "--output-dir", str(Path(tmp) / "runs")]), 0)
                server.assert_not_called()
                completion.assert_not_called()

    def test_old_32k_server_is_rejected_before_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = SimpleNamespace(problem_description="unused", test_cases=["fake.txt"])
            with (
                patch("main.load_task", return_value=data),
                patch("main.ensure_server", return_value={"id": main.MODEL, "max_model_len": 32768}),
                patch("litellm.completion") as completion,
                patch("main.Evaluator") as evaluator,
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self.assertEqual(main.main(["--output-dir", tmp]), 1)
                completion.assert_not_called()
                evaluator.assert_not_called()
            run_dir, = Path(tmp).iterdir()
            state = json.loads((run_dir / "run.json").read_text())
            self.assertEqual(state["generation_requests"], 0)
            self.assertIn("leaves no space for the input", state["error"])

    def test_compile_checks_without_executing_code(self):
        # ast.parse alone accepts return outside a function; compilation must reject it.
        with self.assertRaises(SyntaxError):
            main.validate_code("return 1\n" + VALID_CODE)

    def test_greedy_import_does_not_load_optional_agents(self):
        self.assertNotIn("agents.aide", sys.modules)
        self.assertNotIn("agents.mcts_agent", sys.modules)


class ServerReadinessTests(unittest.TestCase):
    def test_connection_refused_then_ready_does_not_generate(self):
        response = io.StringIO(json.dumps({"data": [{"id": main.MODEL}]}))
        with (
            patch("main.urlopen", side_effect=[URLError(ConnectionRefusedError()), response]) as request,
            patch("main.time.sleep") as sleep,
            patch("litellm.completion") as completion,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            main.ensure_server("http://127.0.0.1:8000/v1", main.MODEL, "local", wait_timeout=600)
            self.assertEqual(request.call_count, 2)
            sleep.assert_called_once()
            completion.assert_not_called()

    def test_wait_timeout_keeps_generation_count_zero(self):
        with (
            patch("main.urlopen", side_effect=URLError(ConnectionRefusedError())) as request,
            patch("main.time.monotonic", side_effect=[0, 0, 601]),
            patch("main.time.sleep") as sleep,
            patch("litellm.completion") as completion,
        ):
            with self.assertRaisesRegex(RuntimeError, "No code-generation request was sent"):
                main.ensure_server("http://127.0.0.1:8000/v1", main.MODEL, "local", wait_timeout=600)
            request.assert_called_once()
            sleep.assert_not_called()
            completion.assert_not_called()

    def test_authentication_error_does_not_wait(self):
        with (
            patch("main.urlopen", side_effect=HTTPError("http://local", 401, "Unauthorized", None, None)) as request,
            patch("main.time.sleep") as sleep,
        ):
            with self.assertRaisesRegex(RuntimeError, "HTTP 401"):
                main.ensure_server("http://127.0.0.1:8000/v1", main.MODEL, "local", wait_timeout=600)
            request.assert_called_once()
            sleep.assert_not_called()


class LocalProtocolTests(unittest.TestCase):
    def test_litellm_sends_qwen_settings_to_local_endpoint(self):
        self.exercise_endpoint(200)

    def test_http_service_error_is_not_retried(self):
        self.exercise_endpoint(503)

    def test_non_thinking_request_disables_thinking_and_uses_its_sampling_settings(self):
        self.exercise_endpoint(200, thinking=False)

    def exercise_endpoint(self, status, thinking=True):
        """Use a static HTTP response; no model or candidate code is executed."""
        calls = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                calls.append((self.path, payload))
                body = json.dumps({
                    "id": "fake-completion", "object": "chat.completion", "created": 0,
                    "model": main.MODEL,
                    "choices": [{"index": 0, "finish_reason": "stop", "message": {
                        "role": "assistant", "content": f"```python\n{VALID_CODE}```", "reasoning_content": "fake reasoning",
                    }}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }).encode()
                if status != 200:
                    body = json.dumps({"error": {"message": "fake unavailability", "type": "server_error"}}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                options = [
                    "--api-base", f"http://127.0.0.1:{server.server_port}/v1", "--generation-timeout", "5",
                ]
                if not thinking:
                    options.append("--no-thinking")
                args = main.make_parser().parse_args(options)
                output = Path(tmp)
                generate = main.make_generator(args, output, "fake-local-key")
                if status == 200:
                    content = generate("test prompt", model=f"openai/{main.MODEL}", reasoning_effort="medium")
                    self.assertIn("def solve", content)
                    self.assertEqual((output / "reasoning.txt").read_text(), "fake reasoning")
                else:
                    import litellm
                    with self.assertRaises(litellm.ServiceUnavailableError):
                        generate("test prompt", model=f"openai/{main.MODEL}", reasoning_effort="medium")
                self.assertEqual(len(calls), 1)
                path, payload = calls[0]
                self.assertEqual(path, "/v1/chat/completions")
                self.assertEqual(payload["model"], main.MODEL)
                self.assertEqual(payload["top_k"], 20)
                self.assertEqual(payload["chat_template_kwargs"]["enable_thinking"], thinking)
                if thinking:
                    self.assertEqual(payload["chat_template_kwargs"]["reasoning_effort"], "medium")
                else:
                    self.assertNotIn("reasoning_effort", payload["chat_template_kwargs"])
                    self.assertEqual(payload["temperature"], 0.7)
                    self.assertEqual(payload["top_p"], 0.8)
                    self.assertEqual(payload["presence_penalty"], 1.5)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
