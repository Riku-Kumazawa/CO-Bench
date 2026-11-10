"""Offline tests for dev-only local-Qwen continuation; no GPU/model requests."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import tsp_continue as cont
from scripts import tsp_worker


CODE = "def solve(**kwargs):\n    return {'tour': list(range(len(kwargs['nodes'])))}\n"


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name) / "continued"
        original = json.loads((cont.SOURCE / "feedback.json").read_text())
        self.info = {"problem_description": "TSP test", "data_dir": str(cont.ROOT / "data"),
                     "sizes": {name: len(value[0]) for name, value in original["results"].items()}}
        self.task_patch = patch("tsp_continue.task_info", return_value=self.info)
        self.task_patch.start()
        self.addCleanup(self.task_patch.stop)
        self.requests = []
        self.events = []
        self.fail_api = False
        self.bad_code = False
        self.contents = {}
        self.finish_reasons = {}
        self.interrupt = None
        self.scores = {}
        self.output_patch = contextlib.redirect_stdout(io.StringIO())
        self.output_patch.__enter__()
        self.addCleanup(self.output_patch.__exit__, None, None, None)

    def prepare(self, submissions=10):
        return cont.prepare(cont.SOURCE, self.folder, submissions)

    def generator(self, args, target, key):
        def generate(prompt, **kwargs):
            self.requests.append((target.name, prompt, args))
            (target / "prompt.txt").write_text(prompt)
            cont.save(target / "request.json", {"model": args.model, "prompt": prompt})
            if self.fail_api:
                raise ConnectionError("fake unavailable server")
            content = self.contents.get(target.name, "No code" if self.bad_code else f"```python\n# {target.name}\n{CODE}```")
            cont.save(target / "response.json", {"choices": [{"finish_reason": self.finish_reasons.get(target.name, "stop"), "message": {
                "content": content, "reasoning_content": "fake reasoning"}}],
                "model": args.model, "usage": {"completion_tokens": 12}})
            return content
        return generate

    def evaluator(self, folder, run, name):
        self.events.append(name)
        if self.interrupt == name:
            self.interrupt = None
            raise KeyboardInterrupt()
        score = self.scores.get(name, .89 + .001 * int(name.split("_")[1]))
        if run["candidates"][name]["code_error"]:
            score = 0
        result = {"dev_score": score, "dev_feedback": f"dev only: {score}", "failed": int(score == 0)}
        cont.save(folder / name / "dev.json", result)
        return result

    def run_it(self):
        return cont.run_continuation(self.folder, self.generator, self.evaluator, lambda *a, **k: {})

    def test_27_requests_condition_local_parents_and_no_test_evaluation(self):
        self.prepare()
        self.assertEqual(self.requests, [])
        self.assertEqual(self.run_it(), 0)
        self.assertEqual(len(self.requests), 27)
        self.assertEqual(len(self.events), 27)
        summary = json.loads((self.folder / "summary.json").read_text())
        self.assertEqual(summary["completed_submissions"], dict(support=10, neutral=10, switch=10))
        self.assertEqual(summary["final_by_condition"], {c: f"{c}_10" for c in cont.INTERVENTIONS})
        run = json.loads((self.folder / "run.json").read_text())
        for name, prompt, args in self.requests:
            condition, number = name.split("_")
            parent = "initial" if number == "02" else f"{condition}_{int(number)-1:02d}"
            self.assertEqual(run["candidates"][name]["parent"], parent)
            self.assertIn((self.folder / parent / "solution.py").read_text(), prompt)
            for other, text in cont.INTERVENTIONS.items():
                if text:
                    self.assertEqual(text in prompt, other == condition)
            self.assertEqual(args.reasoning_effort, "medium")
            self.assertTrue(args.thinking)
            self.assertEqual(args.max_tokens, 32768)
            self.assertEqual(args.seed, 0)
        self.assertEqual(list(self.folder.rglob("test.json")), [])
        self.assertEqual(self.run_it(), 0)
        self.assertEqual(len(self.requests), 27)

    def test_equal_worse_and_zero_score_are_always_the_next_parent(self):
        self.prepare(3)
        initial = cont.load_result(self.folder, "initial")["dev_score"]
        self.scores.update(support_02=initial, neutral_02=initial - .1, switch_02=0)
        self.scores.update(support_03=.1, neutral_03=.2, switch_03=.3)
        self.run_it()
        run = json.loads((self.folder / "run.json").read_text())
        for condition in cont.INTERVENTIONS:
            self.assertEqual(run["candidates"][f"{condition}_03"]["parent"], f"{condition}_02")
        summary = json.loads((self.folder / "summary.json").read_text())
        self.assertEqual(summary["latest_by_condition"], {c: f"{c}_03" for c in cont.INTERVENTIONS})
        self.assertEqual(summary["best_observed_by_dev"], dict.fromkeys(cont.INTERVENTIONS, "initial"))
        self.assertTrue(summary["best_observed_is_diagnostic_only"])
        self.assertTrue(all(row["delta_from_parent"] < 0 for row in summary["rows"]
                            if row["submission"] == 3 and row["condition"] != "switch"))

    def test_evaluation_interruption_resumes_without_regeneration(self):
        self.prepare(3)
        self.interrupt = "neutral_02"
        with self.assertRaises(KeyboardInterrupt):
            self.run_it()
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.run_it(), 0)
        self.assertEqual(len(self.requests), 6)

    def test_unknown_response_never_resends_and_changed_settings_rejected(self):
        self.prepare(2)
        self.fail_api = True
        with self.assertRaisesRegex(RuntimeError, "no retry"):
            self.run_it()
        with self.assertRaisesRegex(RuntimeError, "No automatic resend"):
            self.run_it()
        self.assertEqual(len(self.requests), 1)
        run = json.loads((self.folder / "run.json").read_text())
        run["settings"]["reasoning_effort"] = "low"
        cont.save(self.folder / "run.json", run)
        with self.assertRaisesRegex(RuntimeError, "changed"):
            self.run_it()

    def test_invalid_generation_consumes_slot_and_scores_zero(self):
        self.prepare(3)
        self.bad_code = True
        cont.run_continuation(self.folder, self.generator, cont.evaluate_candidate, lambda *a, **k: {})
        self.assertEqual(len(self.requests), 6)
        summary = json.loads((self.folder / "summary.json").read_text())
        self.assertEqual(summary["latest_by_condition"], {c: f"{c}_03" for c in cont.INTERVENTIONS})
        self.assertEqual(summary["best_observed_by_dev"], dict.fromkeys(cont.INTERVENTIONS, "initial"))
        self.assertEqual(summary["latest_dev_scores"], dict.fromkeys(cont.INTERVENTIONS, 0))
        run = json.loads((self.folder / "run.json").read_text())
        for name in run["candidates"]:
            self.assertEqual(cont.evaluate_candidate(self.folder, run, name)["dev_score"], 0)
            self.assertEqual(cont.load_result(self.folder, name)["failed"], 272)
        for name, prompt, _ in self.requests:
            if name.endswith("_03"):
                self.assertIn("No code", prompt)
                self.assertIn("No Python code block", prompt)
                self.assertNotIn((self.folder / "initial" / "solution.py").read_text(), prompt)

    def test_invalid_syntax_and_empty_truncated_answer_are_forwarded_then_recover(self):
        self.prepare(3)
        self.contents.update(support_02="```python\ndef solve(:\n```", neutral_02="",
                             switch_02="The previous attempt did not produce code.")
        self.finish_reasons["neutral_02"] = "length"
        self.run_it()
        prompts = {name: prompt for name, prompt, _ in self.requests}
        self.assertIn("def solve(:", prompts["support_03"])
        self.assertIn("SyntaxError", prompts["support_03"])
        self.assertIn("No final-answer text was returned", prompts["neutral_03"])
        self.assertIn("finish_reason=length", prompts["neutral_03"])
        self.assertIn(self.contents["switch_02"], prompts["switch_03"])
        self.assertIn("No Python code block", prompts["switch_03"])
        initial_code = (self.folder / "initial" / "solution.py").read_text()
        for condition in cont.INTERVENTIONS:
            prompt = prompts[f"{condition}_03"]
            self.assertNotIn(initial_code, prompt)
            self.assertNotIn("fake reasoning", prompt)
            self.assertGreater(cont.load_result(self.folder, f"{condition}_03")["dev_score"], 0)
        self.assertEqual(len(self.requests), 6)

    def test_resume_after_failed_latest_submission_keeps_its_response_as_parent(self):
        self.prepare(3)
        self.contents["support_02"] = "No solver was produced in this attempt."
        self.interrupt = "neutral_02"
        with self.assertRaises(KeyboardInterrupt):
            self.run_it()
        summary = json.loads((self.folder / "summary.json").read_text())
        self.assertEqual(summary["latest_by_condition"]["support"], "support_02")
        self.assertEqual(summary["best_observed_by_dev"]["support"], "initial")
        self.run_it()
        run = json.loads((self.folder / "run.json").read_text())
        self.assertEqual(run["candidates"]["support_03"]["parent"], "support_02")
        prompt = next(prompt for name, prompt, _ in self.requests if name == "support_03")
        self.assertIn(self.contents["support_02"], prompt)
        self.assertEqual(len(self.requests), 6)

    def test_initial_and_completed_artifact_modification_rejected(self):
        self.prepare(2)
        self.run_it()
        (self.folder / "support_02" / "solution.py").write_text(CODE)
        with self.assertRaisesRegex(RuntimeError, "artifact changed"):
            self.run_it()
        (self.folder / "initial" / "solution.py").write_text(CODE)
        with self.assertRaisesRegex(RuntimeError, "initial artifact changed"):
            self.run_it()


class WorkerTests(unittest.TestCase):
    def test_real_subprocess_scores_a_tiny_tsp_without_test_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = root / "fixture"
            task.mkdir()
            (task / "square.txt").write_text("fixture")
            config = task / "config.py"
            config.write_text("def load_data(path):\n    return []\n\n"
                "def eval_func(nodes, label_tour, tour):\n"
                "    assert sorted(tour) == list(range(len(nodes)))\n"
                "    return sum(((nodes[tour[i]][0]-nodes[tour[(i+1)%len(tour)]][0])**2 + "
                "(nodes[tour[i]][1]-nodes[tour[(i+1)%len(tour)]][1])**2)**.5 for i in range(len(tour)))\n")
            data = SimpleNamespace(src_dir=str(root), task="fixture", config_path=str(config),
                test_cases=["square.txt"], get_dev=lambda: None,
                load_data=lambda path: [{"nodes": [(0, 0), (1, 0), (1, 1), (0, 1)], "label_tour": [0, 1, 2, 3]}],
                norm_score=lambda raw: {name: ([4 / value for value in values], error)
                                       for name, (values, error) in raw.items()})
            with (root / "worker.log").open("w") as log, contextlib.redirect_stdout(log):
                result = tsp_worker.evaluate(CODE, data, 10, root / "dev.rows.json")
            self.assertEqual(result["dev_score"], 1)
            self.assertEqual(result["failed"], 0)
            self.assertNotIn("test_score", result)

    def test_checkpoint_resumes_cases_and_keeps_original_timeout_and_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = root / "task"
            task.mkdir()
            for name in ["a.txt", "b.txt", "config.py"]:
                (task / name).write_text("fake")
            data = SimpleNamespace(src_dir=str(root), task="task", config_path=str(task / "config.py"),
                test_cases=["a.txt", "b.txt"], get_dev=lambda: None,
                load_data=lambda path: [{"nodes": [1], "label_tour": [0]}, {"nodes": [2], "label_tour": [0]}],
                norm_score=lambda raw: raw)
            checkpoint = root / "dev.rows.json"
            calls = []
            def run(instance, code, config, timeout):
                calls.append((instance, timeout))
                if len(calls) == 3:
                    raise KeyboardInterrupt()
                return .5
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(KeyboardInterrupt):
                    tsp_worker.evaluate(CODE, data, 10, checkpoint, run)
                self.assertEqual(len(json.loads(checkpoint.read_text())["rows"]), 2)
                result = tsp_worker.evaluate(CODE, data, 10, checkpoint, run)
            self.assertEqual(len(calls), 5)
            self.assertEqual(result["instances"], 4)
            self.assertEqual(result["dev_score"], .5)
            self.assertNotIn("test_score", result)
            self.assertTrue(all(timeout == 10 and "label_tour" in instance for instance, timeout in calls))


if __name__ == "__main__":
    unittest.main()
