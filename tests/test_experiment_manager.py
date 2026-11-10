"""Offline integration tests for definitions, parent policies and restart safety."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

import experiment
from experiment_manager import definitions as defs, engine, evaluation as ev
from experiment_manager.reports import report
from experiment_manager.storage import digest, hashes, read, save

CODE = "def solve(**kwargs):\n    return {'answer': kwargs['value']}\n"
CONFIG = '''import json
DESCRIPTION = "A tiny test task"
def solve(**kwargs):
    return {'answer': kwargs['value']}
def load_data(path):
    with open(path) as handle:
        return json.load(handle)
def eval_func(answer, **kwargs):
    return answer
def norm_score(results):
    return {name: ([score if isinstance(score, str) else [10, 20, 30][i] / score
                    for i, score in enumerate(scores)], error)
            for name, (scores, error) in results.items()}
def get_dev():
    return {'cases.json': [1]}
'''


class ManagerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        task = self.data / "tiny"
        task.mkdir(parents=True)
        (task / "config.py").write_text(CONFIG)
        save(task / "cases.json", [{"value": 10}, {"value": 20}, {"value": 30}])
        # This file is outside dev and must never be executed.
        save(task / "hidden.json", [{"value": 99}])
        self.exp = self.root / "demo"
        self.draft = self.exp / "drafts/next"
        (self.draft / "prompts").mkdir(parents=True)
        (self.draft / "prompts/refine.txt").write_text(
            "{problem_description}\nCODE:\n{previous_code}\nFEEDBACK:{dev_feedback}\n{intervention}\nTIME:{timeout}")
        (self.draft / "prompts/support.txt").write_text("SUPPORT")
        (self.draft / "prompts/neutral.txt").write_text("")
        initial = self.exp / "initials/one"
        initial.mkdir(parents=True)
        (initial / "solution.py").write_text(CODE)
        self.cfg = {"schema_version": 1, "name": "demo", "task": "tiny", "data_dir": str(self.data),
            "initials": [{"id": "one", "directory": "../../initials/one"}],
            "parent_policies": ["initial", "latest", "best"], "repeats": 1, "steps": 3,
            "prompts": {"template": "prompts/refine.txt", "conditions": {
                "support": "prompts/support.txt", "neutral": "prompts/neutral.txt"}},
            "generation": {}, "evaluation": {"timeout_seconds": 1},
            "seeds": {"base": 101, "paired_across_conditions": True}}
        self.write_cfg()
        self.requests = []
        self.evaluations = []
        self.bad = {}
        self.scores = {}
        self.interrupt_at = None
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def write_cfg(self):
        (self.draft / "experiment.yaml").write_text(yaml.safe_dump(self.cfg, sort_keys=False))

    def prepare(self):
        self.version = defs.freeze(self.draft, "v001", "Initial design")
        self.run_dir = engine.prepare(self.version, "run1")
        return self.run_dir

    def complete(self, request):
        text = request["messages"][0]["content"]
        self.requests.append(request)
        position = len(self.requests)
        content, finish = self.bad.get(position, (f"```python\n# generated {position}\n{CODE}```", "stop"))
        return {"choices": [{"finish_reason": finish, "message": {"content": content, "reasoning_content": "private reasoning"}}],
                "usage": {"completion_tokens": 12}, "model": "fake-model"}

    def evaluator(self, folder, reference, task, settings, error=None):
        path = Path(folder) / reference
        self.evaluations.append(reference)
        if self.interrupt_at == reference:
            self.interrupt_at = None
            raise KeyboardInterrupt()
        if (path / "dev.json").exists():
            return read(path / "dev.json")
        score = self.scores.get(reference, .5 if reference.startswith("initials/") else .6)
        result = ev.failure(task, error) if error else ev.summarize({"cases.json": ([score], None)})
        save(path / "dev.json", result)
        return result

    def run_it(self):
        with patch("experiment_manager.reports.plot"):
            return engine.run(self.run_dir, self.complete, self.evaluator, lambda *a, **kw: {})

    def test_all_policies_conditions_seeds_and_no_cross_branch_parents(self):
        self.prepare()
        # Score progression .6 -> .6 -> .6: best retains the first generated candidate.
        self.run_it()
        manifest = read(self.run_dir / "manifest.json")
        self.assertEqual(len(self.requests), 18)
        self.assertEqual(manifest["status"], "completed")
        for record in manifest["candidates"].values():
            initial = "initials/one"
            if record["step"] == 1 or record["policy"] == "initial":
                expected = initial
            else:
                step = record["step"] - 1 if record["policy"] == "latest" else 1
                expected = f"{record['branch']}/step_{step:03d}"
            self.assertEqual(record["parent"], expected)
            self.assertEqual(record["seed"], 100 + record["step"])
            prompt = (self.run_dir / record["id"] / "prompt.txt").read_text()
            self.assertEqual("SUPPORT" in prompt, record["condition"] == "support")
            self.assertNotIn("private reasoning", prompt)
        count = len(self.requests)
        self.run_it()
        self.assertEqual(len(self.requests), count)

    def test_best_keeps_initial_on_equal_or_worse_and_latest_carries_failure(self):
        self.prepare()
        plan = read(self.run_dir / "plan.json")
        for item in plan:
            if item["step"] == 1:
                self.scores[item["id"]] = .5
        latest_first = next(i for i in plan if i["policy"] == "latest")
        position = plan.index(latest_first) + 1
        self.bad[position] = ("A failed answer with no code", "stop")
        self.run_it()
        records = read(self.run_dir / "manifest.json")["candidates"]
        for record in records.values():
            if record["policy"] == "best" and record["step"] == 2:
                self.assertEqual(record["parent"], "initials/one")
        next_item = next(i for i in plan if i["branch"] == latest_first["branch"] and i["step"] == 2)
        prompt = (self.run_dir / next_item["id"] / "prompt.txt").read_text()
        self.assertIn("A failed answer with no code", prompt)
        self.assertIn("No Python code block", prompt)
        self.assertEqual(read(self.run_dir / latest_first["id"] / "dev.json")["dev_score"], 0)

    def test_syntax_and_truncation_consume_slots(self):
        self.prepare()
        self.bad[1] = ("```python\ndef solve(:\n```", "stop")
        self.bad[2] = ("Incomplete answer", "length")
        self.run_it()
        records = list(read(self.run_dir / "manifest.json")["candidates"].values())
        for record in records[:2]:
            result = read(self.run_dir / record["id"] / "dev.json")
            self.assertEqual(result["dev_score"], 0)
            self.assertTrue(record["code_error"])
        self.assertEqual(len(self.requests), 18)

    def test_inline_reasoning_is_excluded_from_failure_parent(self):
        self.cfg.update(parent_policies=["latest"], steps=2)
        self.write_cfg()
        self.prepare()
        self.bad[1] = ("<think>Do not feed this reasoning</think>Visible failed answer", "length")
        self.run_it()
        record = next(r for r in read(self.run_dir / "manifest.json")["candidates"].values()
                      if r["condition"] == "support" and r["step"] == 2)
        prompt = (self.run_dir / record["id"] / "prompt.txt").read_text()
        self.assertIn("Visible failed answer", prompt)
        self.assertNotIn("Do not feed this reasoning", prompt)

    def test_interrupted_evaluation_reuses_response_prompt_seed_and_parent(self):
        self.prepare()
        item = read(self.run_dir / "plan.json")[1]
        self.interrupt_at = item["id"]
        with self.assertRaises(KeyboardInterrupt):
            self.run_it()
        first_hashes = hashes(self.run_dir / item["id"])
        self.assertEqual(len(self.requests), 2)
        # Editing the working draft cannot affect the already prepared run.
        (self.draft / "prompts/support.txt").write_text("CHANGED")
        self.run_it()
        self.assertEqual(len(self.requests), 18)
        for filename in ("prompt.txt", "request.json", "response.json"):
            self.assertEqual(digest(self.run_dir / item["id"] / filename), first_hashes[filename])

    def test_unknown_response_is_never_resent(self):
        self.prepare()
        calls = []
        def fail(request):
            calls.append(request)
            raise ConnectionError("uncertain")
        with self.assertRaises(ConnectionError):
            engine.run(self.run_dir, fail, self.evaluator, lambda *a, **kw: {})
        with self.assertRaisesRegex(RuntimeError, "No automatic resend"):
            engine.run(self.run_dir, fail, self.evaluator, lambda *a, **kw: {})
        self.assertEqual(len(calls), 1)

    def test_versions_preserve_prompts_lineage_and_snapshots(self):
        self.prepare()
        old_hashes = hashes(self.version)
        draft2 = defs.fork(self.version, "variant")
        (draft2 / "prompts/interventions/support.txt").write_text("NEW INTERVENTION")
        version2 = defs.freeze(draft2, "v002", "Change support prompt")
        self.assertEqual(read(version2 / "version.json")["parent_version"], "v001")
        self.assertEqual(hashes(self.version), old_hashes)
        self.assertIn("NEW INTERVENTION", defs.diff(self.version, version2))
        engine.verify(self.run_dir, runtime=True, data=True)
        (self.version / "prompts/interventions/support.txt").write_text("tampered")
        with self.assertRaisesRegex(RuntimeError, "Published version changed"):
            defs.verify_version(self.version)
        # Published source version changes cannot alter a previously prepared run.
        engine.verify(self.run_dir, runtime=True, data=True)

    def test_repeated_freeze_tracks_lineage_without_overwriting(self):
        version1 = defs.freeze(self.draft, "v001", "First")
        original = hashes(version1)
        (self.draft / "prompts/support.txt").write_text("Version two")
        version2 = defs.freeze(self.draft, "v002", "Second")
        self.assertEqual(read(version2 / "version.json")["parent_version"], "v001")
        self.assertEqual(hashes(version1), original)
        with self.assertRaises(FileExistsError):
            defs.freeze(self.draft, "v001", "Attempt overwrite")

    def test_fixed_artifacts_and_completed_candidate_tampering_are_rejected(self):
        self.prepare()
        self.run_it()
        record = next(iter(read(self.run_dir / "manifest.json")["candidates"].values()))
        (self.run_dir / record["id"] / "solution.py").write_text("tampered")
        with self.assertRaisesRegex(RuntimeError, "Saved artifact changed"):
            engine.verify(self.run_dir)

    def test_repeated_fixed_initial_generation_count_and_seed_pairing(self):
        self.cfg.update(parent_policies=["initial"], repeats=9, steps=1)
        self.write_cfg()
        cfg, task, plan = defs.task_and_plan(self.draft)
        self.assertEqual(len(plan), 18)  # Two conditions in this fixture.
        self.assertEqual([i["seed"] for i in plan[::2]], list(range(101, 110)))
        self.assertEqual([i["seed"] for i in plan[::2]], [i["seed"] for i in plan[1::2]])

    def test_invalid_settings_duplicate_yaml_and_prompt_variables_fail_before_generation(self):
        self.cfg["steps"] = 0
        self.write_cfg()
        with self.assertRaises(ValueError):
            defs.load(self.draft)
        self.cfg["steps"] = 1
        self.write_cfg()
        path = self.draft / "experiment.yaml"
        path.write_text(path.read_text() + "steps: 2\n")
        with self.assertRaisesRegex(ValueError, "Duplicate YAML"):
            defs.load(self.draft)
        self.write_cfg()
        template = self.draft / "prompts/refine.txt"
        template.write_text(template.read_text().replace("{intervention}", "{typo}"))
        with self.assertRaisesRegex(ValueError, "Prompt must contain"):
            defs.load(self.draft)

    def test_cached_initial_reused_only_for_exact_signature(self):
        info = ev.describe("tiny", self.data)
        path = self.exp / "initials/one"
        result = ev.summarize({"cases.json": ([.75], None)})
        result["signature"] = ev.signature(info, digest(path / "solution.py"), {"timeout_seconds": 1})
        save(path / "dev.json", result)
        self.prepare()
        self.assertTrue(read(self.run_dir / "manifest.json")["initials"]["one"]["evaluation_reused"])
        self.run_it()
        self.assertNotIn("initials/one", self.evaluations)
        result["signature"]["timeout_seconds"] = 2
        save(path / "dev.json", result)
        self.assertIsNone(engine.cached_initial(self.draft, defs.load(self.draft), info, self.cfg["initials"][0]))

    def test_real_worker_only_executes_dev_and_keeps_reference_indices(self):
        task = ev.describe("tiny", self.data)
        self.assertEqual(task["dev_indices"], {"cases.json": [1], "hidden.json": []})
        self.prepare()
        # Real subprocess worker, not an implementation-mirroring fake.
        result = engine.evaluate_path(self.run_dir, "initials/one", task, {**defs.DEFAULT_EVALUATION, "timeout_seconds": 1})
        self.assertEqual(result["dev_score"], 1)
        self.assertEqual(result["instances"], 1)
        self.assertNotIn("test_score", result)
        checkpoint = read(self.run_dir / "initials/one/dev.rows.json")
        self.assertEqual([(r["file"], r["index"]) for r in checkpoint["rows"]], [("cases.json", 1)])
        self.assertNotIn("hidden.json", result["results"])

    def test_dev_checkpoint_resumes_without_reexecuting_finished_instances(self):
        task = ev.describe("tiny", self.data)
        checkpoint = self.root / "checkpoint.json"
        calls = []
        def runtime(instance, code, config, timeout):
            calls.append(instance)
            return instance["value"]
        result = ev.evaluate(CODE, task, {**defs.DEFAULT_EVALUATION, "timeout_seconds": 1}, checkpoint, runtime)
        again = ev.evaluate(CODE, task, {**defs.DEFAULT_EVALUATION, "timeout_seconds": 1}, checkpoint, runtime)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["dev_score"], again["dev_score"])

    def test_interrupted_partial_dev_checkpoint_and_data_change_detection(self):
        config = self.data / "tiny/config.py"
        config.write_text(CONFIG.replace("[1]}", "[0, 2]}"))
        task = ev.describe("tiny", self.data)
        checkpoint = self.root / "checkpoint.json"
        calls = []
        def runtime(instance, code, config, timeout):
            calls.append(instance["value"])
            if len(calls) == 2:
                raise KeyboardInterrupt()
            return instance["value"]
        settings = {**defs.DEFAULT_EVALUATION, "timeout_seconds": 1}
        with self.assertRaises(KeyboardInterrupt):
            ev.evaluate(CODE, task, settings, checkpoint, runtime)
        self.assertEqual(len(read(checkpoint)["rows"]), 1)
        result = ev.evaluate(CODE, task, settings, checkpoint, runtime)
        self.assertEqual(calls, [10, 30, 30])
        self.assertEqual(result["dev_score"], 1)
        self.prepare()
        save(self.data / "tiny/cases.json", [{"value": 999}])
        with self.assertRaisesRegex(RuntimeError, "Task data changed"):
            engine.verify(self.run_dir, data=True)

    def test_initial_import_and_cli_workflow_require_no_model(self):
        source = self.root / "candidate.py"
        source.write_text(CODE)
        target = self.root / "initialized"
        draft = experiment.initialize(target, source, "tiny", "other")
        cfg = yaml.safe_load((draft / "experiment.yaml").read_text())
        cfg["data_dir"] = str(self.data)
        (draft / "experiment.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        imported = target / "initials/other"
        self.assertEqual((imported / "solution.py").read_text(), CODE)
        with self.assertRaises(FileExistsError):
            defs.import_initial(target, "other", source, "tiny", self.data)
        with patch("experiment_manager.engine.complete", side_effect=AssertionError("Must not generate")):
            self.assertEqual(experiment.main(["plan", str(draft), "--preview-dir", str(self.root / "preview")]), 0)
            self.assertEqual(experiment.main(["freeze", str(draft), "--version", "v001", "--reason", "CLI check"]), 0)
            self.assertEqual(experiment.main(["prepare", str(target / "versions/v001"), "--run-id", "ready"]), 0)
            self.assertEqual(experiment.main(["check", str(target / "runs/ready")]), 0)
            self.assertEqual(experiment.main(["list", str(target)]), 0)
        self.assertEqual(read(target / "runs/ready/manifest.json")["candidates"], {})

    def test_reports_generate_exportable_figures(self):
        self.cfg.update(parent_policies=["initial", "latest"], steps=2)
        self.write_cfg()
        self.prepare()
        self.run_it()
        summary = report(self.run_dir)
        self.assertEqual(summary["candidates_evaluated"], 8)
        for policy in ("initial", "latest"):
            for extension in ("png", "svg", "pdf"):
                self.assertTrue((self.run_dir / "reports/figures" / f"one_{policy}.{extension}").is_file())


if __name__ == "__main__":
    unittest.main()
