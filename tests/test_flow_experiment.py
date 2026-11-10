"""Offline tests of experiment invariants, SDK requests, failure handling and evaluation."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from openai import OpenAI

import flow_experiment as flow
from scripts.flow_worker import aggregate, load_config, run_instance

CODE = "def solve(**kwargs):\n    return {'job_sequence': list(range(1, kwargs['n'] + 1))}\n"


class FakeAPI:
    """Real SDK, intercepted HTTP transport: no sockets or API credentials."""

    def __init__(self, contents=None, fail_at=None, incomplete_at=None, count=1000):
        self.requests = []
        self.events = []
        self.contents = contents or [f'```python\n{CODE}```'] * 4
        self.fail_at = fail_at
        self.incomplete_at = incomplete_at
        self.count = count
        self.factory_calls = 0

    def factory(self):
        self.factory_calls += 1
        return OpenAI(api_key='test-not-a-real-key', max_retries=0,
                      http_client=httpx.Client(transport=httpx.MockTransport(self.handle)))

    def handle(self, request):
        body = json.loads(request.content)
        if request.url.path.endswith('/input_tokens'):
            return httpx.Response(200, json={'input_tokens': self.count, 'object': 'response.input_tokens'})
        if not request.url.path.endswith('/responses'):
            raise AssertionError(f'Unexpected HTTP request: {request.url}')
        index = len(self.requests)
        self.requests.append(body)
        self.events.append(('generate', index))
        if index == self.fail_at:
            return httpx.Response(500, json={'error': {'message': 'test-not-a-real-key must not leak'}})
        return httpx.Response(200, json={
            'id': f'resp_offline_{index}', 'object': 'response', 'created_at': 0,
            'status': 'incomplete' if index == self.incomplete_at else 'completed',
            'model': flow.MODEL, 'incomplete_details': {'reason': 'max_output_tokens'} if index == self.incomplete_at else None,
            'output': [{'type': 'message', 'role': 'assistant', 'id': f'msg_{index}', 'status': 'completed',
                        'content': [{'type': 'output_text', 'text': self.contents[index], 'annotations': []}]}],
            'usage': {'input_tokens': 1000, 'output_tokens': 10000, 'total_tokens': 11000,
                      'input_tokens_details': {'cached_tokens': 0},
                      'output_tokens_details': {'reasoning_tokens': 8000}},
        })


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name) / 'experiment'
        self.api = FakeAPI()
        # Real task metadata, synthetic scores; integration coverage without 480 solver runs.
        self.info = flow.task_info()
        self.dev_scores = dict(initial=.7, support=.8, neutral=.6, switch=.75)
        self.test_scores = dict(initial=.72, support=.65, neutral=.95, switch=.8)
        self.failed_initial = False
        self.partial_initial = False
        self.patch_plot = patch('flow_experiment.plot_results')
        self.patch_plot.start()
        self.addCleanup(self.patch_plot.stop)

    def evaluator(self, source, split, folder):
        target = folder / f'{split}.json'
        if target.exists():
            return json.loads(target.read_text())
        name = folder.name
        self.api.events.append(('evaluate', name, split))
        value = (self.dev_scores if split == 'dev' else self.test_scores)[name]
        rows = [{'file': filename, 'index': i, 'status': 'ok', 'score': value, 'error': None}
                for filename, indices in self.info['splits'][split].items() for i in indices]
        if name == 'initial' and (self.failed_initial or self.partial_initial):
            for row in (rows if self.failed_initial else rows[:1]):
                row.update(status='error', score=0.0, error='Example constraint error')
        result = aggregate(rows)
        flow.save(target, result)
        return result

    def execute(self, resume=False):
        return flow.run_experiment(self.folder, resume=resume, client_provider=self.api.factory, evaluator=self.evaluator)

    def test_four_calls_same_parent_only_intervention_differs_no_test_leakage(self):
        self.assertEqual(self.execute(), 0)
        self.assertEqual(len(self.api.requests), 4)
        bodies = self.api.requests
        neutral = bodies[2]['input'][0]['content']
        for name, index in [('support', 1), ('switch', 3)]:
            self.assertEqual(bodies[index]['input'][0]['content'].replace(flow.INTERVENTIONS[name], ''), neutral)
        self.assertIn(CODE.strip(), neutral)
        self.assertNotIn('distinctly different', neutral)
        self.assertNotIn('abandon the previous method', neutral)
        self.assertNotIn('TEST', neutral)
        for body in bodies:
            self.assertEqual(body['model'], 'gpt-6.1-sol')
            self.assertEqual(body['max_output_tokens'], 32768)
            self.assertEqual(body['prompt_cache_options'], {'mode': 'explicit'})
            self.assertEqual(body['service_tier'], 'default')
            self.assertFalse(body['store'])
            self.assertNotIn('previous_response_id', body)
            self.assertNotIn('temperature', body)
        first_test = next(i for i, event in enumerate(self.api.events) if event[-1] == 'test')
        last_generation = max(i for i, event in enumerate(self.api.events) if event[0] == 'generate')
        self.assertGreater(first_test, last_generation)
        summary = json.loads((self.folder / 'summary.json').read_text())
        self.assertEqual(summary['selection_by_dev'], {'support': 'support', 'neutral': 'initial', 'switch': 'switch'})
        self.assertAlmostEqual(summary['cost_usd_estimate_or_reservation'], .408)
        self.assertEqual(self.execute(resume=True), 0)
        self.assertEqual(len(self.api.requests), 4)
        self.assertEqual(self.api.factory_calls, 1)

    def test_initial_invalid_code_stops_after_one_generation(self):
        self.api.contents[0] = 'No code'
        self.assertEqual(self.execute(), 2)
        self.assertEqual(len(self.api.requests), 1)
        self.assertFalse(any(e[0] == 'evaluate' for e in self.api.events))
        self.assertEqual(json.loads((self.folder / 'run.json').read_text())['status'], 'stopped_invalid_initial')

    def test_initial_all_case_failure_stops_but_partial_failure_branches(self):
        self.failed_initial = True
        self.assertEqual(self.execute(), 2)
        self.assertEqual(len(self.api.requests), 1)
        self.folder = Path(self.temp.name) / 'partial'
        self.api = FakeAPI()
        self.failed_initial, self.partial_initial = False, True
        self.assertEqual(self.execute(), 0)
        self.assertEqual(len(self.api.requests), 4)
        self.assertIn('Example constraint error', self.api.requests[2]['input'][0]['content'])

    def test_invalid_improvement_is_zero_without_repair_other_arms_continue(self):
        self.api.contents[1] = '```python\ndef solve(:\n```'
        self.api.incomplete_at = 2
        self.assertEqual(self.execute(), 0)
        self.assertEqual(len(self.api.requests), 4)
        for name in ['support', 'neutral']:
            for split in ['dev', 'test']:
                result = json.loads((self.folder / name / f'{split}.json').read_text())
                self.assertEqual(result['score'], 0)
                self.assertEqual(result['successful'], 0)

    def test_api_error_does_not_retry_and_cannot_be_resent_on_resume(self):
        self.api.fail_at = 1
        with self.assertRaisesRegex(RuntimeError, 'no retry'):
            self.execute()
        self.assertEqual(len(self.api.requests), 2)
        with self.assertRaisesRegex(RuntimeError, 'No automatic resend'):
            self.execute(resume=True)
        self.assertEqual(len(self.api.requests), 2)
        self.assertNotIn('test-not-a-real-key', (self.folder / 'run.json').read_text())

    def test_evaluation_interruption_reuses_saved_response(self):
        calls = 0

        def interrupted(source, split, folder):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt()
            return self.evaluator(source, split, folder)

        with self.assertRaises(KeyboardInterrupt):
            flow.run_experiment(self.folder, client_provider=self.api.factory, evaluator=interrupted)
        self.assertEqual(len(self.api.requests), 2)
        self.assertEqual(self.execute(resume=True), 0)
        self.assertEqual(len(self.api.requests), 4)

    def test_budget_or_input_limit_stops_before_generation(self):
        with patch.dict(flow.SETTINGS, budget_usd=.01):
            with self.assertRaisesRegex(RuntimeError, 'budget'):
                self.execute()
        self.assertEqual(len(self.api.requests), 0)
        self.folder = Path(self.temp.name) / 'long'
        self.api.count = flow.MAX_INPUT + 1
        with self.assertRaisesRegex(RuntimeError, 'input tokens'):
            self.execute()
        self.assertEqual(len(self.api.requests), 0)

    def test_resume_rejects_changed_configuration(self):
        self.execute()
        with patch.dict(flow.SETTINGS, seed=1):
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                self.execute(resume=True)

    def test_plot_and_csv_are_generated(self):
        self.execute()
        self.patch_plot.stop()
        run = json.loads((self.folder / 'run.json').read_text())
        flow.write_summary(self.folder, run)
        for filename in ['scores.csv', 'size_scores.csv', 'scores.png', 'scores.pdf']:
            self.assertGreater((self.folder / filename).stat().st_size, 100)


class WorkerTests(unittest.TestCase):
    def test_mutating_solver_input_does_not_change_scoring_instance(self):
        config = load_config(flow.TASK / 'config.py')
        instance = {'n': 3, 'm': 2, 'matrix': [[1, 2], [3, 1], [2, 4]], 'lower_bound': 5}
        source = "def solve(**kwargs):\n    kwargs['matrix'][0][0] = 100000\n    return {'job_sequence': [1, 2, 3]}\n"
        result = run_instance(source, instance, config.eval_func, 3, 42)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['score'], config.eval_func(**instance, job_sequence=[1, 2, 3]))

    def test_timeout_errors_and_valid_solution(self):
        config = load_config(flow.TASK / 'config.py')
        instance = {'n': 3, 'm': 2, 'matrix': [[1, 2], [3, 1], [2, 4]], 'lower_bound': 5}
        valid = run_instance(CODE, instance, config.eval_func, 3, 42)
        self.assertEqual(valid['status'], 'ok')
        expected = config.eval_func(**instance, job_sequence=[1, 2, 3])
        self.assertEqual(valid['score'], expected)
        bad = run_instance("def solve(**kwargs):\n    return {'job_sequence': [1, 1, 1]}\n", instance, config.eval_func, 3, 42)
        self.assertEqual(bad['status'], 'error')
        slow = run_instance('while True: pass', instance, config.eval_func, .2, 42)
        self.assertEqual(slow['status'], 'timeout')

    def test_aggregation_keeps_failed_cases_and_equal_file_weight(self):
        result = aggregate([{'file': 'a', 'score': 1, 'status': 'ok'},
                            {'file': 'a', 'score': 0, 'status': 'error'},
                            {'file': 'b', 'score': 1, 'status': 'ok'}])
        self.assertEqual(result['score'], .75)
        self.assertEqual(result['valid_rate'], 2 / 3)

    def test_environment_does_not_forward_credentials(self):
        with patch.dict(os.environ, OPENAI_API_KEY='test-secret', COBENCH_API_KEY='another-secret'):
            env = flow.clean_environment()
        self.assertNotIn('OPENAI_API_KEY', env)
        self.assertNotIn('COBENCH_API_KEY', env)
        self.assertEqual(env['OPENBLAS_NUM_THREADS'], '1')


if __name__ == '__main__':
    unittest.main()
