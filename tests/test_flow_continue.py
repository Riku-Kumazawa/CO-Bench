"""Offline continuation tests: lineage, limits, imported results and resumption."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import flow_experiment as base
import flow_continue as cont
from test_flow_experiment import CODE, FakeAPI


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.destination = self.root / 'pilot', self.root / 'continued'
        self.plot_patch = patch('flow_experiment.plot_results')
        self.plot_patch.start()
        self.addCleanup(self.plot_patch.stop)
        self.cont_plot = patch('flow_continue.plot')
        self.cont_plot.start()
        self.addCleanup(self.cont_plot.stop)
        self.info = base.task_info()
        self.api = FakeAPI(contents=[f'```python\n# generated {i}\n{CODE}```' for i in range(32)])
        self.events = []
        self.special_scores = {}
        self.interrupt_at = None
        base.run_experiment(self.source, client_provider=self.api.factory, evaluator=self.evaluator)
        self.api = FakeAPI(contents=[f'```python\n# continuation {i}\n{CODE}```' for i in range(32)])
        self.source_hashes = {str(p.relative_to(self.source)): base.digest(p)
                              for p in self.source.rglob('*') if p.is_file() and p.name != '.lock'}

    def evaluator(self, source, split, folder):
        target = folder / f'{split}.json'
        if target.exists():
            return json.loads(target.read_text())
        name = folder.name
        self.events.append((name, split, len(self.api.requests)))
        if self.interrupt_at == (name, split):
            self.interrupt_at = None
            raise KeyboardInterrupt()
        condition = name.split('_')[0]
        round_number = int(name.split('_')[1]) if '_' in name else 2
        value = {'initial': .7, 'support': .8, 'neutral': .85, 'switch': .69}[condition]
        if '_' in name:
            value += .001 * round_number
        if split == 'test':
            value = .99 if condition == 'switch' else value - .01
        value = self.special_scores.get((name, split), value)
        result = base.aggregate([{'file': filename, 'index': i, 'score': value,
                                  'status': 'ok', 'error': None, 'elapsed_seconds': 0}
                                 for filename, indices in self.info['splits'][split].items() for i in indices])
        base.save(target, result)
        return result

    def prepare(self, submissions=10, budget=10):
        self.events.clear()
        cont.prepare(self.source, self.destination, submissions, budget)

    def run_it(self):
        return cont.run_continuation(self.destination, client_provider=self.api.factory, evaluator=self.evaluator)

    def test_prepare_imports_without_generation_or_evaluation_and_preserves_source(self):
        self.prepare()
        self.assertEqual(len(self.api.requests), 0)
        self.assertEqual(self.events, [])
        run = json.loads((self.destination / 'run.json').read_text())
        self.assertEqual(run['settings']['total_generations'], 28)
        self.assertEqual(len(run['candidates']), 4)
        self.assertEqual(cont.check(self.destination), 0)
        self.assertEqual(self.source_hashes, {str(p.relative_to(self.source)): base.digest(p)
                                            for p in self.source.rglob('*') if p.is_file() and p.name != '.lock'})

    def test_24_new_requests_ten_submissions_each_parent_selection_is_condition_local(self):
        self.prepare()
        self.assertEqual(self.run_it(), 0)
        self.assertEqual(len(self.api.requests), 24)
        run = json.loads((self.destination / 'run.json').read_text())
        summary = json.loads((self.destination / 'summary.json').read_text())
        self.assertEqual(summary['completed_submissions'], dict(support=10, neutral=10, switch=10))
        self.assertEqual(summary['generation_requests_attempted'], 28)
        self.assertEqual(run['candidates']['support_03']['parent'], 'support')
        self.assertEqual(run['candidates']['neutral_03']['parent'], 'neutral')
        self.assertEqual(run['candidates']['switch_03']['parent'], 'initial')
        self.assertEqual(run['candidates']['support_04']['parent'], 'support_03')
        # switch test performance is .99, but every dev score remains below initial .7.
        self.assertEqual(run['candidates']['switch_10']['parent'], 'initial')
        for index, item in enumerate(run['plan'][4:]):
            record = run['candidates'][item['name']]
            prompt = self.api.requests[index]['input'][0]['content']
            parent_code = (self.destination / record['parent'] / 'solution.py').read_text()
            self.assertIn(parent_code, prompt)
            if item['condition'] == 'neutral':
                self.assertNotIn(base.INTERVENTIONS['support'], prompt)
                self.assertNotIn(base.INTERVENTIONS['switch'], prompt)
            else:
                self.assertIn(base.INTERVENTIONS[item['condition']], prompt)
        new_tests = [event for event in self.events if event[1] == 'test']
        self.assertTrue(all(count == 24 for _, _, count in new_tests))
        self.assertEqual(self.run_it(), 0)
        self.assertEqual(len(self.api.requests), 24)
        self.assertEqual(self.api.factory_calls, 1)
        self.assertEqual(self.source_hashes, {str(p.relative_to(self.source)): base.digest(p)
                                            for p in self.source.rglob('*') if p.is_file() and p.name != '.lock'})

    def test_failed_or_equal_proposal_does_not_replace_incumbent(self):
        self.prepare(submissions=4)
        self.api.contents[0] = 'No code'
        self.special_scores['neutral_03', 'dev'] = .85
        self.run_it()
        run = json.loads((self.destination / 'run.json').read_text())
        self.assertEqual(len(self.api.requests), 6)
        self.assertEqual(run['candidates']['support_04']['parent'], 'support')
        self.assertEqual(run['candidates']['neutral_04']['parent'], 'neutral')
        self.assertEqual(cont.load_result(self.destination / 'support_03', 'dev')['score'], 0)

    def test_interrupted_evaluation_resumes_without_duplicate_generation(self):
        self.prepare(submissions=4)
        self.interrupt_at = 'neutral_03', 'dev'
        with self.assertRaises(KeyboardInterrupt):
            self.run_it()
        self.assertEqual(len(self.api.requests), 2)
        self.assertEqual(self.run_it(), 0)
        self.assertEqual(len(self.api.requests), 6)

    def test_unknown_api_result_cannot_be_resent_and_error_is_saved(self):
        self.prepare(submissions=3)
        self.api.fail_at = 1
        with self.assertRaisesRegex(RuntimeError, 'no retry'):
            self.run_it()
        self.assertEqual(len(self.api.requests), 2)
        with self.assertRaisesRegex(RuntimeError, 'No automatic resend'):
            self.run_it()
        self.assertEqual(len(self.api.requests), 2)
        self.assertNotIn('test-not-a-real-key', (self.destination / 'run.json').read_text())

    def test_budget_prevents_generation_and_import_integrity_is_checked(self):
        self.prepare(submissions=3, budget=.42)
        with self.assertRaisesRegex(RuntimeError, 'budget'):
            self.run_it()
        self.assertEqual(len(self.api.requests), 0)
        (self.destination / 'support' / 'dev.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'artifact changed'):
            self.run_it()

    def test_plot_and_separate_cost_table(self):
        self.prepare(submissions=3)
        self.run_it()
        self.cont_plot.stop()
        run = json.loads((self.destination / 'run.json').read_text())
        cont.report(self.destination, run)
        self.assertGreater((self.destination / 'scores.png').stat().st_size, 100)
        self.assertEqual(len((self.destination / 'costs.csv').read_text().splitlines()), 8)


if __name__ == '__main__':
    unittest.main()
