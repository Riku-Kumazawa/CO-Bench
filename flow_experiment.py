"""One shared Flow shop initial solver, followed by three independent interventions."""

import argparse
import ast
import contextlib
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import textwrap
import time

from scripts.flow_worker import aggregate, load_config, save

ROOT = Path(__file__).resolve().parent
TASK = ROOT / 'data' / 'Flow shop scheduling'
MODEL = 'gpt-6.1-sol'
MAX_OUTPUT = 32768
MAX_INPUT = 20000
INPUT_PRICE = 2.0  # USD / 1M tokens; Standard, short context, 2026-11-08.
OUTPUT_PRICE = 10.0
INTERVENTIONS = {
    'support': 'I suggest continuing to refine the current algorithmic approach.',
    'neutral': '',
    'switch': 'I suggest exploring a different algorithmic approach.',
}
NAMES = ('initial', *INTERVENTIONS)
SETTINGS = {'model': MODEL, 'reasoning_effort': 'medium', 'max_output_tokens': MAX_OUTPUT,
            'max_input_tokens': MAX_INPUT, 'timeout_seconds': 10, 'cpu_num': 1, 'seed': 0,
            'budget_usd': 2.0, 'service_tier': 'default', 'generation_timeout_seconds': 1800,
            'input_price_per_million': INPUT_PRICE, 'output_price_per_million': OUTPUT_PRICE,
            'price_date': '2026-11-08', 'prompt_cache_mode': 'explicit_no_breakpoints',
            'evaluation_repetitions': 1, 'candidate_order': list(NAMES),
            'score_aggregation': 'mean of file means; errors are zero',
            'selection': 'dev score; ties retain initial', 'schema_version': 1}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def template(kind):
    return (ROOT / 'prompts' / f'flow_{kind}.txt').read_text()


def make_prompt(problem, condition='initial', code='', feedback=''):
    return template('initial' if condition == 'initial' else 'refine').format(
        problem_description=problem, previous_code=code, dev_feedback=feedback,
        intervention=INTERVENTIONS.get(condition, ''), timeout=SETTINGS['timeout_seconds'])


def feedback_text(result):
    groups = {}
    for row in result['rows']:
        groups.setdefault(row['file'], []).append(
            f'{row["score"]:.3f}' if row['status'] == 'ok' else row['error'])
    return '\n'.join(f'{name} -> Scores: {values!r}' for name, values in groups.items()) + f'\nAvg Score {result["score"]}'


def task_info():
    config_path = TASK / 'config.py'
    config = load_config(config_path)
    dev = config.get_dev()
    files = sorted(p.name for p in TASK.glob('*.txt'))
    if files != sorted(dev):
        raise ValueError('Data files differ from the configured dev split')
    splits = {'dev': {}, 'test': {}}
    for name in files:
        instances = config.load_data(str(TASK / name))
        ids = dev[name]
        if len(set(ids)) != len(ids) or not ids or not set(ids) < set(range(len(instances))):
            raise ValueError(f'Invalid dev/test split: {name}')
        splits['dev'][name] = sorted(ids)
        splits['test'][name] = [i for i in range(len(instances)) if i not in ids]
    return {'problem_description': f'{config.DESCRIPTION}\n\n# Implement in Solve Function\n\n{inspect.getsource(config.solve).rstrip()}',
            'splits': splits,
            'data_hashes': {name: digest(TASK / name) for name in ['config.py', *files]}}


def fingerprints():
    names = ['flow_experiment.py', 'scripts/flow_worker.py', 'prompts/flow_initial.txt',
             'prompts/flow_refine.txt', 'uv.lock']
    return {name: digest(ROOT / name) for name in names}


def packages():
    return {name: importlib.metadata.version(name) for name in
            ['openai', 'numpy', 'scipy', 'ortools', 'matplotlib']}


def clean_environment():
    env = {key: os.environ[key] for key in ['PATH', 'LANG', 'LC_ALL', 'LD_LIBRARY_PATH', 'TMPDIR'] if key in os.environ}
    env.update({key: '1' for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']})
    env.update(PYTHONHASHSEED='0', PYTHONUNBUFFERED='1')
    return env


def evaluate_candidate(source, split, folder):
    target = folder / f'{split}.json'
    if target.exists():
        return json.loads(target.read_text())
    # Exec a new interpreter: no API client or API key is inherited in process memory.
    with (folder / f'{split}.log').open('a') as log:
        subprocess.run([sys.executable, str(ROOT / 'scripts' / 'flow_worker.py'),
                        '--config', str(TASK / 'config.py'), '--source', str(source),
                        '--split', split, '--timeout', str(SETTINGS['timeout_seconds']),
                        '--seed', str(SETTINGS['seed']), '--output', str(target)],
                       env=clean_environment(), stdout=log, stderr=log, check=True, cwd=ROOT)
    return json.loads(target.read_text())


def failed_result(info, split, message):
    rows = [{'file': name, 'index': i, 'status': 'invalid_code', 'score': 0.0, 'error': message,
             'elapsed_seconds': 0.0} for name, indices in info['splits'][split].items() for i in indices]
    return aggregate(rows)


def extract_code(response):
    text = ''.join(part.get('text', '') for item in response.get('output', [])
                   if item.get('type') == 'message' for part in item.get('content', [])
                   if part.get('type') == 'output_text')
    if response.get('status') != 'completed':
        return text, None, f'Response status: {response.get("status")}; details: {response.get("incomplete_details")}'
    matches = re.findall(r'```python\s*\n?(.*?)```', text, flags=re.DOTALL)
    if not matches:
        return text, None, 'No Python code block'
    code = textwrap.dedent(matches[0]).strip() + '\n'
    try:
        tree = ast.parse(code)
        compile(tree, 'solution.py', 'exec')
        if not any(isinstance(node, ast.FunctionDef) and node.name == 'solve' for node in tree.body):
            raise ValueError('No top-level solve function')
    except (SyntaxError, ValueError) as exc:
        return text, code, f'{type(exc).__name__}: {exc}'
    return text, code, None


def request_body(prompt):
    return {'model': MODEL, 'input': [{'role': 'user', 'content': prompt}],
            'reasoning': {'effort': 'medium'}, 'max_output_tokens': MAX_OUTPUT,
            'service_tier': 'default', 'store': False,
            'prompt_cache_options': {'mode': 'explicit'}, 'truncation': 'disabled'}


def client_factory():
    from openai import OpenAI
    if not os.environ.get('OPENAI_API_KEY'):
        raise RuntimeError('Set OPENAI_API_KEY in this terminal before running')
    return OpenAI(api_key=os.environ['OPENAI_API_KEY'], base_url='https://api.openai.com/v1',
                  max_retries=0, timeout=SETTINGS['generation_timeout_seconds'])


def usage_cost(response, reserved):
    usage = response.get('usage')
    if not usage:
        return reserved, 'reserved_maximum_usage_missing'
    return (usage['input_tokens'] * INPUT_PRICE + usage['output_tokens'] * OUTPUT_PRICE) / 1e6, 'standard_no_cache_estimate'


def total_cost(run):
    return sum(item.get('cost_usd', item.get('reserved_usd', 0)) for item in run['candidates'].values())


def generate(run, run_dir, name, prompt, get_client):
    folder = run_dir / name
    folder.mkdir(exist_ok=True)
    response_path = folder / 'response.json'
    record = run['candidates'].get(name)
    body = request_body(prompt)
    if record is not None:
        if json.loads((folder / 'request.json').read_text()) != body:
            raise RuntimeError('Saved request differs from the current prompt; refusing to resume')
        if not response_path.exists():
            raise RuntimeError(f'{name}: request was already attempted but no response was saved. No automatic resend.')
        response = json.loads(response_path.read_text())
    else:
        (folder / 'prompt.txt').write_text(prompt)
        save(folder / 'request.json', body)
        client = get_client()
        count = client.responses.input_tokens.count(model=MODEL, input=body['input'], reasoning=body['reasoning'])
        tokens = count.input_tokens
        if tokens > MAX_INPUT:
            raise RuntimeError(f'{name}: {tokens} input tokens exceeds limit {MAX_INPUT}; no generation sent')
        reserved = (tokens * INPUT_PRICE + MAX_OUTPUT * OUTPUT_PRICE) / 1e6
        if total_cost(run) + reserved > SETTINGS['budget_usd']:
            raise RuntimeError('Insufficient remaining budget for a full response; no generation sent')
        record = {'status': 'request_started', 'started_at': now(), 'input_tokens_counted': tokens,
                  'reserved_usd': reserved}
        run['candidates'][name] = record
        save(run_dir / 'run.json', run)  # Durable before the potentially billable call.
        started = time.monotonic()
        print(f'{name}: generating (request {len(run["candidates"])}/4, maximum ${reserved:.4f})', flush=True)
        try:
            response = client.responses.create(**body).model_dump(mode='json')
            save(response_path, response)  # Save raw API response before parsing or evaluation.
        except Exception as exc:
            # API exceptions can contain credentials; save class/status only.
            record.update(status='api_error_or_unknown', error_type=type(exc).__name__,
                          http_status=getattr(exc, 'status_code', None), finished_at=now())
            save(run_dir / 'run.json', run)
            raise RuntimeError(f'{name}: API request failed ({type(exc).__name__}); no retry') from None
        record['generation_seconds'] = time.monotonic() - started
    record['cost_usd'], record['cost_basis'] = usage_cost(response, record['reserved_usd'])
    record.update(status='response_saved', usage=response.get('usage'),
                  response_id=response.get('id'), response_model=response.get('model'), finished_at=now())
    text, code, error = extract_code(response)
    (folder / 'response.txt').write_text(text)
    if code is not None:
        path = folder / 'solution.py'
        if path.exists() and path.read_text() != code:
            raise RuntimeError('Saved solution was modified; refusing to reuse its evaluation')
        path.write_text(code)
    record['code_error'] = error
    save(run_dir / 'run.json', run)
    return error


def write_summary(run_dir, run):
    results = {}
    for name in NAMES:
        results[name] = {split: json.loads(path.read_text()) for split in ['dev', 'test']
                         if (path := run_dir / name / f'{split}.json').exists()}
    initial = results['initial']
    summary = {'status': run['status'], 'generation_requests_attempted': len(run['candidates']),
               'cost_usd_estimate_or_reservation': total_cost(run), 'candidates': {}, 'selection_by_dev': {}}
    rows = []
    for name, splits in results.items():
        summary['candidates'][name] = {split: {k: v for k, v in value.items() if k != 'rows'} for split, value in splits.items()}
        if name != 'initial' and 'dev' in splits and 'dev' in initial:
            winner = name if splits['dev']['score'] > initial['dev']['score'] else 'initial'
            summary['selection_by_dev'][name] = winner
        for split, value in splits.items():
            record = run['candidates'].get(name, {})
            rows.append({'candidate': name, 'submission': 1 if name == 'initial' else 2, 'split': split,
                         'score': value['score'], 'valid_rate': value['valid_rate'],
                         'failed': value['failed'], 'cost_usd': record.get('cost_usd', ''),
                         'generation_seconds': record.get('generation_seconds', '')})
    save(run_dir / 'summary.json', summary)
    if rows:
        with (run_dir / 'scores.csv').open('w', newline='') as out:
            writer = csv.DictWriter(out, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with (run_dir / 'size_scores.csv').open('w', newline='') as out:
            writer = csv.writer(out)
            writer.writerow(['candidate', 'split', 'file', 'score'])
            for name, splits in results.items():
                for split, value in splits.items():
                    for filename, score in value['by_size'].items():
                        writer.writerow([name, split, filename, score])
    plot_results(run_dir, results, summary['selection_by_dev'])


def plot_results(run_dir, results, winners):
    if not results['initial']:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'support': '#0072B2', 'neutral': '#009E73', 'switch': '#D55E00'}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), squeeze=False)
    for row, split in enumerate(['dev', 'test']):
        for col, title in enumerate(['Submitted candidate', 'Retained by dev score']):
            ax = axes[row][col]
            ax.set(title=f'{split}: {title}', xlabel='Submission', ylabel='Normalized score', xticks=[1, 2])
            ax.grid(alpha=.25)
            if split not in results['initial']:
                ax.text(.5, .5, 'Not evaluated', ha='center', transform=ax.transAxes)
                continue
            first = results['initial'][split]['score']
            ax.scatter([1], [first], color='black', zorder=4)
            for name in INTERVENTIONS:
                selected = name if col == 0 else winners.get(name)
                if name not in winners or selected is None or split not in results[selected]:
                    continue
                second = results[selected][split]['score']
                ax.plot([1, 2], [first, second], 'o-', label=name, color=colors[name], alpha=.8)
            if ax.lines:
                ax.legend()
    fig.suptitle('Flow shop pilot: one shared initial solver, one improvement per condition')
    fig.tight_layout()
    fig.savefig(run_dir / 'scores.png', dpi=180)
    fig.savefig(run_dir / 'scores.pdf')
    plt.close(fig)


@contextlib.contextmanager
def run_lock(run_dir):
    with (run_dir / '.lock').open('w') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This experiment is already running') from None
        yield


def run_experiment(run_dir, resume=False, client_provider=client_factory, evaluator=evaluate_candidate):
    info = task_info()
    if not resume:
        run_dir.mkdir(parents=True, exist_ok=False)
    elif not (run_dir / 'run.json').exists():
        raise RuntimeError('Resume requires an existing run.json')
    with run_lock(run_dir):
        if resume:
            run = json.loads((run_dir / 'run.json').read_text())
            if (run['settings'] != SETTINGS or run['fingerprints'] != fingerprints()
                    or run['task'] != info or run['packages'] != packages()):
                raise RuntimeError('Code, prompts, data, settings or packages changed; refusing to resume')
        else:
            run = {'started_at': now(), 'status': 'prepared', 'settings': SETTINGS,
                   'task': info, 'fingerprints': fingerprints(), 'packages': packages(),
                   'python_version': sys.version, 'candidates': {}}
            save(run_dir / 'run.json', run)
        client = None

        def get_client():
            nonlocal client
            if client is None:
                client = client_provider()
            return client

        print(f'Artifacts: {run_dir}', flush=True)
        initial = None
        try:
            run['status'] = 'running'
            for name in NAMES:
                folder = run_dir / name
                prompt = make_prompt(info['problem_description'], name,
                                     (run_dir / 'initial' / 'solution.py').read_text() if initial else '',
                                     feedback_text(initial) if initial else '')
                error = generate(run, run_dir, name, prompt, get_client)
                print(f'{name}: evaluating dev', flush=True)
                if error:
                    result = failed_result(info, 'dev', error)
                    save(folder / 'dev.json', result)
                else:
                    result = evaluator(folder / 'solution.py', 'dev', folder)
                (folder / 'dev_feedback.txt').write_text(feedback_text(result))
                run['candidates'][name]['status'] = 'dev_evaluated'
                save(run_dir / 'run.json', run)
                if name == 'initial':
                    initial = result
                    if error or result['successful'] == 0:
                        run['status'] = 'stopped_invalid_initial'
                        return 2
            # Test data are evaluated only after all four generation attempts completed.
            for name in NAMES:
                folder = run_dir / name
                print(f'{name}: evaluating test', flush=True)
                error = run['candidates'][name]['code_error']
                if error:
                    save(folder / 'test.json', failed_result(info, 'test', error))
                else:
                    evaluator(folder / 'solution.py', 'test', folder)
                run['candidates'][name]['status'] = 'evaluated'
            run['status'] = 'completed'
            return 0
        except KeyboardInterrupt:
            run['status'] = 'interrupted'
            raise
        except Exception:
            run['status'] = 'blocked_or_failed'
            raise
        finally:
            run['updated_at'] = now()
            save(run_dir / 'run.json', run)
            if client is not None:
                client.close()
            write_summary(run_dir, run)


def preflight(output, online=False):
    info = task_info()
    installed = packages()
    counts = {split: sum(map(len, files.values())) for split, files in info['splits'].items()}
    output.mkdir(parents=True, exist_ok=True)
    prompt = make_prompt(info['problem_description'])
    (output / 'initial.txt').write_text(prompt)
    for name in INTERVENTIONS:
        (output / f'{name}.txt').write_text(make_prompt(info['problem_description'], name,
                                                      '<SAME INITIAL CODE>', '<SAME DEV FEEDBACK>'))
    report = {'settings': SETTINGS, 'instances': counts, 'sizes': len(info['splits']['dev']),
              'packages': installed, 'api_key_present': bool(os.environ.get('OPENAI_API_KEY')),
              'api_access_checked': False,
              'maximum_generation_cost_usd': 4 * (MAX_INPUT * INPUT_PRICE + MAX_OUTPUT * OUTPUT_PRICE) / 1e6}
    if online:
        with client_factory() as client:
            client.models.retrieve(MODEL)
            counted = client.responses.input_tokens.count(model=MODEL, input=request_body(prompt)['input'],
                                                         reasoning={'effort': 'medium'})
            report.update(api_access_checked=True, initial_input_tokens=counted.input_tokens)
    save(output / 'check.json', report)
    print(json.dumps(report, indent=2))
    print(f'Prompt previews: {output}. No generation request was sent.')
    return 0


def smoke(output):
    """No model requests: evaluate the identity permutation on all dev/test cases."""
    task_info()
    packages()
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'solution.py'
    source.write_text("def solve(**kwargs):\n    return {'job_sequence': list(range(1, kwargs['n'] + 1))}\n")
    for split in ['dev', 'test']:
        result = evaluate_candidate(source, split, output)
        print(f'{split}: score={result["score"]:.6f}, valid={result["successful"]}/{result["instances"]}')
        if result['failed']:
            raise RuntimeError('Identity solver smoke test failed; inspect worker logs')
    print('Offline evaluation smoke test passed. No API calls.')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('check', help='Validate data/packages and preview prompts; no generation')
    check.add_argument('--online', action='store_true', help='Also check model access and count initial tokens')
    check.add_argument('--output-dir', type=Path, default=ROOT / 'runs' / 'flow_preflight')
    smoke_parser = sub.add_parser('smoke', help='Offline evaluation with a fixed identity solver')
    smoke_parser.add_argument('--output-dir', type=Path, required=True)
    start = sub.add_parser('run', help='Start the paid four-generation experiment')
    start.add_argument('--output-dir', type=Path, required=True)
    resume = sub.add_parser('resume', help='Reuse saved responses; never resend an uncertain request')
    resume.add_argument('run_dir', type=Path)
    report = sub.add_parser('report', help='Rebuild CSV/plots without API calls or evaluations')
    report.add_argument('run_dir', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'check':
            return preflight(args.output_dir.resolve(), args.online)
        if args.command == 'smoke':
            return smoke(args.output_dir.resolve())
        if args.command == 'report':
            folder = args.run_dir.resolve()
            with run_lock(folder):
                write_summary(folder, json.loads((folder / 'run.json').read_text()))
            return 0
        return run_experiment((args.run_dir if args.command == 'resume' else args.output_dir).resolve(),
                              resume=args.command == 'resume')
    except KeyboardInterrupt:
        print('Interrupted. Saved responses will be reused on resume.', file=sys.stderr)
        return 130
    except Exception as exc:
        # Do not print raw SDK error bodies or tracebacks that might contain a key.
        message = str(exc) if isinstance(exc, (RuntimeError, ValueError, FileExistsError)) else type(exc).__name__
        print(f'Experiment stopped: {message}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
