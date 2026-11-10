"""Continue a completed Flow shop pilot to N submissions per feedback condition."""

import argparse
import copy
import csv
import json
import math
from pathlib import Path
import shutil
import sys
import time

import flow_experiment as base

DEFAULT_SUBMISSIONS = 10
DEFAULT_BUDGET = 10.0


def plan_for(submissions):
    if submissions < 2:
        raise ValueError('At least 2 submissions per condition are required')
    plan = [{'name': 'initial', 'condition': 'initial', 'submission': 1}]
    for submission in range(2, submissions + 1):
        for condition in base.INTERVENTIONS:
            name = condition if submission == 2 else f'{condition}_{submission:02d}'
            plan.append({'name': name, 'condition': condition, 'submission': submission})
    return plan


def fingerprints():
    return {**base.fingerprints(), 'flow_continue.py': base.digest(Path(__file__))}


def load_result(folder, split):
    path = folder / f'{split}.json'
    return json.loads(path.read_text()) if path.exists() else None


def verify_results(folder, info, split):
    result = load_result(folder, split)
    if result is None:
        raise RuntimeError(f'Missing imported {split} result: {folder.name}')
    expected = {(name, i) for name, indices in info['splits'][split].items() for i in indices}
    keys = [(r['file'], r['index']) for r in result['rows']]
    if len(keys) != len(expected) or set(keys) != expected:
        raise RuntimeError(f'Imported {folder.name}/{split} does not match the full split')
    if base.aggregate(result['rows']) != result:
        raise RuntimeError(f'Imported {folder.name}/{split} aggregation is inconsistent')


def prepare(source, destination, submissions=DEFAULT_SUBMISSIONS, budget=DEFAULT_BUDGET):
    """Read-only import of the source, with no model or evaluator calls."""
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError('Budget must be a finite positive USD amount')
    plan = plan_for(submissions)
    if destination == source or source in destination.parents:
        raise ValueError('Choose an output directory outside the source experiment')
    if destination.exists():
        raise FileExistsError(f'Output directory already exists: {destination}')
    with base.run_lock(source):
        original = json.loads((source / 'run.json').read_text())
        if original['status'] != 'completed' or set(original['candidates']) != set(base.NAMES):
            raise RuntimeError('Source must be a completed four-generation pilot')
        info = base.task_info()
        if (original['settings'] != base.SETTINGS or original['fingerprints'] != base.fingerprints()
                or original['task'] != info or original['packages'] != base.packages()):
            raise RuntimeError('Source settings, code, data or packages differ from this installation')
        for name in base.NAMES:
            folder = source / name
            response = json.loads((folder / 'response.json').read_text())
            _, code, error = base.extract_code(response)
            if error != original['candidates'][name]['code_error']:
                raise RuntimeError(f'Imported code status mismatch: {name}')
            if code is not None and (folder / 'solution.py').read_text() != code:
                raise RuntimeError(f'Imported solution was modified: {name}')
            for split in ['dev', 'test']:
                verify_results(folder, info, split)
        if load_result(source / 'initial', 'dev')['successful'] == 0:
            raise RuntimeError('Initial solver has no successful dev cases')
        destination.mkdir(parents=True)
        for name in base.NAMES:
            shutil.copytree(source / name, destination / name)
        base.save(destination / 'source_run.json', original)
    imported_hashes = {str(path.relative_to(destination)): base.digest(path)
                       for name in base.NAMES for path in (destination / name).rglob('*') if path.is_file()}
    imported_hashes['source_run.json'] = base.digest(destination / 'source_run.json')
    cost = base.total_cost(original)
    new_count = len(plan) - 4
    upper = cost + new_count * (base.MAX_INPUT * base.INPUT_PRICE + base.MAX_OUTPUT * base.OUTPUT_PRICE) / 1e6
    run = {'schema_version': 1, 'status': 'prepared', 'started_at': base.now(),
           'source_directory': str(source), 'source_manifest_sha256': base.digest(source / 'run.json'),
           'settings': {**copy.deepcopy(base.SETTINGS), 'budget_usd': budget,
                        'submissions_per_condition': submissions, 'total_generations': len(plan),
                        'candidate_order': [item['name'] for item in plan],
                        'selection': 'strict dev improvement; ties retain incumbent',
                        'feedback_schedule': 'same directional message on every improvement',
                        'price_verified_date': '2026-11-09'},
           'plan': plan, 'task': info, 'fingerprints': fingerprints(), 'packages': base.packages(),
           'python_version': sys.version, 'imported_names': list(base.NAMES),
           'imported_hashes': imported_hashes, 'candidates': copy.deepcopy(original['candidates']),
           'maximum_total_cost_usd': upper,
           'expected_additional_cost_from_pilot_usd': new_count * cost / 4}
    base.save(destination / 'run.json', run)
    report(destination, run)
    print(f'Prepared: {destination}\nSubmissions per condition: {submissions}; total generations: {len(plan)}')
    print(f'Imported: 4; new generations: {new_count}; no API or evaluation calls.')
    print(f'Total budget: ${budget:.2f}; maximum total at token caps: ${upper:.6f}')
    return run


def verify_run(folder, run):
    expected_settings = {**base.SETTINGS, 'budget_usd': run['settings']['budget_usd'],
                         'submissions_per_condition': run['settings']['submissions_per_condition'],
                         'total_generations': run['settings']['total_generations'],
                         'candidate_order': [item['name'] for item in run['plan']],
                         'selection': 'strict dev improvement; ties retain incumbent',
                         'feedback_schedule': 'same directional message on every improvement',
                         'price_verified_date': '2026-11-09'}
    if (run['schema_version'] != 1 or run['settings'] != expected_settings
            or run['plan'] != plan_for(run['settings']['submissions_per_condition'])
            or run['fingerprints'] != fingerprints() or run['packages'] != base.packages()
            or run['task'] != base.task_info() or run['python_version'] != sys.version):
        raise RuntimeError('Continuation code, prompts, settings, data or environment changed; refusing to run')
    if run['settings']['total_generations'] != len(run['plan']):
        raise RuntimeError('Generation limit differs from the plan')
    if not set(run['candidates']) <= {p['name'] for p in run['plan']}:
        raise RuntimeError('Unknown candidates in the continuation manifest')
    for name, expected in run['imported_hashes'].items():
        if not (folder / name).is_file() or base.digest(folder / name) != expected:
            raise RuntimeError(f'Imported artifact changed: {name}')


def incumbent_after(folder, incumbent, candidate):
    old, new = load_result(folder / incumbent, 'dev'), load_result(folder / candidate, 'dev')
    return candidate if new['score'] > old['score'] else incumbent


def generate(folder, run, item, parent, get_client):
    """One durable request slot per planned candidate; uncertain slots are never resent."""
    name, condition = item['name'], item['condition']
    target = folder / name
    target.mkdir(exist_ok=True)
    prompt = base.make_prompt(run['task']['problem_description'], condition,
                              (folder / parent / 'solution.py').read_text(),
                              base.feedback_text(load_result(folder / parent, 'dev')))
    body = base.request_body(prompt)
    record = run['candidates'].get(name)
    if record is not None:
        if (record['parent'] != parent or record['condition'] != condition
                or record['submission'] != item['submission']
                or json.loads((target / 'request.json').read_text()) != body):
            raise RuntimeError(f'Saved parent or request differs: {name}')
        if not (target / 'response.json').exists():
            raise RuntimeError(f'{name}: request may have been sent, but no response was saved. No automatic resend.')
        response = json.loads((target / 'response.json').read_text())
    else:
        (target / 'prompt.txt').write_text(prompt)
        base.save(target / 'request.json', body)
        client = get_client()
        count = client.responses.input_tokens.count(model=base.MODEL, input=body['input'], reasoning=body['reasoning'])
        tokens = count.input_tokens
        if tokens > base.MAX_INPUT:
            raise RuntimeError(f'{name}: input exceeds {base.MAX_INPUT} tokens; no generation sent')
        reserved = (tokens * base.INPUT_PRICE + base.MAX_OUTPUT * base.OUTPUT_PRICE) / 1e6
        if base.total_cost(run) + reserved > run['settings']['budget_usd']:
            raise RuntimeError('Remaining budget cannot cover maximum output; no generation sent')
        record = {'status': 'request_started', 'parent': parent, 'condition': condition,
                  'submission': item['submission'], 'started_at': base.now(),
                  'input_tokens_counted': tokens, 'reserved_usd': reserved}
        run['candidates'][name] = record
        base.save(folder / 'run.json', run)
        print(f'{name}: generation {len(run["candidates"])}/{len(run["plan"])}; parent={parent}; max=${reserved:.4f}', flush=True)
        started = time.monotonic()
        try:
            response = client.responses.create(**body).model_dump(mode='json')
            base.save(target / 'response.json', response)
        except Exception as exc:
            record.update(status='api_error_or_unknown', error_type=type(exc).__name__,
                          http_status=getattr(exc, 'status_code', None))
            base.save(folder / 'run.json', run)
            raise RuntimeError(f'{name}: API request failed ({type(exc).__name__}); no retry') from None
        record['generation_seconds'] = time.monotonic() - started
    record['cost_usd'], record['cost_basis'] = base.usage_cost(response, record['reserved_usd'])
    record.update(status='response_saved', usage=response.get('usage'), response_id=response.get('id'),
                  response_model=response.get('model'), finished_at=base.now())
    text, code, error = base.extract_code(response)
    (target / 'response.txt').write_text(text)
    if code is not None:
        source = target / 'solution.py'
        if source.exists() and source.read_text() != code:
            raise RuntimeError(f'Saved solution was modified: {name}')
        source.write_text(code)
    record['code_error'] = error
    base.save(folder / 'run.json', run)
    return error


def evaluate(folder, name, split, run, evaluator):
    target = folder / name
    existing = load_result(target, split)
    if existing is not None:
        return existing
    error = run['candidates'][name]['code_error']
    if error:
        result = base.failed_result(run['task'], split, error)
        base.save(target / f'{split}.json', result)
    else:
        print(f'{name}: evaluating {split}', flush=True)
        result = evaluator(target / 'solution.py', split, target)
    if split == 'dev':
        (target / 'dev_feedback.txt').write_text(base.feedback_text(result))
    return result


def run_continuation(folder, client_provider=base.client_factory, evaluator=base.evaluate_candidate):
    with base.run_lock(folder):
        run = json.loads((folder / 'run.json').read_text())
        verify_run(folder, run)
        client = None

        def get_client():
            nonlocal client
            if client is None:
                client = client_provider()
            return client

        incumbents = {name: incumbent_after(folder, 'initial', name) for name in base.INTERVENTIONS}
        try:
            run['status'] = 'running'
            run.pop('last_error', None)
            base.save(folder / 'run.json', run)
            for item in run['plan'][4:]:
                name, condition = item['name'], item['condition']
                parent = incumbents[condition]
                generate(folder, run, item, parent, get_client)
                evaluate(folder, name, 'dev', run, evaluator)
                incumbents[condition] = incumbent_after(folder, parent, name)
                run['candidates'][name].update(status='dev_evaluated', retained=incumbents[condition])
                base.save(folder / 'run.json', run)
            # Imported test results are kept; no new test evaluations until all new generation is done.
            for item in run['plan'][4:]:
                evaluate(folder, item['name'], 'test', run, evaluator)
                run['candidates'][item['name']]['status'] = 'evaluated'
                base.save(folder / 'run.json', run)
            run['status'] = 'completed'
            return 0
        except KeyboardInterrupt:
            run.update(status='interrupted', last_error={'type': 'KeyboardInterrupt'})
            raise
        except Exception as exc:
            run.update(status='blocked_or_failed', last_error={'type': type(exc).__name__})
            if isinstance(exc, RuntimeError):
                run['last_error']['message'] = str(exc)
            raise
        finally:
            run['updated_at'] = base.now()
            base.save(folder / 'run.json', run)
            if client is not None:
                client.close()
            report(folder, run)


def trajectories(folder, run):
    lines = {condition: [] for condition in base.INTERVENTIONS}
    results = {item['name']: {split: result for split in ['dev', 'test']
                             if (result := load_result(folder / item['name'], split)) is not None}
               for item in run['plan']}
    if 'dev' not in results['initial']:
        return results, lines
    for condition in lines:
        selected = 'initial'
        lines[condition].append({'submission': 1, 'candidate': 'initial', 'parent': None, 'retained': selected})
        for item in run['plan']:
            if item['condition'] != condition or 'dev' not in results[item['name']]:
                continue
            candidate = item['name']
            parent = selected
            if results[candidate]['dev']['score'] > results[selected]['dev']['score']:
                selected = candidate
            lines[condition].append({'submission': item['submission'], 'candidate': candidate,
                                     'parent': parent, 'retained': selected})
    return results, lines


def report(folder, run):
    results, lines = trajectories(folder, run)
    completed = {c: len(points) for c, points in lines.items()}
    summary = {'status': run['status'], 'target_submissions_per_condition': run['settings']['submissions_per_condition'],
               'target_total_generations': len(run['plan']), 'generation_requests_attempted': len(run['candidates']),
               'imported_generations': 4, 'new_generation_requests_attempted': len(run['candidates']) - 4,
               'completed_submissions': completed, 'cost_usd_estimate_or_reservation': base.total_cost(run),
               'additional_cost_usd_estimate_or_reservation': base.total_cost(run) - sum(run['candidates'][n]['cost_usd'] for n in base.NAMES),
               'selection_by_dev': {c: points[-1]['retained'] for c, points in lines.items() if points},
               'trajectories': lines,
               'candidates': {n: {s: {k: v for k, v in r.items() if k != 'rows'} for s, r in splits.items()}
                              for n, splits in results.items() if splits}}
    base.save(folder / 'summary.json', summary)
    records = []
    sizes = []
    for condition, points in lines.items():
        for point in points:
            name, kept = point['candidate'], point['retained']
            for split in ['dev', 'test']:
                if split not in results[name]:
                    continue
                candidate, selected = results[name][split], results[kept].get(split)
                records.append({'condition': condition, 'submission': point['submission'], 'candidate': name,
                                'parent': point['parent'], 'split': split, 'score': candidate['score'],
                                'valid_rate': candidate['valid_rate'], 'failed': candidate['failed'],
                                'retained': kept, 'retained_score': selected['score'] if selected else ''})
                for size, score in candidate['by_size'].items():
                    sizes.append({'condition': condition, 'submission': point['submission'],
                                  'candidate': name, 'split': split, 'size': size, 'score': score})
    costs = [{'candidate': name, 'imported': name in base.NAMES,
              'cost_usd': record.get('cost_usd', record.get('reserved_usd')),
              'generation_seconds': record.get('generation_seconds'),
              'input_tokens': (record.get('usage') or {}).get('input_tokens'),
              'output_tokens': (record.get('usage') or {}).get('output_tokens')}
             for name, record in run['candidates'].items()]
    for filename, rows in [('scores.csv', records), ('size_scores.csv', sizes), ('costs.csv', costs)]:
        if rows:
            with (folder / filename).open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    plot(folder, results, lines, run['settings']['submissions_per_condition'])
    return summary


def plot(folder, results, lines, target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'support': '#0072B2', 'neutral': '#009E73', 'switch': '#D55E00'}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), squeeze=False)
    for row, split in enumerate(['dev', 'test']):
        for col, title in enumerate(['Submitted candidate', 'Retained by dev score']):
            ax = axes[row][col]
            ax.set(title=f'{split}: {title}', xlabel='Submission', ylabel='Normalized score',
                   xticks=range(1, target + 1), xlim=(.7, target + .3))
            ax.grid(alpha=.25)
            for condition, points in lines.items():
                selected = [(p['submission'], results[p['candidate'] if col == 0 else p['retained']].get(split)) for p in points]
                # Missing intermediate test results remain gaps, rather than connecting across them.
                ax.plot([x for x, _ in selected], [r['score'] if r else float('nan') for _, r in selected],
                        'o-', label=condition, color=colors[condition], alpha=.8)
            ax.legend()
    fig.suptitle(f'Flow shop: one shared initial solver, {target} submissions per feedback condition')
    fig.tight_layout()
    fig.savefig(folder / 'scores.png', dpi=180)
    fig.savefig(folder / 'scores.pdf')
    plt.close(fig)


def check(folder):
    with base.run_lock(folder):
        run = json.loads((folder / 'run.json').read_text())
        verify_run(folder, run)
        summary = report(folder, run)
    print(json.dumps({key: summary[key] for key in ['status', 'target_submissions_per_condition',
                     'target_total_generations', 'generation_requests_attempted', 'completed_submissions',
                     'selection_by_dev', 'cost_usd_estimate_or_reservation']}, indent=2))
    print(f'Budget: ${run["settings"]["budget_usd"]:.2f}; maximum total: ${run["maximum_total_cost_usd"]:.6f}')
    print('Offline check passed. No API or evaluator calls.')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare', help='Import a completed four-generation pilot without API calls')
    prep.add_argument('--source', type=Path, required=True)
    prep.add_argument('--output-dir', type=Path, required=True)
    prep.add_argument('--submissions', type=int, default=DEFAULT_SUBMISSIONS)
    prep.add_argument('--budget-usd', type=float, default=DEFAULT_BUDGET)
    for command in ['check', 'run', 'resume', 'report']:
        item = sub.add_parser(command)
        item.add_argument('run_dir', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare':
            prepare(args.source.resolve(), args.output_dir.resolve(), args.submissions, args.budget_usd)
            return 0
        if args.command == 'check':
            return check(args.run_dir.resolve())
        if args.command == 'report':
            with base.run_lock(args.run_dir):
                report(args.run_dir, json.loads((args.run_dir / 'run.json').read_text()))
            return 0
        return run_continuation(args.run_dir.resolve())
    except KeyboardInterrupt:
        print('Interrupted. Saved responses will be reused; uncertain requests will not be resent.', file=sys.stderr)
        return 130
    except Exception as exc:
        message = str(exc) if isinstance(exc, (RuntimeError, ValueError, FileExistsError)) else type(exc).__name__
        print(f'Continuation stopped: {message}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
