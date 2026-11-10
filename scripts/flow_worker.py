"""Credential-free Flow shop evaluation worker (process isolation, not a sandbox)."""

import argparse
import contextlib
import copy
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import signal
import tempfile
import time


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def load_config(path):
    spec = importlib.util.spec_from_file_location('flow_config', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def child(connection, source, instance, eval_func, seed):
    os.setsid()
    # Native and Python output must not fill a pipe or pollute experiment logs.
    with open(os.devnull, 'w') as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
        try:
            import numpy as np
            random.seed(seed)
            np.random.seed(seed)
            with tempfile.TemporaryDirectory(prefix='flow-candidate-') as work:
                os.chdir(work)
                namespace = {}
                exec(compile(source, 'solution.py', 'exec'), namespace)
                # Only the documented solver inputs; reference bounds stay with the evaluator.
                solution = namespace['solve'](n=instance['n'], m=instance['m'], matrix=copy.deepcopy(instance['matrix']))
                if not isinstance(solution, dict):
                    raise ValueError('solve must return a dictionary')
                score = float(eval_func(**instance, **solution))
                if not math.isfinite(score) or score < 0:
                    raise ValueError('Evaluator returned an invalid score')
                result = {'status': 'ok', 'score': score, 'error': None}
        except BaseException as exc:
            result = {'status': 'error', 'score': 0.0,
                      'error': f'{type(exc).__name__}: {str(exc)[:2000]}'}
        connection.send(result)
        connection.close()


def run_instance(source, instance, eval_func, timeout, seed):
    context = mp.get_context('fork')  # This process was exec'd with a clean environment.
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=child, args=(send, source, instance, eval_func, seed))
    started = time.monotonic()
    process.start()
    send.close()
    try:
        if receive.poll(timeout):
            try:
                result = receive.recv()
            except EOFError:
                result = {'status': 'error', 'score': 0.0, 'error': 'Worker exited without a result'}
        else:
            result = {'status': 'timeout', 'score': 0.0, 'error': f'Timeout ({timeout}s)'}
    finally:
        # Kill the whole candidate process group, including leftover solver subprocesses.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        if process.is_alive():
            process.kill()
        process.join()
        receive.close()
        process.close()
    result['elapsed_seconds'] = time.monotonic() - started
    return result


def aggregate(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row['file'], []).append(row)
    scores = {name: sum(r['score'] for r in group) / len(group) for name, group in groups.items()}
    successful = sum(r['status'] == 'ok' for r in rows)
    return {'score': sum(scores.values()) / len(scores) if scores else 0.0,
            'instances': len(rows), 'successful': successful, 'failed': len(rows) - successful,
            'valid_rate': successful / len(rows) if rows else 0.0, 'by_size': scores, 'rows': rows}


def evaluate(config_path, data_dir, source, split, timeout, seed, checkpoint):
    config = load_config(config_path)
    dev = config.get_dev()
    expected = []
    for filename in sorted(dev):
        instances = config.load_data(str(Path(data_dir) / filename))
        for index, instance in enumerate(instances):
            if (index in dev[filename]) == (split == 'dev'):
                expected.append((filename, index, instance))
    rows = json.loads(checkpoint.read_text()) if checkpoint.exists() else []
    if [(r['file'], r['index']) for r in rows] != [(f, i) for f, i, _ in expected[:len(rows)]]:
        raise ValueError('Checkpoint does not match the fixed evaluation split')
    if len(rows) > len(expected):
        raise ValueError('Too many checkpoint rows')
    for filename, index, instance in expected[len(rows):]:
        case_seed = int.from_bytes(hashlib.sha256(f'{seed}:{filename}:{index}'.encode()).digest()[:4], 'big')
        result = run_instance(source, instance, config.eval_func, timeout, case_seed)
        rows.append({'file': filename, 'index': index, 'n': instance['n'], 'm': instance['m'],
                     'seed': case_seed, **result})
        save(checkpoint, rows)
        print(f'{split}: {len(rows)}/{len(expected)} {filename}[{index}] {result["status"]}', flush=True)
    return aggregate(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--split', choices=['dev', 'test'], required=True)
    parser.add_argument('--timeout', type=float, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Pin this worker and every descendant to one available CPU.
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    result = evaluate(args.config, args.config.parent, args.source.read_text(), args.split,
                      args.timeout, args.seed, args.output.with_suffix('.rows.json'))
    save(args.output, result)


if __name__ == '__main__':
    main()
