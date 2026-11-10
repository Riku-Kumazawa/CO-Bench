"""Evaluate TSP development cases with the original CO-Bench instance runner."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from evaluation import Evaluator, get_data
from evaluation.evaluate import evaluate_instance
from evaluation.utils import FileLock, ParallelRun, average_score
from main import result_counts


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def summarize(data, results):
    score = average_score(results, list(results))
    evaluator = object.__new__(Evaluator)
    evaluator.feedback_length = 64
    return {"dev_score": score, "dev_feedback": evaluator.get_feedback(results, score),
            "results": results, **result_counts(results),
            "by_size": {name: average_score({name: value}, [name]) for name, value in results.items()}}


def evaluate(source, data, timeout, checkpoint, runner=None):
    # TSP has no configured split. Keep exactly the original 272 inputs.
    if data.get_dev() is not None:
        raise ValueError("This continuation requires the original, unsplit TSP data")
    paths = {name: Path(data.src_dir) / data.task / name for name in data.test_cases}
    signature = {"code": hashlib.sha256(source.encode()).hexdigest(), "timeout": timeout,
                 "data": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()},
                 "config": hashlib.sha256(Path(data.config_path).read_bytes()).hexdigest()}
    state = json.loads(checkpoint.read_text()) if checkpoint.exists() else {"signature": signature, "rows": []}
    if state["signature"] != signature:
        raise RuntimeError("Evaluation checkpoint differs from code, timeout or data")
    expected = [(name, i, instance) for name, path in paths.items()
                for i, instance in enumerate(data.load_data(str(path)))]
    rows = state["rows"]
    if (len(rows) > len(expected) or [(r["file"], r["index"]) for r in rows]
            != [(name, i) for name, i, _ in expected[:len(rows)]]):
        raise RuntimeError("Evaluation checkpoint does not match the case order")
    runtime = runner or ParallelRun(evaluate_instance).run_instance_with_timeout
    with FileLock():
        for name, index, instance in expected[len(rows):]:
            started = time.monotonic()
            # Preserve original execution, inputs, 10s timeout + original 1s grace.
            raw_score = runtime(instance, source, data.config_path, timeout)
            rows.append({"file": name, "index": index, "raw_score": raw_score,
                         "elapsed_seconds": time.monotonic() - started})
            save(checkpoint, state)
            print(f"dev {len(rows)}/{len(expected)} {name}[{index}]: {raw_score}", flush=True)
    raw = {}
    for name in paths:
        raw[name] = ([r["raw_score"] for r in rows if r["file"] == name], None)
    return summarize(data, data.norm_score(raw))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = get_data("Travelling salesman problem", src_dir=str(args.data_dir))
    result = evaluate(args.source.read_text(), data, args.timeout, args.output.with_suffix(".rows.json"))
    save(args.output, result)


if __name__ == "__main__":
    main()
