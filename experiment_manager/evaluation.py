"""Run only development instances, retaining original normalization indices."""

import argparse
import hashlib
import math
from pathlib import Path
import time

from evaluation import Evaluator, get_data
from evaluation.evaluate import evaluate_instance
from evaluation.utils import FileLock, ParallelRun, average_score, filter_dev
from main import result_counts
from experiment_manager.storage import ROOT, digest, environment, read, save


def describe(task, data_dir, unsplit_policy="all_as_dev"):
    data_dir = Path(data_dir).resolve()
    config = (data_dir / task / "config.py").resolve()
    if not config.is_relative_to(data_dir) or not config.is_file():
        raise ValueError(f"Task config missing: {task}")
    data = get_data(task, src_dir=str(data_dir))
    dev = data.get_dev()
    if dev is None and unsplit_policy != "all_as_dev":
        raise ValueError("This task has no dev split; specify unsplit_policy: all_as_dev")
    sizes, indices, file_hashes = {}, {}, {}
    for name in data.test_cases:
        path = (data_dir / task / name).resolve()
        if not path.is_relative_to(data_dir / task):
            raise ValueError(f"Case path escapes task directory: {name}")
        size = len(data.load_data(str(path)))
        sizes[name] = size
        file_hashes[name] = digest(path)
        # Match CO-Bench filter_dev's special treatment of an empty index list.
        selected = range(size) if dev is None else (dev.get(name) or [0]) if name in dev else []
        selected = set(selected)
        indices[name] = [i for i in range(size) if i in selected]
    if not sum(map(len, indices.values())):
        raise ValueError("No development instances selected")
    return {"task": task, "data_dir": str(data_dir), "sizes": sizes, "dev_indices": indices,
            "split": "all_as_dev" if dev is None else "configured_dev",
            "problem_description": data.problem_description,
            "hashes": {"config.py": digest(config), **file_hashes}}


def signature(task, code_hash, settings):
    return {"solution_sha256": code_hash, "task": task["task"], "hashes": task["hashes"],
            "sizes": task["sizes"], "dev_indices": task["dev_indices"],
            "timeout_seconds": settings["timeout_seconds"], "cpu_num": 1,
            "environment": environment(),
            "runner_hashes": {n: digest(ROOT / n) for n in (
                "evaluation/utils.py", "evaluation/evaluate.py", "evaluation/controller.py",
                "experiment_manager/evaluation.py")},
            "runner": "CO-Bench ParallelRun timeout+1; dev-only; original-index normalization"}


def summarize(results, feedback_length=64):
    if not results:
        raise ValueError("Empty dev results")
    score = float(average_score(results, list(results)))
    if not math.isfinite(score):
        raise ValueError("Non-finite dev score")
    evaluator = object.__new__(Evaluator)
    evaluator.feedback_length = feedback_length
    return {"dev_score": score, "dev_feedback": evaluator.get_feedback(results, score),
            "results": results, **result_counts(results),
            "by_file": {n: average_score({n: v}, [n]) for n, v in results.items()}}


def failure(task, error, feedback_length=64):
    return summarize({name: ([error] * len(indices), None)
                      for name, indices in task["dev_indices"].items() if indices}, feedback_length)


def evaluate(source, task, settings, checkpoint, runner=None):
    current = describe(task["task"], task["data_dir"], task["split"])
    if current != task:
        raise RuntimeError("Task data or split changed")
    data = get_data(task["task"], src_dir=task["data_dir"])
    code_hash = hashlib.sha256(source.encode()).hexdigest()
    sig = signature(task, code_hash, settings)
    checkpoint = Path(checkpoint)
    state = read(checkpoint) if checkpoint.exists() else {"signature": sig, "rows": []}
    if state["signature"] != sig:
        raise RuntimeError("Evaluation checkpoint signature changed")
    keys = [(name, i) for name, indices in task["dev_indices"].items() for i in indices]
    rows = state["rows"]
    if len(rows) > len(keys) or [(r["file"], r["index"]) for r in rows] != keys[:len(rows)]:
        raise RuntimeError("Evaluation checkpoint order changed")
    runtime = runner or ParallelRun(evaluate_instance).run_instance_with_timeout
    with FileLock():
        loaded = {}
        for name, index in keys[len(rows):]:
            if name not in loaded:
                loaded[name] = data.load_data(str(Path(data.src_dir) / data.task / name))
            started = time.monotonic()
            raw = runtime(loaded[name][index], source, data.config_path, settings["timeout_seconds"])
            rows.append({"file": name, "index": index, "raw_score": raw,
                         "elapsed_seconds": time.monotonic() - started})
            save(checkpoint, state)
            print(f"dev {len(rows)}/{len(keys)} {name}[{index}]: {raw}", flush=True)
    # Keep unexecuted test positions as strings during normalization so reference
    # arrays are indexed exactly as in the original evaluator. Never report them.
    raw = {name: (["Not evaluated (outside dev)"] * size, None) for name, size in task["sizes"].items()}
    for row in rows:
        raw[row["file"]][0][row["index"]] = row["raw_score"]
    normalized = data.norm_score(raw)
    dev = {n: [i for i in idx] for n, idx in task["dev_indices"].items() if idx}
    result = summarize(filter_dev(normalized, dev), settings["feedback_length"])
    result["signature"] = sig
    result["evaluation_seconds"] = sum(r["elapsed_seconds"] for r in rows)
    return result


def worker():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("task", type=Path)
    parser.add_argument("settings", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.source.read_text(), read(args.task), read(args.settings),
                      args.output.with_suffix(".rows.json"))
    save(args.output, result)


if __name__ == "__main__":
    worker()
