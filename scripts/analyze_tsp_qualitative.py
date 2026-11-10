"""Audit saved TSP candidates; write separate diagnostics without changing scores."""

import ast
from collections import Counter
import contextlib
import csv
import faulthandler
import hashlib
import importlib.util
from importlib.metadata import version
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/tsp_qwen_latest_10_submissions"
OUT = RUN / "qualitative_analysis"
TASK = ROOT / "data/Travelling salesman problem"
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
for key in THREAD_VARS:
    os.environ[key] = "1"


def config():
    spec = importlib.util.spec_from_file_location("tsp_config_audit", TASK / "config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def namespace(name):
    path = RUN / name / "solution.py"
    scope = {"__name__": "candidate_audit", "__file__": str(path)}
    exec(compile(path.read_text(), str(path), "exec"), scope)
    return scope


def replay_one(name, filename, index):
    faulthandler.enable()
    faulthandler.dump_traceback_later(9)
    started = time.monotonic()
    solution = None
    result = {"candidate": name, "file": filename, "index": index}
    try:
        c = config()
        instance = c.load_data(str(TASK / filename))[index]
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
            scope = namespace(name)
            solution = scope["solve"](**instance)
            score = c.eval_func(**instance, **solution)
        result.update(status="ok", raw_objective=float(score))
    except BaseException as exc:
        frames = []
        tb = exc.__traceback__
        while tb:
            frame = tb.tb_frame
            if str(RUN) in frame.f_code.co_filename:
                local = frame.f_locals
                values = {key: str(local[key]) for key in
                          ["n", "i", "j", "a", "k", "c", "p", "q", "pos", "start", "end", "seg_len"]
                          if key in local and not hasattr(local[key], "shape")}
                tour = local.get("tour")
                if tour is not None:
                    with contextlib.suppress(Exception):
                        values.update(tour_length=len(tour), tour_unique=len(set(tour)))
                frames.append({"function": frame.f_code.co_name, "line": tb.tb_lineno, "locals": values})
            tb = tb.tb_next
        result.update(status="error", error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc(), candidate_frames=frames)
    finally:
        faulthandler.cancel_dump_traceback_later()
    if isinstance(solution, dict) and "tour" in solution:
        tour = solution["tour"]
        result.update(returned_tour_length=len(tour), returned_tour_unique=len(set(tour)))
    result["diagnostic_elapsed_seconds"] = time.monotonic() - started
    print(json.dumps(result, ensure_ascii=False))


def replay_failures(manifest):
    jobs = []
    for name in manifest["candidates"]:
        result = json.loads((RUN / name / "dev.json").read_text())
        for filename in ["tsp500_test_concorde.txt", "tsp1000_test_concorde.txt", "tsp10000_test_concorde.txt"]:
            values = result["results"][filename][0]
            failed = next((i for i, value in enumerate(values) if isinstance(value, str)), None)
            if failed is not None:
                jobs.append((name, filename, failed, values[failed]))
                break
    # neutral_03 has a separate defect in the large-instance distance calculation.
    jobs.append(("neutral_03", "tsp10000_test_concorde.txt", 0,
                 json.loads((RUN / "neutral_03/dev.json").read_text())["results"]["tsp10000_test_concorde.txt"][0][0]))
    env = {key: os.environ[key] for key in ["PATH", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "TMPDIR"] if key in os.environ}
    env.update({key: "1" for key in THREAD_VARS})
    results = []
    for name, filename, index, original in jobs:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--one", name, filename, str(index)],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
                                   cwd=ROOT, start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=12)
            item = json.loads(stdout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            item = {"candidate": name, "file": filename, "index": index, "status": "diagnostic_timeout_12s"}
        item["original_error"] = original
        if stderr:
            item["diagnostic_stack_or_stderr"] = stderr
        results.append(item)
        print(name, filename, index, item["status"], item.get("error", ""), flush=True)
    return results


def small_probes():
    import numpy as np
    from ortools.constraint_solver import pywrapcp
    probes = []
    for name, function, args in [
        ("support_04", "_reverse_cyclic", (np.arange(8), 5, 1, 8)),
        ("support_04", "_perform_single_move", (np.arange(8), 6, 1, 8)),
        ("support_06", "_perform_2_move", (np.arange(10), 6, 1, 10)),
    ]:
        try:
            value = namespace(name)[function](*args)
            probes.append({"candidate": name, "function": function, "returned": str(value)})
        except Exception as exc:
            probes.append({"candidate": name, "function": function, "error": str(exc), "traceback": traceback.format_exc()})
    api = {"ortools_version": version("ortools"),
           "RoutingModel.DefaultSearchParameters_exists": hasattr(pywrapcp.RoutingModel, "DefaultSearchParameters"),
           "pywrapcp.DefaultRoutingSearchParameters_exists": hasattr(pywrapcp, "DefaultRoutingSearchParameters")}
    params = pywrapcp.DefaultRoutingSearchParameters()
    api["search_log_field_exists"] = "search_log" in params.DESCRIPTOR.fields_by_name
    coords = np.random.default_rng(7).random((20, 2))
    for name in ["support_08", "support_09", "support_10"]:
        scope = namespace(name)
        caught = []
        def trace(frame, event, arg):
            if event == "exception" and frame.f_code.co_filename == str(RUN / name / "solution.py"):
                caught.append({"function": frame.f_code.co_name, "line": frame.f_lineno,
                               "error_type": arg[0].__name__, "error": str(arg[1])})
            return trace
        old_trace = sys.gettrace()
        sys.settrace(trace)
        try:
            value = scope["_solve_ortools"](coords, 20, time.time() + 5)
            returned = "None" if value is None else "tour"
        except Exception as exc:
            returned = f"{type(exc).__name__}: {exc}"
        finally:
            sys.settrace(old_trace)
        probes.append({"candidate": name, "function": "_solve_ortools", "returned": returned, "exceptions": caught})
    # Isolate one unchanged Or-opt pass with a deterministic clock.
    scope = namespace("support_02")
    class OnePassClock:
        calls = 0
        def time(self):
            self.calls += 1
            return 0 if self.calls <= 2 else 2
    scope["time"] = OnePassClock()
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    tour = np.argsort(np.arctan2(coords[:, 1] - coords[:, 1].mean(), coords[:, 0] - coords[:, 0].mean()))
    before = float(dist[tour, np.roll(tour, -1)].sum())
    after_tour = scope["_or_opt_single"](tour.copy(), dist, len(tour), 1)
    after = float(dist[after_tour, np.roll(after_tour, -1)].sum())
    probes.append({"candidate": "support_02", "function": "_or_opt_single", "fixture": "20 random points, seed 7, one pass",
                   "cost_before": before, "cost_after": after, "valid_permutation": sorted(after_tour.tolist()) == list(range(20))})
    for name, function in [("switch_06", "two_opt"), ("switch_10", "_two_opt_fast")]:
        scope = namespace(name)
        class BoundedClock:
            calls = 0
            def time(self):
                self.calls += 1
                return 0 if self.calls <= 20 else 2
        scope["time"] = BoundedClock()
        snapshots = []
        def inspect_return(frame, event, arg):
            if event == "return" and frame.f_code.co_name == function:
                local = frame.f_locals
                tour_at_return = local["tour"]
                cache = local.get("ec", local.get("e"))
                actual = dist[tour_at_return, np.roll(tour_at_return, -1)]
                snapshot = {"edge_cache_mismatches": int(np.count_nonzero(np.abs(cache - actual) > 1e-10))}
                if "next_tour" in local:
                    snapshot["next_tour_mismatches"] = int(np.count_nonzero(local["next_tour"] != np.roll(tour_at_return, -1)))
                snapshots.append(snapshot)
            return inspect_return
        old_trace = sys.gettrace()
        sys.settrace(inspect_return)
        try:
            scope[function](np.random.default_rng(8).permutation(20), dist, 20, 1)
        finally:
            sys.settrace(old_trace)
        probes.append({"candidate": name, "function": function, "fixture": "20 points, bounded deterministic clock",
                       "return_cache_checks": snapshots})
    return {"api": api, "probes": probes}


def sweep_comparison():
    import numpy as np
    c = config()
    results = {}
    for filename in sorted(p.name for p in TASK.glob("*.txt")):
        costs = []
        for instance in c.load_data(str(TASK / filename)):
            coords = np.asarray(instance["nodes"])
            center = coords.mean(axis=0)
            tour = np.argsort(np.arctan2(coords[:, 1] - center[1], coords[:, 0] - center[0]))
            costs.append(float(np.linalg.norm(coords[tour] - coords[np.roll(tour, -1)], axis=1).sum()))
        scores = c.norm_score({filename: (costs, None)})[filename][0]
        for name in ["support_08", "support_09", "support_10"]:
            actual = json.loads((RUN / name / "dev.json").read_text())["results"][filename][0]
            diffs = [abs(a - b) for a, b in zip(actual, scores)]
            results.setdefault(name, {})[filename] = {"cases": len(scores), "sweep_score_mean": float(np.mean(scores)),
                "original_score_mean": float(np.mean(actual)), "matching_sweep_within_1e_10": sum(d < 1e-10 for d in diffs)}
    return results


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        replay_one(sys.argv[2], sys.argv[3], int(sys.argv[4]))
        return
    manifest = json.loads((RUN / "run.json").read_text())
    OUT.mkdir(exist_ok=True)
    rows = []
    for name in ["initial", *manifest["candidates"]]:
        result = json.loads((RUN / name / "dev.json").read_text())
        source = RUN / name / "solution.py"
        text = source.read_text() if source.exists() else ""
        errors = Counter(value for values, _ in result["results"].values() for value in values if isinstance(value, str))
        rows.append({"candidate": name, "dev_score": result["dev_score"], "failed": result["failed"],
                     "code_lines": len(text.splitlines()), "code_sha256": hashlib.sha256(text.encode()).hexdigest(),
                     "errors": dict(errors), "by_size": result["by_size"],
                     "plan": (RUN / name / "response.txt").read_text().split("```")[0].strip()})
    results = {"scope": "Saved scores plus limited diagnostic replays; original experiment outputs are unchanged.",
               "candidates": rows, "replays": replay_failures(manifest),
               "small_probes": small_probes(), "sweep_comparison": sweep_comparison()}
    (OUT / "evidence.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(OUT / "evidence.json", flush=True)


if __name__ == "__main__":
    main()
