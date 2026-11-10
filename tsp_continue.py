"""Continue one Qwen TSP solver using the latest submission in each condition."""

import argparse
import contextlib
import csv
import fcntl
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import time

import psutil

import main as single
from agents.utils import extract_code_blocks
from evaluation import get_data
from scripts.tsp_worker import save, summarize

ROOT = Path(__file__).resolve().parent
TASK = "Travelling salesman problem"
INTERVENTIONS = {
    "support": "I suggest continuing to refine the current algorithmic approach.",
    "neutral": "",
    "switch": "I suggest exploring a different algorithmic approach.",
}
SOURCE = ROOT / "runs" / "20261106-230111-b22819f8"
OUTPUT = ROOT / "runs" / "tsp_qwen_latest_10_submissions"
TEMPLATE = ROOT / "prompts" / "flow_refine.txt"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plan_for(submissions):
    if submissions < 2:
        raise ValueError("At least two submissions per condition are required")
    return [{"name": f"{condition}_{number:02d}", "condition": condition, "submission": number}
            for number in range(2, submissions + 1) for condition in INTERVENTIONS]


def parent_for(item):
    """Parent selection never depends on scores or whether a submission succeeded."""
    return "initial" if item["submission"] == 2 else f"{item['condition']}_{item['submission'] - 1:02d}"


def fingerprints():
    names = ["tsp_continue.py", "scripts/tsp_worker.py", "main.py", "agents/utils.py",
             "evaluation/controller.py", "evaluation/evaluate.py", "evaluation/utils.py",
             "prompts/flow_refine.txt", "scripts/serve_qwen.sh", "uv.lock"]
    return {name: digest(ROOT / name) for name in names}


def environment():
    return {"python": sys.version, "packages": {name: version(name) for name in single.PACKAGES}}


def task_info(data_dir):
    data = get_data(TASK, src_dir=str(data_dir))
    if data.get_dev() is not None:
        raise ValueError("Keep the original TSP inputs; no new dev/test split is configured")
    files = {name: Path(data.src_dir) / TASK / name for name in data.test_cases}
    return {"problem_description": data.problem_description,
            "data_dir": str(data_dir), "split": "original unsplit inputs, dev only",
            "sizes": {name: len(data.load_data(str(path))) for name, path in files.items()},
            "hashes": {"config.py": digest(data.config_path), **{name: digest(p) for name, p in files.items()}}}


@contextlib.contextmanager
def run_lock(folder):
    with (folder / ".lock").open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("This experiment is already running") from None
        yield


def settings_from(source):
    old = json.loads((source / "run.json").read_text())
    request = json.loads((source / "request.json").read_text())
    if (old["task"] != TASK or old["model"] != single.MODEL
            or old["status"] not in ("completed", "completed_with_errors")):
        raise ValueError("Source must be a completed, single-candidate Qwen TSP run")
    expected = {"max_tokens": 32768, "temperature": 1.0, "top_p": .95, "presence_penalty": 0.0,
                "seed": 0, "timeout": 7200, "num_retries": 0, "max_retries": 0,
                "drop_params": False, "extra_body": {"top_k": 20, "min_p": 0.0,
                    "repetition_penalty": 1.0, "chat_template_kwargs": {
                        "enable_thinking": True, "reasoning_effort": "medium"}}}
    if ({key: request.get(key) for key in expected} != expected
            or old["timeout_seconds_per_instance"] != 10 or old["cpu_num"] != 1):
        raise ValueError("Source differs from the agreed generation or evaluation settings")
    return {"model": old["model"], "api_base": old["api_base"], "request_parameters": expected,
            "thinking": True, "reasoning_effort": "medium", "timeout": 10, "cpu_num": 1,
            "seed": 0, "max_tokens": 32768, "generation_timeout": 7200,
            "server_wait_timeout": 600, "evaluation_split": "dev only; original full TSP set",
            "selection": "latest submission regardless of dev score or failure",
            "failure_feedback": "immediate previous code or final-answer text, plus its dev errors; no fallback",
            "order": list(INTERVENTIONS), "enforce_eager": False,
            "checkpoint_revision": old["local_checkpoint_revision"]}


def prepare(source=SOURCE, output=OUTPUT, submissions=10):
    source, output = source.resolve(), output.resolve()
    plan = plan_for(submissions)
    settings = settings_from(source)
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    data_dir = ROOT / "data"
    info = task_info(data_dir)
    old = json.loads((source / "feedback.json").read_text())
    result = summarize(None, old["results"])
    if (set(old["results"]) != set(info["sizes"])
            or any(len(old["results"][name][0]) != count for name, count in info["sizes"].items())
            or not math.isclose(result["dev_score"], old["dev_score"], rel_tol=0, abs_tol=1e-12)):
        raise ValueError("Source dev results do not match the current evaluation cases")
    single.validate_code((source / "solution.py").read_text())
    output.mkdir(parents=True)
    initial = output / "initial"
    initial.mkdir()
    for name in ["solution.py", "request.json", "response.json", "response.txt", "reasoning.txt", "prompt.txt", "run.json"]:
        if (source / name).exists():
            shutil.copy2(source / name, initial / name)
    # Only the inherited dev results are exposed to this continuation.
    result["dev_feedback"] = old["dev_feedback"]
    save(initial / "dev.json", result)
    (initial / "dev_feedback.txt").write_text(result["dev_feedback"])
    run = {"schema_version": 2, "status": "prepared", "started_at": single.now(),
           "source_directory": str(source), "source_feedback_sha256": digest(source / "feedback.json"),
           "settings": settings, "submissions_per_condition": submissions, "plan": plan,
           "task": info, "fingerprints": fingerprints(), "environment": environment(),
           "initial_hashes": {p.name: digest(p) for p in initial.iterdir()},
           "server": {"gpu_ids": "2,3,4,5", "dtype": "bfloat16", "tensor_parallel_size": 4,
                      "max_model_len": 65536, "enforce_eager": False,
                      "note": "Inherited initial code/result used eager mode; new generations use CUDA Graphs."},
           "candidates": {}}
    save(output / "run.json", run)
    report(output, run)
    print(f"Prepared {output}: {len(plan)} new requests, initial dev={result['dev_score']:.6f}")
    print("No generation requests or solver evaluations were performed.")
    return run


def verify(folder, run):
    if (run["schema_version"] != 2 or run["fingerprints"] != fingerprints()
            or run["environment"] != environment()
            or run["task"] != task_info(Path(run["task"]["data_dir"]))
            or run["plan"] != plan_for(run["submissions_per_condition"])
            or run["settings"] != settings_from(folder / "initial")):
        raise RuntimeError("Code, prompts, data, environment or plan changed; refusing to resume")
    for name, expected in run["initial_hashes"].items():
        if digest(folder / "initial" / name) != expected:
            raise RuntimeError(f"Imported initial artifact changed: {name}")
    if set(run["candidates"]) - {item["name"] for item in run["plan"]}:
        raise RuntimeError("Unexpected candidates in manifest")


def load_result(folder, name):
    path = folder / name / "dev.json"
    return json.loads(path.read_text()) if path.exists() else None


def prompt_for(run, folder, condition, parent):
    result = load_result(folder, parent)
    if result is None:
        raise RuntimeError(f"The latest parent has no completed dev evaluation: {parent}")
    record = run["candidates"].get(parent, {})
    feedback = result["dev_feedback"]
    if parent == "initial" or record.get("code_sha256"):
        previous = (folder / parent / "solution.py").read_text()
    else:
        # Use the saved response as the source of truth, including an incomplete answer.
        # Prior reasoning is not fed back, just as for successful candidates.
        response = json.loads((folder / parent / "response.json").read_text())
        previous_text = response["choices"][0]["message"].get("content") or ""
        previous = ("No usable Python code was extracted from the immediately preceding submission.\n"
                    "Its final-answer text was:\n\n" + (previous_text or "(No final-answer text was returned.)"))
    if record.get("code_error"):
        feedback += (f"\n\nThe immediately preceding submission failed: {record['code_error']}\n"
                     "Produce a complete, valid solver in this submission, using the preceding attempt and errors.")
    return TEMPLATE.read_text().format(problem_description=run["task"]["problem_description"],
        previous_code=previous, dev_feedback=feedback,
        intervention=INTERVENTIONS[condition], timeout=run["settings"]["timeout"])


def options(run):
    settings = run["settings"]
    return single.make_parser().parse_args(["--task", TASK, "--model", settings["model"],
        "--api-base", settings["api_base"], "--reasoning-effort", settings["reasoning_effort"],
        "--max-tokens", str(settings["max_tokens"]), "--generation-timeout", str(settings["generation_timeout"]),
        "--timeout", str(settings["timeout"]), "--cpu-num", "1", "--seed", str(settings["seed"])])


def generate(folder, run, item, parent, generator_factory=single.make_generator):
    name = item["name"]
    target = folder / name
    target.mkdir(exist_ok=True)
    prompt = prompt_for(run, folder, item["condition"], parent)
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    record = run["candidates"].get(name)
    if record:
        if record["parent"] != parent or record["prompt_sha256"] != prompt_hash:
            raise RuntimeError(f"Parent or prompt changed: {name}")
        if not (target / "response.json").exists():
            raise RuntimeError(f"{name}: request already attempted without a saved response. No automatic resend.")
        if record.get("response_sha256") and digest(target / "response.json") != record["response_sha256"]:
            raise RuntimeError(f"Saved response changed: {name}")
    else:
        record = {**item, "parent": parent, "prompt_sha256": prompt_hash,
                  "status": "request_started", "started_at": single.now()}
        run["candidates"][name] = record
        save(folder / "run.json", run)
        print(f"{name}: generating {len(run['candidates'])}/{len(run['plan'])}, parent={parent}", flush=True)
        started = time.monotonic()
        try:
            generator_factory(options(run), target, os.environ.get("COBENCH_API_KEY", "local"))(
                prompt, model=f"openai/{run['settings']['model']}", reasoning_effort="medium")
        except Exception as exc:
            if not (target / "response.json").exists():
                record.update(status="api_error_or_unknown", error_type=type(exc).__name__)
                save(folder / "run.json", run)
                raise RuntimeError(f"{name}: generation failed ({type(exc).__name__}); no retry") from None
            if not isinstance(exc, ValueError):
                raise
        record["generation_seconds"] = time.monotonic() - started
    response = json.loads((target / "response.json").read_text())
    choice = response["choices"][0]
    message = choice["message"]
    content = message.get("content") or ""
    (target / "response.txt").write_text(content)
    if message.get("reasoning_content"):
        (target / "reasoning.txt").write_text(message["reasoning_content"])
    record.update(usage=response.get("usage"), finish_reason=choice["finish_reason"],
                  response_model=response.get("model"), response_fingerprint=response.get("system_fingerprint"))
    error = None
    code = None
    try:
        if choice["finish_reason"] != "stop":
            raise ValueError(f"Generation ended with finish_reason={choice['finish_reason']}")
        blocks = extract_code_blocks(content.rsplit("</think>", 1)[-1])
        if not blocks:
            raise ValueError("No Python code block")
        code = textwrap.dedent(blocks[0])
        single.validate_code(code)
    except (ValueError, SyntaxError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    if code is not None:
        path = target / "solution.py"
        if path.exists() and path.read_text() != code:
            raise RuntimeError(f"Saved solution changed: {name}")
        path.write_text(code)
    record.update(code_error=error, status="response_saved", response_sha256=digest(target / "response.json"),
                  code_sha256=digest(target / "solution.py") if code is not None else None)
    save(folder / "run.json", run)


def evaluate_candidate(folder, run, name):
    target = folder / name
    existing = load_result(folder, name)
    if existing is not None:
        return existing
    error = run["candidates"][name]["code_error"]
    if error:
        raw = {case: ([error] * count, None) for case, count in run["task"]["sizes"].items()}
        result = summarize(None, raw)
        save(target / "dev.json", result)
    else:
        env = {key: os.environ[key] for key in ["PATH", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "TMPDIR"] if key in os.environ}
        env.update({key: "1" for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]})
        env["PYTHONUNBUFFERED"] = "1"
        with (target / "dev.log").open("a") as log:
            process = subprocess.Popen([sys.executable, str(ROOT / "scripts/tsp_worker.py"),
                "--source", str(target / "solution.py"), "--data-dir", run["task"]["data_dir"],
                "--timeout", str(run["settings"]["timeout"]), "--output", str(target / "dev.json")],
                env=env, cwd=ROOT, stdout=log, stderr=log)
            try:
                returncode = process.wait()
                if returncode:
                    raise RuntimeError(f"Evaluation worker failed ({returncode}); inspect {target / 'dev.log'}")
            except BaseException:
                with contextlib.suppress(psutil.NoSuchProcess):
                    for child in psutil.Process(process.pid).children(recursive=True):
                        with contextlib.suppress(psutil.NoSuchProcess):
                            child.kill()
                if process.poll() is None:
                    process.kill()
                process.wait()
                raise
        result = load_result(folder, name)
    (target / "dev_feedback.txt").write_text(result["dev_feedback"])
    return result


def report(folder, run):
    initial = load_result(folder, "initial")
    latest = {condition: "initial" for condition in INTERVENTIONS}
    best_observed = dict(latest)  # Reporting only; never used by generation or parent selection.
    rows = [{"condition": condition, "submission": 1, "candidate": "initial", "parent": "",
             "dev_score": initial["dev_score"], "failed": initial["failed"],
             "delta_from_parent": "", "delta_from_initial": 0.0,
             "best_observed_candidate": "initial", "best_observed_dev_score": initial["dev_score"]}
            for condition in INTERVENTIONS]
    completed = {condition: 1 for condition in INTERVENTIONS}
    for item in run["plan"]:
        result = load_result(folder, item["name"])
        if result is None:
            continue
        condition = item["condition"]
        parent = parent_for(item)
        parent_result = load_result(folder, parent)
        if parent_result is None:
            raise RuntimeError(f"Missing latest parent result: {parent}")
        latest[condition] = item["name"]
        if result["dev_score"] > load_result(folder, best_observed[condition])["dev_score"]:
            best_observed[condition] = item["name"]
        best = best_observed[condition]
        completed[condition] += 1
        rows.append({"condition": condition, "submission": item["submission"], "candidate": item["name"],
                     "parent": parent, "dev_score": result["dev_score"], "failed": result["failed"],
                     "delta_from_parent": result["dev_score"] - parent_result["dev_score"],
                     "delta_from_initial": result["dev_score"] - initial["dev_score"],
                     "best_observed_candidate": best, "best_observed_dev_score": load_result(folder, best)["dev_score"]})
    summary = {"status": run["status"], "new_generation_requests_attempted": len(run["candidates"]),
               "new_generation_requests_planned": len(run["plan"]), "completed_submissions": completed,
               "parent_policy": run["settings"]["selection"],
               "latest_by_condition": latest,
               "latest_dev_scores": {c: load_result(folder, name)["dev_score"] for c, name in latest.items()},
               "final_by_condition": {c: name if completed[c] == run["submissions_per_condition"] else None
                                      for c, name in latest.items()},
               "best_observed_by_dev": best_observed, "best_observed_is_diagnostic_only": True, "rows": rows}
    save(folder / "summary.json", summary)
    with (folder / "scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    usage_rows = [{"candidate": "initial", "imported": True, "generation_seconds": "",
                   **(json.loads((folder / "initial" / "response.json").read_text()).get("usage") or {})}]
    usage_rows.extend({"candidate": name, "imported": False,
                       "generation_seconds": value.get("generation_seconds", ""),
                       **(value.get("usage") or {})} for name, value in run["candidates"].items())
    columns = list(dict.fromkeys(key for row in usage_rows for key in row))
    with (folder / "usage.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(usage_rows)
    return summary


def run_continuation(folder, generator_factory=single.make_generator, evaluator=evaluate_candidate, readiness=single.ensure_server):
    with run_lock(folder):
        run = json.loads((folder / "run.json").read_text())
        verify(folder, run)
        server_checked = False
        try:
            run.update(status="running", updated_at=single.now())
            run.pop("last_error", None)
            save(folder / "run.json", run)
            for item in run["plan"]:
                name, condition = item["name"], item["condition"]
                parent = parent_for(item)
                existing = load_result(folder, name)
                if existing is not None:
                    record = run["candidates"][name]
                    if record["parent"] != parent:
                        raise RuntimeError(f"Parent selection changed: {name}")
                    for filename, key in [("solution.py", "code_sha256"), ("response.json", "response_sha256"), ("dev.json", "dev_sha256")]:
                        if record.get(key) and digest(folder / name / filename) != record[key]:
                            raise RuntimeError(f"Saved candidate artifact changed: {name}/{filename}")
                else:
                    if name not in run["candidates"] and not server_checked:
                        print("Checking local Qwen server (no generation request)...", flush=True)
                        served = readiness(run["settings"]["api_base"], run["settings"]["model"],
                            os.environ.get("COBENCH_API_KEY", "local"), wait_timeout=run["settings"]["server_wait_timeout"])
                        if isinstance(served, dict) and served.get("max_model_len", 65536) <= run["settings"]["max_tokens"]:
                            raise RuntimeError("Server context limit leaves no input space")
                        run["server_runtime"] = {"checked_at": single.now(), "served_model": served}
                        server_checked = True
                    generate(folder, run, item, parent, generator_factory)
                    print(f"{name}: evaluating dev; progress: {folder / name / 'dev.log'}", flush=True)
                    existing = evaluator(folder, run, name)
                run["candidates"][name].update(status="evaluated", next_parent=name,
                    dev_sha256=digest(folder / name / "dev.json"), finished_at=single.now())
                save(folder / "run.json", run)
                report(folder, run)
                print(f"{name}: dev={existing['dev_score']:.6f}, failed={existing['failed']}, next_parent={name}", flush=True)
            run["status"] = "completed"
            return 0
        except KeyboardInterrupt:
            run.update(status="interrupted", last_error={"type": "KeyboardInterrupt"})
            raise
        except Exception as exc:
            run.update(status="stopped", last_error={"type": type(exc).__name__})
            if isinstance(exc, (RuntimeError, ValueError)):
                run["last_error"]["message"] = str(exc)
            raise
        finally:
            run["updated_at"] = single.now()
            save(folder / "run.json", run)
            report(folder, run)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Import the initial solver/results without generation or evaluation")
    prep.add_argument("--source", type=Path, default=SOURCE)
    prep.add_argument("--output-dir", type=Path, default=OUTPUT)
    prep.add_argument("--submissions", type=int, default=10)
    for command in ["check", "run", "resume", "report"]:
        command_parser = sub.add_parser(command)
        command_parser.add_argument("run_dir", type=Path, nargs="?", default=OUTPUT)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            prepare(args.source, args.output_dir, args.submissions)
            return 0
        folder = args.run_dir.resolve()
        if args.command in ("run", "resume"):
            return run_continuation(folder)
        with run_lock(folder):
            run = json.loads((folder / "run.json").read_text())
            verify(folder, run)
            summary = report(folder, run)
        print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))
        print("No model requests or solver evaluations were performed.")
        return 0
    except KeyboardInterrupt:
        print("Interrupted. Saved responses and evaluation checkpoints will be reused.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Continuation stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
