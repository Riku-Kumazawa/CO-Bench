"""Prepare and execute a fixed experiment plan without duplicate submissions."""

import contextlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import time
from uuid import uuid4

import psutil

import main as single
from agents.utils import extract_code_blocks
from experiment_manager import definitions as defs
from experiment_manager import evaluation as ev
from experiment_manager.storage import ROOT, digest, environment, hashes, implementation, lock, read, save, verify_hashes


def initial_path(folder, cfg, initial):
    item = next(i for i in cfg["initials"] if i["id"] == initial)
    return Path(folder) / "snapshot" / item["directory"]


def initial_ref(name):
    return f"initials/{name}"


def artifact_path(folder, reference):
    return Path(folder) / reference


def cached_initial(folder, cfg, task, item):
    path = Path(folder) / item["directory"]
    result_path = path / "dev.json"
    if not result_path.exists():
        return None
    result = read(result_path)
    if result.get("signature") != ev.signature(task, digest(path / "solution.py"), cfg["evaluation"]):
        return None
    results = result.get("results", {})
    expected = {n: len(idx) for n, idx in task["dev_indices"].items() if idx}
    if (set(results) != set(expected)
            or any(len(results[n][0]) != count for n, count in expected.items())):
        return None
    clean = ev.summarize(results, cfg["evaluation"]["feedback_length"])
    return {**clean, "signature": result["signature"]}


def plan_summary(folder):
    cfg, task, plan = defs.task_and_plan(folder)
    initial_evaluations = [i["id"] for i in cfg["initials"] if cached_initial(folder, cfg, task, i) is None]
    return {"name": cfg["name"], "task": cfg["task"], "parent_policies": cfg["parent_policies"],
            "initials": [i["id"] for i in cfg["initials"]], "conditions": cfg["prompts"]["conditions"],
            "repeats": cfg["repeats"], "steps": cfg["steps"], "new_generations": len(plan),
            "candidate_dev_evaluations": len(plan), "dev_instances_per_evaluation": sum(map(len, task["dev_indices"].values())),
            "initials_requiring_evaluation": initial_evaluations,
            "initials_reusing_evaluation": [i["id"] for i in cfg["initials"] if i["id"] not in initial_evaluations],
            "interventions": {n: (Path(folder) / p).read_text() for n, p in cfg["prompts"]["conditions"].items()},
            "first_candidates": plan[:len(cfg["prompts"]["conditions"])]}


def prepare(version, run_id=None):
    version = Path(version).resolve()
    defs.verify_version(version)
    if version.parent.name != "versions":
        raise ValueError("Prepare requires <experiment>/versions/<version>")
    experiment = version.parent.parent
    run_id = run_id or time.strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:8]
    defs.identifier(run_id)
    cfg, task, plan = defs.task_and_plan(version)
    with lock(experiment):
        runs = experiment / "runs"
        runs.mkdir(exist_ok=True)
        dest = runs / run_id
        if dest.exists():
            raise FileExistsError(f"Run already exists: {dest}")
        temporary = runs / f".preparing-{uuid4().hex}"
        temporary.mkdir()
        try:
            shutil.copytree(version, temporary / "snapshot")
            save(temporary / "plan.json", plan)
            save(temporary / "task.json", task)
            save(temporary / "evaluation.json", cfg["evaluation"])
            state = {"schema_version": 1, "status": "prepared", "created_at": single.now(),
                     "version": version.name, "version_metadata": read(version / "version.json"),
                     "snapshot_hashes": hashes(temporary / "snapshot"),
                     "fixed_hashes": {n: digest(temporary / n) for n in ("plan.json", "task.json", "evaluation.json")},
                     "implementation": implementation(), "environment": environment(),
                     "local_checkpoint": single.checkpoint_info()[1],
                     "candidates": {}, "initials": {}}
            for item in cfg["initials"]:
                name = item["id"]
                target = temporary / initial_ref(name)
                target.mkdir(parents=True)
                shutil.copy2(initial_path(temporary, cfg, name) / "solution.py", target / "solution.py")
                cached = cached_initial(version, cfg, task, item)
                record = {"code_sha256": digest(target / "solution.py"), "evaluation_reused": cached is not None}
                if cached:
                    save(target / "dev.json", cached)
                    record.update(status="evaluated", dev_sha256=digest(target / "dev.json"))
                else:
                    record["status"] = "awaiting_evaluation"
                state["initials"][name] = record
            save(temporary / "manifest.json", state)
            temporary.rename(dest)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    from experiment_manager.reports import report
    report(dest)
    return dest


def verify(folder, runtime=False, data=False):
    folder = Path(folder)
    state = read(folder / "manifest.json")
    verify_hashes(folder / "snapshot", state["snapshot_hashes"])
    for name, expected in state["fixed_hashes"].items():
        if digest(folder / name) != expected:
            raise RuntimeError(f"Fixed run file changed: {name}")
    cfg = defs.load(folder / "snapshot")
    if runtime and (state["implementation"] != implementation() or state["environment"] != environment()):
        raise RuntimeError("Execution code or environment changed; create a new run")
    task = read(folder / "task.json")
    if data and task != ev.describe(cfg["task"], cfg["data_dir"], cfg["evaluation"]["unsplit_policy"]):
        raise RuntimeError("Task data changed")
    plan = read(folder / "plan.json")
    planned = {i["id"] for i in plan}
    if set(state["candidates"]) - planned:
        raise RuntimeError("Unexpected candidate in manifest")
    for name, record in state["initials"].items():
        _verify_artifacts(folder / initial_ref(name), record)
    for name, record in state["candidates"].items():
        _verify_artifacts(folder / name, record)
    return state, cfg, task, plan


def _verify_artifacts(path, record):
    for filename, key in [("solution.py", "code_sha256"), ("response.json", "response_sha256"),
                          ("dev.json", "dev_sha256"), ("prompt.txt", "prompt_sha256"), ("request.json", "request_sha256")]:
        if record.get(key) and (not (path / filename).is_file() or digest(path / filename) != record[key]):
            raise RuntimeError(f"Saved artifact changed: {path / filename}")


def select_parent(folder, state, item):
    initial = initial_ref(item["initial"])
    if item["policy"] == "initial" or item["step"] == 1:
        return initial
    previous = sorted((r for r in state["candidates"].values()
                       if r["branch"] == item["branch"] and r["step"] < item["step"]), key=lambda r: r["step"])
    if len(previous) != item["step"] - 1 or any(r["status"] != "evaluated" for r in previous):
        raise RuntimeError("Previous branch steps have not completed")
    if item["policy"] == "latest":
        return previous[-1]["id"]
    best = initial
    score = read(artifact_path(folder, best) / "dev.json")["dev_score"]
    for record in previous:
        # Non-executable final-answer text cannot be a best-code parent.
        if record.get("code_error"):
            continue
        candidate = read(artifact_path(folder, record["id"]) / "dev.json")["dev_score"]
        if candidate > score:  # Strictly greater: retain earlier best on ties.
            best, score = record["id"], candidate
    return best


def render_prompt(folder, cfg, task, parent, condition, state):
    path = artifact_path(folder, parent)
    result = read(path / "dev.json")
    record = state["candidates"].get(parent, {})
    if (path / "solution.py").exists():
        previous = (path / "solution.py").read_text()
    else:
        response = read(path / "response.json")
        content = final_answer(response["choices"][0]["message"]) or "(No final-answer text was returned.)"
        previous = "No usable Python code was extracted. The preceding final-answer text was:\n\n" + content
    feedback = result["dev_feedback"]
    if record.get("code_error"):
        feedback += f"\n\nThe parent submission failed: {record['code_error']}\nProduce a complete, valid solver."
    snapshot = Path(folder) / "snapshot"
    return (snapshot / cfg["prompts"]["template"]).read_text().format(
        problem_description=task["problem_description"], previous_code=previous,
        dev_feedback=feedback, intervention=(snapshot / cfg["prompts"]["conditions"][condition]).read_text(),
        timeout=cfg["evaluation"]["timeout_seconds"])


def final_answer(message):
    content = message.get("content") or ""
    if "</think>" in content:
        return content.rsplit("</think>", 1)[-1]
    if content.lstrip().startswith("<think>"):
        return ""
    return content


def request_for(cfg, item, prompt):
    g = cfg["generation"]
    return {"model": f"openai/{g['model']}", "api_base": g["api_base"],
            "messages": [{"role": "user", "content": prompt}], "seed": item["seed"],
            "max_tokens": g["max_tokens"], "temperature": g["temperature"], "top_p": g["top_p"],
            "presence_penalty": g["presence_penalty"], "timeout": g["timeout_seconds"],
            "num_retries": 0, "max_retries": 0, "drop_params": False,
            "extra_body": {"top_k": g["top_k"], "min_p": g["min_p"], "repetition_penalty": g["repetition_penalty"],
                "chat_template_kwargs": {"enable_thinking": g["thinking"],
                    **({"reasoning_effort": g["reasoning_effort"]} if g["thinking"] else {})}}}


def complete(request):
    import litellm
    return litellm.completion(api_key=os.environ.get("COBENCH_API_KEY", "local"), **request).model_dump(mode="json")


def generate(folder, state, cfg, task, item, parent, completion):
    path = Path(folder) / item["id"]
    path.mkdir(parents=True, exist_ok=True)
    prompt = render_prompt(folder, cfg, task, parent, item["condition"], state)
    request = request_for(cfg, item, prompt)
    record = state["candidates"].get(item["id"])
    if record:
        if (record["parent"] != parent or (path / "prompt.txt").read_text() != prompt
                or read(path / "request.json") != request):
            raise RuntimeError("Parent, prompt or request changed")
        if not (path / "response.json").exists():
            raise RuntimeError("Request already attempted without a saved response. No automatic resend.")
    else:
        (path / "prompt.txt").write_text(prompt)
        save(path / "request.json", request)
        record = {**item, "parent": parent, "parent_selection_reason": item["policy"],
                  "status": "request_started", "started_at": single.now(),
                  "prompt_sha256": digest(path / "prompt.txt"), "request_sha256": digest(path / "request.json")}
        state["candidates"][item["id"]] = record
        save(Path(folder) / "manifest.json", state)
        started = time.monotonic()
        try:
            save(path / "response.json", completion(request))
        except BaseException:
            record["status"] = "response_unknown"
            save(Path(folder) / "manifest.json", state)
            raise
        record["generation_seconds"] = time.monotonic() - started
    response = read(path / "response.json")
    choice = response["choices"][0]
    message = choice["message"]
    content = message.get("content") or ""
    (path / "response.txt").write_text(content)
    if message.get("reasoning_content"):
        (path / "reasoning.txt").write_text(message["reasoning_content"])
    code, error = None, None
    try:
        if choice["finish_reason"] != "stop":
            raise ValueError(f"Generation ended with finish_reason={choice['finish_reason']}")
        blocks = extract_code_blocks(final_answer(message))
        if not blocks:
            raise ValueError("No Python code block")
        code = textwrap.dedent(blocks[0])
        single.validate_code(code)
    except (ValueError, SyntaxError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    if code is not None:
        (path / "solution.py").write_text(code)
    record.update(status="response_saved", code_error=error, finish_reason=choice["finish_reason"],
                  usage=response.get("usage"), response_sha256=digest(path / "response.json"),
                  code_sha256=digest(path / "solution.py") if code is not None else None,
                  response_model=response.get("model"), response_fingerprint=response.get("system_fingerprint"))
    save(Path(folder) / "manifest.json", state)
    return record


def evaluate_path(folder, reference, task, settings, error=None):
    path = Path(folder) / reference
    if (path / "dev.json").exists():
        return read(path / "dev.json")
    if error:
        result = ev.failure(task, error, settings["feedback_length"])
        save(path / "dev.json", result)
    else:
        env = {key: os.environ[key] for key in ("PATH", "LANG", "LC_ALL", "LD_LIBRARY_PATH", "TMPDIR") if key in os.environ}
        env.update({k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
        env.update(PYTHONUNBUFFERED="1", LITELLM_LOCAL_MODEL_COST_MAP="True")
        with (path / "dev.log").open("a") as log:
            process = subprocess.Popen([sys.executable, "-m", "experiment_manager.evaluation", str(path / "solution.py"),
                str(Path(folder) / "task.json"), str(Path(folder) / "evaluation.json"), str(path / "dev.json")],
                env=env, cwd=ROOT, stdout=log, stderr=log, start_new_session=True)
            try:
                code = process.wait()
                if code:
                    raise RuntimeError(f"Evaluation worker exited {code}; inspect {path / 'dev.log'}")
            except BaseException:
                with contextlib.suppress(psutil.NoSuchProcess):
                    for child in psutil.Process(process.pid).children(recursive=True):
                        with contextlib.suppress(psutil.NoSuchProcess):
                            child.kill()
                if process.poll() is None:
                    process.kill()
                process.wait()
                raise
        result = read(path / "dev.json")
    (path / "dev_feedback.txt").write_text(result["dev_feedback"])
    return result


def run(folder, completion=complete, evaluator=evaluate_path, readiness=single.ensure_server):
    folder = Path(folder).resolve()
    from experiment_manager.reports import report
    with lock(folder):
        state, cfg, task, plan = verify(folder, runtime=True, data=True)
        try:
            state.update(status="running", updated_at=single.now())
            state.pop("last_error", None)
            save(folder / "manifest.json", state)
            for name, record in state["initials"].items():
                if record["status"] != "evaluated":
                    print(f"Evaluating initial {name} (dev only)", flush=True)
                    evaluator(folder, initial_ref(name), task, cfg["evaluation"])
                    record.update(status="evaluated", dev_sha256=digest(folder / initial_ref(name) / "dev.json"))
                    save(folder / "manifest.json", state)
            server_checked = False
            for item in plan:
                parent = select_parent(folder, state, item)
                existing = state["candidates"].get(item["id"])
                if existing and existing["status"] == "evaluated":
                    if existing["parent"] != parent:
                        raise RuntimeError("Saved parent selection changed")
                    continue
                if not existing and not server_checked:
                    g = cfg["generation"]
                    served = readiness(g["api_base"], g["model"], os.environ.get("COBENCH_API_KEY", "local"),
                                       wait_timeout=g["server_wait_seconds"])
                    if isinstance(served, dict) and served.get("max_model_len", 65536) <= g["max_tokens"]:
                        raise RuntimeError("Server context limit leaves no input space")
                    state["server_runtime"] = served
                    server_checked = True
                print(f"{item['id']}: parent={parent}, seed={item['seed']}", flush=True)
                record = generate(folder, state, cfg, task, item, parent, completion)
                result = evaluator(folder, item["id"], task, cfg["evaluation"], record["code_error"])
                record.update(status="evaluated", dev_sha256=digest(folder / item["id"] / "dev.json"), finished_at=single.now())
                save(folder / "manifest.json", state)
                report(folder, figures=False)
                print(f"dev={result['dev_score']:.6f}, failed={result['failed']}", flush=True)
            state["status"] = "completed"
        except BaseException as exc:
            state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "stopped",
                         last_error={"type": type(exc).__name__, "message": str(exc)})
            raise
        finally:
            state["updated_at"] = single.now()
            save(folder / "manifest.json", state)
            report(folder, figures=False)
        report(folder, figures=True)
    return 0
