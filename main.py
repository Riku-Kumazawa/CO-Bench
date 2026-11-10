"""Generate one solver with local Qwen, evaluate it once, and retain artifacts."""

import argparse
import ast
import json
import os
import sys
import textwrap
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from uuid import uuid4
from zoneinfo import ZoneInfo

# Importing LiteLLM must not fetch its remote model-pricing map.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

from agents import GreedyRefine
from evaluation import Evaluator, get_data

ROOT = Path(__file__).resolve().parent
MODEL = "Qwen/Qwen3.8-27B"
PACKAGES = ("litellm", "openai", "psutil", "cloudpickle", "numpy", "scipy", "ortools", "tqdm")


def now():
    return datetime.now(ZoneInfo("Asia/Tokyo")).isoformat()


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n")


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return number


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check local preparation without contacting the model server")
    parser.add_argument("--task", default="Aircraft landing")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--model", default=os.environ.get("COBENCH_MODEL", MODEL))
    parser.add_argument("--api-base", default=os.environ.get("COBENCH_API_BASE", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "xhigh"), default="medium")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True, help="enable Qwen thinking; use --no-thinking for a faster first smoke run")
    parser.add_argument("--max-tokens", type=positive_int, default=32768, help="output token limit, including thinking tokens")
    parser.add_argument("--generation-timeout", type=positive_int, default=7200, help="LLM request timeout in seconds")
    parser.add_argument("--server-wait-timeout", type=nonnegative_int, default=600, help="wait for model API readiness in seconds (0: fail immediately)")
    parser.add_argument("--timeout", type=positive_int, default=10, help="evaluation timeout per instance in seconds")
    parser.add_argument("--cpu-num", type=positive_int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--code", type=Path, help="evaluate a saved solver without contacting the model")
    return parser


def load_task(args):
    src_dir = args.data_dir.resolve()
    config = (src_dir / args.task / "config.py").resolve()
    if not config.is_relative_to(src_dir) or not config.is_file():
        raise ValueError(f"Task config not found under {src_dir}: {args.task}")
    data = get_data(args.task, src_dir=str(src_dir))
    if not data.test_cases:
        raise ValueError(f"No instances found for {args.task}")
    return data


def checkpoint_info():
    model_dir = Path(os.environ.get("COBENCH_MODEL_DIR", ROOT / "models/Qwen3.8-27B"))
    manifest = model_dir / "checkpoint.json"
    if not manifest.is_file():
        return model_dir, None
    return model_dir, json.loads(manifest.read_text())


def check_preparation(args):
    data = load_task(args)
    sizes = {case: len(data.load_data(str(Path(data.src_dir) / data.task / case))) for case in data.test_cases}
    dev = data.get_dev()
    if dev is None:
        split = "no dev/test split (both scores use all instances)"
    else:
        dev_count = sum(len(dev.get(case, [])) for case in sizes)
        split = f"dev={dev_count}, test={sum(sizes.values()) - dev_count}"
    print(f"Task: {data.task}; files={len(sizes)}, instances={sum(sizes.values())}, {split}")
    ok = True
    for name in PACKAGES:
        try:
            print(f"  {name}: {version(name)}")
        except PackageNotFoundError:
            print(f"  {name}: MISSING")
            ok = False
    model_dir, manifest = checkpoint_info()
    if manifest:
        for name, size in manifest["shards"].items():
            path = model_dir / name
            if not path.is_file() or path.stat().st_size != size:
                print(f"  Missing/incomplete checkpoint shard: {name}")
                ok = False
        print(f"Checkpoint: {manifest['repo_id']}@{manifest['revision']}")
    else:
        print("Checkpoint: not prepared; run uv run --frozen python scripts/prepare_qwen.py")
        ok = False
    vllm_python = Path(os.environ.get("COBENCH_VLLM_PYTHON", "/home/rocklee/qwen-vllm/.venv/bin/python"))
    if not vllm_python.is_file():
        print(f"vLLM Python: MISSING ({vllm_python})")
        ok = False
    else:
        print(f"vLLM Python: {vllm_python}")
    ninja = vllm_python.parent / "ninja"
    if not os.access(ninja, os.X_OK):
        print(f"ninja: MISSING ({ninja})")
        ok = False
    else:
        print(f"ninja: {ninja}")
    print("No server was started, and no solver was generated or evaluated.")
    return 0 if ok else 1


def ensure_server(api_base, model, api_key, wait_timeout=0):
    url = urlsplit(api_base)
    if url.scheme not in ("http", "https") or not url.netloc or url.username or url.password or url.query or url.fragment:
        raise ValueError("api-base must be an http(s) URL without credentials, query, or fragment")
    request = Request(api_base.rstrip("/") + "/models", headers={"Authorization": f"Bearer {api_key}"})
    deadline = time.monotonic() + wait_timeout
    next_progress = 0.0
    while True:
        try:
            remaining = max(0.0, deadline - time.monotonic())
            with urlopen(request, timeout=min(10.0, remaining) if wait_timeout and remaining else 10) as response:
                available = json.load(response)
            break
        except (URLError, TimeoutError, ConnectionError) as exc:
            # Authentication and URL errors require a configuration fix, not waiting.
            if isinstance(exc, HTTPError) and exc.code not in (502, 503, 504):
                raise RuntimeError(f"Model API returned HTTP {exc.code} at {api_base}; check the URL and authentication.") from exc
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"Model API unavailable at {api_base} after waiting {wait_timeout}s. "
                    "Check the serve_qwen.sh terminal for startup progress or errors. "
                    "No code-generation request was sent."
                ) from exc
            if time.monotonic() >= next_progress:
                print(f"Waiting for model API: {api_base} ({remaining:.0f}s remaining; no generation request sent)", flush=True)
                next_progress = time.monotonic() + 10
            time.sleep(min(2.0, remaining))
    names = {item["id"] for item in available.get("data", [])}
    if model not in names:
        raise ValueError(f"Requested model {model!r} is not served. Available: {sorted(names)}")
    return next(item for item in available["data"] if item["id"] == model)


def make_generator(args, output, api_key):
    def generate(prompt, *, model, reasoning_effort):
        import litellm

        request = {
            "model": model,
            "api_base": args.api_base,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_tokens,
            "temperature": 1.0 if args.thinking else 0.7,
            "top_p": 0.95 if args.thinking else 0.8,
            "presence_penalty": 0.0 if args.thinking else 1.5,
            "seed": args.seed,
            "timeout": args.generation_timeout,
            "num_retries": 0,
            "max_retries": 0,
            "drop_params": False,
            "extra_body": {
                "top_k": 20,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "chat_template_kwargs": {
                    "enable_thinking": args.thinking,
                    **({"reasoning_effort": reasoning_effort} if args.thinking else {}),
                },
            },
        }
        save_json(output / "request.json", request)
        (output / "prompt.txt").write_text(prompt)
        # One request only: no fallbacks, retries, repair calls, or refinement.
        response = litellm.completion(api_key=api_key, **request)
        save_json(output / "response.json", response.model_dump(mode="json"))
        choice = response.choices[0]
        content = choice.message.content or ""
        (output / "response.txt").write_text(content)
        reasoning = getattr(choice.message, "reasoning_content", None)
        if reasoning:
            (output / "reasoning.txt").write_text(reasoning)
        if choice.finish_reason == "length":
            raise ValueError("Generation reached max-tokens. Response saved; no retry or evaluation was performed.")
        if not content.strip():
            raise ValueError("The model returned no final answer. Response saved; no retry was performed.")
        # Also tolerate a server that returns thinking inline rather than separately.
        if "</think>" in content:
            content = content.rsplit("</think>", 1)[1]
        return content

    return generate


def validate_code(code):
    tree = ast.parse(code, filename="solution.py")
    compile(tree, "solution.py", "exec")
    if not any(isinstance(node, ast.FunctionDef) and node.name == "solve" for node in tree.body):
        raise ValueError("The generated code does not define a top-level solve function.")


def result_counts(results):
    total = failed = 0
    for scores, error in results.values():
        if scores is None:
            total += 1
            failed += 1
        else:
            total += len(scores)
            failed += len(scores) if error else sum(isinstance(value, str) for value in scores)
    return {"instances": total, "successful": total - failed, "failed": failed}


def run_task(args):
    data = load_task(args)
    stamp = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d-%H%M%S")
    output = args.output_dir.resolve() / f"{stamp}-{uuid4().hex[:8]}"
    output.mkdir(parents=True)
    _, checkpoint = checkpoint_info()
    state = {
        "status": "preparing", "started_at": now(), "task": args.task,
        "model": args.model, "api_base": args.api_base,
        "local_checkpoint_revision": checkpoint["revision"] if checkpoint and args.model == MODEL else None,
        "agent": "GreedyRefine" if not args.code else None,
        "candidate_limit": 1, "candidate_count": 0, "generation_requests": 0, "refinement_iterations": 0,
        "timeout_seconds_per_instance": args.timeout, "cpu_num": args.cpu_num,
        "max_tokens": args.max_tokens, "generation_timeout_seconds": args.generation_timeout,
        "server_wait_timeout_seconds": args.server_wait_timeout,
        "thinking": args.thinking,
        "reasoning_effort": args.reasoning_effort if args.thinking else None, "seed": args.seed,
        "code_source": str(args.code.resolve()) if args.code else "generated",
        "packages": {name: version(name) for name in PACKAGES},
    }
    save_json(output / "run.json", state)
    print(f"Artifacts: {output}", flush=True)
    try:
        if args.code:
            code = args.code.read_text()
        else:
            api_key = os.environ.get("COBENCH_API_KEY", "local")
            served_model = ensure_server(args.api_base, args.model, api_key, wait_timeout=args.server_wait_timeout)
            context_limit = served_model.get("max_model_len")
            if isinstance(context_limit, int):
                state["server_context_limit"] = context_limit
                if args.max_tokens >= context_limit:
                    raise ValueError(
                        f"Output limit {args.max_tokens} leaves no space for the input within "
                        f"the server context limit {context_limit}. Restart serve_qwen.sh "
                        "with a larger COBENCH_MAX_MODEL_LEN (default: 65536)."
                    )
            agent = GreedyRefine(
                problem_description=data.problem_description, timeout=args.timeout,
                model=f"openai/{args.model}", reasoning_effort=args.reasoning_effort,
                additional_prompt=(
                    "Use only the Python standard library, NumPy, SciPy, or OR-Tools. "
                    "Other solver packages are not installed. Use one CPU thread. "
                    "Return the best feasible solution before the timeout."
                ),
                llm_callable=make_generator(args, output, api_key),
            )
            state["status"] = "generating"
            state["generation_requests"] = 1
            save_json(output / "run.json", state)
            print("Generating one solver (no retries or refinement)...", flush=True)
            code = agent.step()
        code = textwrap.dedent(code)
        (output / "solution.py").write_text(code)
        state["candidate_count"] = 1
        state["status"] = "validating"
        validate_code(code)
        state["status"] = "evaluating"
        save_json(output / "run.json", state)
        print(f"Evaluating {args.task}: {len(data.test_cases)} files; timeout={args.timeout}s per instance", flush=True)
        evaluator = Evaluator(data, timeout=args.timeout, cpu_num=args.cpu_num)
        feedback = evaluator.evaluate(code)
        save_json(output / "feedback.json", asdict(feedback))
        (output / "feedback.txt").write_text(
            f"DEV\n{feedback.dev_feedback}\n\nTEST\n{feedback.test_feedback}\n"
        )
        counts = result_counts(feedback.results)
        state.update(counts)
        state.update(score=feedback.score, dev_score=feedback.dev_score, test_score=feedback.test_score)
        state["status"] = "completed" if not counts["failed"] else "completed_with_errors"
        print(f"dev_score={feedback.dev_score:.6f}, test_score={feedback.test_score:.6f}")
        print(f"Instances: {counts['successful']} successful, {counts['failed']} failed")
        return 0 if not counts["failed"] else 2
    except KeyboardInterrupt:
        state["status"] = "interrupted"
        print("Interrupted; available artifacts have been retained.", file=sys.stderr)
        return 130
    except Exception as exc:
        phase = state["status"]
        state["status"] = f"{phase}_failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        (output / "error.txt").write_text(traceback.format_exc())
        print(f"{state['status']}: {state['error']}", file=sys.stderr)
        print(f"Details: {output / 'error.txt'}", file=sys.stderr)
        return 1
    finally:
        state["finished_at"] = now()
        save_json(output / "run.json", state)


def main(argv=None):
    args = make_parser().parse_args(argv)
    # Keep candidate execution within the selected CPU budget.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    try:
        return check_preparation(args) if args.check else run_task(args)
    except Exception as exc:
        print(f"Preparation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
