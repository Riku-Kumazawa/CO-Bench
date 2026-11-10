"""Editable drafts, immutable versions, initial artifacts and experiment plans."""

import difflib
import itertools
from pathlib import Path
import shutil
import string
from uuid import uuid4
from urllib.parse import urlsplit

import yaml

import main as single
from experiment_manager import evaluation as ev
from experiment_manager.storage import ROOT, digest, environment, hashes, identifier, lock, read, save, verify_hashes


class UniqueLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)

DEFAULT_GENERATION = {"model": single.MODEL, "api_base": "http://127.0.0.1:8000/v1",
    "thinking": True, "reasoning_effort": "medium", "max_tokens": 32768,
    "temperature": 1.0, "top_p": .95, "top_k": 20, "min_p": 0.0,
    "presence_penalty": 0.0, "repetition_penalty": 1.0,
    "timeout_seconds": 7200, "server_wait_seconds": 600}
DEFAULT_EVALUATION = {"split": "dev", "unsplit_policy": "all_as_dev",
                      "timeout_seconds": 10, "cpu_num": 1, "feedback_length": 64}


def keys(value, allowed, label):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise ValueError(f"Invalid or unknown {label} settings: {value}")


def positive(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def load(folder):
    folder = Path(folder).resolve()
    if (folder / "version.json").exists():
        verify_version(folder)
    cfg = yaml.load((folder / "experiment.yaml").read_text(), Loader=UniqueLoader)
    keys(cfg, ["schema_version", "name", "task", "data_dir", "initials", "parent_policies",
               "repeats", "steps", "prompts", "generation", "seeds", "evaluation"], "experiment")
    if cfg.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    identifier(cfg["name"])
    if not isinstance(cfg.get("task"), str) or not cfg["task"]:
        raise ValueError("task must be a nonempty string")
    for field in ("repeats", "steps"):
        positive(cfg[field], field)
    policies = cfg["parent_policies"]
    if (not isinstance(policies, list) or not policies or len(set(policies)) != len(policies)
            or set(policies) - {"initial", "latest", "best"}):
        raise ValueError("parent_policies must contain unique initial/latest/best entries")
    keys(cfg["prompts"], ["template", "conditions"], "prompts")
    conditions = cfg["prompts"]["conditions"]
    if not isinstance(conditions, dict) or not conditions:
        raise ValueError("At least one condition is required")
    for name, path in conditions.items():
        identifier(name)
        (folder / path).read_text()
    template = (folder / cfg["prompts"]["template"]).read_text()
    fields = set()
    for _, name, spec, conversion in string.Formatter().parse(template):
        if name is not None:
            if spec or conversion:
                raise ValueError("Prompt formatting options are not supported")
            fields.add(name)
    allowed = {"problem_description", "previous_code", "dev_feedback", "intervention", "timeout"}
    if fields != allowed:
        raise ValueError(f"Prompt must contain exactly these variables: {sorted(allowed)}; found {sorted(fields)}")
    if not isinstance(cfg["initials"], list) or not cfg["initials"]:
        raise ValueError("At least one initial solution is required")
    seen = set()
    for item in cfg["initials"]:
        keys(item, ["id", "directory"], "initial")
        name = identifier(item["id"])
        if name in seen:
            raise ValueError(f"Duplicate initial: {name}")
        seen.add(name)
        initial = (folder / item["directory"]).resolve()
        single.validate_code((initial / "solution.py").read_text())
        if (initial / "provenance.json").exists():
            provenance = read(initial / "provenance.json")
            if provenance["task"] != cfg["task"]:
                raise ValueError(f"Initial {name} belongs to a different task")
            verify_hashes(initial, {**provenance["artifact_hashes"], "provenance.json": digest(initial / "provenance.json")})
    generation = dict(cfg.get("generation", {}))
    keys(generation, DEFAULT_GENERATION, "generation")
    cfg["generation"] = {**DEFAULT_GENERATION, **generation}
    g = cfg["generation"]
    for field in ["max_tokens", "timeout_seconds"]:
        positive(g[field], f"generation.{field}")
    if type(g["server_wait_seconds"]) is not int or g["server_wait_seconds"] < 0:
        raise ValueError("server_wait_seconds must be a nonnegative integer")
    if type(g["thinking"]) is not bool or g["reasoning_effort"] not in ("low", "medium", "high", "xhigh"):
        raise ValueError("Invalid thinking/reasoning_effort")
    import math
    for field in ["temperature", "top_p", "min_p", "presence_penalty", "repetition_penalty"]:
        if type(g[field]) not in (int, float) or not math.isfinite(g[field]):
            raise ValueError(f"Invalid generation.{field}")
    if (g["temperature"] < 0 or not 0 < g["top_p"] <= 1 or not 0 <= g["min_p"] <= 1
            or g["repetition_penalty"] <= 0 or type(g["top_k"]) is not int or g["top_k"] < 1):
        raise ValueError("Invalid sampling parameters")
    if not isinstance(g["model"], str) or not g["model"] or not isinstance(g["api_base"], str):
        raise ValueError("model and api_base must be strings")
    url = urlsplit(g["api_base"])
    if (url.scheme not in ("http", "https") or not url.netloc or url.username or url.password
            or url.query or url.fragment):
        raise ValueError("api_base must be an http(s) URL without credentials, query or fragment")
    evaluation = dict(cfg.get("evaluation", {}))
    keys(evaluation, DEFAULT_EVALUATION, "evaluation")
    cfg["evaluation"] = {**DEFAULT_EVALUATION, **evaluation}
    e = cfg["evaluation"]
    if e["split"] != "dev" or e["cpu_num"] != 1 or e["unsplit_policy"] not in ("all_as_dev", "error"):
        raise ValueError("Only dev evaluation with one CPU thread is supported")
    positive(e["timeout_seconds"], "evaluation.timeout_seconds")
    positive(e["feedback_length"], "evaluation.feedback_length")
    seeds = cfg.get("seeds", {})
    keys(seeds, ["base", "paired_across_conditions"], "seeds")
    cfg["seeds"] = {"base": 101, "paired_across_conditions": True, **seeds}
    if type(cfg["seeds"]["base"]) is not int or cfg["seeds"]["base"] < 0 or type(cfg["seeds"]["paired_across_conditions"]) is not bool:
        raise ValueError("Invalid seed settings")
    cfg["data_dir"] = str((folder / cfg.get("data_dir", str(ROOT / "data"))).resolve())
    return cfg


def version_hashes(folder):
    return {k: v for k, v in hashes(folder).items() if k != "version.json"}


def verify_version(folder):
    folder = Path(folder)
    meta = read(folder / "version.json")
    if version_hashes(folder) != meta["hashes"]:
        raise RuntimeError(f"Published version changed: {folder}. Create a new version instead.")
    return meta


def bundle(source, target):
    """Copy referenced assets; the result has no dependency on editable originals."""
    source, target = Path(source).resolve(), Path(target)
    cfg = load(source)
    target.mkdir(parents=True, exist_ok=False)
    (target / "source.yaml").write_text((source / "experiment.yaml").read_text())
    for key, rel in [("template", cfg["prompts"]["template"])]:
        destination = target / "prompts/refine.txt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text((source / rel).read_text())
        cfg["prompts"][key] = "prompts/refine.txt"
    for name, rel in list(cfg["prompts"]["conditions"].items()):
        dest = Path("prompts/interventions") / f"{name}.txt"
        (target / dest).parent.mkdir(parents=True, exist_ok=True)
        (target / dest).write_text((source / rel).read_text())
        cfg["prompts"]["conditions"][name] = str(dest)
    for item in cfg["initials"]:
        src = (source / item["directory"]).resolve()
        hashes(src)  # Refuse symlinked initial artifacts before copytree dereferences them.
        dest = Path("initials") / item["id"]
        shutil.copytree(src, target / dest)
        item["directory"] = str(dest)
    (target / "experiment.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    return cfg


def freeze(draft, version, reason, parent=None):
    draft = Path(draft).resolve()
    if draft.parent.name != "drafts":
        raise ValueError("Draft must be under <experiment>/drafts/<name>")
    experiment = draft.parent.parent
    identifier(version)
    if not reason.strip():
        raise ValueError("A version change reason is required")
    if parent is None and (draft / "draft.json").exists():
        parent = read(draft / "draft.json").get("parent_version")
    if parent:
        identifier(parent)
        verify_version(experiment / "versions" / parent)
    with lock(experiment):
        versions = experiment / "versions"
        versions.mkdir(exist_ok=True)
        dest = versions / version
        if dest.exists():
            raise FileExistsError(f"Version already exists: {dest}")
        temporary = versions / f".preparing-{uuid4().hex}"
        try:
            bundle(draft, temporary)
            if (draft / "README.md").exists():
                shutil.copy2(draft / "README.md", temporary / "README.md")
            save(temporary / "version.json", {"schema_version": 1, "version": version,
                "parent_version": parent, "reason": reason, "created_at": single.now(),
                "hashes": version_hashes(temporary)})
            temporary.rename(dest)
            save(draft / "draft.json", {"parent_version": version, "last_frozen_version": version,
                                        "updated_at": single.now()})
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return dest


def fork(source, name):
    source = Path(source).resolve()
    meta = verify_version(source)
    identifier(name)
    dest = source.parent.parent / "drafts" / name
    with lock(source.parent.parent):
        bundle(source, dest)
        if (source / "README.md").exists():
            shutil.copy2(source / "README.md", dest / "README.md")
        save(dest / "draft.json", {"parent_version": meta["version"], "created_at": single.now()})
    return dest


def import_initial(experiment, name, source, task, data_dir=ROOT / "data"):
    identifier(name)
    source, experiment = Path(source).resolve(), Path(experiment).resolve()
    code = source if source.is_file() else source / "solution.py"
    single.validate_code(code.read_text())
    provenance = {"source": str(source), "imported_at": single.now(), "task": task,
                  "evaluation_reused": False}
    with lock(experiment):
        dest = experiment / "initials" / name
        if dest.exists():
            raise FileExistsError(f"Initial ID already exists: {dest}")
        temporary = experiment / "initials" / f".importing-{uuid4().hex}"
        temporary.mkdir(parents=True)
        try:
            shutil.copy2(code, temporary / "solution.py")
            if source.is_dir():
                for filename in ("request.json", "response.json", "response.txt", "reasoning.txt", "prompt.txt", "run.json"):
                    if (source / filename).exists():
                        shutil.copy2(source / filename, temporary / filename)
                # New manager artifacts carry a full evaluation signature.
                if (source / "dev.json").exists():
                    result = read(source / "dev.json")
                    if result.get("signature", {}).get("solution_sha256") == digest(code):
                        save(temporary / "dev.json", result)
                        provenance["evaluation_reused"] = True
                # Import this project's earlier TSP run only when the enclosing
                # manifest proves current data hashes and the initial code/result.
                if not provenance["evaluation_reused"] and source.name == "initial" and (source.parent / "run.json").exists():
                    old = read(source.parent / "run.json")
                    info = ev.describe(task, data_dir)
                    artifacts = old.get("initial_hashes", {})
                    current_env = environment()
                    old_env = old.get("environment", {})
                    same_env = (old_env.get("python") == current_env["python"] and all(
                        old_env.get("packages", {}).get(n) == current_env["packages"][n] for n in single.PACKAGES))
                    if (old.get("schema_version") == 2 and old.get("task", {}).get("hashes") == info["hashes"]
                            and same_env
                            and artifacts.get("solution.py") == digest(code)
                            and (source / "dev.json").exists()
                            and artifacts.get("dev.json") == digest(source / "dev.json")
                            and old["settings"]["cpu_num"] == 1):
                        settings = {"timeout_seconds": old["settings"]["timeout"]}
                        result = ev.summarize(read(source / "dev.json")["results"])
                        result["signature"] = ev.signature(info, digest(code), settings)
                        save(temporary / "dev.json", result)
                        save(temporary / "legacy_manifest.json", old)
                        provenance["evaluation_reused"] = True
            provenance["artifact_hashes"] = hashes(temporary)
            save(temporary / "provenance.json", provenance)
            temporary.rename(dest)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return dest


def task_and_plan(folder):
    folder = Path(folder).resolve()
    cfg = load(folder)
    task = ev.describe(cfg["task"], cfg["data_dir"], cfg["evaluation"]["unsplit_policy"])
    plan = []
    combinations = list(itertools.product(cfg["initials"], cfg["parent_policies"]))
    # Round-robin conditions for each step, keeping all branches independent.
    for initial, policy in combinations:
        for repeat in range(1, cfg["repeats"] + 1):
            for step in range(1, cfg["steps"] + 1):
                for condition_index, condition in enumerate(cfg["prompts"]["conditions"]):
                    seed = cfg["seeds"]["base"] + (repeat - 1) * cfg["steps"] + step - 1
                    if not cfg["seeds"]["paired_across_conditions"]:
                        seed += condition_index * cfg["repeats"] * cfg["steps"]
                    branch = f"branches/{initial['id']}/{policy}/{condition}/repeat_{repeat:03d}"
                    plan.append({"id": f"{branch}/step_{step:03d}", "branch": branch,
                        "initial": initial["id"], "policy": policy, "condition": condition,
                        "repeat": repeat, "step": step, "seed": seed})
    return cfg, task, plan


def diff(left, right):
    left, right = Path(left), Path(right)
    for folder in (left, right):
        load(folder)
    a, b = hashes(left), hashes(right)
    output = []
    for name in sorted(set(a) | set(b)):
        if name in ("version.json", "draft.json", "source.yaml") or a.get(name) == b.get(name):
            continue
        if name.endswith((".txt", ".yaml", ".md")):
            first = (left / name).read_text().splitlines(keepends=True) if name in a else []
            second = (right / name).read_text().splitlines(keepends=True) if name in b else []
            output.extend(difflib.unified_diff(first, second, fromfile=f"{left.name}/{name}", tofile=f"{right.name}/{name}"))
        else:
            output.append(f"{name}: {a.get(name, '(absent)')} -> {b.get(name, '(absent)')}\n")
    return "".join(output) or "No definition differences.\n"
