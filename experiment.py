"""Manage versioned CO-Bench experiments. See RUN_EXPERIMENTS.md."""

import argparse
import json
from pathlib import Path
import sys

import yaml

from experiment_manager import definitions as defs
from experiment_manager import engine
from experiment_manager.reports import report
from experiment_manager.storage import ROOT, lock, read, save


def initialize(folder, source, task, initial_id):
    folder = Path(folder).resolve()
    if folder.exists():
        raise FileExistsError(f"Experiment already exists: {folder}")
    defs.identifier(folder.name)
    folder.mkdir(parents=True)
    initial = defs.import_initial(folder, initial_id, source, task)
    draft = folder / "drafts/next"
    prompts = draft / "prompts/interventions"
    prompts.mkdir(parents=True)
    (draft / "prompts/refine.txt").write_text((ROOT / "prompts/flow_refine.txt").read_text())
    interventions = {"support": "I suggest continuing to refine the current algorithmic approach.",
                     "neutral": "", "switch": "I suggest exploring a different algorithmic approach."}
    for name, text in interventions.items():
        (prompts / f"{name}.txt").write_text(text)
    cfg = {"schema_version": 1, "name": folder.name, "task": task, "data_dir": str(ROOT / "data"),
           "initials": [{"id": initial_id, "directory": f"../../initials/{initial_id}"}],
           "parent_policies": ["initial"], "repeats": 9, "steps": 1,
           "prompts": {"template": "prompts/refine.txt", "conditions": {
               n: f"prompts/interventions/{n}.txt" for n in interventions}},
           "generation": defs.DEFAULT_GENERATION, "seeds": {"base": 101, "paired_across_conditions": True},
           "evaluation": defs.DEFAULT_EVALUATION}
    # next is two levels beneath the experiment directory.
    (draft / "experiment.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    (draft / "README.md").write_text("# 実験の目的\n\n同じ初期解に対する3種類の介入の影響を比較する。\n\n"
        "初期解の選定理由、仮説、主指標、変更理由を実行前に記入する。\n")
    save(draft / "draft.json", {"parent_version": None})
    (folder / "README.md").write_text("# 実験管理\n\n編集は drafts/、確定した設定は versions/、実行結果は runs/ に保存する。\n"
        "操作方法はリポジトリの RUN_EXPERIMENTS.md を参照。\n")
    defs.load(draft)
    return draft


def preview(folder, output):
    cfg, task, _ = defs.task_and_plan(folder)
    folder, output = Path(folder).resolve(), Path(output)
    if output.exists():
        raise FileExistsError(f"Preview output already exists: {output}")
    output.mkdir(parents=True)
    item = cfg["initials"][0]
    code = (folder / item["directory"] / "solution.py").read_text()
    cached = engine.cached_initial(folder, cfg, task, item)
    feedback = cached["dev_feedback"] if cached else "[Initial dev evaluation pending; no evaluation was performed by plan.]"
    for condition, path in cfg["prompts"]["conditions"].items():
        text = (folder / cfg["prompts"]["template"]).read_text().format(
            problem_description=task["problem_description"], previous_code=code, dev_feedback=feedback,
            intervention=(folder / path).read_text(), timeout=cfg["evaluation"]["timeout_seconds"])
        (output / f"{condition}.txt").write_text(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Create an editable experiment; no model/evaluation calls")
    init.add_argument("experiment", type=Path)
    init.add_argument("--source", type=Path, required=True)
    init.add_argument("--task", default="Travelling salesman problem")
    init.add_argument("--initial-id", default="qwen_tsp_01")
    imp = commands.add_parser("import-initial", help="Import code and available evaluation provenance")
    imp.add_argument("experiment", type=Path)
    imp.add_argument("initial_id")
    imp.add_argument("--source", type=Path, required=True)
    imp.add_argument("--task", required=True)
    imp.add_argument("--data-dir", type=Path, default=ROOT / "data")
    freeze = commands.add_parser("freeze", help="Publish an immutable definition version")
    freeze.add_argument("draft", type=Path)
    freeze.add_argument("--version", required=True)
    freeze.add_argument("--reason", required=True)
    freeze.add_argument("--parent")
    fork = commands.add_parser("fork", help="Copy a published version to a new editable draft")
    fork.add_argument("version", type=Path)
    fork.add_argument("--draft", required=True)
    difference = commands.add_parser("diff", help="Compare configuration, prompts and initial artifacts")
    difference.add_argument("left", type=Path)
    difference.add_argument("right", type=Path)
    plan = commands.add_parser("plan", help="Validate and display exact new-generation counts")
    plan.add_argument("definition", type=Path)
    plan.add_argument("--preview-dir", type=Path)
    prepare = commands.add_parser("prepare", help="Snapshot a published version; no model/evaluation calls")
    prepare.add_argument("version", type=Path)
    prepare.add_argument("--run-id")
    listing = commands.add_parser("list", help="List saved versions and runs")
    listing.add_argument("experiment", type=Path)
    for name in ("run", "resume", "check", "report"):
        command = commands.add_parser(name)
        command.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            print(initialize(args.experiment, args.source, args.task, args.initial_id))
        elif args.command == "import-initial":
            print(defs.import_initial(args.experiment, args.initial_id, args.source, args.task, args.data_dir))
        elif args.command == "freeze":
            print(defs.freeze(args.draft, args.version, args.reason, args.parent))
        elif args.command == "fork":
            print(defs.fork(args.version, args.draft))
        elif args.command == "diff":
            print(defs.diff(args.left, args.right), end="")
        elif args.command == "plan":
            print(json.dumps(engine.plan_summary(args.definition), ensure_ascii=False, indent=2))
            if args.preview_dir:
                preview(args.definition, args.preview_dir)
                print(f"Prompt previews saved to {args.preview_dir}")
            print("No model requests or solver evaluations were performed.")
        elif args.command == "prepare":
            print(engine.prepare(args.version, args.run_id))
            print("No model requests or solver evaluations were performed.")
        elif args.command == "list":
            versions = []
            for path in sorted((args.experiment / "versions").glob("*/version.json")):
                meta = defs.verify_version(path.parent)
                versions.append({k: meta[k] for k in ("version", "parent_version", "reason", "created_at")})
            runs = [{"run": p.parent.name, **{k: read(p)[k] for k in ("version", "status", "created_at")}}
                    for p in sorted((args.experiment / "runs").glob("*/manifest.json"))]
            print(json.dumps({"versions": versions, "runs": runs}, ensure_ascii=False, indent=2))
        elif args.command in ("run", "resume"):
            return engine.run(args.run_dir)
        else:
            with lock(args.run_dir):
                engine.verify(args.run_dir, runtime=args.command == "check", data=args.command == "check")
                print(json.dumps(report(args.run_dir, figures=args.command == "report"), ensure_ascii=False, indent=2))
            print("No model requests or solver evaluations were performed.")
        return 0
    except KeyboardInterrupt:
        print("Interrupted. Saved responses/checkpoints will be reused.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Experiment stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
