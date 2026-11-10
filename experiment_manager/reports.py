"""Rebuildable tables and plots; reporting never selects the next parent."""

import csv
from collections import defaultdict
from pathlib import Path
import statistics

from experiment_manager.storage import read, save


def write_csv(path, rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def report(folder, figures=True):
    from experiment_manager.engine import verify
    folder = Path(folder)
    state, cfg, task, plan = verify(folder)
    output = folder / "reports"
    output.mkdir(exist_ok=True)
    rows, usage = [], []
    branches, grouped = defaultdict(list), defaultdict(list)
    for item in plan:
        record = state["candidates"].get(item["id"])
        if not record or record["status"] != "evaluated":
            continue
        result = read(folder / item["id"] / "dev.json")
        parent = read(folder / record["parent"] / "dev.json")
        initial = read(folder / "initials" / item["initial"] / "dev.json")
        row = {**item, "parent": record["parent"], "dev_score": result["dev_score"],
               "initial_dev_score": initial["dev_score"], "parent_dev_score": parent["dev_score"],
               "delta_from_initial": result["dev_score"] - initial["dev_score"],
               "delta_from_parent": result["dev_score"] - parent["dev_score"],
               "instances": result["instances"], "failed": result["failed"],
               "all_cases_failed": result["failed"] == result["instances"],
               "generation_seconds": record.get("generation_seconds", ""),
               "evaluation_seconds": result.get("evaluation_seconds", ""),
               "code_error": record.get("code_error") or "", "finish_reason": record.get("finish_reason", "")}
        rows.append(row)
        branches[item["branch"]].append(row)
        grouped[(item["initial"], item["policy"], item["condition"])].append(row)
        usage.append({"id": item["id"], "seed": item["seed"],
                      "generation_seconds": record.get("generation_seconds", ""), **(record.get("usage") or {})})
    branch_summary = {}
    for branch, values in branches.items():
        first, last = values[0], values[-1]
        best_id, best_score = f"initials/{first['initial']}", first["initial_dev_score"]
        for value in values:
            if value["dev_score"] > best_score:
                best_id, best_score = value["id"], value["dev_score"]
        branch_summary[branch] = {"completed_steps": len(values), "latest": last["id"],
            "latest_dev_score": last["dev_score"], "final": last["id"] if last["step"] == cfg["steps"] else None,
            "best_observed": best_id, "best_observed_dev_score": best_score}
    comparisons = []
    for (initial, policy, condition), values in grouped.items():
        count = len(values)
        comparisons.append({"initial": initial, "policy": policy, "condition": condition, "candidates": count,
            "mean_dev_score": statistics.mean(v["dev_score"] for v in values),
            "median_dev_score": statistics.median(v["dev_score"] for v in values),
            "mean_delta_from_initial": statistics.mean(v["delta_from_initial"] for v in values),
            "improvement_rate": sum(v["delta_from_initial"] > 0 for v in values) / count,
            "all_cases_failed_rate": sum(v["all_cases_failed"] for v in values) / count,
            "case_failure_rate": sum(v["failed"] for v in values) / sum(v["instances"] for v in values)})
    summary = {"status": state["status"], "version": state["version"],
               "new_generations_planned": len(plan), "requests_attempted": len(state["candidates"]),
               "candidates_evaluated": len(rows), "initials": state["initials"],
               "branches": branch_summary, "comparisons": comparisons,
               "note": "Steps in a chain and case failures are not independent experiment repetitions."}
    save(output / "summary.json", summary)
    write_csv(output / "scores.csv", rows)
    write_csv(output / "usage.csv", usage)
    write_csv(output / "comparisons.csv", comparisons)
    if figures and rows:
        plot(output / "figures", rows)
    return summary


def plot(output, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(exist_ok=True)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["initial"], row["policy"])].append(row)
    for (initial, policy), values in groups.items():
        fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
        conditions = list(dict.fromkeys(v["condition"] for v in values))
        for index, condition in enumerate(conditions):
            subset = [v for v in values if v["condition"] == condition]
            color = f"C{index % 10}"
            if policy == "initial":
                ax.scatter([index] * len(subset), [v["dev_score"] for v in subset],
                           label=condition, alpha=.65, color=color)
            else:
                repeats = sorted({v["repeat"] for v in subset})
                for position, repeat in enumerate(repeats):
                    chain = sorted((v for v in subset if v["repeat"] == repeat), key=lambda v: v["step"])
                    ax.plot([0] + [v["step"] for v in chain],
                            [chain[0]["initial_dev_score"]] + [v["dev_score"] for v in chain],
                            "o-", color=color, alpha=.7, label=condition if position == 0 else None)
        ax.axhline(values[0]["initial_dev_score"], color="gray", linestyle="--", label="Initial")
        if policy == "initial":
            ax.set_xticks(range(len(conditions)), conditions)
            ax.set_xlabel("Intervention condition (independent generations)")
        else:
            ax.set_xlabel("New submission step (0 = initial)")
        ax.set_ylabel("Dev score")
        ax.set_title(f"{initial}: parent={policy}")
        ax.grid(alpha=.2)
        ax.legend()
        for extension in ("png", "svg", "pdf"):
            fig.savefig(output / f"{initial}_{policy}.{extension}", dpi=180)
        plt.close(fig)
