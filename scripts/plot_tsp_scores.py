"""Plot submitted TSP dev scores, without substituting best-so-far scores."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parents[1]
SERIES = {
    "support": ("方針支持", "#0072B2", "o"),
    "neutral": ("追加指示なし", "#009E73", "s"),
    "switch": ("方針転換", "#D55E00", "^"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, nargs="?", default=ROOT / "runs/tsp_qwen_latest_10_submissions")
    args = parser.parse_args()
    folder = args.run_dir.resolve()
    manifest = json.loads((folder / "run.json").read_text())
    with (folder / "scores.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    target = manifest["submissions_per_condition"]
    points = {}
    for condition in SERIES:
        selected = sorted((row for row in rows if row["condition"] == condition), key=lambda row: int(row["submission"]))
        if [int(row["submission"]) for row in selected] != list(range(1, target + 1)):
            raise ValueError(f"Incomplete or duplicate submission sequence: {condition}")
        for row in selected:
            result = json.loads((folder / row["candidate"] / "dev.json").read_text())
            if not math.isclose(float(row["dev_score"]), result["dev_score"], rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"CSV and candidate score differ: {row['candidate']}")
        points[condition] = selected

    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"font.size": 12, "axes.unicode_minus": False,
                         "pdf.fonttype": 42, "svg.fonttype": "path"})
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=180)
    fig.subplots_adjust(left=.09, right=.92, bottom=.17, top=.85)
    ax.set_facecolor("#FBFCFE")
    baseline = float(points["support"][0]["dev_score"])
    ax.axhline(baseline, color="#94A3B8", linestyle="--", linewidth=1, alpha=.7, zorder=1)
    for condition, (label, color, marker) in SERIES.items():
        selected = points[condition]
        x = [int(row["submission"]) for row in selected]
        y = [float(row["dev_score"]) for row in selected]
        ax.plot(x, y, label=label, color=color, marker=marker, linewidth=2.1,
                markersize=6.5, markeredgecolor="white", markeredgewidth=.7, zorder=3)
        offset = -12 if condition == "switch" else 9
        ax.annotate(f"{y[-1]:.3f}", (x[-1], y[-1]), xytext=(9, offset),
                    textcoords="offset points", color=color, fontsize=11, fontweight="bold")
    ax.set(xlabel="提出回数", ylabel="devスコア", xlim=(.8, target + .7), ylim=(-.035, 1.02))
    ax.set_xticks(range(1, target + 1))
    ax.yaxis.set_major_locator(MultipleLocator(.1))
    ax.grid(axis="y", color="#D9E1EB", linewidth=.8)
    ax.grid(axis="x", color="#E6EBF2", linewidth=.6, alpha=.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#B4BECD")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.015), ncols=3, frameon=False)
    fig.suptitle("TSP：フィードバック別のスコア推移", x=.09, y=.97, ha="left", fontsize=18, fontweight="bold")
    fig.text(.09, .055, "Qwen3.8-27B ｜ 各条件で直前の提出を使用 ｜ 提出1は共通初期解", color="#475569", fontsize=10)
    fig.text(.09, .025, "各提出のdevスコア（全272件）。失敗による0点を含み、最高スコアへの置き換えは行わない。", color="#64748B", fontsize=9)
    for suffix in ("png", "pdf", "svg"):
        output = folder / f"dev_score_vs_submissions.{suffix}"
        fig.savefig(output, facecolor="white", dpi=200)
        print(output)
    plt.close(fig)


if __name__ == "__main__":
    main()
