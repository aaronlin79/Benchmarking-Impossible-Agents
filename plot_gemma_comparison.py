"""Draw the AgentAbstain model comparison as a Matplotlib dot chart."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter


TEST_PAIRS = 52
TEST_REQUESTS = 104

# Recorded scores on the seed-42 test split. Counts are shared across models.
RUNS = [
    ("Na\u00efve Bayes", "Word-count baseline",
     {"correct_requests": 58, "auroc": 0.5824704142011834, "correct_pairs": 6, "pause_hits": 6}),
    ("Prompted Gemma", "Original Ollama prompt",
     {"correct_requests": 52, "auroc": 0.4604289940828402, "correct_pairs": 0, "pause_hits": 0}),
    ("Two-choice Gemma", "Before fine-tuning",
     {"correct_requests": 56, "auroc": 0.5445636094674556, "correct_pairs": 4, "pause_hits": 4}),
    ("Gemma with LoRA", "Fine-tuned on single requests",
     {"correct_requests": 59, "auroc": 0.6159393491124261, "correct_pairs": 7, "pause_hits": 9}),
    ("Gemma with matched pairs", "Fine-tuned on paired requests",
     {"correct_requests": 61, "auroc": 0.6640162721893491, "correct_pairs": 9, "pause_hits": 12}),
]


def plot_comparison(output):
    ink = "#182B36"
    muted = "#465D6A"
    grid = "#DFE7EB"
    accent = "#13756A"
    reference = "#7D8B92"
    marker_colors = [reference, "#8A97A0", "#69829A", "#346F9E", accent]
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.facecolor": "white",
        "text.color": ink,
    })
    fig = plt.figure(figsize=(13.5, 5.75), dpi=200)
    fig.text(0.043, 0.921, "Gemma training comparison", fontsize=24, fontweight="bold", va="top")
    fig.text(0.043, 0.842, "AgentAbstain  \u00b7  104 requests in 52 held-out pairs  \u00b7  Request text only",
             fontsize=12, color=muted, va="top")

    lefts = [0.28, 0.454, 0.628, 0.802]
    bottom, panel_height, panel_width = 0.247, 0.476, 0.15
    y_min, y_max = -0.49, 4.53
    def y_to_fig(y):
        return bottom + panel_height * (y - y_min) / (y_max - y_min)

    fig.add_artist(Rectangle(
        (0.031, y_to_fig(-0.44)),
        0.938,
        y_to_fig(0.44) - y_to_fig(-0.44),
        transform=fig.transFigure,
        color="#EAF5F2",
        linewidth=0,
        zorder=-1,
    ))
    fig.text(0.043, 0.754, "Model", fontsize=13.2, fontweight="bold", va="bottom")
    for index, (name, description, _) in enumerate(RUNS):
        y = y_to_fig(4 - index)
        best = index == len(RUNS) - 1
        fig.text(0.043, y + 0.007, name, fontsize=13.1,
                 color=accent if best else ink, fontweight="bold", va="center")
        fig.text(0.043, y - 0.027, description, fontsize=9.8, color=muted, va="center")

    def accuracy(data):
        return data["correct_requests"] / TEST_REQUESTS

    metrics = [
        ("Overall accuracy", "Correct requests", accuracy,
         (0.445, 0.657), [0.45, 0.50, 0.55, 0.60, 0.65],
         lambda val: f"{val:.1%}", lambda val: f"{val:.0%}"),
        ("AUROC", "Ranking score", lambda data: data["auroc"],
         (0.398, 0.795), [0.4, 0.5, 0.6, 0.7],
         lambda val: f"{val:.3f}", lambda val: f"{val:.1f}"),
        ("Both versions right", "Pairs out of 52", lambda data: data["correct_pairs"],
         (-0.18, 12.55), [0, 4, 8, 12],
         lambda val: f"{val}/{TEST_PAIRS}", lambda val: f"{val:.0f}"),
        ("Pause requests caught", "At up to 5% false alarms", lambda data: data["pause_hits"] / TEST_PAIRS,
         (-0.005, 0.342), [0, 0.10, 0.20, 0.30],
         lambda val: f"{val:.1%}", lambda val: f"{val:.0%}"),
    ]

    for x, (title, subtitle, value, limits, ticks, point_label, axis_label) in zip(lefts, metrics):
        ax = fig.add_axes([x, bottom, panel_width, panel_height], facecolor="none")
        fig.text(x, 0.754, title, fontsize=13.2, fontweight="bold", va="bottom")
        fig.text(x, 0.721, subtitle, fontsize=9.8, color=muted, va="bottom")
        ax.set(xlim=limits, ylim=(y_min, y_max), xticks=ticks, yticks=[])
        ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, pos, formatter=axis_label: formatter(val)))
        ax.grid(axis="x", color=grid, linewidth=0.7)
        ax.axvline(value(RUNS[0][2]), color=reference, linewidth=1.1, linestyle=(0, (1.5, 2.9)), zorder=1)
        for spine in ("top", "left", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(grid)
        ax.tick_params(axis="x", labelsize=9.4, colors=muted, length=3, pad=4)

        for index, (_, _, result) in enumerate(RUNS):
            y = 4 - index
            score = value(result)
            is_best = index == len(RUNS) - 1
            ax.scatter(
                score, y,
                marker="D" if is_best else "o",
                s=67 if is_best else 49,
                color="white" if index == 0 else marker_colors[index],
                edgecolors=marker_colors[index],
                linewidths=1.35,
                zorder=3,
                clip_on=False,
            )
            ax.annotate(point_label(score), (score, y), xytext=(8, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=10.7,
                        fontweight="bold" if is_best else "normal",
                        color=accent if is_best else ink, zorder=4,
                        bbox={"facecolor": "#EAF5F2" if is_best else "white",
                              "edgecolor": "none", "pad": 0.6})
    fig.add_artist(plt.Line2D([0.043, 0.957], [0.171, 0.171], transform=fig.transFigure,
                             color=grid, linewidth=0.85))
    fig.text(0.043, 0.13, "Dotted lines mark the na\u00efve Bayes score in each panel.",
             color=muted, fontsize=10.1, va="center")
    fig.text(0.043, 0.079, "At the selected decision cutoff, matched-pair Gemma got 17 of 52 pause requests right. "
             "Na\u00efve Bayes got 26.", color=muted, fontsize=10.1, va="center")

    fig.savefig(output, metadata={"Title": "Gemma training comparison on 52 held-out AgentAbstain pairs"})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("gemma-comparison.png"),
        help="Image to save (default: gemma-comparison.png beside this script)",
    )
    args = parser.parse_args()
    plot_comparison(args.output)
    print(f"Saved {args.output.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
