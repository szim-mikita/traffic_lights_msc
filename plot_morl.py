"""Plot evaluated MORL return vectors saved by train_morl.py."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def load_results(path: Path) -> tuple[str, tuple[str, ...], np.ndarray]:
    with path.open(encoding="utf-8") as file:
        results = json.load(file)

    components = tuple(results["environment"]["reward_components"])
    returns = np.asarray(results["evaluation"]["returns"], dtype=np.float64)
    if not components:
        raise ValueError(f"{path}: result contains no reward components.")
    if returns.ndim != 2 or returns.shape[1] != len(components) or not len(returns):
        raise ValueError(
            f"{path}: expected a non-empty return matrix with "
            f"{len(components)} objective columns."
        )
    if not np.all(np.isfinite(returns)):
        raise ValueError(f"{path}: evaluation returns must be finite.")
    label = results.get("policy", path.stem.removesuffix("_results"))
    return label, components, returns


def pareto_mask(returns: np.ndarray) -> np.ndarray:
    """Mark returns not dominated when larger reward values are preferred."""
    dominates = np.all(returns[:, None, :] >= returns[None, :, :], axis=2)
    dominates &= np.any(returns[:, None, :] > returns[None, :, :], axis=2)
    return ~np.any(dominates, axis=0)


def plot_results(
    results_files: list[Path], output_dir: Path, prefix: str, show: bool = False
) -> tuple[Path, Path]:
    runs = [load_results(path) for path in results_files]
    components = runs[0][1]
    if any(run[1] != components for run in runs[1:]):
        raise ValueError("All result files must have the same ordered reward components.")

    labels = [run[0] for run in runs]
    all_returns = np.concatenate([run[2] for run in runs], axis=0)
    all_labels = np.concatenate(
        [np.full(len(run[2]), index, dtype=np.int32) for index, run in enumerate(runs)]
    )
    front_mask = pareto_mask(all_returns)

    objective_count = len(components)
    figure, axes = plt.subplots(
        objective_count,
        objective_count,
        figsize=(4 * objective_count, 4 * objective_count),
        squeeze=False,
    )
    display_labels = [
        label if len(label) <= 24 else f"{label[:21]}..."
        for label in labels
    ]
    colors = plt.get_cmap("tab10")
    for row in range(objective_count):
        for column in range(objective_count):
            axis = axes[row, column]
            if row == column:
                for run_index, (_, _, returns) in enumerate(runs):
                    axis.hist(
                        returns[:, column],
                        bins="auto",
                        alpha=0.45,
                        color=colors(run_index % 10),
                        label=labels[run_index],
                    )
                axis.set_xlabel(components[column])
                axis.set_ylabel("Evaluation episodes")
            else:
                for run_index, (_, _, returns) in enumerate(runs):
                    axis.scatter(
                        returns[:, column],
                        returns[:, row],
                        color=colors(run_index % 10),
                        alpha=0.75,
                        label=labels[run_index],
                    )
                front_points = all_returns[front_mask]
                axis.scatter(
                    front_points[:, column],
                    front_points[:, row],
                    facecolors="none",
                    edgecolors="black",
                    linewidths=1.2,
                    label="Non-dominated",
                )
                axis.set_xlabel(components[column])
                axis.set_ylabel(components[row])
            axis.grid(True, alpha=0.25)

    handles, legend_labels = axes[0, -1].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=min(4, len(handles)),
        )
    figure.suptitle("MORL evaluation returns (higher is better)", y=1.04)
    figure.tight_layout(rect=(0, 0, 1, 0.91))

    distribution_figure, distribution_axes = plt.subplots(
        1, objective_count, figsize=(5 * objective_count, 4), squeeze=False
    )
    for objective, axis in enumerate(distribution_axes[0]):
        axis.boxplot(
            [run[2][:, objective] for run in runs],
            tick_labels=display_labels,
            showmeans=True,
        )
        axis.set_title(components[objective])
        axis.set_ylabel("Episode return")
        axis.tick_params(axis="x", labelrotation=25, labelsize=8)
        axis.grid(True, axis="y", alpha=0.25)
    distribution_figure.suptitle("MORL return distributions")
    distribution_figure.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    return_path = output_dir / f"{prefix}_pairs.png"
    distribution_path = output_dir / f"{prefix}_distributions.png"
    figure.savefig(return_path, dpi=200, bbox_inches="tight")
    distribution_figure.savefig(distribution_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)
    plt.close(distribution_figure)
    print(f"Saved objective-pair plot: {return_path}")
    print(f"Saved return distributions: {distribution_path}")
    return return_path, distribution_path


def plot_state_trace(trace_path: Path, output_path: Path) -> Path:
    with trace_path.open(encoding="utf-8") as file:
        trace = json.load(file)

    trajectories = trace["states"]
    if not trajectories or any(not points for points in trajectories.values()):
        raise ValueError(f"{trace_path}: state trajectories must be non-empty.")
    main_states = list(
        dict.fromkeys(
            point["state"]
            for points in trajectories.values()
            for point in points
            if point.get("kind", "other") == "main"
        )
    )
    if not main_states:
        raise ValueError(f"{trace_path}: trajectories contain no main states.")
    state_ids = {state: index for index, state in enumerate(main_states)}
    figure, axes = plt.subplots(
        1,
        len(trajectories),
        figsize=(8 * len(trajectories), max(5, 2 + 0.45 * len(main_states))),
        sharey=True,
    )
    axes = np.atleast_1d(axes)
    for axis, (policy, points) in zip(axes, trajectories.items()):
        times = [point["time"] for point in points]
        main_state = next(
            point["state"]
            for point in points
            if point.get("kind", "other") == "main"
        )
        states = []
        transition_spans = []
        for index, point in enumerate(points):
            if point.get("kind", "other") == "main":
                main_state = point["state"]
            states.append(state_ids[main_state])
            end_time = (
                points[index + 1]["time"]
                if index + 1 < len(points)
                else trace["simulation_time"]
            )
            if point.get("kind", "other") == "transition" and end_time > point["time"]:
                if transition_spans and transition_spans[-1][1] == point["time"]:
                    transition_spans[-1] = (transition_spans[-1][0], end_time)
                else:
                    transition_spans.append((point["time"], end_time))
        for start, end in transition_spans:
            axis.axvspan(
                start,
                end,
                facecolor="0.8",
                edgecolor="0.55",
                alpha=0.35,
                hatch="///",
                linewidth=0,
                zorder=0,
            )
        axis.step(times, states, where="post", linewidth=2, zorder=2)
        axis.set_title(policy)
        axis.set_xlabel("Simulation time (seconds)")
        axis.set_xlim(0, trace["simulation_time"])
        axis.set_yticks(range(len(main_states)))
        axis.set_yticklabels(
            [f"M{index}: {state}" for index, state in enumerate(main_states)],
            family="monospace",
            fontsize=8,
        )
        axis.grid(True, axis="x", alpha=0.3)
    axes[0].set_ylabel("Selectable main green state")
    figure.legend(
        handles=[
            Line2D([0], [0], color="C0", linewidth=2, label="Main state"),
            Patch(
                facecolor="0.8",
                edgecolor="0.55",
                hatch="///",
                label="Intergreen transition",
            ),
        ],
        loc="upper center",
        ncol=2,
    )
    figure.suptitle(
        f"Selectable main states and intergreen periods — seed {trace['seed']}",
        y=0.99,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved state trajectory plot: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path, help="PCN results JSON file(s).")
    parser.add_argument("--output-dir", type=Path, default=Path("plots"))
    parser.add_argument("--prefix", default="morl_returns")
    parser.add_argument("--show", action="store_true", help="Display plots interactively.")
    parser.add_argument(
        "--state-trace",
        type=Path,
        default=None,
        help="Optional trajectory JSON produced by compare_morl.py.",
    )
    args = parser.parse_args()
    plot_results(args.results, args.output_dir, args.prefix, show=args.show)
    if args.state_trace:
        plot_state_trace(
            args.state_trace,
            args.output_dir / f"{args.prefix}_state_trajectory.png",
        )


if __name__ == "__main__":
    main()
