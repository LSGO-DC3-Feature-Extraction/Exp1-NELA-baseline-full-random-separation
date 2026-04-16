from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common import DATA_DIR, OUTPUT_DIR, configure_matplotlib, ensure_dirs, problem_output_dir, save_dataframe


SUMMARY_PATH = DATA_DIR / "experiment_summary.csv"
EPOCH_PATH = DATA_DIR / "epoch_progress.csv"
SLICE_PATH = DATA_DIR / "slice_progress.csv"
VECTOR_PATH = DATA_DIR / "vector_progress.csv"


def maybe_build_data() -> None:
    from build_report_data import main as build_data_main

    build_data_main()


def require_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [path for path in (SUMMARY_PATH, EPOCH_PATH, SLICE_PATH, VECTOR_PATH) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing normalized data tables. Run `python .\\pics\\build_report_data.py` first or use --build-data.\n"
            + "\n".join(str(path) for path in missing)
        )
    return (
        pd.read_csv(SUMMARY_PATH),
        pd.read_csv(EPOCH_PATH),
        pd.read_csv(SLICE_PATH),
        pd.read_csv(VECTOR_PATH),
    )


def annotate_bars(ax: plt.Axes, bars, values: list[float]) -> None:
    ymin, ymax = ax.get_ylim()
    offset = (ymax - ymin) * 0.015 if ymax > ymin else 0.1
    for bar, value in zip(bars, values):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            rotation=0,
        )


def plot_final_ranking(problem_df: pd.DataFrame, output_dir: Path) -> None:
    ranked = problem_df.sort_values(["optimized_y", "exp_id"], na_position="last").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(16, 8))
    bars = ax.bar(ranked["exp_label"], ranked["optimized_y"], color="#4C78A8")
    annotate_bars(ax, bars, ranked["optimized_y"].tolist())
    ax.set_title(f"Problem {int(ranked['problem_id'].iloc[0])}: Final Optimized Value Ranking")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("optimized_y")
    ax.tick_params(axis="x", rotation=35)
    fig.savefig(output_dir / "01_final_ranking.png")
    plt.close(fig)


def plot_grouped_comparison(problem_df: pd.DataFrame, output_dir: Path) -> None:
    grouped = problem_df.sort_values(["slice_length", "agent", "use_fe", "exp_id"]).copy().reset_index(drop=True)
    slice_lengths = sorted(grouped["slice_length"].dropna().unique().tolist())
    variants = grouped[["agent", "use_fe"]].drop_duplicates().sort_values(["agent", "use_fe"]).values.tolist()
    x_positions = list(range(len(slice_lengths)))
    width = 0.8 / max(len(variants), 1)
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
    hatches = {True: "///", False: ""}

    fig, ax = plt.subplots(figsize=(16, 8))
    for idx, (agent, use_fe) in enumerate(variants):
        subset = grouped[(grouped["agent"] == agent) & (grouped["use_fe"] == use_fe)]
        heights = []
        for slice_length in slice_lengths:
            match = subset[subset["slice_length"] == slice_length]
            heights.append(match["optimized_y"].iloc[0] if not match.empty else math.nan)
        offsets = [x + (idx - (len(variants) - 1) / 2) * width for x in x_positions]
        ax.bar(
            offsets,
            heights,
            width=width,
            label=f"{agent} | {'FE on' if bool(use_fe) else 'FE off'}",
            color=colors[idx % len(colors)],
            hatch=hatches[bool(use_fe)],
        )
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(value) for value in slice_lengths])
    ax.set_xlabel("slice_length")
    ax.set_ylabel("optimized_y")
    ax.set_title(f"Problem {int(grouped['problem_id'].iloc[0])}: Grouped Comparison")
    ax.legend()
    fig.savefig(output_dir / "02_grouped_comparison.png")
    plt.close(fig)


def plot_epoch_progress(problem_summary: pd.DataFrame, problem_epoch: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 8))
    merged = problem_epoch.merge(
        problem_summary[["exp_id", "exp_label"]],
        on=["exp_id", "exp_label"],
        how="left",
    ).sort_values(["exp_id", "epoch"])
    for exp_label, exp_df in merged.groupby("exp_label"):
        exp_df = exp_df.sort_values("epoch")
        ax.plot(exp_df["epoch"], exp_df["epoch_mean_best"], marker="o", linewidth=2, label=exp_label)
        if {"epoch_best", "epoch_worst"}.issubset(exp_df.columns):
            ax.fill_between(
                exp_df["epoch"],
                exp_df["epoch_best"],
                exp_df["epoch_worst"],
                alpha=0.10,
            )
    ax.set_title(f"Problem {int(problem_summary['problem_id'].iloc[0])}: Optimization Evolution")
    ax.set_xlabel("epoch")
    ax.set_ylabel("epoch_mean_best")
    ax.legend()
    fig.savefig(output_dir / "03_epoch_progress.png")
    plt.close(fig)


def plot_vector_distance_evolution(problem_summary: pd.DataFrame, problem_vector: pd.DataFrame, output_dir: Path) -> bool:
    fig, ax = plt.subplots(figsize=(16, 8))
    if problem_vector.empty:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No vector reconstruction data is available for this problem.",
            ha="center",
            va="center",
            fontsize=16,
        )
        fig.savefig(output_dir / "04_vector_distance_evolution.png")
        plt.close(fig)
        return False

    merged = problem_vector.merge(
        problem_summary[["exp_id", "exp_label"]],
        on=["exp_id", "exp_label"],
        how="left",
    ).sort_values(["exp_id", "fes_anchor"])
    for exp_label, exp_df in merged.groupby("exp_label"):
        exp_df = exp_df.sort_values("fes_anchor")
        ax.plot(exp_df["fes_anchor"], exp_df["vector_l2_norm"], marker="o", linewidth=2, label=exp_label)
    ax.set_title(f"Problem {int(problem_summary['problem_id'].iloc[0])}: Full-Vector Distance Evolution")
    ax.set_xlabel("fes_anchor")
    ax.set_ylabel("vector_l2_norm")
    ax.legend()
    fig.savefig(output_dir / "04_vector_distance_evolution.png")
    plt.close(fig)
    return True


def plot_vector_path_ratio(problem_summary: pd.DataFrame, problem_vector: pd.DataFrame, output_dir: Path) -> bool:
    fig, ax = plt.subplots(figsize=(16, 8))
    if problem_vector.empty:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No vector path-ratio data is available for this problem.",
            ha="center",
            va="center",
            fontsize=16,
        )
        fig.savefig(output_dir / "05_vector_path_ratio.png")
        plt.close(fig)
        return False

    merged = problem_vector.merge(
        problem_summary[["exp_id", "exp_label"]],
        on=["exp_id", "exp_label"],
        how="left",
    ).sort_values(["exp_id", "fes_anchor"])
    for exp_label, exp_df in merged.groupby("exp_label"):
        exp_df = exp_df.sort_values("fes_anchor")
        final_ratio = exp_df["path_length_ratio"].iloc[-1]
        ax.plot(
            exp_df["fes_anchor"],
            exp_df["path_length_ratio"],
            marker="o",
            linewidth=2,
            label=f"{exp_label} | final={final_ratio:.2f}",
        )
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_title(f"Problem {int(problem_summary['problem_id'].iloc[0])}: Full-Vector Path Length Ratio")
    ax.set_xlabel("fes_anchor")
    ax.set_ylabel("path_length_ratio")
    ax.legend()
    fig.savefig(output_dir / "05_vector_path_ratio.png")
    plt.close(fig)
    return True


def cleanup_legacy_outputs(output_dir: Path) -> None:
    for name in (
        "04_time_comparison.png",
        "05_top3_slice_curve.png",
        "06_gain_vs_baseline.png",
    ):
        path = output_dir / name
        if path.exists():
            path.unlink()


def build_index(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for problem_id, problem_df in summary_df.groupby("problem_id"):
        best_df = problem_df.sort_values(["optimized_y", "exp_id"], na_position="last")
        rows.append(
            {
                "problem_id": int(problem_id),
                "experiment_count": int(len(problem_df)),
                "has_time_data": bool(problem_df["has_time_data"].fillna(False).any()),
                "best_exp_id": int(best_df.iloc[0]["exp_id"]),
                "best_optimized_y": float(best_df.iloc[0]["optimized_y"]),
            }
        )
    index_df = pd.DataFrame(
        rows,
        columns=["problem_id", "experiment_count", "has_time_data", "best_exp_id", "best_optimized_y"],
    )
    if not index_df.empty:
        index_df = index_df.sort_values("problem_id").reset_index(drop=True)
    return index_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-data", action="store_true", help="Build normalized data before plotting.")
    args = parser.parse_args()

    ensure_dirs()
    configure_matplotlib()
    if args.build_data:
        maybe_build_data()

    summary_df, epoch_df, slice_df, vector_df = require_data()
    if summary_df.empty:
        raise RuntimeError("No experiments were found in normalized data.")

    for problem_id, problem_summary in summary_df.groupby("problem_id"):
        problem_epoch = epoch_df[epoch_df["problem_id"] == problem_id].copy()
        problem_vector = vector_df[vector_df["problem_id"] == problem_id].copy()
        output_dir = problem_output_dir(int(problem_id))
        cleanup_legacy_outputs(output_dir)

        plot_final_ranking(problem_summary, output_dir)
        plot_grouped_comparison(problem_summary, output_dir)
        plot_epoch_progress(problem_summary, problem_epoch, output_dir)
        plot_vector_distance_evolution(problem_summary, problem_vector, output_dir)
        plot_vector_path_ratio(problem_summary, problem_vector, output_dir)

    index_df = build_index(summary_df)
    save_dataframe(index_df, OUTPUT_DIR / "index.csv")
    print(f"Saved plots for {summary_df['problem_id'].nunique()} problems into {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
