from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str((ROOT / "artifacts" / ".mplconfig").resolve()))
(ROOT / "artifacts" / ".mplconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str((ROOT / "artifacts" / ".cache").resolve()))
(ROOT / "artifacts" / ".cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build plots for assignment report.")
    parser.add_argument("--summary-csv", type=str, default="artifacts/summary.csv")
    parser.add_argument("--plots-dir", type=str, default="artifacts/plots")
    return parser.parse_args()


def plot_n_mels(summary: pd.DataFrame, plots_dir: Path) -> None:
    part = summary[summary["experiment_type"] == "n_mels"].sort_values("n_mels")
    if part.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(part["n_mels"], part["test_accuracy"], marker="o")
    ax.set_title("Test Accuracy vs n_mels")
    ax.set_xlabel("n_mels")
    ax.set_ylabel("test_accuracy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "n_mels_vs_test_accuracy.png", dpi=180)


def plot_groups(summary: pd.DataFrame, plots_dir: Path) -> None:
    part = summary[summary["experiment_type"] == "groups"].sort_values("groups")
    if part.empty:
        return

    for metric in ["mean_epoch_time_sec", "params", "flops", "test_accuracy"]:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(part["groups"], part[metric], marker="o")
        ax.set_title(f"{metric} vs groups")
        ax.set_xlabel("groups")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / f"groups_vs_{metric}.png", dpi=180)


def plot_train_loss_curves(summary: pd.DataFrame, plots_dir: Path) -> None:
    n_mels_rows = summary[summary["experiment_type"] == "n_mels"]
    if n_mels_rows.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    for _, row in n_mels_rows.iterrows():
        history = pd.read_csv(row["history_csv"])
        ax.plot(history["epoch"], history["train_loss"], marker="o", label=row["run_name"])
    ax.set_title("Train Loss Curves (n_mels runs)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("train_loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "train_loss_n_mels_runs.png", dpi=180)


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.summary_csv)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_n_mels(summary, plots_dir)
    plot_groups(summary, plots_dir)
    plot_train_loss_curves(summary, plots_dir)
    print(f"Saved plots to: {plots_dir}")


if __name__ == "__main__":
    main()
