from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import build_dataloaders
from src.model import SpeechYesNoCNN, count_trainable_params, estimate_flops
from src.train import EpochMetrics, train_one_experiment


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train yes/no SpeechCommands models.")
    parser.add_argument("--data-root", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--n-mels-list", type=str, default="20,40,80")
    parser.add_argument("--groups-list", type=str, default="2,4,8,16")
    parser.add_argument("--baseline-n-mels", type=int, default=80)
    parser.add_argument("--no-download", action="store_true", help="Do not download dataset if absent.")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def history_to_frame(history: list[EpochMetrics], run_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_name": run_name,
                "epoch": m.epoch,
                "train_loss": m.train_loss,
                "val_accuracy": m.val_accuracy,
                "epoch_time_sec": m.epoch_time_sec,
            }
            for m in history
        ]
    )


def run_single(
    *,
    run_name: str,
    n_mels: int,
    groups: int,
    loaders,
    args: argparse.Namespace,
    device: torch.device,
    logs_dir: Path,
    checkpoints_dir: Path,
) -> dict:
    model = SpeechYesNoCNN(n_mels=n_mels, groups=groups)
    params = count_trainable_params(model)
    flops = estimate_flops(model, input_shape=(1, 16000), device=device)
    history, test_accuracy = train_one_experiment(
        model=model,
        train_loader=loaders.train,
        val_loader=loaders.val,
        test_loader=loaders.test,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    best_val = max(x.val_accuracy for x in history)
    mean_epoch_time = sum(x.epoch_time_sec for x in history) / len(history)

    history_df = history_to_frame(history, run_name=run_name)
    history_path = logs_dir / f"{run_name}_history.csv"
    history_df.to_csv(history_path, index=False)

    checkpoint_path = checkpoints_dir / f"{run_name}.pt"
    torch.save(model.state_dict(), checkpoint_path)

    return {
        "run_name": run_name,
        "n_mels": n_mels,
        "groups": groups,
        "epochs": args.epochs,
        "params": params,
        "flops": flops,
        "best_val_accuracy": best_val,
        "test_accuracy": test_accuracy,
        "mean_epoch_time_sec": mean_epoch_time,
        "history_csv": str(history_path),
        "checkpoint": str(checkpoint_path),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    logs_dir = output_dir / "logs"
    checkpoints_dir = output_dir / "checkpoints"
    logs_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    loaders = build_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        fixed_num_samples=16000,
        download=not args.no_download,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples,
    )

    summary_rows: list[dict] = []

    for n_mels in parse_int_list(args.n_mels_list):
        run_name = f"mels_{n_mels}_groups_1"
        print(f"Running: {run_name}")
        row = run_single(
            run_name=run_name,
            n_mels=n_mels,
            groups=1,
            loaders=loaders,
            args=args,
            device=device,
            logs_dir=logs_dir,
            checkpoints_dir=checkpoints_dir,
        )
        row["experiment_type"] = "n_mels"
        summary_rows.append(row)

    for groups in parse_int_list(args.groups_list):
        run_name = f"mels_{args.baseline_n_mels}_groups_{groups}"
        print(f"Running: {run_name}")
        row = run_single(
            run_name=run_name,
            n_mels=args.baseline_n_mels,
            groups=groups,
            loaders=loaders,
            args=args,
            device=device,
            logs_dir=logs_dir,
            checkpoints_dir=checkpoints_dir,
        )
        row["experiment_type"] = "groups"
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows).sort_values(["experiment_type", "n_mels", "groups"])
    summary_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
