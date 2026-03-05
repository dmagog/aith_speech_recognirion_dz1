from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import torch
import torchaudio

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str((ROOT / "artifacts" / ".mplconfig").resolve()))
(ROOT / "artifacts" / ".mplconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str((ROOT / "artifacts" / ".cache").resolve()))
(ROOT / "artifacts" / ".cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

from melbanks import LogMelFilterBanks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare custom LogMelFilterBanks with torchaudio.")
    parser.add_argument("--wav-path", type=str, default=None, help="Path to wav file (16 kHz expected).")
    parser.add_argument(
        "--output-plot",
        type=str,
        default="artifacts/plots/logmel_comparison.png",
        help="Output plot path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.wav_path:
        signal, sr = torchaudio.load(args.wav_path)
        if sr != 16000:
            signal = torchaudio.functional.resample(signal, sr, 16000)
    else:
        torch.manual_seed(0)
        signal = torch.randn(1, 16000)

    melspec = torchaudio.transforms.MelSpectrogram(hop_length=160, n_mels=80)(signal)
    reference = torch.log(melspec + 1e-6)
    custom = LogMelFilterBanks()(signal)

    print(f"Reference shape: {tuple(reference.shape)}")
    print(f"Custom shape:    {tuple(custom.shape)}")
    print(f"allclose:        {torch.allclose(reference, custom)}")
    print(f"max abs diff:    {(reference - custom).abs().max().item():.8f}")

    plot_path = Path(args.output_plot)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    ref_img = reference[0].cpu().numpy()
    custom_img = custom[0].cpu().numpy()
    diff_img = (reference[0] - custom[0]).abs().cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    axes[0].imshow(ref_img, origin="lower", aspect="auto")
    axes[0].set_title("torchaudio log-mel")
    axes[1].imshow(custom_img, origin="lower", aspect="auto")
    axes[1].set_title("custom log-mel")
    axes[2].imshow(diff_img, origin="lower", aspect="auto")
    axes[2].set_title("|difference|")
    for axis in axes:
        axis.set_xlabel("frames")
        axis.set_ylabel("mels")
    fig.savefig(plot_path, dpi=180)
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
