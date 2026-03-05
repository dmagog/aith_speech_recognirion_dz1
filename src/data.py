from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torchaudio
from torch.nn import functional as nnf
from torch.utils.data import DataLoader, Dataset, Subset
from torchaudio.datasets import SPEECHCOMMANDS


YES_NO_LABELS = {"yes": 1, "no": 0}


class YesNoSpeechCommands(Dataset):
    def __init__(self, root: str | Path, subset: str, download: bool = True) -> None:
        self.dataset = SPEECHCOMMANDS(root=str(root), subset=subset, download=download)
        self.indices: list[int] = []
        for index in range(len(self.dataset)):
            sample = self.dataset.get_metadata(index)
            label = sample[2]
            if label in YES_NO_LABELS:
                self.indices.append(index)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        waveform, sample_rate, label, *_ = self.dataset[self.indices[index]]
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        waveform = waveform.squeeze(0)
        target = YES_NO_LABELS[label]
        return waveform, target


def build_collate_fn(fixed_num_samples: int = 16000) -> Callable:
    def collate(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        waveforms: list[torch.Tensor] = []
        labels: list[int] = []
        for waveform, label in batch:
            if waveform.numel() < fixed_num_samples:
                padding = fixed_num_samples - waveform.numel()
                waveform = nnf.pad(waveform, (0, padding))
            elif waveform.numel() > fixed_num_samples:
                waveform = waveform[:fixed_num_samples]
            waveforms.append(waveform)
            labels.append(label)
        return torch.stack(waveforms, dim=0), torch.tensor(labels, dtype=torch.long)

    return collate


@dataclass
class SpeechCommandsLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader


def build_dataloaders(
    data_root: str | Path,
    batch_size: int = 64,
    num_workers: int = 2,
    fixed_num_samples: int = 16000,
    download: bool = True,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
) -> SpeechCommandsLoaders:
    collate_fn = build_collate_fn(fixed_num_samples=fixed_num_samples)

    train_dataset = YesNoSpeechCommands(data_root, subset="training", download=download)
    val_dataset = YesNoSpeechCommands(data_root, subset="validation", download=download)
    test_dataset = YesNoSpeechCommands(data_root, subset="testing", download=download)

    if max_train_samples is not None:
        train_dataset = Subset(train_dataset, range(min(max_train_samples, len(train_dataset))))
    if max_val_samples is not None:
        val_dataset = Subset(val_dataset, range(min(max_val_samples, len(val_dataset))))
    if max_test_samples is not None:
        test_dataset = Subset(test_dataset, range(min(max_test_samples, len(test_dataset))))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return SpeechCommandsLoaders(train=train_loader, val=val_loader, test=test_loader)
