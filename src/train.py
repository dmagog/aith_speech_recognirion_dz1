from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_accuracy: float
    epoch_time_sec: float


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = (torch.sigmoid(logits) >= 0.5).long()
    return (predictions == targets).float().mean().item()


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for waveforms, labels in loader:
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            logits = model(waveforms)
            predictions = (torch.sigmoid(logits) >= 0.5).long()
            correct += (predictions == labels).sum().item()
            total += labels.numel()
    return correct / max(total, 1)


def train_one_experiment(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> tuple[list[EpochMetrics], float]:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    history: list[EpochMetrics] = []

    for epoch in range(1, epochs + 1):
        model.train()
        start = time.perf_counter()
        running_loss = 0.0
        num_batches = 0
        for waveforms, labels in train_loader:
            waveforms = waveforms.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(waveforms)
            loss = criterion(logits, labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        epoch_time = time.perf_counter() - start
        train_loss = running_loss / max(num_batches, 1)
        val_accuracy = evaluate(model, val_loader, device=device)
        history.append(
            EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_accuracy=val_accuracy,
                epoch_time_sec=epoch_time,
            )
        )

    test_accuracy = evaluate(model, test_loader, device=device)
    return history, test_accuracy
