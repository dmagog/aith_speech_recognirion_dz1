from __future__ import annotations

import torch
from torch import nn

from melbanks import LogMelFilterBanks


class SpeechYesNoCNN(nn.Module):
    def __init__(self, n_mels: int = 80, groups: int = 1) -> None:
        super().__init__()
        if 64 % groups != 0:
            raise ValueError(f"'groups' must divide 64, got {groups}")

        self.features = LogMelFilterBanks(n_mels=n_mels)
        self.encoder = nn.Sequential(
            nn.Conv1d(n_mels, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 64, kernel_size=3, padding=1, groups=groups, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.encoder(x)
        x = x.squeeze(-1)
        logits = self.classifier(x).squeeze(-1)
        return logits


def count_trainable_params(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def estimate_flops(
    model: nn.Module,
    input_shape: tuple[int, int] = (1, 16000),
    device: torch.device | str = "cpu",
) -> int:
    flops = 0
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    def conv1d_hook(module: nn.Conv1d, inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        nonlocal flops
        x = inputs[0]
        batch_size = x.shape[0]
        out_length = output.shape[-1]
        kernel = module.kernel_size[0]
        in_channels = module.in_channels
        out_channels = module.out_channels
        macs = batch_size * out_length * out_channels * (in_channels // module.groups) * kernel
        flops += 2 * macs

    def linear_hook(module: nn.Linear, inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        nonlocal flops
        x = inputs[0]
        batch_size = x.shape[0]
        macs = batch_size * module.in_features * module.out_features
        flops += 2 * macs

    for module in model.modules():
        if isinstance(module, nn.Conv1d):
            hooks.append(module.register_forward_hook(conv1d_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(*input_shape, device=device)
        _ = model(dummy)

    for hook in hooks:
        hook.remove()

    return int(flops)
