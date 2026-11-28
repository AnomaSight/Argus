"""Frozen WideResNet-50 backbone returning layer2 and layer3 (task A1.3)."""

from __future__ import annotations

import os

import torch
from torch import nn
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2


def resolve_device(device: str = "auto") -> torch.device:
    """ "auto" uses $ARGUS_DEVICE if set, otherwise the first available of CUDA, MPS and CPU."""
    if device == "auto":
        device = os.environ.get("ARGUS_DEVICE") or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    return torch.device(device)


def device_name(device: torch.device) -> str:
    """C2's device string, e.g. "cuda:Tesla T4", "mps" or "cpu"."""
    if device.type == "cuda":
        return f"cuda:{torch.cuda.get_device_name(device)}"
    return device.type


class Backbone(nn.Module):
    """wide_resnet50_2 up to layer3, eval mode, no gradients. layer4 is never run.

    weights="IMAGENET1K_V1" matches PatchCore's torchvision weights; None gives random
    weights (tests only, so no download is needed).
    """

    def __init__(self, weights: str | None = "IMAGENET1K_V1"):
        super().__init__()
        net = wide_resnet50_2(weights=Wide_ResNet50_2_Weights[weights] if weights else None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.layer2, self.layer3 = net.layer2, net.layer3
        self.eval().requires_grad_(False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        l2 = self.layer2(self.stem(x))
        return l2, self.layer3(l2)
