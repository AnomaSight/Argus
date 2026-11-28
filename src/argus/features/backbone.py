"""Frozen WideResNet-50 backbone returning layer2 and layer3 (task A1.3)."""

from __future__ import annotations

import os
import re

import torch
from torch import nn
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

WEIGHTS = "IMAGENET1K_V1"
_DEVICE_RE = re.compile(r"cpu|mps|cuda(:\d+)?")


def available_devices() -> list[str]:
    """Device strings usable on this machine, preferred first: cuda[:N], mps, cpu."""
    out = []
    if torch.cuda.is_available():
        out += ["cuda"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if torch.backends.mps.is_available():
        out.append("mps")
    return out + ["cpu"]


def resolve_device(device: str = "auto") -> torch.device:
    """C3's device semantics: "cuda", "cuda:N", "mps", "cpu" or "auto".

    "auto" uses $ARGUS_DEVICE if it is set (and not "auto"), otherwise the first available of
    CUDA, MPS and CPU. Raises ValueError if the requested device does not exist here, naming
    where the request came from, instead of failing later at ``.to()``.
    """
    device = str(device).strip()
    source, fix = f"device={device!r}", "pass device='auto' or one of"
    if device == "auto":
        env = os.environ.get("ARGUS_DEVICE", "").strip()
        if not env or env == "auto":
            return torch.device(available_devices()[0])
        device, source = env, f"ARGUS_DEVICE={env!r}"
        fix = "unset ARGUS_DEVICE (or fix it in your shell or .env) or set it to one of"
    choices = ", ".join(available_devices())
    if not _DEVICE_RE.fullmatch(device):
        raise ValueError(
            f"{source} is not a device; expected cuda, cuda:N, mps, cpu or auto. "
            f"To fix: {fix} {choices}."
        )
    kind, _, index = device.partition(":")
    if kind == "cuda" and not torch.cuda.is_available():
        problem = "CUDA is not available on this machine"
    elif kind == "cuda" and index and int(index) >= torch.cuda.device_count():
        problem = f"there are only {torch.cuda.device_count()} CUDA device(s)"
    elif kind == "mps" and not torch.backends.mps.is_available():
        problem = "MPS (Apple Silicon) is not available on this machine"
    else:
        return torch.device(device)
    raise ValueError(f"{source} asks for {device}, but {problem}. To fix: {fix} {choices}.")


def device_name(device: torch.device) -> str:
    """C2's device string, e.g. "cuda:Tesla T4", "mps" or "cpu"."""
    if device.type == "cuda":
        return f"cuda:{torch.cuda.get_device_name(device)}"
    return device.type


class Backbone(nn.Module):
    """wide_resnet50_2 up to layer3, always in eval mode, no gradients. layer4 is never run.

    ``weights="IMAGENET1K_V1"`` is what the official PatchCore code loads
    (``models.wide_resnet50_2(pretrained=True)``). Nothing else is accepted: torchvision's
    ``IMAGENET1K_V2`` is a differently trained network, so its features are not PatchCore's.
    ``weights=None`` gives random weights, for tests only (no download).

    Returns the outputs of layer2 and layer3, the same tensors the official
    ``NetworkFeatureAggregator`` takes from forward hooks on the full network.
    """

    def __init__(self, weights: str | None = WEIGHTS):
        super().__init__()
        if weights not in (WEIGHTS, None):
            raise ValueError(
                f"weights={weights!r}: only {WEIGHTS!r} (PatchCore's weights) or None "
                "(random, tests only) is supported"
            )
        net = wide_resnet50_2(weights=Wide_ResNet50_2_Weights[weights] if weights else None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.layer2, self.layer3 = net.layer2, net.layer3
        self.eval().requires_grad_(False)

    def train(self, mode: bool = True) -> Backbone:
        # Always eval: in train mode BatchNorm would update its ImageNet statistics from our
        # images and the features would drift silently, even with requires_grad off.
        return super().train(False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        l2 = self.layer2(self.stem(x))
        return l2, self.layer3(l2)
