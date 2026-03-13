from __future__ import annotations

import os
from typing import Any

import pytest
import torch
from pydantic import Field
from torch.utils.data import DataLoader, Subset

import pydantic_torch.nn as nn
from examples.vit import VisionTransformer

torchvision = pytest.importorskip("torchvision")
datasets = torchvision.datasets
transforms = torchvision.transforms


def vit_nano_mnist(num_classes: int = 10) -> VisionTransformer:
    # 28x28 with patch 7 => 4x4 = 16 patches (very small token count)
    return VisionTransformer(
        img_size=28,
        patch_size=7,
        in_chans=1,
        num_classes=num_classes,
        embed_dim=32,     # tiny
        depth=2,          # tiny
        num_heads=4,      # 32 % 4 == 0
        mlp_ratio=2.0,    # smaller MLP
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    )

def _build_mnist_loaders(root: str) -> tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    allow_download = os.getenv("PYDANTIC_TORCH_TEST_DOWNLOAD_MNIST", "0") == "1"

    try:
        train_ds = datasets.MNIST(root=root, train=True, download=allow_download, transform=transform)
        test_ds = datasets.MNIST(root=root, train=False, download=allow_download, transform=transform)
    except RuntimeError as exc:
        pytest.skip(
            "MNIST dataset not found locally. "
            "Set PYDANTIC_TORCH_TEST_DOWNLOAD_MNIST=1 to allow downloading during this test."
        )

    train_subset = Subset(train_ds, range(1024))
    test_subset = Subset(test_ds, range(512))

    train_loader = DataLoader(train_subset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=64, shuffle=False)
    return train_loader, test_loader


def test_train_simple_mnist() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = _build_mnist_loaders(root=".data")

    model = vit_nano_mnist().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    losses: list[float] = []
    for step, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        opt.step()
        opt.zero_grad()
        # losses.append(loss.item())

    # head = sum(losses[:5]) / len(losses[:5])
    # tail = sum(losses[-5:]) / len(losses[-5:])
    # assert tail <= head

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images).argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.numel()

    accuracy = correct / total
    assert accuracy >= 0.20
