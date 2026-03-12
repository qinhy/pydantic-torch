from __future__ import annotations

from typing import Any, Literal

import torch
from pydantic import Field

import pydantic_torch.nn as nn
from pydantic_torch.containers import ModuleList


class Projection(nn.Module):
    in_channels: int = Field(default=64, ge=1)
    out_channels: int = Field(default=64, ge=1)
    stride: int = Field(default=1, ge=1)

    conv: nn.Conv2d = Field(default=None)
    norm: nn.BatchNorm2d = Field(default=None)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self.conv = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=self.stride,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(num_features=self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv(x))


class BasicBlock(nn.Module):
    expansion: int = 1

    in_channels: int = Field(default=64, ge=1)
    out_channels: int = Field(default=64, ge=1)
    stride: int = Field(default=1, ge=1)

    conv1: nn.Conv2d = Field(default=None)
    norm1: nn.BatchNorm2d = Field(default=None)
    relu: nn.ReLU = Field(default=None)
    conv2: nn.Conv2d = Field(default=None)
    norm2: nn.BatchNorm2d = Field(default=None)
    shortcut: Any = Field(default=None)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.BatchNorm2d(num_features=self.out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.BatchNorm2d(num_features=self.out_channels)

        if self.in_channels == self.out_channels and self.stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = Projection(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                stride=self.stride,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.relu(x + residual)


class Bottleneck(nn.Module):
    expansion: int = 4

    in_channels: int = Field(default=64, ge=1)
    bottleneck_channels: int = Field(default=64, ge=1)
    stride: int = Field(default=1, ge=1)

    conv1: nn.Conv2d = Field(default=None)
    norm1: nn.BatchNorm2d = Field(default=None)
    conv2: nn.Conv2d = Field(default=None)
    norm2: nn.BatchNorm2d = Field(default=None)
    conv3: nn.Conv2d = Field(default=None)
    norm3: nn.BatchNorm2d = Field(default=None)
    relu: nn.ReLU = Field(default=None)
    shortcut: Any = Field(default=None)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        out_channels = self.bottleneck_channels * self.expansion

        self.conv1 = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.bottleneck_channels,
            kernel_size=1,
            bias=False,
        )
        self.norm1 = nn.BatchNorm2d(num_features=self.bottleneck_channels)
        self.conv2 = nn.Conv2d(
            in_channels=self.bottleneck_channels,
            out_channels=self.bottleneck_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.BatchNorm2d(num_features=self.bottleneck_channels)
        self.conv3 = nn.Conv2d(
            in_channels=self.bottleneck_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
        )
        self.norm3 = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU(inplace=True)

        if self.in_channels == out_channels and self.stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = Projection(
                in_channels=self.in_channels,
                out_channels=out_channels,
                stride=self.stride,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        x = self.norm3(self.conv3(x))
        return self.relu(x + residual)


class ResNet(nn.Module):
    num_classes: int = Field(default=1000, ge=1)
    variant: Literal["resnet18", "resnet50"] = Field(default="resnet18")

    stem_conv: nn.Conv2d = Field(default=None)
    stem_norm: nn.BatchNorm2d = Field(default=None)
    stem_relu: nn.ReLU = Field(default=None)
    stem_pool: nn.MaxPool2d = Field(default=None)
    stages: ModuleList = Field(default=None)
    head_pool: nn.AdaptiveAvgPool2d = Field(default=None)
    head: nn.Linear = Field(default=None)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self.stem_conv = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.stem_norm = nn.BatchNorm2d(num_features=64)
        self.stem_relu = nn.ReLU(inplace=True)
        self.stem_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if self.variant == "resnet18":
            self.stages = self._make_basic_stages()
            head_in_features = 512
        else:
            self.stages = self._make_bottleneck_stages()
            head_in_features = 2048

        self.head_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.head = nn.Linear(in_features=head_in_features, out_features=self.num_classes)

    def _make_basic_stages(self) -> ModuleList:
        return ModuleList(
            mods=[
                ModuleList(
                    mods=[
                        BasicBlock(in_channels=64, out_channels=64),
                        BasicBlock(in_channels=64, out_channels=64),
                    ]
                ),
                ModuleList(
                    mods=[
                        BasicBlock(in_channels=64, out_channels=128, stride=2),
                        BasicBlock(in_channels=128, out_channels=128),
                    ]
                ),
                ModuleList(
                    mods=[
                        BasicBlock(in_channels=128, out_channels=256, stride=2),
                        BasicBlock(in_channels=256, out_channels=256),
                    ]
                ),
                ModuleList(
                    mods=[
                        BasicBlock(in_channels=256, out_channels=512, stride=2),
                        BasicBlock(in_channels=512, out_channels=512),
                    ]
                ),
            ]
        )

    def _make_bottleneck_stages(self) -> ModuleList:
        return ModuleList(
            mods=[
                ModuleList(
                    mods=[
                        Bottleneck(in_channels=64, bottleneck_channels=64),
                        Bottleneck(in_channels=256, bottleneck_channels=64),
                        Bottleneck(in_channels=256, bottleneck_channels=64),
                    ]
                ),
                ModuleList(
                    mods=[
                        Bottleneck(in_channels=256, bottleneck_channels=128, stride=2),
                        Bottleneck(in_channels=512, bottleneck_channels=128),
                        Bottleneck(in_channels=512, bottleneck_channels=128),
                        Bottleneck(in_channels=512, bottleneck_channels=128),
                    ]
                ),
                ModuleList(
                    mods=[
                        Bottleneck(in_channels=512, bottleneck_channels=256, stride=2),
                        Bottleneck(in_channels=1024, bottleneck_channels=256),
                        Bottleneck(in_channels=1024, bottleneck_channels=256),
                        Bottleneck(in_channels=1024, bottleneck_channels=256),
                        Bottleneck(in_channels=1024, bottleneck_channels=256),
                        Bottleneck(in_channels=1024, bottleneck_channels=256),
                    ]
                ),
                ModuleList(
                    mods=[
                        Bottleneck(in_channels=1024, bottleneck_channels=512, stride=2),
                        Bottleneck(in_channels=2048, bottleneck_channels=512),
                        Bottleneck(in_channels=2048, bottleneck_channels=512),
                    ]
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_pool(self.stem_relu(self.stem_norm(self.stem_conv(x))))
        for stage in self.stages:
            for block in stage:
                x = block(x)
        x = self.head_pool(x)
        x = torch.flatten(x, 1)
        return self.head(x)


def resnet18(num_classes: int = 1000) -> ResNet:
    return ResNet(num_classes=num_classes, variant="resnet18")


def resnet50(num_classes: int = 1000) -> ResNet:
    return ResNet(num_classes=num_classes, variant="resnet50")


def main() -> None:
    x = torch.randn(2, 3, 224, 224)

    model18 = resnet18(num_classes=100)
    out18 = model18(x)
    print("resnet18 output shape:", tuple(out18.shape))

    model50 = resnet50(num_classes=100)
    out50 = model50(x)
    print("resnet50 output shape:", tuple(out50.shape))


if __name__ == "__main__":
    main()
