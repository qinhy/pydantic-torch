from __future__ import annotations

import os
from typing import Any, Literal, Union

import torch
from pydantic import Field, field_validator
from torch.utils.data import DataLoader, Subset

import pydantic_torch.nn as nn

class BasicBlock(nn.Module):
    expansion: int = 1

    in_channels: int = Field(default=64, ge=1)
    out_channels: int = Field(default=64, ge=1)
    stride: int = Field(default=1, ge=1)

    act: nn.Acts.types = Field(default_factory=lambda: nn.ReLU(inplace=True))
    conv1: nn.Conv2dNormAct = Field(default=None)
    conv2: nn.Conv2dNorm = Field(default=None)
    shortcut: Union[nn.Identity, nn.Conv2dNorm] = Field(default=nn.Identity())

    @field_validator("act", mode="before")
    @classmethod
    def parse_act(cls, v: Any) -> nn.Acts.types:
        return nn.Acts.parse(v)

    @field_validator("shortcut", mode="before")
    @classmethod
    def parse_shortcut(cls, v: Any) -> Union[nn.Identity, nn.Conv2dNorm]:
        return nn.Cls_parse(v, {"Identity": nn.Identity, "Conv2dNorm": nn.Conv2dNorm})

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)

        norm_dd = lambda c: dict(
            bias=False,
            norm=nn.BatchNorm2d(num_features=c, device=self.device, dtype=self.dtype),
            device=self.device,
            dtype=self.dtype,
        )

        norm_act_dd = lambda c: dict(
            **norm_dd(c),
            act=self.act.clone(device=self.device, dtype=self.dtype),
        )
        
        self.conv1 = nn.Conv2dNormAct(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            **norm_act_dd(self.out_channels),
        )
        self.conv2 = nn.Conv2dNorm(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            padding=1,
            **norm_dd(self.out_channels),
        )

        if self.in_channels == self.out_channels and self.stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2dNorm(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=self.stride,
                **norm_dd(self.out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv2(self.conv1(x))
        return self.act(x + residual)

class Bottleneck(nn.Module):
    expansion: int = 4

    in_channels: int = Field(default=64, ge=1)
    bottleneck_channels: int = Field(default=64, ge=1)
    stride: int = Field(default=1, ge=1)

    act: nn.Acts.types = Field(default_factory=lambda: nn.ReLU(inplace=True))
    conv1: nn.Conv2dNormAct = Field(default=None)
    conv2: nn.Conv2dNormAct = Field(default=None)
    conv3: nn.Conv2dNorm = Field(default=None)
    shortcut: Union[nn.Identity, nn.Conv2dNorm] = Field(default=nn.Identity())

    @field_validator("act", mode="before")
    @classmethod
    def parse_act(cls, v: Any) -> nn.Acts.types:
        return nn.Acts.parse(v)

    @field_validator("shortcut", mode="before")
    @classmethod
    def parse_shortcut(cls, v: Any) -> Union[nn.Identity, nn.Conv2dNorm]:
        if v is None:
            return v
        return nn.Cls_parse(v, {"Identity": nn.Identity, "Conv2dNorm": nn.Conv2dNorm})

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        out_channels = self.bottleneck_channels * self.expansion

        norm_dd = lambda c: dict(
            bias=False,
            norm=nn.BatchNorm2d(num_features=c, device=self.device, dtype=self.dtype),
            device=self.device,
            dtype=self.dtype,
        )

        norm_act_dd = lambda c: dict(
            **norm_dd(c),
            act=self.act.clone(device=self.device, dtype=self.dtype),
        )

        self.conv1 = nn.Conv2dNormAct(
            in_channels=self.in_channels,
            out_channels=self.bottleneck_channels,
            kernel_size=1,
            **norm_act_dd(self.bottleneck_channels),
        )

        self.conv2 = nn.Conv2dNormAct(
            in_channels=self.bottleneck_channels,
            out_channels=self.bottleneck_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            **norm_act_dd(self.bottleneck_channels),
        )

        self.conv3 = nn.Conv2dNorm(
            in_channels=self.bottleneck_channels,
            out_channels=out_channels,
            kernel_size=1,
            **norm_dd(out_channels),
        )

        if self.in_channels == out_channels and self.stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2dNorm(
                in_channels=self.in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=self.stride,
                **norm_dd(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv3(self.conv2(self.conv1(x)))
        return self.act(x + residual)

class ResidualStage(nn.Module):
    blocks: list[Union[BasicBlock, Bottleneck]] = Field(default_factory=list)

    @field_validator("blocks", mode="before")
    @classmethod
    def parse_blocks(cls, v: Any) -> list[Union[BasicBlock, Bottleneck]]:
        if isinstance(v, list):
            return [nn.Cls_parse(block, {"BasicBlock": BasicBlock, "Bottleneck": Bottleneck}) for block in v]
        return v

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        for index, block in enumerate(self.blocks):
            self.add_module(str(index), block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x

class ResNet(nn.Module):
    num_classes: int = Field(default=1000, ge=1)
    variant: Literal["resnet18", "resnet50"] = Field(default="resnet18")

    act: nn.Acts.types = Field(default_factory=lambda: nn.ReLU(inplace=True))
    stem: nn.Conv2dNormAct = Field(default=None)
    stem_pool: Union[nn.Identity, nn.MaxPool2d] = Field(default=None)
    stage1: ResidualStage = Field(default=None)
    stage2: ResidualStage = Field(default=None)
    stage3: ResidualStage = Field(default=None)
    stage4: ResidualStage = Field(default=None)
    head_pool: nn.AdaptiveAvgPool2d = Field(default=None)
    head: nn.Linear = Field(default=None)

    @field_validator("act", mode="before")
    @classmethod
    def parse_act(cls, v: Any) -> nn.Acts.types:
        return nn.Acts.parse(v)

    @field_validator("stem_pool", mode="before")
    @classmethod
    def parse_stem_pool(cls, v: Any) -> Union[nn.Identity, nn.MaxPool2d]:
        if v is None:
            return v
        return nn.Cls_parse(v, {"Identity": nn.Identity, "MaxPool2d": nn.MaxPool2d})
    
    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if self.stem is None:
            self.stem = nn.Conv2dNormAct(
                in_channels=3,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
                norm=nn.BatchNorm2d(num_features=64, device=self.device, dtype=self.dtype),
                act=self.act.clone(device=self.device, dtype=self.dtype),
                device=self.device,
                dtype=self.dtype,
            )
        if self.stem_pool is None:
            self.stem_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if self.variant == "resnet18":
            stages = self._make_basic_stages()
            head_in_features = 512
        else:
            stages = self._make_bottleneck_stages()
            head_in_features = 2048

        self.stage1, self.stage2, self.stage3, self.stage4 = stages
        self.head_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.head = nn.Linear(
            in_features=head_in_features,
            out_features=self.num_classes,
            device=self.device,
            dtype=self.dtype,
        )

    def _make_basic_stages(self) -> tuple[ResidualStage, ResidualStage, ResidualStage, ResidualStage]:
        args = lambda:dict(act=self.act.clone(device=self.device, dtype=self.dtype), device=self.device, dtype=self.dtype)
        return (
            ResidualStage(blocks=[
                    BasicBlock(in_channels=64, out_channels=64, **args()),
                    BasicBlock(in_channels=64, out_channels=64, **args()),
            ]),
            ResidualStage(blocks=[
                    BasicBlock(in_channels=64, out_channels=128, stride=2, **args()),
                    BasicBlock(in_channels=128, out_channels=128, **args()),
            ]),
            ResidualStage(blocks=[
                    BasicBlock(in_channels=128, out_channels=256, stride=2, **args()),
                    BasicBlock(in_channels=256, out_channels=256, **args()),
            ]),
            ResidualStage(blocks=[
                    BasicBlock(in_channels=256, out_channels=512, stride=2, **args()),
                    BasicBlock(in_channels=512, out_channels=512, **args()),
            ]),
        )

    def _make_bottleneck_stages(self) -> tuple[ResidualStage, ResidualStage, ResidualStage, ResidualStage]:
        args = lambda:dict(act=self.act.clone(device=self.device, dtype=self.dtype), device=self.device, dtype=self.dtype)
        return (
            ResidualStage(blocks=[
                    Bottleneck(in_channels=64, bottleneck_channels=64, **args()),
                    Bottleneck(in_channels=256, bottleneck_channels=64, **args()),
                    Bottleneck(in_channels=256, bottleneck_channels=64, **args()),
            ]),
            ResidualStage(blocks=[
                    Bottleneck(in_channels=256, bottleneck_channels=128, stride=2, **args()),
                    Bottleneck(in_channels=512, bottleneck_channels=128, **args()),
                    Bottleneck(in_channels=512, bottleneck_channels=128, **args()),
                    Bottleneck(in_channels=512, bottleneck_channels=128, **args()),
            ]),
            ResidualStage(blocks=[
                    Bottleneck(in_channels=512, bottleneck_channels=256, stride=2, **args()),
                    Bottleneck(in_channels=1024, bottleneck_channels=256, **args()),
                    Bottleneck(in_channels=1024, bottleneck_channels=256, **args()),
                    Bottleneck(in_channels=1024, bottleneck_channels=256, **args()),
                    Bottleneck(in_channels=1024, bottleneck_channels=256, **args()),
                    Bottleneck(in_channels=1024, bottleneck_channels=256, **args()),
            ]),
            ResidualStage(blocks=[
                    Bottleneck(in_channels=1024, bottleneck_channels=512, stride=2, **args()),
                    Bottleneck(in_channels=2048, bottleneck_channels=512, **args()),
                    Bottleneck(in_channels=2048, bottleneck_channels=512, **args()),
            ]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_pool(self.stem(x))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.head_pool(x)
        x = torch.flatten(x, 1)
        return self.head(x)

def resnet18(num_classes: int = 1000) -> ResNet:
    return ResNet(num_classes=num_classes, variant="resnet18")

def resnet50(num_classes: int = 1000) -> ResNet:
    return ResNet(num_classes=num_classes, variant="resnet50")

def mnist_resnet18(device: str = "cpu") -> ResNet:
    return ResNet(
        num_classes=10,
        variant="resnet18",
        stem=nn.Conv2dNormAct(
            in_channels=1,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm=nn.BatchNorm2d(num_features=64, device=device),
            act=nn.ReLU(inplace=True, device=device),
            device=device,
        ),
        stem_pool=nn.Identity(),
        device=device,
    )

def build_mnist_loaders(root: str = ".data") -> tuple[DataLoader, DataLoader]:
    try:
        import torchvision
    except ImportError as exc:
        raise RuntimeError("torchvision is required for the MNIST example") from exc

    allow_download = os.getenv("PYDANTIC_TORCH_TEST_DOWNLOAD_MNIST", "0") == "1"
    transform = torchvision.transforms.ToTensor()

    train_ds = torchvision.datasets.MNIST(
        root=root,
        train=True,
        download=allow_download,
        transform=transform,
    )
    test_ds = torchvision.datasets.MNIST(
        root=root,
        train=False,
        download=allow_download,
        transform=transform,
    )

    train_subset = Subset(train_ds, range(1024*4))
    test_subset = Subset(test_ds, range(512))
    return (
        DataLoader(train_subset, batch_size=64, shuffle=True),
        DataLoader(test_subset, batch_size=64, shuffle=False),
    )

def train_mnist() -> None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader = build_mnist_loaders()

    model = mnist_resnet18(device=device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print("mnist loss:", round(loss.item(), 4))
    
    model = model.clone()
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

    print("mnist accuracy:", round(correct / total, 4))

def main() -> None:
    x = torch.randn(2, 3, 224, 224)

    model18 = resnet18(num_classes=100)
    restored18 = ResNet(**model18.model_dump())
    out18 = restored18(x)
    print("resnet18 output shape:", tuple(out18.shape))

    model50 = resnet50(num_classes=100)
    restored50 = ResNet(**model50.model_dump())
    out50 = restored50(x)
    print("resnet50 output shape:", tuple(out50.shape))

    train_mnist()

if __name__ == "__main__":
    main()
