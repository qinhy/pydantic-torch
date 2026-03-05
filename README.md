# pydantic-torch

`pydantic-torch` combines:

- Pydantic v2 field validation
- PyTorch `nn.Module` behavior (parameters, buffers, hooks, `.to()`, `state_dict()`, etc.)

It is useful when you want strict config validation and standard PyTorch module ergonomics in one model class.

## Install

```bash
uv sync --extra dev
```

## Quick Start

```python
from typing import Any

import torch
from pydantic import Field
import pydantic_torch.nn as nn


class ThreeLayerNet(nn.Module):
    in_features: int = Field(default=10, ge=1)
    hidden_features: int = Field(default=8, ge=1)
    out_features: int = Field(default=5, ge=1)

    linear1: nn.Linear = Field(default=None)
    linear2: nn.Linear = Field(default=None)
    linear3: nn.Linear = Field(default=None)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self.linear1 = nn.Linear(in_features=self.in_features, out_features=self.hidden_features)
        self.linear2 = nn.Linear(in_features=self.hidden_features, out_features=self.hidden_features)
        self.linear3 = nn.Linear(in_features=self.hidden_features, out_features=self.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear3(self.linear2(self.linear1(x)))


model = ThreeLayerNet(in_features=10, hidden_features=16, out_features=2)
x = torch.randn(4, 10)
y = model(x)
print(y.shape)  # torch.Size([4, 2])
```

## Run the Full Example

A complete usage script is included at `how_to_use.py`:

```bash
uv run python how_to_use.py
```

It demonstrates:

- Pydantic validation errors
- Forward + optimizer step
- `state_dict()` save/load
- A small Vision Transformer configuration

## Vision Transformer Helper

This repo includes a ViT implementation in `pydantic_torch/vit.py`:

```python
import torch
from pydantic_torch.vit import vit_base_patch16_224

model = vit_base_patch16_224(num_classes=1000)
x = torch.randn(2, 3, 224, 224)
y = model(x)
print(y.shape)  # torch.Size([2, 1000])
```

## Run Tests

```bash
uv run pytest
```

MNIST training integration test (optional):

```bash
PYDANTIC_TORCH_TEST_DOWNLOAD_MNIST=1 uv run pytest tests/test_mnist_train.py -q
```
