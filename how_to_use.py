from __future__ import annotations

from typing import Any

import torch
from pydantic import Field, ValidationError
from examples.vit import VisionTransformer
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


def main() -> None:
    print("1) Build a validated model")
    model = ThreeLayerNet(in_features=10, hidden_features=16, out_features=2)
    x = torch.randn(4, 10)
    y = model(x)
    print(f"forward output shape: {tuple(y.shape)}")

    print("\n2) Validation error example")
    try:
        _ = ThreeLayerNet(in_features=0, hidden_features=16, out_features=2)
    except ValidationError as exc:
        first_error = exc.errors()[0]
        print(f"validation failed: {first_error['loc']} -> {first_error['msg']}")

    print("\n3) Optimizer step")
    target = torch.randn(4, 2)
    loss = torch.nn.functional.mse_loss(y, target)
    loss.backward()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    opt.step()
    opt.zero_grad()
    print(f"loss: {loss.item():.6f}")

    print("\n4) Save/load with state_dict")
    state = model.state_dict()
    restored = ThreeLayerNet(in_features=10, hidden_features=16, out_features=2)
    restored.load_state_dict(state)
    restored_out = restored(x)
    print(f"restored output shape: {tuple(restored_out.shape)}")

    print("\n5) Small Vision Transformer example")
    vit = VisionTransformer(
        img_size=32,
        patch_size=8,
        in_chans=3,
        num_classes=10,
        embed_dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        drop_path_rate=0.0,
    )
    img = torch.randn(2, 3, 32, 32)
    logits = vit(img)
    print(f"vit output shape: {tuple(logits.shape)}")
    print(vit.model_dump())
    print(VisionTransformer(**vit.model_dump()))
    print(vit.clone())


if __name__ == "__main__":
    main()
