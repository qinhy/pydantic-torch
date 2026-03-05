from __future__ import annotations
import torch
from pydantic_torch.vit import vit_base_patch16_224

model = vit_base_patch16_224(num_classes=1000)

def test_basic_forward_shape() -> None:
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.shape == (2, 1000)