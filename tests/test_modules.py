from __future__ import annotations

import copy
import io
import os
from typing import Any

import pytest
import torch
from pydantic import ValidationError, Field

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


def test_basic_forward_shape() -> None:
    net = ThreeLayerNet(in_features=10, out_features=5)
    x = torch.randn(2, 10)
    y = net(x)
    assert y.shape == (2, 5)

def test_clone()-> None:
    net = ThreeLayerNet(in_features=10, out_features=5)
    net2 = net.clone()
    x = torch.randn(2, 10)
    assert net.uuid != net2.uuid
    assert (net(x) == net2(x)).all().item()

def test_save_load_file()-> None:
    net = ThreeLayerNet(in_features=10, out_features=5)
    x = torch.randn(2, 10)
    res1 = net(x)
    net.save_file("test_ThreeLayerNet.pt")
    net2 = ThreeLayerNet.load_file("test_ThreeLayerNet.pt")
    res2 = net2(x)
    os.remove("test_ThreeLayerNet.pt")
    assert (res1 == res2).all().item()

def test_pydantic_validation() -> None:
    with pytest.raises(ValidationError):
        ThreeLayerNet(in_features=0, out_features=5)


def test_parameter_registration() -> None:
    net = ThreeLayerNet(in_features=10, out_features=5)
    state = net.state_dict()
    assert state

    names = [name for name, _ in net.named_parameters()]
    assert "linear1.weight" in names
    assert "linear1.bias" in names


def test_backward_and_optimizer_step() -> None:
    net = ThreeLayerNet(in_features=10, out_features=5)
    net(torch.randn(4, 10)).sum().backward()

    assert net.linear1 is not None
    assert net.linear1 is not None
    assert net.linear1.weight.grad is not None

    before = net.linear1.weight.detach().clone()
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    opt.step()
    after = net.linear1.weight.detach().clone()
    assert not torch.equal(before, after)


def test_state_dict_roundtrip() -> None:
    net = ThreeLayerNet(in_features=10, out_features=5)
    x = torch.randn(3, 10)
    y0 = net(x).detach()

    net2 = ThreeLayerNet(in_features=10, out_features=5)
    net2.load_state_dict(net.state_dict())
    y1 = net2(x).detach()

    assert torch.allclose(y0, y1)


def test_hooks_fire() -> None:
    net = ThreeLayerNet(in_features=10, out_features=5)
    calls = {"pre": 0, "post": 0}

    def pre_hook(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        del module, inputs
        calls["pre"] += 1

    def post_hook(
        module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        del module, inputs, output
        calls["post"] += 1

    pre = net.register_forward_pre_hook(pre_hook)
    post = net.register_forward_hook(post_hook)

    _ = net(torch.randn(2, 10))

    pre.remove()
    post.remove()

    assert calls == {"pre": 1, "post": 1}


def test_named_modules_and_buffers() -> None:
    net = ThreeLayerNet(in_features=10, out_features=5)

    named_mods = dict(net.named_modules())
    assert "linear1" in named_mods
    assert any(mod is net.linear1 for mod in net.modules())

    net.register_buffer("running", torch.zeros(3))
    state = net.state_dict()
    assert "running" in state
    assert state["running"].shape == (3,)

    dumped = net.model_dump()
    assert "running" not in dumped


def test_device_and_dtype_transfer() -> None:
    net = ThreeLayerNet(in_features=10, out_features=5)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net = net.to(device)
    assert next(net.parameters()).device == device

    net = net.double()
    assert next(net.parameters()).dtype == torch.float64

    x = torch.randn(4, 10, device=device, dtype=torch.float64)
    y = net(x)
    assert y.dtype == torch.float64
    assert y.device == device


def test_pydantic_json_roundtrip_structure() -> None:
    net = ThreeLayerNet(in_features=10, hidden_features=7, out_features=5)
    payload = net.model_dump_json()

    restored = ThreeLayerNet.model_validate_json(payload)

    assert restored.in_features == 10
    assert restored.hidden_features == 7
    assert restored.out_features == 5
    assert isinstance(restored.linear1, nn.Linear)


def test_deepcopy_independence() -> None:
    net = ThreeLayerNet(in_features=10, out_features=5)
    clone = copy.deepcopy(net)

    assert net is not clone
    assert net.linear1 is not None and clone.linear1 is not None
    assert net.linear1 is not None and clone.linear1 is not None
    assert net.linear1.weight.data_ptr() != clone.linear1.weight.data_ptr()


def test_torch_save_load_state_dict_bytesio() -> None:
    net = ThreeLayerNet(in_features=10, out_features=5).double()

    buf = io.BytesIO()
    torch.save(net.state_dict(), buf)
    buf.seek(0)

    loaded_state = torch.load(buf, map_location="cpu")
    restored = ThreeLayerNet(in_features=10, out_features=5).double()
    restored.load_state_dict(loaded_state)

    x = torch.randn(4, 10, dtype=torch.float64)
    y0 = net.to("cpu")(x).detach()
    y1 = restored(x).detach()

    assert torch.allclose(y0, y1)


def test_jit_trace() -> None:
    net = ThreeLayerNet(in_features=10, out_features=5).double()

    traced = torch.compile(net)
    out = traced(torch.randn(2, 10, dtype=torch.float64))

    assert out.shape == (2, 5)

def test_linear_bias_false() -> None:
    linear = nn.Linear(in_features=6, out_features=2, bias=False)
    y = linear(torch.randn(3, 6))
    assert y.shape == (3, 2)
    assert linear.bias is None

def test_conv2d_forward_and_config() -> None:
    conv = nn.Conv2d(
        in_channels=3,
        out_channels=4,
        kernel_size=3,
        stride=2,
        padding=1,
        bias=False,
    )

    x = torch.randn(2, 3, 8, 8)
    y = conv(x)

    assert y.shape == (2, 4, 4, 4)
    assert conv.bias is None
    assert conv.model_dump()["in_channels"] == 3
    assert conv.model_dump()["out_channels"] == 4


def test_relu_forward_and_config() -> None:
    relu = nn.ReLU(inplace=True)
    x = torch.tensor([[-1.0, 2.0]])
    y = relu(x)

    assert torch.equal(y, torch.tensor([[0.0, 2.0]]))
    assert relu.model_dump()["inplace"] is True


def test_batchnorm2d_forward_and_config() -> None:
    norm = nn.BatchNorm2d(num_features=4, affine=False, track_running_stats=True)
    x = torch.randn(2, 4, 8, 8)
    y = norm(x)

    assert y.shape == x.shape
    assert norm.weight is None
    assert norm.bias is None
    assert norm.running_mean is not None
    assert norm.model_dump()["num_features"] == 4


def test_maxpool2d_forward_and_config() -> None:
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    x = torch.randn(2, 3, 8, 8)
    y = pool(x)

    assert y.shape == (2, 3, 4, 4)
    assert pool.model_dump()["kernel_size"] == 2
    assert pool.model_dump()["stride"] == 2


def test_adaptive_avg_pool2d_forward_and_config() -> None:
    pool = nn.AdaptiveAvgPool2d(output_size=(2, 3))
    x = torch.randn(2, 3, 8, 8)
    y = pool(x)

    assert y.shape == (2, 3, 2, 3)
    assert pool.model_dump()["output_size"] == (2, 3)


def test_embedding_forward_and_config() -> None:
    embedding = nn.Embedding(
        num_embeddings=10,
        embedding_dim=4,
        padding_idx=0,
        freeze=True,
    )
    x = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)
    y = embedding(x)

    assert y.shape == (2, 3, 4)
    assert embedding.weight is not None
    assert embedding.weight.requires_grad is False
    assert embedding.model_dump()["num_embeddings"] == 10
    assert embedding.model_dump()["embedding_dim"] == 4
    assert embedding.model_dump()["padding_idx"] == 0
