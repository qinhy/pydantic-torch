from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, Union
from uuid import uuid4

import torch
from pydantic import BaseModel, Field
from pydantic.config import ConfigDict


class Module(torch.nn.Module):
    class Conf(BaseModel):
        model_config = ConfigDict(extra="forbid")
        uuid: str = Field(default=None)

        def model_post_init(self, __context: Any) -> None:
            self.uuid = str(uuid4())
            super().model_post_init(__context)

    def __init__(self, config: Conf | dict = None):
        super().__init__()
        self._config = config

    def model_dump(self) -> dict:
        self._config.uuid = f"{self.__class__.__name__}:{self._config.uuid.split(':')[-1]}"
        return self._config.model_dump()

    def clone(self, **kwargs):
        state = deepcopy(self.state_dict())
        config = self.Conf.model_validate({**self.model_dump(), **kwargs})
        model = self.__class__(config)
        model.load_state_dict(state)
        return model

    def save_file(self, path: str | Path, meta: Optional[dict] = None) -> None:
        meta = {} if meta is None else dict(meta)

        reserved = {"model", "model_dump"}
        overlap = reserved & meta.keys()
        if overlap:
            raise ValueError(f"meta contains reserved keys: {sorted(overlap)}")

        state = {
            "model": self.state_dict(),
            "model_dump": self.model_dump(),
            **meta,
        }
        torch.save(state, path)

    @classmethod
    def load_file(cls, path: str | Path, map_location=None):
        state = torch.load(path, map_location=map_location)
        weights = state.get("model", {})
        config_dict = state.get("model_dump", {})

        config = cls.Conf.model_validate(config_dict)
        model = cls(config)
        model.load_state_dict(weights)

        meta = {k: v for k, v in state.items() if k not in {"model", "model_dump"}}
        return model, meta


class Linear(Module, torch.nn.Linear):
    class Conf(Module.Conf):
        in_features: int = Field(default=10, ge=1)
        out_features: int = Field(default=5, ge=1)
        bias: bool = Field(default=True)

    def __init__(self, config: Conf | dict):
        config = self.Conf.model_validate(config)

        torch.nn.Linear.__init__(
            self,
            in_features=config.in_features,
            out_features=config.out_features,
            bias=config.bias,
        )
        self._config = config


class Conv2d(Module, torch.nn.Conv2d):
    class Conf(Module.Conf):
        in_channels: int = Field(default=1, ge=1)
        out_channels: int = Field(default=1, ge=1)
        kernel_size: Union[int, Tuple[int, int]] = Field(default=3)
        stride: Union[int, Tuple[int, int]] = Field(default=1)
        padding: Union[str, int, Tuple[int, int]] = Field(default=0)
        dilation: Union[int, Tuple[int, int]] = Field(default=1)
        groups: int = Field(default=1, ge=1)
        padding_mode: str = Field(default="zeros")
        bias: bool = Field(default=True)

    def __init__(self, config: Conf | dict):
        config = self.Conf.model_validate(config)

        torch.nn.Conv2d.__init__(
            self,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding,
            dilation=config.dilation,
            groups=config.groups,
            bias=config.bias,
            padding_mode=config.padding_mode,
        )
        self._config = config

class LayerNorm(Module, torch.nn.LayerNorm):
    class Conf(Module.Conf):
        normalized_shape: Union[int, List[int], Tuple[int, ...]]
        eps: float = Field(default=1e-5, ge=0.0)
        elementwise_affine: bool = Field(default=True)

    def __init__(self, config: Conf | dict):
        config = self.Conf.model_validate(config)

        torch.nn.LayerNorm.__init__(
            self,
            normalized_shape=config.normalized_shape,
            eps=config.eps,
            elementwise_affine=config.elementwise_affine,
        )
        self._config = config


class BatchNorm2d(Module, torch.nn.BatchNorm2d):
    class Conf(Module.Conf):
        num_features: int = Field(default=1, ge=1)
        eps: float = Field(default=1e-5, ge=0.0)
        momentum: Optional[float] = Field(default=0.1)
        affine: bool = Field(default=True)
        track_running_stats: bool = Field(default=True)

    def __init__(self, config: Conf | dict):
        config = self.Conf.model_validate(config)

        torch.nn.BatchNorm2d.__init__(
            self,
            num_features=config.num_features,
            eps=config.eps,
            momentum=config.momentum,
            affine=config.affine,
            track_running_stats=config.track_running_stats,
        )
        self._config = config


class MaxPool2d(Module, torch.nn.MaxPool2d):
    class Conf(Module.Conf):
        kernel_size: Union[int, Tuple[int, int]] = Field(default=2)
        stride: Optional[Union[int, Tuple[int, int]]] = Field(default=None)
        padding: Union[int, Tuple[int, int]] = Field(default=0)
        dilation: Union[int, Tuple[int, int]] = Field(default=1)
        return_indices: bool = Field(default=False)
        ceil_mode: bool = Field(default=False)

    def __init__(self, config: Conf | dict):
        config = self.Conf.model_validate(config)

        torch.nn.MaxPool2d.__init__(
            self,
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding,
            dilation=config.dilation,
            return_indices=config.return_indices,
            ceil_mode=config.ceil_mode,
        )
        self._config = config


class AdaptiveAvgPool2d(Module, torch.nn.AdaptiveAvgPool2d):
    class Conf(Module.Conf):
        output_size: Union[int, Tuple[Optional[int], Optional[int]]] = Field(default=1)

    def __init__(self, config: Conf | dict):
        config = self.Conf.model_validate(config)

        torch.nn.AdaptiveAvgPool2d.__init__(
            self,
            output_size=config.output_size,
        )
        self._config = config


class Embedding(Module, torch.nn.Embedding):
    class Conf(Module.Conf):
        num_embeddings: int = Field(default=1, ge=1)
        embedding_dim: int = Field(default=1, ge=1)
        padding_idx: Optional[int] = Field(default=None)
        max_norm: Optional[float] = Field(default=None)
        norm_type: float = Field(default=2.0)
        scale_grad_by_freq: bool = Field(default=False)
        sparse: bool = Field(default=False)
        freeze: bool = Field(default=False)

    def __init__(self, config: Conf | dict):
        config = self.Conf.model_validate(config)

        torch.nn.Embedding.__init__(
            self,
            num_embeddings=config.num_embeddings,
            embedding_dim=config.embedding_dim,
            padding_idx=config.padding_idx,
            max_norm=config.max_norm,
            norm_type=config.norm_type,
            scale_grad_by_freq=config.scale_grad_by_freq,
            sparse=config.sparse,
        )
        self.weight.requires_grad_(not config.freeze)
        self._config = config


class GELU(Module, torch.nn.GELU):
    class Conf(Module.Conf):
        approximate: Literal["none", "tanh"] = Field(default="none")

    def __init__(self, config: Conf | dict = None):
        config = self.Conf() if config is None else self.Conf.model_validate(config)

        torch.nn.GELU.__init__(
            self,
            approximate=config.approximate,
        )
        self._config = config


class ReLU(Module, torch.nn.ReLU):
    class Conf(Module.Conf):
        inplace: bool = Field(default=False)

    def __init__(self, config: Conf | dict = None):
        config = self.Conf() if config is None else self.Conf.model_validate(config)

        torch.nn.ReLU.__init__(
            self,
            inplace=config.inplace,
        )
        self._config = config


class Dropout(Module, torch.nn.Dropout):
    class Conf(Module.Conf):
        p: float = Field(default=0.5, ge=0.0, le=1.0)
        inplace: bool = Field(default=False)

    def __init__(self, config: Conf | dict = None):
        config = self.Conf() if config is None else self.Conf.model_validate(config)

        torch.nn.Dropout.__init__(
            self,
            p=config.p,
            inplace=config.inplace,
        )
        self._config = config


class DropPath(Module):
    class Conf(Module.Conf):
        drop_prob: float = Field(default=0.0, ge=0.0, le=1.0)

    def __init__(self, config: Conf | dict = None):
        super().__init__()
        config = self.Conf() if config is None else self.Conf.model_validate(config)
        self._config = config

    @property
    def drop_prob(self) -> float:
        return self._config.drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask / keep_prob


class Identity(Module, torch.nn.Identity):
    class Conf(Module.Conf):
        pass

    def __init__(self, config: Conf | dict = None):
        config = self.Conf() if config is None else self.Conf.model_validate(config)

        torch.nn.Identity.__init__(self)
        self._config = config
        
if __name__ == "__main__":
    linear = Linear(
        Linear.Conf(
            in_features=8,
            out_features=4,
            bias=True,
        )
    )

    x = torch.randn(2, 8)
    y = linear(x)

    print(y.shape)  # torch.Size([2, 4])
    print(linear.model_dump())
    # {'in_features': 8, 'out_features': 4, 'bias': True}
    linear2 = linear.clone()

    print(linear2.model_dump())
    # same config as linear
    linear = Linear(Linear.Conf(
            in_features=8,
            out_features=4,
            bias=True,
        ))

    linear.save_file("linear.pt", meta={"tag": "demo", "epoch": 3})
    loaded_linear, meta = Linear.load_file("linear.pt")

    print(loaded_linear)
    print(loaded_linear.model_dump())
    print(meta)
    # {'tag': 'demo', 'epoch': 3}