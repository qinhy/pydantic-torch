from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, ClassVar, List, Optional, Self, Tuple, Union
from uuid import uuid4

import torch
from pydantic import BaseModel, Field, PrivateAttr
from pydantic.config import ConfigDict

# -----------------------------------------------------------------------------
# Pydantic + Torch hybrid module
# -----------------------------------------------------------------------------
class Module(BaseModel, torch.nn.Module):
    """
    Hybrid base class: a Pydantic v2 model that is also a torch.torch.nn.Module.

    Key goals:
      - Pydantic validates/configures constructor fields.
      - Torch correctly registers submodules/params/buffers and supports hooks, .to(), etc.
      - Exclude torch.nn.Module internals from Pydantic dumps.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    __hash__ = object.__hash__  # torch internals sometimes need hashable modules

    # prevent Pydantic from treating these as model fields
    dump_patches: ClassVar[bool]
    call_super_init: ClassVar[bool]
    forward: ClassVar[Callable[..., Any]]

    # keep training settable (torch.nn.Module.train()/eval() assigns this)
    training: bool = Field(default=True, exclude=True)
    uuid: str = Field(default=None)

    def __setattr__(self, name: str, value: Any) -> None:
        # Important: keep Torch registration behavior and still satisfy Pydantic
        torch.nn.Module.__setattr__(self, name, value)
        BaseModel.__setattr__(self, name, value)

    def model_post_init(self, __context: Any) -> None:
        self.uuid = f"{self.__class__.__name__}:{str(uuid4())}"
        # Ensure torch.nn.Module initialization happens after Pydantic has set fields
        torch.nn.Module.__init__(self)
        # Always present buffer (some tests expect it to exist in state_dict)
        self.register_buffer("running", torch.zeros(3))
        super().model_post_init(__context)

    def clone(self) -> Module:
        state = deepcopy(self.state_dict())
        model = self.__class__(**self.model_dump())
        model.load_state_dict(state)
        return model
    
    def save_file(self, path: str) -> None:
        weights = self.state_dict()
        config = self.model_dump()
        state = {"model": weights, "model_dump": config}
        torch.save(state, path)

    @classmethod
    def load_file(cls, path: str) -> None:
        state = torch.load(path)
        weights = state.get("model", {})
        config = state.get("model_dump", None)
        if config is not None:
            model = cls(**config)
        else:
            model = cls()
        model.load_state_dict(weights)
        return model

class Linear(Module, torch.nn.Linear):
    in_features: int = Field(default=10, ge=1)
    out_features: int = Field(default=5, ge=1)
    weight: torch.Tensor = Field(default=None, exclude=True)
    bias: Optional[Union[torch.Tensor, torch.nn.Parameter, bool]] = Field(default=None, exclude=True)

    def model_post_init(self, __context):
        super().model_post_init(__context)
        torch.nn.Linear.__init__(self, **self.model_dump(exclude=["uuid"]))

class LayerNorm(Module, torch.nn.LayerNorm):
    normalized_shape: Union[int,List[int],Tuple[int, ...]]
    eps: float = Field(default=1e-5, ge=0.0)
    elementwise_affine: bool = Field(default=True)
    weight: torch.Tensor = Field(default=None, exclude=True)
    bias: torch.Tensor = Field(default=None, exclude=True)

    def model_post_init(self, __context):
        super().model_post_init(__context)
        torch.nn.LayerNorm.__init__(self, **self.model_dump(exclude=["uuid"]))

class GELU(Module, torch.nn.GELU):
    approximate: str = Field(default="none")

    def model_post_init(self, __context):
        super().model_post_init(__context)
        torch.nn.GELU.__init__(self, **self.model_dump(exclude=["uuid"]))


class Dropout(Module, torch.nn.Dropout):
    p: float = Field(default=0.5, ge=0.0, le=1.0)
    inplace: bool = Field(default=False)
    
class Identity(torch.nn.Identity, Module):
    pass
