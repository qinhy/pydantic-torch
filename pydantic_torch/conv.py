from pydantic import field_validator

from .modules import *

# advnace api
def Cls_parse(v: Any, cls_dict: dict[str, type]) -> Any:
    if isinstance(v, tuple(cls_dict.values())):
        return v
    if not isinstance(v, dict):
        raise TypeError("expected a module instance or serialized module dict")
    raw_uuid = v.get("uuid")
    if not isinstance(raw_uuid, str) or ":" not in raw_uuid:
        raise ValueError("serialized module must include uuid like 'ClassName:...'")
    kind = raw_uuid.split(":", 1)[0]
    module_cls = cls_dict.get(kind)
    if module_cls is None:
        raise ValueError(f"unknown module type: {kind}")
    return module_cls.model_validate(v)

class Acts:
    types = Union[ReLU, GELU]
    cls = {"ReLU": ReLU, "GELU": GELU, }
    @staticmethod
    def parse(v):return Cls_parse(v,Acts.cls)

class Norms:
    types = Union[LayerNorm, BatchNorm2d]
    cls = {"LayerNorm": LayerNorm, "BatchNorm2d": BatchNorm2d, }
    @staticmethod
    def parse(v):return Cls_parse(v,Norms.cls)

class Conv2dAct(Conv2d):
    act: Acts.types = Field(default=ReLU())

    @field_validator("act", mode="before")
    @classmethod
    def parse_act(cls, v):
        return Acts.parse(v)

    def forward(self, x):
        return self.act(super().forward(x))

class Conv2dNorm(Conv2d):
    norm: Norms.types

    @field_validator("norm", mode="before")
    @classmethod
    def parse_norm(cls, v):
        return Norms.parse(v)

    def forward(self, x):
        return self.norm(super().forward(x))

class Conv2dNormAct(Conv2dNorm):
    act: Acts.types
    @field_validator("act", mode="before")
    @classmethod
    def parse_act(cls, v):
        return Acts.parse(v)
    
    def forward(self, x):
        return self.act(super().forward(x))

