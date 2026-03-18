from pydantic import field_validator

from .modules import *
from .containers import ModuleList

class Module(Module):pass
class Linear(Linear):pass
class Conv2d(Conv2d):pass
class LayerNorm(LayerNorm):pass
class BatchNorm2d(BatchNorm2d):pass
class MaxPool2d(MaxPool2d):pass
class AdaptiveAvgPool2d(AdaptiveAvgPool2d):pass
class Embedding(Embedding):pass
class GELU(GELU):pass
class ReLU(ReLU):pass
class Dropout(Dropout):pass
class Identity(Identity):pass

class ModuleList(ModuleList):pass


# advnace api

def Cls_parse(v,types,cls_dict):
    # already parsed
    if isinstance(v, types):
        return v
    if not isinstance(v, dict):
        raise TypeError("obj must be a dict or an instance")
    raw_uuid = v.get("uuid")
    if not isinstance(raw_uuid, str) or ":" not in raw_uuid:
        raise ValueError("obj.uuid must look like 'LayerNorm:...' or 'GELU:...'")
    kind = raw_uuid.split(":", 1)[0]
    obj_cls = cls_dict.get(kind)
    if obj_cls is None:
        raise ValueError(f"Unknown obj type: {kind}")
    return obj_cls.model_validate(v)

class Acts:
    types = Union[ReLU, GELU]
    cls = {"ReLU": ReLU, "GELU": GELU, }
    @staticmethod
    def parse(v):return Cls_parse(v,Acts.types,Acts.cls)

class Norms:
    types = Union[LayerNorm, BatchNorm2d]
    cls = {"LayerNorm": LayerNorm, "BatchNorm2d": BatchNorm2d, }
    @staticmethod
    def parse(v):return Cls_parse(v,Norms.types,Norms.cls)

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

