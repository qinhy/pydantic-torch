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
class Acts:
    types = Union[ReLU, GELU]
    cls = {"ReLU": ReLU, "GELU": GELU, }

    @staticmethod
    def Acts_parse(v):    
        # already parsed
        if isinstance(v, Acts.types):
            return v
        # raw input must be a dict with uuid like "ReLU:..."
        if not isinstance(v, dict):
            raise TypeError("act must be a dict or an instance")
        raw_uuid = v.get("uuid")
        if not isinstance(raw_uuid, str) or ":" not in raw_uuid:
            raise ValueError("act.uuid must look like 'ReLU:...' or 'GELU:...'")
        kind = raw_uuid.split(":", 1)[0]
        act_cls = Acts.cls.get(kind)
        if act_cls is None:
            raise ValueError(f"Unknown act type: {kind}")
        return act_cls.model_validate(v)
    
class Conv2dAct(Conv2d):
    act: Acts.types = Field(default=ReLU())

    @field_validator("act", mode="before")
    @classmethod
    def parse_act(cls, v):
        return Acts.Acts_parse(v)

    def forward(self, x):
        return self.act(super().forward(x))