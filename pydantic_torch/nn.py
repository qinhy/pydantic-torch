from .modules import *
from .conv import *
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
class Acts(Acts):pass
class Norms(Norms):pass
class Conv2dAct(Conv2dAct):pass
class Conv2dNorm(Conv2dNorm):pass
class Conv2dNormAct(Conv2dNormAct):pass


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