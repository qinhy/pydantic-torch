from .modules import *
# from .conv import *
from .utils import Cls_parse

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

# class ModuleList(ModuleList):pass


# # advnace api
class Acts:
    types = Union[ReLU.Conf, GELU.Conf]
    cls = {"ReLU": ReLU, "GELU": GELU, }
    @staticmethod
    def parse(v:types):return Cls_parse(v,Acts.cls)

# class Norms(Norms):pass
# class Conv2dAct(Conv2dAct):pass
# class Conv2dNorm(Conv2dNorm):pass
# class Conv2dNormAct(Conv2dNormAct):pass
