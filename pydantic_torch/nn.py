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