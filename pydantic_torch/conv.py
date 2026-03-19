from pydantic import field_validator

from .modules import *
from .utils import Cls_parse

# advnace api
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

