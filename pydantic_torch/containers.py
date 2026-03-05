from __future__ import annotations

from itertools import chain
import operator
from typing import Any, Callable, ClassVar, Dict, Iterable, Iterator, List, Optional, Self, overload
from collections import abc as container_abcs, OrderedDict

from pydantic import Field, PrivateAttr

import pydantic_torch.vit as vit
from .modules import *

class nn:
    class Module(Module):pass
    class Linear(Linear):pass
    class LayerNorm(LayerNorm):pass
    class GELU(GELU):pass
    class Dropout(Dropout):pass
    class Identity(Identity):pass

class ModuleList(Module):
    r"""Holds submodules in a list.

    :class:`~torch.nn.ModuleList` can be indexed like a regular Python list, but
    modules it contains are properly registered, and will be visible by all
    :class:`~torch.nn.Module` methods.

    Args:
        modules (iterable, optional): an iterable of modules to add

    Example::

        class MyModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

            def forward(self, x):
                # ModuleList can act as an iterable, or be indexed using ints
                for i, l in enumerate(self.linears):
                    x = self.linears[i // 2](x) + l(x)
                return x
    """

    mods: Optional[List[Any]] = Field(default=None)
    _modules: Dict[str, Module] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context):
        super().model_post_init(__context)
        if self.mods is not None:
            tmp = self + self.mods
            self.mods = tmp.mods
            self._modules = tmp._modules

    def _get_abs_string_index(self, idx):
        """Get the absolute index for the list of modules."""
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError(f"index {idx} is out of range")
        if idx < 0:
            idx += len(self)
        return str(idx)

    @overload
    def __getitem__(self, idx: slice) -> ModuleList: ...

    @overload
    def __getitem__(self, idx: int) -> Module: ...

    def __getitem__(self, idx: int | slice) -> Module | ModuleList:
        if isinstance(idx, slice):
            return self.__class__(list(self._modules.values())[idx])
        else:
            return self._modules[self._get_abs_string_index(idx)]

    def __setitem__(self, idx: int, module: Module) -> None:
        idx = self._get_abs_string_index(idx)
        return setattr(self, str(idx), module)

    def __delitem__(self, idx: int | slice) -> None:
        if isinstance(idx, slice):
            for k in range(len(self._modules))[idx]:
                delattr(self, str(k))
        else:
            delattr(self, self._get_abs_string_index(idx))
        # To preserve numbering, self._modules is being reconstructed with modules after deletion
        str_indices = [str(i) for i in range(len(self._modules))]
        self._modules = OrderedDict(
            zip(str_indices, self._modules.values(), strict=True)
        )

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def __iadd__(self, modules: Iterable[Module]) -> Self:
        return self.extend(modules)

    def __add__(self, other: Iterable[Module]) -> ModuleList:
        combined = ModuleList()
        for i, module in enumerate(chain(self, other)):
            if type(module) is dict:
                name = module["uuid"].split(":")[0]
                module_cls = nn.__dict__.get(name,
                        vit.__dict__.get(name))
                assert module_cls is not None
                module = module_cls(**module)

            combined.add_module(str(i), module)        
        combined.mods = [v for k,v in combined._modules.items()]
        return combined

    def __repr__(self) -> str:
        """Return a custom repr for ModuleList that compresses repeated module representations."""
        list_of_reprs = [repr(item) for item in self]
        if len(list_of_reprs) == 0:
            return self._get_name() + "()"

        start_end_indices = [[0, 0]]
        repeated_blocks = [list_of_reprs[0]]
        for i, r in enumerate(list_of_reprs[1:], 1):
            if r == repeated_blocks[-1]:
                start_end_indices[-1][1] += 1
                continue

            start_end_indices.append([i, i])
            repeated_blocks.append(r)

        lines = []
        main_str = self._get_name() + "("
        for (start_id, end_id), b in zip(
            start_end_indices, repeated_blocks, strict=True
        ):
            local_repr = f"({start_id}): {b}"  # default repr

            if start_id != end_id:
                n = end_id - start_id + 1
                local_repr = f"({start_id}-{end_id}): {n} x {b}"

            # Copied from torch.nn.modules.module, required for a custom __repr__ for ModuleList
            def _addindent(s_, numSpaces):
                s = s_.split("\n")
                # don't do anything for single-line stuff
                if len(s) == 1:
                    return s_
                first = s.pop(0)
                s = [(numSpaces * " ") + line for line in s]
                s = "\n".join(s)
                s = first + "\n" + s
                return s
            local_repr = _addindent(local_repr, 2)
            lines.append(local_repr)

        main_str += "\n  " + "\n  ".join(lines) + "\n"
        main_str += ")"
        return main_str

    def __dir__(self) -> list[str]:
        keys = super().__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def insert(self, index: int, module: Module) -> None:
        r"""Insert a given module before a given index in the list.

        Args:
            index (int): index to insert.
            module (nn.Module): module to insert
        """
        for i in range(len(self._modules), index, -1):
            self._modules[str(i)] = self._modules[str(i - 1)]
        self._modules[str(index)] = module

    def append(self, module: Module) -> Self:
        r"""Append a given module to the end of the list.

        Args:
            module (nn.Module): module to append
        """
        self.add_module(str(len(self)), module)
        return self

    def pop(self, key: int | slice) -> Module:
        v = self[key]
        del self[key]
        return v

    def extend(self, modules: Iterable[Module]) -> Self:
        r"""Append modules from a Python iterable to the end of the list.

        Args:
            modules (iterable): iterable of modules to append
        """
        if not isinstance(modules, container_abcs.Iterable):
            raise TypeError(
                "ModuleList.extend should be called with an "
                "iterable, but got " + type(modules).__name__
            )
        offset = len(self)
        for i, module in enumerate(modules):
            self.add_module(str(offset + i), module)
        return self
