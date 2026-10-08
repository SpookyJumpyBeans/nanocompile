"""The element types a graph can hold.

Five, and only the ones the model needs: ``f32`` for everything numeric,
``i64`` for token IDs and positions, ``i32`` and ``i8`` for phase 6, ``bool``
for masks. There is no bfloat16. The weight file is bf16 on disk, but nanoinfer
widens it to float32 at load, and the phase 1 gate is agreement with nanoinfer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DType:
    name: str
    numpy: np.dtype

    @property
    def is_float(self) -> bool:
        return self.numpy.kind == "f"

    @property
    def is_int(self) -> bool:
        return self.numpy.kind in "iu"

    @property
    def is_numeric(self) -> bool:
        return self.is_float or self.is_int

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


f32 = DType("f32", np.dtype(np.float32))
i64 = DType("i64", np.dtype(np.int64))
i32 = DType("i32", np.dtype(np.int32))
i8 = DType("i8", np.dtype(np.int8))
bool_ = DType("bool", np.dtype(np.bool_))

ALL = (f32, i64, i32, i8, bool_)
_BY_NUMPY = {d.numpy: d for d in ALL}
_BY_NAME = {d.name: d for d in ALL}


def from_numpy(dtype: np.dtype) -> DType:
    try:
        return _BY_NUMPY[np.dtype(dtype)]
    except KeyError:
        raise TypeError(f"numpy dtype {np.dtype(dtype)} has no graph equivalent") from None


def from_name(name: str) -> DType:
    try:
        return _BY_NAME[name]
    except KeyError:
        raise TypeError(f"unknown dtype {name!r}") from None
