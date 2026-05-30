from __future__ import annotations

from typing import Sequence, TypeVar, Annotated

UnsignedInt = Annotated[int, "Value must be >= 0"]

import numpy as np
import numpy.typing as npt

# Supported scalar types
T = TypeVar("Inexact", np.float32, np.float64, np.complex64, np.complex128)

# Enumerations

class Wavelet:
    # Daubechies
    Daubechies1: Wavelet
    Daubechies2: Wavelet
    Daubechies3: Wavelet
    Daubechies4: Wavelet
    Daubechies5: Wavelet
    Daubechies6: Wavelet
    Daubechies7: Wavelet
    Daubechies8: Wavelet
    Daubechies9: Wavelet
    Daubechies10: Wavelet
    # Symlets
    Symlet4: Wavelet
    Symlet5: Wavelet
    Symlet6: Wavelet
    # Coiflets
    Coiflet2: Wavelet
    Coiflet3: Wavelet
    # Biorthogonal
    Bior1_3: Wavelet
    Bior1_5: Wavelet
    Bior2_2: Wavelet
    Bior2_4: Wavelet
    Bior2_6: Wavelet
    Bior2_8: Wavelet
    Bior3_1: Wavelet
    Bior3_3: Wavelet
    Bior3_5: Wavelet
    Bior3_7: Wavelet
    Bior3_9: Wavelet
    Bior4_2: Wavelet
    Bior4_4: Wavelet
    Bior4_6: Wavelet
    Bior5_5: Wavelet
    Bior6_8: Wavelet
    # CDF
    CDF5_3: Wavelet
    CDF9_7: Wavelet

    def width(self) -> int: ...

class BoundaryCondition:
    Zero: BoundaryCondition
    Periodic: BoundaryCondition
    Constant: BoundaryCondition
    Symmetric: BoundaryCondition
    Reflect: BoundaryCondition
    Antisymmetric: BoundaryCondition
    Smooth: BoundaryCondition
    Antireflect: BoundaryCondition

# Utilities

def max_level(wavelet: Wavelet, n: int) -> int: ...
def max_level_nd(
    wavelet: Wavelet,
    shape: Sequence[int],
    axes: int | Sequence[int] | None = None,
) -> int: ...
def get_dwt_shape(
    wavelet: Wavelet,
    shape: Sequence[int],
    *,
    axes: int | Sequence[int] | None = None,
    level: UnsignedInt = 0,
) -> tuple[int, ...]: ...

# LWT

def lwt(
    wavelet: Wavelet,
    x: npt.NDArray[T],
    *,
    bc: BoundaryCondition = BoundaryCondition.Symmetric,
    axes: int | Sequence[int] | None = None,
    level: UnsignedInt = 0,
    out: npt.NDArray[T] | None = None,
) -> npt.NDArray[T]: ...
def ilwt(
    wavelet: Wavelet,
    x: npt.NDArray[T],
    *,
    bc: BoundaryCondition = BoundaryCondition.Symmetric,
    axes: int | Sequence[int] | None = None,
    level: UnsignedInt = 0,
    out: npt.NDArray[T] | None = None,
) -> npt.NDArray[T]: ...
def lwt_adj(
    wavelet: Wavelet,
    x: npt.NDArray[T],
    *,
    bc: BoundaryCondition = BoundaryCondition.Symmetric,
    axes: int | Sequence[int] | None = None,
    level: UnsignedInt = 0,
    out: npt.NDArray[T] | None = None,
) -> npt.NDArray[T]: ...
def ilwt_adj(
    wavelet: Wavelet,
    x: npt.NDArray[T],
    *,
    bc: BoundaryCondition = BoundaryCondition.Symmetric,
    axes: int | Sequence[int] | None = None,
    level: UnsignedInt = 0,
    out: npt.NDArray[T] | None = None,
) -> npt.NDArray[T]: ...

# DWT

def dwt(
    wavelet: Wavelet,
    x: npt.NDArray[T],
    *,
    bc: BoundaryCondition = BoundaryCondition.Symmetric,
    axes: int | Sequence[int] | None = None,
    level: UnsignedInt = 0,
    out: npt.NDArray[T] | None = None,
) -> npt.NDArray[T]: ...
def idwt(
    wavelet: Wavelet,
    x: npt.NDArray[T],
    out: npt.NDArray[T] | Sequence[int],
    *,
    bc: BoundaryCondition = BoundaryCondition.Symmetric,
    axes: int | Sequence[int] | None = None,
    level: UnsignedInt = 0,
) -> npt.NDArray[T]: ...
def dwt_adj(
    wavelet: Wavelet,
    x: npt.NDArray[T],
    out: npt.NDArray[T] | Sequence[int],
    *,
    bc: BoundaryCondition = BoundaryCondition.Symmetric,
    axes: int | Sequence[int] | None = None,
    level: UnsignedInt = 0,
) -> npt.NDArray[T]: ...
def idwt_adj(
    wavelet: Wavelet,
    x: npt.NDArray[T],
    *,
    bc: BoundaryCondition = BoundaryCondition.Symmetric,
    axes: int | Sequence[int] | None = None,
    level: UnsignedInt = 0,
    out: npt.NDArray[T] | None = None,
) -> npt.NDArray[T]: ...

# DWT (periodic boundary condition)

def dwt_per(
    wavelet: Wavelet,
    x: npt.NDArray[T],
    *,
    axes: int | Sequence[int] | None = None,
    level: UnsignedInt = 0,
    out: npt.NDArray[T] | None = None,
) -> npt.NDArray[T]: ...
def idwt_per(
    wavelet: Wavelet,
    x: npt.NDArray[T],
    *,
    axes: int | Sequence[int] | None = None,
    level: UnsignedInt = 0,
    out: npt.NDArray[T] | None = None,
) -> npt.NDArray[T]: ...
def dwt_per_adj(
    wavelet: Wavelet,
    x: npt.NDArray[T],
    *,
    axes: int | Sequence[int] | None = None,
    level: UnsignedInt = 0,
    out: npt.NDArray[T] | None = None,
) -> npt.NDArray[T]: ...
def idwt_per_adj(
    wavelet: Wavelet,
    x: npt.NDArray[T],
    *,
    axes: int | Sequence[int] | None = None,
    level: UnsignedInt = 0,
    out: npt.NDArray[T] | None = None,
) -> npt.NDArray[T]: ...
