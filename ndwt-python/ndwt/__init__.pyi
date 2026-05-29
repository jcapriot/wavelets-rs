from __future__ import annotations

from typing import Sequence, overload, TypeVar, Annotated

UnsignedInt = Annotated[int, "Value must be >= 0"]

import numpy as np
import numpy.typing as npt

# ── Supported scalar types ────────────────────────────────────────────────────

_SCT = TypeVar("_SCT", np.float32, np.float64, np.complex64, np.complex128)

# ── Enumerations ──────────────────────────────────────────────────────────────

class Wavelets:

    # Daubechies
    Daubechies1: Wavelets
    Daubechies2: Wavelets
    Daubechies3: Wavelets
    Daubechies4: Wavelets
    Daubechies5: Wavelets
    Daubechies6: Wavelets
    Daubechies7: Wavelets
    Daubechies8: Wavelets
    Daubechies9: Wavelets
    Daubechies10: Wavelets
    # Symlets
    Symlet4: Wavelets
    Symlet5: Wavelets
    Symlet6: Wavelets
    # Coiflets
    Coiflet2: Wavelets
    Coiflet3: Wavelets
    # Biorthogonal
    Bior1_3: Wavelets
    Bior1_5: Wavelets
    Bior2_2: Wavelets
    Bior2_4: Wavelets
    Bior2_6: Wavelets
    Bior2_8: Wavelets
    Bior3_1: Wavelets
    Bior3_3: Wavelets
    Bior3_5: Wavelets
    Bior3_7: Wavelets
    Bior3_9: Wavelets
    Bior4_2: Wavelets
    Bior4_4: Wavelets
    Bior4_6: Wavelets
    Bior5_5: Wavelets
    Bior6_8: Wavelets
    # CDF
    CDF5_3: Wavelets
    CDF9_7: Wavelets

    def width(self) -> int:
        ...

class BoundaryCondition:

    Zero: BoundaryCondition
    Periodic: BoundaryCondition
    Constant: BoundaryCondition
    Symmetric: BoundaryCondition
    Reflect: BoundaryCondition
    Antisymmetric: BoundaryCondition
    Smooth: BoundaryCondition
    Antireflect: BoundaryCondition

# ── Utilities ─────────────────────────────────────────────────────────────────

def max_level(wavelet: Wavelets, n: int) -> int:
    ...

def max_level_nd(
    wavelet: Wavelets,
    shape: Sequence[int],
    axes: int | Sequence[int] | None = None,
) -> int:
    ...

def get_dwt_shape(
    wavelet: Wavelets,
    shape: Sequence[int],
    *,
    axes: int | Sequence[int] | None = None,
    level: UnsignedInt = 0,
) -> tuple[int, ...]:
    ...

# ── LWT ───────────────────────────────────────────────────────────────────────

def lwt(
    wavelet: Wavelets,
    x: npt.NDArray[_SCT],
    *,
    bc: BoundaryCondition = BoundaryCondition.Symmetric,
    axes: int | Sequence[int]  | None = None,
    level: UnsignedInt = 0,
    out: npt.NDArray[_SCT] | None = None,
) -> npt.NDArray[_SCT]:
    ...

def ilwt(
    wavelet: Wavelets,
    x: npt.NDArray[_SCT],
    *,
    bc: BoundaryCondition = BoundaryCondition.Symmetric,
    axes: int | Sequence[int] | None = None,
    level: UnsignedInt = 0,
    out: npt.NDArray[_SCT] | None = None,
) -> npt.NDArray[_SCT]:
    ...

def lwt_adj(
    wavelet: Wavelets,
    x: npt.NDArray[_SCT],
    *,
    bc: BoundaryCondition = BoundaryCondition.Symmetric,
    axes: int | Sequence[int] | None = None,
    level: UnsignedInt = 0,
    out: npt.NDArray[_SCT] | None = None,
) -> npt.NDArray[_SCT]:
    ...

def ilwt_adj(
    wavelet: Wavelets,
    x: npt.NDArray[_SCT],
    *,
    bc: BoundaryCondition = BoundaryCondition.Symmetric,
    axes: int | Sequence[int] | None = None,
    level: UnsignedInt = 0,
    out: npt.NDArray[_SCT] | None = None,
) -> npt.NDArray[_SCT]:
    ...

# ── DWT (general boundary condition) ─────────────────────────────────────────

def dwt(
    wavelet: Wavelets,
    x: npt.NDArray[_SCT],
    *,
    bc: BoundaryCondition = BoundaryCondition.Symmetric,
    axes: int | Sequence[int] | None = None,
    level: UnsignedInt = 0,
    out: npt.NDArray[_SCT] | None = None,
) -> npt.NDArray[_SCT]:
    ...

def idwt(
    wavelet: Wavelets,
    x: npt.NDArray[_SCT],
    out: npt.NDArray[_SCT] | Sequence[int],
    *,
    bc: BoundaryCondition = BoundaryCondition.Symmetric,
    axes: int | Sequence[int] | None = None,
    level: UnsignedInt = 0,
) -> npt.NDArray[_SCT]:
    ...

def dwt_adj(
    wavelet: Wavelets,
    x: npt.NDArray[_SCT],
    out: npt.NDArray[_SCT] | Sequence[int],
    *,
    bc: BoundaryCondition = BoundaryCondition.Symmetric,
    axes: int | Sequence[int] | None = None,
    level: UnsignedInt = 0,
) -> npt.NDArray[_SCT]:
    ...

def idwt_adj(
    wavelet: Wavelets,
    x: npt.NDArray[_SCT],
    *,
    bc: BoundaryCondition = BoundaryCondition.Symmetric,
    axes: int | Sequence[int] | None = None,
    level: UnsignedInt = 0,
    out: npt.NDArray[_SCT] | None = None,
) -> npt.NDArray[_SCT]:
    ...

# ── DWT (periodic boundary condition) ────────────────────────────────────────

def dwt_per(
    wavelet: Wavelets,
    x: npt.NDArray[_SCT],
    *,
    axes: int | Sequence[int] | None = None,
    level: UnsignedInt = 0,
    out: npt.NDArray[_SCT] | None = None,
) -> npt.NDArray[_SCT]:
    ...

def idwt_per(
    wavelet: Wavelets,
    x: npt.NDArray[_SCT],
    *,
    axes: int | Sequence[int] | None = None,
    level: UnsignedInt = 0,
    out: npt.NDArray[_SCT] | None = None,
) -> npt.NDArray[_SCT]:
    ...

def dwt_per_adj(
    wavelet: Wavelets,
    x: npt.NDArray[_SCT],
    *,
    axes: int | Sequence[int] | None = None,
    level: UnsignedInt = 0,
    out: npt.NDArray[_SCT] | None = None,
) -> npt.NDArray[_SCT]:
    ...

def idwt_per_adj(
    wavelet: Wavelets,
    x: npt.NDArray[_SCT],
    *,
    axes: int | Sequence[int] | None = None,
    level: UnsignedInt = 0,
    out: npt.NDArray[_SCT] | None = None,
) -> npt.NDArray[_SCT]:
    ...
