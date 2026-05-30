"""
ndwt — N-dimensional wavelet transforms for NumPy arrays.

Provides forward, inverse, and adjoint wavelet transforms via two families:

* **LWT** (Lifting Wavelet Transform): in-place factored form; output shape
  always equals input shape.
* **DWT** (Discrete Wavelet Transform): classic convolution/subsampling form;
  output shape is *larger* than the input for non-periodic boundary conditions.

Both families operate on ``float32``, ``float64``, ``complex64``, and
``complex128`` NumPy arrays and support N-D transforms along any subset of
axes. The operation is parallelized for N-D transforms when N>2.

Enumerations
------------
Wavelets
    Supported wavelet families (Daubechies, Symlets, Coiflets, Biorthogonal,
    CDF).  Every transform function accepts a ``Wavelets`` member as its first
    argument.
BoundaryCondition
    Signal extension modes used by non-periodic transforms: ``Zero``,
    ``Periodic``, ``Constant``, ``Symmetric``, ``Reflect``,
    ``Antisymmetric``, ``Smooth``, ``Antireflect``.

LWT
---
lwt(wavelet, x, *, bc, axes, level, out)
    Forward lifting wavelet transform.
ilwt(wavelet, x, *, bc, axes, level, out)
    Inverse lifting wavelet transform.
lwt_adj(wavelet, x, *, bc, axes, level, out)
    Adjoint of the forward LWT.
ilwt_adj(wavelet, x, *, bc, axes, level, out)
    Adjoint of the inverse LWT.

DWT (general boundary condition)
---------------------------------
dwt(wavelet, x, *, bc, axes, level, out)
    Forward DWT; output is larger than input.
idwt(wavelet, x, out, *, bc, axes, level)
    Inverse DWT; ``out`` supplies the target shape or buffer.
dwt_adj(wavelet, x, out, *, bc, axes, level)
    Adjoint of the forward DWT; ``out`` supplies the target shape or buffer.
idwt_adj(wavelet, x, *, bc, axes, level, out)
    Adjoint of the inverse DWT; output is larger than input.

DWT (periodic boundary condition)
-----------------------------------
dwt_per(wavelet, x, *, axes, level, out)
    Forward periodic DWT; output shape equals input shape.
idwt_per(wavelet, x, *, axes, level, out)
    Inverse periodic DWT.
dwt_per_adj(wavelet, x, *, axes, level, out)
    Adjoint of the forward periodic DWT.
idwt_per_adj(wavelet, x, *, axes, level, out)
    Adjoint of the inverse periodic DWT.

Utilities
---------
get_dwt_shape(wavelet, shape, *, axes, level)
    Compute the output shape of a forward DWT without running the transform.
max_level(wavelet, n)
    Maximum decomposition levels for a 1-D signal of length *n*.
max_level_nd(wavelet, shape, axes)
    Maximum decomposition levels for an N-D signal.
"""

from ._ndwt_ext import *
