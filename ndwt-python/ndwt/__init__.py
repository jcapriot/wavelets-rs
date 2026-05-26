import numpy as np

from ._ndwt_ext import (
    BoundaryCondition,
    Wavelets,
    adj_forward_transform_c32,
    adj_forward_transform_c64,
    adj_forward_transform_f32,
    adj_forward_transform_f64,
    adj_inverse_transform_c32,
    adj_inverse_transform_c64,
    adj_inverse_transform_f32,
    adj_inverse_transform_f64,
    forward_transform_c32,
    forward_transform_c64,
    forward_transform_f32,
    forward_transform_f64,
    inverse_transform_c32,
    inverse_transform_c64,
    inverse_transform_f32,
    inverse_transform_f64,
)

_FORWARD = {
    np.dtype("float32"): forward_transform_f32,
    np.dtype("float64"): forward_transform_f64,
    np.dtype("complex64"): forward_transform_c32,
    np.dtype("complex128"): forward_transform_c64,
}

_INVERSE = {
    np.dtype("float32"): inverse_transform_f32,
    np.dtype("float64"): inverse_transform_f64,
    np.dtype("complex64"): inverse_transform_c32,
    np.dtype("complex128"): inverse_transform_c64,
}

_ADJ_FORWARD = {
    np.dtype("float32"): adj_forward_transform_f32,
    np.dtype("float64"): adj_forward_transform_f64,
    np.dtype("complex64"): adj_forward_transform_c32,
    np.dtype("complex128"): adj_forward_transform_c64,
}

_ADJ_INVERSE = {
    np.dtype("float32"): adj_inverse_transform_f32,
    np.dtype("float64"): adj_inverse_transform_f64,
    np.dtype("complex64"): adj_inverse_transform_c32,
    np.dtype("complex128"): adj_inverse_transform_c64,
}

_SUPPORTED_DTYPES = "float32, float64, complex64, complex128"


def _run_transform(dispatch, x, wavelet, axes, level, bc):
    x = np.asarray(x)
    fn = dispatch.get(x.dtype)
    if fn is None:
        raise TypeError(
            f"Unsupported dtype {x.dtype!r}. Supported: {_SUPPORTED_DTYPES}"
        )
    if isinstance(axes, int):
        axes = [axes]
    actual_level = max(level, 0)
    y = np.empty_like(x)
    fn(wavelet, x, y, bc, axes, actual_level)
    return y


def lwt(x, wavelet, *, axes=None, level=0, bc=BoundaryCondition.Symmetric):
    """Forward lifting wavelet transform."""
    return _run_transform(_FORWARD, x, wavelet, axes, level, bc)


def ilwt(x, wavelet, *, axes=None, level=0, bc=BoundaryCondition.Symmetric):
    """Inverse lifting wavelet transform."""
    return _run_transform(_INVERSE, x, wavelet, axes, level, bc)


def lwt_adj(x, wavelet, *, axes=None, level=0, bc=BoundaryCondition.Symmetric):
    """Adjoint forward lifting wavelet transform."""
    return _run_transform(_ADJ_FORWARD, x, wavelet, axes, level, bc)


def ilwt_adj(x, wavelet, *, axes=None, level=0, bc=BoundaryCondition.Symmetric):
    """Adjoint inverse lifting wavelet transform."""
    return _run_transform(_ADJ_INVERSE, x, wavelet, axes, level, bc)
