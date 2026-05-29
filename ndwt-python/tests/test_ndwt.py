"""
Round-trip, adjoint, and error tests for the ndwt Python bindings.

Run after `maturin develop` inside ndwt-python/:

    pytest tests/
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
from typing import Any

import ndwt
from ndwt import (
    BoundaryCondition,
    Wavelets,
    dwt,
    dwt_adj,
    dwt_per,
    dwt_per_adj,
    get_dwt_shape,
    idwt,
    idwt_adj,
    idwt_per,
    idwt_per_adj,
    ilwt,
    ilwt_adj,
    lwt,
    lwt_adj,
    max_level,
    max_level_nd,
)

# ── Parametrisation ───────────────────────────────────────────────────────────

# Representative wavelets across families.
WAVELETS = [
    Wavelets.Daubechies4,
    Wavelets.Symlet4,
    Wavelets.Bior2_2,
    Wavelets.CDF9_7,
]

REAL_DTYPES: list[Any] = [np.float32, np.float64]
COMPLEX_DTYPES: list[Any] = [np.complex64, np.complex128]
ALL_DTYPES = REAL_DTYPES + COMPLEX_DTYPES

BCS = [BoundaryCondition.Symmetric, BoundaryCondition.Periodic, BoundaryCondition.Zero]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _rng_array(shape, dtype, seed=0):
    rng = np.random.default_rng(seed)
    if np.issubdtype(dtype, np.complexfloating):
        arr = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    else:
        arr = rng.standard_normal(shape)
    arr = arr + 10 # Shift the array values away from zero for beter rtol commparisons.
    return arr.astype(dtype)


def _inner(a, b):
    """Conjugate inner product <a, b> = Σ conj(aᵢ) bᵢ."""
    return np.vdot(a.ravel(), b.ravel())


def _tol(dtype):
    """Absolute and relative tolerances appropriate for *dtype*."""
    if dtype in (np.float32, np.complex64):
        return {"atol": 1e-4, "rtol": 1e-2}
    return {"atol": 1e-10, "rtol": 1e-10}




@pytest.fixture(params=WAVELETS)
def wavelet(request):
    return request.param

@pytest.fixture(params=ALL_DTYPES)
def dtype(request):
    return request.param

@pytest.fixture(params=BCS)
def bc(request):
    return request.param

@pytest.fixture(params=[63, 64])
def size(request):
    return request.param

# ── Utility tests ─────────────────────────────────────────────────────────────

class TestUtilities:
    def test_max_level_increases_with_n(self):
        w = Wavelets.Daubechies4
        assert max_level(w, 8) <= max_level(w, 16) <= max_level(w, 64)

    def test_max_level_zero_for_short_signal(self):
        assert max_level(Wavelets.Daubechies4, 1) == 0

    def test_max_level_nd_is_min_across_axes(self):
        w = Wavelets.Daubechies4
        shape = (64, 32)
        l_0 = max_level(w, shape[0])
        l_1 = max_level(w, shape[1])
        assert max_level_nd(w, shape) == min(l_0, l_1)

    def test_max_level_nd_single_axis(self):
        w = Wavelets.Daubechies4
        shape = (64, 32)
        assert max_level_nd(w, shape, axes=0) == max_level(w, shape[0])
        assert max_level_nd(w, shape, axes=1) == max_level(w, shape[1])

    def test_get_dwt_shape_expands(self):
        w = Wavelets.Daubechies4
        shape = (64,)
        out_shape = get_dwt_shape(w, shape)
        assert out_shape != shape
        assert out_shape[0] > shape[0]

    def test_get_dwt_shape_explicit_level(self):
        w = Wavelets.Daubechies4
        shape = (64,)
        # Different levels must produce different shapes.
        s1 = get_dwt_shape(w, shape, level=1)
        s2 = get_dwt_shape(w, shape, level=2)
        assert s1 != s2

    def test_get_dwt_shape_2d(self):
        w = Wavelets.Daubechies4
        shape = (32, 64)
        out_shape = get_dwt_shape(w, shape)
        assert len(out_shape) == 2
        assert out_shape[0] > shape[0]
        assert out_shape[1] > shape[1]

    def test_get_dwt_shape_single_axis(self):
        w = Wavelets.Daubechies4
        shape = (32, 64)
        out_shape = get_dwt_shape(w, shape, axes=1)
        assert out_shape[0] == shape[0]   # untransformed axis unchanged
        assert out_shape[1] > shape[1]

    def test_wavelet_width_positive(self, wavelet):
        assert wavelet.width() > 0

    def test_wavelet_width_known_value(self):
        # Daubechies-4 has 8 coefficients.
        assert Wavelets.Daubechies4.width() == 8


# ── LWT tests ─────────────────────────────────────────────────────────────────
class TestLWT:
    def test_roundtrip_1d(self, wavelet, dtype, bc, size):
        x = _rng_array((size,), dtype)
        npt.assert_allclose(ilwt(wavelet, lwt(wavelet, x, bc=bc), bc=bc), x, **_tol(dtype))

    def test_roundtrip_2d(self, wavelet, dtype, bc, size):
        x = _rng_array((size, size), dtype)
        npt.assert_allclose(ilwt(wavelet, lwt(wavelet, x, bc=bc), bc=bc), x, **_tol(dtype))

    @pytest.mark.parametrize("axis", [0, 1, -1, -2, -3])
    def test_roundtrip_2d_single_axis(self, wavelet, dtype, bc, size, axis):
        x = _rng_array((size, size), dtype)
        npt.assert_allclose(ilwt(wavelet, lwt(wavelet, x, axes=axis, bc=bc), axes=axis, bc=bc), x, **_tol(dtype))

    def test_output_shape_unchanged(self, wavelet, dtype):
        x = _rng_array((64,), dtype)
        assert lwt(wavelet, x).shape == x.shape

    def test_out_parameter(self, wavelet, dtype):
        x = _rng_array((64,), dtype)
        buf = np.empty_like(x)
        y = lwt(wavelet, x, out=buf)
        assert y is buf
        npt.assert_allclose(ilwt(wavelet, y), x, **_tol(dtype))

    def test_adjoint_forward(self, wavelet, dtype, bc, size):
        """⟨lwt(x), y⟩ == ⟨x, lwt_adj(y)⟩"""
        x = _rng_array((size,), dtype, seed=0)
        y = _rng_array((size,), dtype, seed=1)
        lhs = _inner(lwt(wavelet, x, bc=bc), y)
        rhs = _inner(x, lwt_adj(wavelet, y, bc=bc))
        npt.assert_allclose(lhs, rhs, **_tol(dtype))

    def test_adjoint_inverse(self, wavelet, dtype, bc, size):
        """⟨ilwt(x), y⟩ == ⟨x, ilwt_adj(y)⟩"""
        x = _rng_array((size,), dtype, seed=0)
        y = _rng_array((size,), dtype, seed=1)
        lhs = _inner(ilwt(wavelet, x, bc=bc), y)
        rhs = _inner(x, ilwt_adj(wavelet, y, bc=bc))
        npt.assert_allclose(lhs, rhs, **_tol(dtype))

    def test_adjoint_forward_2d(self, wavelet, dtype, bc, size):
        x = _rng_array((size, size), dtype, seed=0)
        y = _rng_array((size, size), dtype, seed=1)
        lhs = _inner(lwt(wavelet, x, bc=bc), y)
        rhs = _inner(x, lwt_adj(wavelet, y, bc=bc))
        npt.assert_allclose(lhs, rhs, **_tol(dtype))

    @pytest.mark.parametrize("func", [lwt, ilwt, lwt_adj, ilwt_adj])
    @pytest.mark.parametrize("shape", [(5,), (5, 5), (4, 5, 2)])
    def test_small_copy_over(self, func, shape, dtype):
        """The transform functions should just copy to the output if max_level is 0."""
        wavelet = Wavelets.Daubechies10  # has a width of 20
        x_in = _rng_array(shape, dtype, seed=2)
        assert max_level_nd(wavelet, x_in.shape) == 0
        x_2 = func(wavelet, x_in)
        npt.assert_equal(x_in, x_2)

# ── DWT tests ─────────────────────────────────────────────────────────────────

class TestDWT:
    def test_roundtrip_1d(self, wavelet, dtype, bc, size):
        x = _rng_array((size,), dtype)
        y = dwt(wavelet, x, bc=bc)
        npt.assert_allclose(idwt(wavelet, y, x.shape, bc=bc), x, **_tol(dtype))

    def test_roundtrip_2d(self, wavelet, dtype, bc, size):
        x = _rng_array((size, size), dtype)
        y = dwt(wavelet, x, bc=bc)
        npt.assert_allclose(idwt(wavelet, y, x.shape, bc=bc), x, **_tol(dtype))

    def test_roundtrip_single_axis(self, wavelet, bc, dtype, size):
        x = _rng_array((size, size), dtype)
        y = dwt(wavelet, x, axes=1, bc=bc)
        npt.assert_allclose(idwt(wavelet, y, x.shape, axes=1, bc=bc), x, **_tol(dtype))

    def test_output_shape_matches_get_dwt_shape(self, wavelet, size):
        x = _rng_array((size,), np.float64)
        y = dwt(wavelet, x)
        assert y.shape == get_dwt_shape(wavelet, x.shape)

    def test_output_shape_2d_matches_get_dwt_shape(self, wavelet):
        x = _rng_array((32, 48), np.float64)
        y = dwt(wavelet, x)
        assert y.shape == get_dwt_shape(wavelet, x.shape)

    def test_out_parameter(self, wavelet, dtype, size):
        x = _rng_array((size,), dtype)
        out_shape = get_dwt_shape(wavelet, x.shape)
        buf = np.empty(out_shape, dtype=dtype)
        y = dwt(wavelet, x, out=buf)
        assert y is buf

    def test_idwt_with_preallocated_out(self, wavelet, dtype, size):
        x = _rng_array((size,), dtype)
        y = dwt(wavelet, x)
        buf = np.empty_like(x)
        x_rec = idwt(wavelet, y, buf)
        assert x_rec is buf
        npt.assert_allclose(x_rec, x, **_tol(dtype))

    def test_adjoint_forward(self, wavelet, dtype, bc, size):
        """⟨dwt(x), y_big⟩ == ⟨x, dwt_adj(y_big, x.shape)⟩"""
        x = _rng_array((size,), dtype, seed=0)
        out_shape = get_dwt_shape(wavelet, x.shape)
        y = _rng_array(out_shape, dtype, seed=1)
        lhs = _inner(dwt(wavelet, x, bc=bc), y)
        rhs = _inner(x, dwt_adj(wavelet, y, x.shape, bc=bc))
        npt.assert_allclose(lhs, rhs, **_tol(dtype))


    def test_adjoint_inverse(self, wavelet, dtype, bc, size):
        """⟨idwt(x_big, shape), y⟩ == ⟨x_big, idwt_adj(y)⟩"""
        orig_shape = (size,)
        y = _rng_array(orig_shape, dtype, seed=0)
        big_shape = get_dwt_shape(wavelet, orig_shape)
        x_big = _rng_array(big_shape, dtype, seed=1)
        lhs = _inner(idwt(wavelet, x_big, orig_shape, bc=bc), y)
        rhs = _inner(x_big, idwt_adj(wavelet, y, bc=bc))
        npt.assert_allclose(lhs, rhs, **_tol(dtype))

 
    def test_adjoint_forward_2d(self, wavelet, dtype, bc, size):
        x = _rng_array((size, size), dtype, seed=0)
        out_shape = get_dwt_shape(wavelet, x.shape)
        y = _rng_array(out_shape, dtype, seed=1)
        lhs = _inner(dwt(wavelet, x, bc=bc), y)
        rhs = _inner(x, dwt_adj(wavelet, y, x.shape, bc=bc))
        npt.assert_allclose(lhs, rhs, **_tol(dtype))

    @pytest.mark.parametrize("func", [dwt, idwt, dwt_adj, idwt_adj])
    @pytest.mark.parametrize("shape", [(5,), (5, 5), (4, 5, 2)])
    def test_small_copy_over(self, func, shape, dtype):
        """The transform functions should just copy to the output if max_level is 0."""
        wavelet = Wavelets.Daubechies10  # has a width of 20
        x_in = _rng_array(shape, dtype, seed=2)
        assert max_level_nd(wavelet, x_in.shape) == 0
        assert get_dwt_shape(wavelet, x_in.shape, level=0) == x_in.shape
        x_2 = np.zeros_like(x_in)
        func(wavelet, x_in, out=x_2)
        npt.assert_equal(x_in, x_2)


# ── DWT-per tests ─────────────────────────────────────────────────────────────

class TestDWTPer:
    def test_roundtrip_1d(self, wavelet, dtype, size):
        x = _rng_array((size,), dtype)
        npt.assert_allclose(idwt_per(wavelet, dwt_per(wavelet, x)), x, **_tol(dtype))

    def test_roundtrip_2d(self, wavelet, dtype, size):
        x = _rng_array((size, size), dtype)
        npt.assert_allclose(idwt_per(wavelet, dwt_per(wavelet, x)), x, **_tol(dtype))

    def test_roundtrip_single_axis(self, wavelet, dtype, size):
        x = _rng_array((size, size), dtype)
        npt.assert_allclose(idwt_per(wavelet, dwt_per(wavelet, x, axes=0), axes=0), x, **_tol(dtype))

    def test_shape_preserved(self, wavelet, dtype):
        x = _rng_array((32, 64), dtype)
        assert dwt_per(wavelet, x).shape == x.shape

    def test_out_parameter(self, wavelet, dtype):
        x = _rng_array((64,), dtype)
        buf = np.empty_like(x)
        y = dwt_per(wavelet, x, out=buf)
        assert y is buf

    def test_adjoint_forward(self, wavelet, dtype, size):
        """⟨dwt_per(x), y⟩ == ⟨x, dwt_per_adj(y)⟩"""
        x = _rng_array((size,), dtype, seed=0)
        y = _rng_array((size,), dtype, seed=1)
        lhs = _inner(dwt_per(wavelet, x), y)
        rhs = _inner(x, dwt_per_adj(wavelet, y))
        npt.assert_allclose(lhs, rhs, **_tol(dtype))

    def test_adjoint_inverse(self, wavelet, dtype, size):
        """⟨idwt_per(x), y⟩ == ⟨x, idwt_per_adj(y)⟩"""
        x = _rng_array((size,), dtype, seed=0)
        y = _rng_array((size,), dtype, seed=1)
        lhs = _inner(idwt_per(wavelet, x), y)
        rhs = _inner(x, idwt_per_adj(wavelet, y))
        npt.assert_allclose(lhs, rhs, **_tol(dtype))

    def test_adjoint_forward_2d(self, wavelet, dtype, size):
        x = _rng_array((size, size), dtype, seed=0)
        y = _rng_array((size, size), dtype, seed=1)
        lhs = _inner(dwt_per(wavelet, x), y)
        rhs = _inner(x, dwt_per_adj(wavelet, y))
        npt.assert_allclose(lhs, rhs, **_tol(dtype))

    @pytest.mark.parametrize("func", [dwt_per, idwt_per, dwt_per_adj, idwt_per_adj])
    @pytest.mark.parametrize("shape", [(5,), (5, 5), (4, 5, 2)])
    def test_small_copy_over(self, func, shape, dtype):
        """The transform functions should just copy to the output if max_level is 0."""
        wavelet = Wavelets.Daubechies10  # has a width of 20
        x_in = _rng_array(shape, dtype, seed=2)
        assert max_level_nd(wavelet, x_in.shape) == 0
        x_2 = func(wavelet, x_in)
        npt.assert_equal(x_in, x_2)


# ── Error tests ───────────────────────────────────────────────────────────────

class TestErrors:
    def test_lwt_invalid_dtype(self):
        """Unsupported dtype raises TypeError."""
        x = np.ones((64,), dtype=np.int32)
        with pytest.raises(TypeError):
            lwt(Wavelets.Daubechies4, x)

    def test_dwt_invalid_dtype(self):
        x = np.ones((64,), dtype=np.int32)
        with pytest.raises(TypeError):
            dwt(Wavelets.Daubechies4, x)

    def test_dwt_per_invalid_dtype(self):
        x = np.ones((64,), dtype=np.int32)
        with pytest.raises(TypeError):
            dwt_per(Wavelets.Daubechies4, x)

    def test_lwt_axis_out_of_range(self):
        """Axis beyond array dimensionality raises ValueError."""
        x = np.ones((64,), dtype=np.float64)
        with pytest.raises(ValueError):
            lwt(Wavelets.Daubechies4, x, axes=2)

    def test_dwt_axis_out_of_range(self):
        x = np.ones((64,), dtype=np.float64)
        with pytest.raises(ValueError):
            dwt(Wavelets.Daubechies4, x, axes=2)

    def test_dwt_per_axis_out_of_range(self):
        x = np.ones((64,), dtype=np.float64)
        with pytest.raises(ValueError):
            dwt_per(Wavelets.Daubechies4, x, axes=2)

    def test_lwt_out_dtype_mismatch(self):
        """Mismatched dtypes between x and out raise ValueError."""
        x = np.ones((64,), dtype=np.float32)
        out = np.empty((64,), dtype=np.float64)
        with pytest.raises(ValueError):
            lwt(Wavelets.Daubechies4, x, out=out)

    def test_dwt_out_dtype_mismatch(self):
        x = np.ones((64,), dtype=np.float32)
        out = np.empty(get_dwt_shape(Wavelets.Daubechies4, (64,)), dtype=np.float64)
        with pytest.raises(ValueError):
            dwt(Wavelets.Daubechies4, x, out=out)

    def test_max_level_nd_axis_out_of_range(self):
        with pytest.raises(ValueError):
            max_level_nd(Wavelets.Daubechies4, (64,), axes=5)

    def test_get_dwt_shape_axis_out_of_range(self):
        with pytest.raises(ValueError):
            get_dwt_shape(Wavelets.Daubechies4, (64,), axes=5)
