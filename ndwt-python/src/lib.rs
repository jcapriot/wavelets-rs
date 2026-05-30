use pyo3::prelude::{pyclass, pymethods, pymodule};

mod dwt;
mod lwt;
use ndwt::boundarys;
use ndwt_macros::generate_wavelet_enum;
use ndwt_macros::generate_wavelet_match_arms;
use num_complex::{Complex32 as c32, Complex64 as c64};
use numpy::{PyReadonlyArrayDyn, PyReadwriteArrayDyn};
use pyo3::FromPyObject;

generate_wavelet_enum! {
    Wavelet,
    (Clone, Copy, Debug, PartialEq, Eq, Hash),
    {
        /// Wavelets supported for transformations.
        ///
        /// Members
        /// -------
        /// Daubechies[1-10]
        ///     Daubechies family wavelets. Daubechies1 is equivalent to the Haar wavelet.
        /// Symlet[4-6]
        ///     Near-symmetric wavelets, least-asymmetric modifications of Daubechies wavelets.
        ///     Symlets[1-3] are equivalent to the corresponding Daubechies wavelet.
        /// Coiflet[1-3]
        ///     Wavelets with vanishing moments for both analysis and synthethesis filters.
        /// Bior[A_B]
        ///     Biorthogonal wavelets use separate analysis and synthesis filters. The naming
        ///     convention `BiorA_B` refers to the order of the synthesis/analysis filter pair.
        ///     A is in the range 1-6, and B is between 1 and 9, but only certain combinations
        ///     of the two are supported. Generally, they must both either be even or odd.
        /// CDF5_3, CDF9_7
        ///     The Cohen–Daubechies–Feauveau wavelets variants (also biorthogonal).
        #[pyclass(from_py_object)]
    }
}

impl Wavelet {
    fn as_ndwt_wavelet(&self) -> ndwt::Wavelet {
        generate_wavelet_match_arms! {Self, self, {ndwt::Wavelet::#wvlt,}}
    }
}

#[pymethods]
impl Wavelet {
    /// Number of filter coefficients for this wavelet.
    ///
    /// Returns
    /// -------
    /// int
    ///     Filter width (number of taps).
    fn width(&self) -> usize {
        self.as_ndwt_wavelet().width()
    }
}

/// Boundary extension mode used by non-periodic wavelet transforms.
///
/// Determines how the signal is extended beyond its edges before
/// convolution with the wavelet filter.
///
/// Members
/// -------
/// Zero
///     Extend with zeros (zero-padding).
/// Periodic
///     Treat the signal as periodic (circular extension).
/// Constant
///     Repeat the boundary sample value.
/// Symmetric
///     Even (mirror) reflection about the boundary sample.
/// Reflect
///     Odd reflection about a virtual point just outside the boundary.
/// Antisymmetric
///     Even reflection with sign flip.
/// Smooth
///     Linear extrapolation from the two outermost samples.
/// Antireflect
///     Odd-symmetric extension that preserves the first derivative.
///
/// Notes
/// -----
/// In the LWT transform, these extension modes are seperately applied to
/// the even and odd elements, and applied at each lifting step independently so
/// that each update step remains upper (or lower) triangular and trivially invertible.
#[pyclass(from_py_object)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BoundaryCondition {
    Zero,
    Periodic,
    Constant,
    Symmetric,
    Reflect,
    Antisymmetric,
    Smooth,
    Antireflect,
}

impl BoundaryCondition {
    fn as_ndwt_boundary_condition(&self) -> boundarys::BoundaryCondition {
        match self {
            Self::Zero => boundarys::BoundaryCondition::Zero,
            Self::Periodic => boundarys::BoundaryCondition::Periodic,
            Self::Constant => boundarys::BoundaryCondition::Constant,
            Self::Symmetric => boundarys::BoundaryCondition::Symmetric,
            Self::Reflect => boundarys::BoundaryCondition::Reflect,
            Self::Antisymmetric => boundarys::BoundaryCondition::Antisymmetric,
            Self::Smooth => boundarys::BoundaryCondition::Smooth,
            Self::Antireflect => boundarys::BoundaryCondition::Antireflect,
        }
    }
}

// Axes normalization shared by both macro families.
macro_rules! normalize_axes {
    ($axes:ident, $ndim:ident) => {
        $axes
            .map(|v| match v {
                ValOrVec::Val(i) => {
                    let v = if i < 0 {
                        i.rem_euclid($ndim as isize) as usize
                    } else {
                        i as usize
                    };
                    vec![v]
                }
                ValOrVec::Vector(v) => v
                    .into_iter()
                    .map(|i| {
                        if i < 0 {
                            i.rem_euclid($ndim as isize) as usize
                        } else {
                            i as usize
                        }
                    })
                    .collect::<Vec<_>>(),
            })
            .unwrap_or_else(|| (0..$ndim).collect::<Vec<_>>())
    };
}

// Check axis dims.
macro_rules! check_axes {
    ($axes:ident, $ndim:ident) => {
        if $axes.iter().any(|&ax| ax >= $ndim) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "request axis is beyond the dimensionality of shape.",
            ));
        }
    };
}

pub(crate) use check_axes;
pub(crate) use normalize_axes;

#[derive(FromPyObject)]
enum ValOrVec<T> {
    #[pyo3(transparent)]
    Val(T),
    #[pyo3(transparent)]
    Vector(Vec<T>),
}

#[derive(FromPyObject)]
enum ReadArray<'py> {
    #[pyo3(transparent)]
    Float32(PyReadonlyArrayDyn<'py, f32>), // Matches Python int
    #[pyo3(transparent)]
    Float64(PyReadonlyArrayDyn<'py, f64>), // Matches Python str
    #[pyo3(transparent)]
    Complex32(PyReadonlyArrayDyn<'py, c32>), // Matches Python str
    #[pyo3(transparent)]
    Complex64(PyReadonlyArrayDyn<'py, c64>), // Matches Python str
}

#[derive(FromPyObject)]
enum ReadWriteArray<'py> {
    #[pyo3(transparent)]
    Float32(PyReadwriteArrayDyn<'py, f32>), // Matches Python int
    #[pyo3(transparent)]
    Float64(PyReadwriteArrayDyn<'py, f64>), // Matches Python str
    #[pyo3(transparent)]
    Complex32(PyReadwriteArrayDyn<'py, c32>), // Matches Python str
    #[pyo3(transparent)]
    Complex64(PyReadwriteArrayDyn<'py, c64>), // Matches Python str
}

#[derive(FromPyObject)]
enum ShapeOrOutArray<'py> {
    #[pyo3(transparent)]
    Shape(Vec<usize>),
    #[pyo3(transparent)]
    Float32(PyReadwriteArrayDyn<'py, f32>), // Matches Python int
    #[pyo3(transparent)]
    Float64(PyReadwriteArrayDyn<'py, f64>), // Matches Python str
    #[pyo3(transparent)]
    Complex32(PyReadwriteArrayDyn<'py, c32>), // Matches Python str
    #[pyo3(transparent)]
    Complex64(PyReadwriteArrayDyn<'py, c64>), // Matches Python str
}

/// N-dimensional wavelet transforms for NumPy arrays.
///
/// See the ``ndwt`` package docstring for a full overview.
#[pymodule]
mod _ndwt_ext {
    use pyo3::prelude::{pyfunction, Py, PyAny, PyErr, PyResult, Python};
    use pyo3::types::PyTuple;

    #[pymodule_export]
    use super::Wavelet;

    #[pymodule_export]
    use super::BoundaryCondition;

    use super::{ReadArray, ReadWriteArray, ShapeOrOutArray, ValOrVec};

    /// Maximum decomposition levels for a 1-D signal.
    ///
    /// Parameters
    /// ----------
    /// wavelet : Wavelet
    ///     Wavelet family.
    /// n : int
    ///     Signal length.
    ///
    /// Returns
    /// -------
    /// int
    ///     Maximum number of decomposition levels.  Returns ``0`` when
    ///     the signal is too short for even one level.
    #[pyfunction]
    fn max_level(wavelet: Wavelet, n: usize) -> usize {
        wavelet.as_ndwt_wavelet().max_level(n)
    }

    /// Maximum decomposition levels for an N-D signal.
    ///
    /// Returns the minimum of :func:`max_level` across all transformed
    /// axes — i.e. the largest level that can be applied to every axis
    /// simultaneously.
    ///
    /// Parameters
    /// ----------
    /// wavelet : Wavelet
    ///     Wavelet family.
    /// shape : sequence of int
    ///     Array shape.
    /// axes : int | sequence of int, optional
    ///     Axes to consider.  ``None`` (default) means all axes.
    ///
    /// Returns
    /// -------
    /// int
    ///     Maximum number of decomposition levels.
    #[pyfunction]
    #[pyo3(signature = (wavelet, shape, *, axes=None))]
    fn max_level_nd(
        wavelet: Wavelet,
        shape: Vec<usize>,
        axes: Option<ValOrVec<isize>>,
    ) -> PyResult<usize> {
        let ndim = shape.len();
        let axes = normalize_axes!(axes, ndim);
        check_axes!(axes, ndim);
        let width = wavelet.as_ndwt_wavelet().width();
        Ok(ndwt::max_level_nd(width, &shape, &axes))
    }

    /// Output shape of a forward (or adjoint-inverse) multi-level DWT.
    ///
    /// For non-periodic transforms the coefficients from all decomposition
    /// levels are packed into a single expanded array; use this function to
    /// determine its shape before calling :func:`dwt` or :func:`idwt_adj`.
    ///
    /// Parameters
    /// ----------
    /// wavelet : Wavelet
    ///     Wavelet family.
    /// shape : sequence of int
    ///     Input array shape.
    /// axes : int | sequence of int, optional
    ///     Axes to transform.  ``None`` (default) means all axes.
    /// level : int, optional
    ///     Number of decomposition levels.  ``0`` (default) selects the
    ///     maximum possible level for the given ``wavelet`` and ``shape``.
    ///
    /// Returns
    /// -------
    /// tuple of int
    ///     Output shape for the forward DWT.
    #[pyfunction]
    #[pyo3(signature = (wavelet, shape, *, axes=None, level=0))]
    fn get_dwt_shape(
        py: Python,
        wavelet: Wavelet,
        shape: Vec<usize>,
        axes: Option<ValOrVec<isize>>,
        level: usize,
    ) -> PyResult<Py<PyTuple>> {
        let ndim = shape.len();
        let axes = normalize_axes!(axes, ndim);
        check_axes!(axes, ndim);
        let width = wavelet.width();
        let level = if level > 0 {
            level
        } else {
            ndwt::max_level_nd(width, &shape, &axes)
        };

        let v = ndwt::dwt::driver::get_transform_shape(&shape, &axes, level, width, false);

        let x = PyTuple::new(py, v).unwrap();

        Ok(x.unbind())
    }

    /// Forward multi-level Lifting Wavelet Transform (LWT).
    ///
    /// The output has the same shape as the input; coefficients from every
    /// decomposition level are stored in-place within the array.
    ///
    /// Parameters
    /// ----------
    /// wavelet : Wavelet
    ///     Wavelet family.
    /// x : numpy.ndarray
    ///     Input array.  Supported dtypes: ``float32``, ``float64``,
    ///     ``complex64``, ``complex128``.
    /// bc : BoundaryCondition, optional
    ///     Boundary extension mode.  Default is ``BoundaryCondition.Symmetric``.
    /// axes : int | sequence of int, optional
    ///     Axes along which to transform.  ``None`` (default) means all axes.
    /// level : int, optional
    ///     Number of decomposition levels.  ``0`` (default) selects the
    ///     maximum possible level.
    /// out : numpy.ndarray, optional
    ///     Pre-allocated output array with the same shape and dtype as ``x``.
    ///     If ``None`` (default) a new array is allocated.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     LWT coefficient array.  Same shape and dtype as ``x``.
    #[pyfunction]
    #[pyo3(signature = (wavelet, x, *, bc=BoundaryCondition::Symmetric, axes=None, level=0, out=None), text_signature = "(wavelet, x, *, bc=BoundaryCondition.Symmetric, axes=None, level=0, out=None)")]
    fn lwt(
        py: Python,
        wavelet: Wavelet,
        x: ReadArray,
        bc: BoundaryCondition,
        axes: Option<ValOrVec<isize>>,
        level: usize,
        out: Option<ReadWriteArray>,
    ) -> PyResult<Py<PyAny>> {
        match (x, out) {
            (ReadArray::Float32(x), Some(ReadWriteArray::Float32(y))) => {
                crate::lwt::lwt(py, wavelet, x, Some(y), bc, axes, level)
            }
            (ReadArray::Float64(x), Some(ReadWriteArray::Float64(y))) => {
                crate::lwt::lwt(py, wavelet, x, Some(y), bc, axes, level)
            }
            (ReadArray::Complex32(x), Some(ReadWriteArray::Complex32(y))) => {
                crate::lwt::lwt(py, wavelet, x, Some(y), bc, axes, level)
            }
            (ReadArray::Complex64(x), Some(ReadWriteArray::Complex64(y))) => {
                crate::lwt::lwt(py, wavelet, x, Some(y), bc, axes, level)
            }
            (ReadArray::Float32(x), None) => crate::lwt::lwt(py, wavelet, x, None, bc, axes, level),
            (ReadArray::Float64(x), None) => crate::lwt::lwt(py, wavelet, x, None, bc, axes, level),
            (ReadArray::Complex32(x), None) => {
                crate::lwt::lwt(py, wavelet, x, None, bc, axes, level)
            }
            (ReadArray::Complex64(x), None) => {
                crate::lwt::lwt(py, wavelet, x, None, bc, axes, level)
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "`input` and `output` arrays must be the same type",
            )),
        }
    }

    /// Inverse multi-level Lifting Wavelet Transform (LWT).
    ///
    /// Reconstructs the original signal from its LWT representation.
    /// The output has the same shape as the input.
    ///
    /// Parameters
    /// ----------
    /// wavelet : Wavelet
    ///     Wavelet family used for the forward transform.
    /// x : numpy.ndarray
    ///     LWT coefficient array.  Supported dtypes: ``float32``, ``float64``,
    ///     ``complex64``, ``complex128``.
    /// bc : BoundaryCondition, optional
    ///     Boundary extension mode.  Must match the forward transform.
    ///     Default is ``BoundaryCondition.Symmetric``.
    /// axes : int | sequence of int, optional
    ///     Axes to reconstruct.  ``None`` (default) means all axes.
    /// level : int, optional
    ///     Number of decomposition levels.  ``0`` (default) selects the
    ///     maximum possible level.
    /// out : numpy.ndarray, optional
    ///     Pre-allocated output array.  If ``None`` (default) a new array
    ///     is allocated.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Reconstructed signal.  Same shape and dtype as ``x``.
    #[pyfunction]
    #[pyo3(signature = (wavelet, x, *, bc=BoundaryCondition::Symmetric, axes=None, level=0, out=None), text_signature = "(wavelet, x, *, bc=BoundaryCondition.Symmetric, axes=None, level=0, out=None)")]
    fn ilwt(
        py: Python,
        wavelet: Wavelet,
        x: ReadArray,
        bc: BoundaryCondition,
        axes: Option<ValOrVec<isize>>,
        level: usize,
        out: Option<ReadWriteArray>,
    ) -> PyResult<Py<PyAny>> {
        match (x, out) {
            (ReadArray::Float32(x), Some(ReadWriteArray::Float32(y))) => {
                crate::lwt::ilwt(py, wavelet, x, Some(y), bc, axes, level)
            }
            (ReadArray::Float64(x), Some(ReadWriteArray::Float64(y))) => {
                crate::lwt::ilwt(py, wavelet, x, Some(y), bc, axes, level)
            }
            (ReadArray::Complex32(x), Some(ReadWriteArray::Complex32(y))) => {
                crate::lwt::ilwt(py, wavelet, x, Some(y), bc, axes, level)
            }
            (ReadArray::Complex64(x), Some(ReadWriteArray::Complex64(y))) => {
                crate::lwt::ilwt(py, wavelet, x, Some(y), bc, axes, level)
            }
            (ReadArray::Float32(x), None) => {
                crate::lwt::ilwt(py, wavelet, x, None, bc, axes, level)
            }
            (ReadArray::Float64(x), None) => {
                crate::lwt::ilwt(py, wavelet, x, None, bc, axes, level)
            }
            (ReadArray::Complex32(x), None) => {
                crate::lwt::ilwt(py, wavelet, x, None, bc, axes, level)
            }
            (ReadArray::Complex64(x), None) => {
                crate::lwt::ilwt(py, wavelet, x, None, bc, axes, level)
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "`input` and `output` arrays must be the same type",
            )),
        }
    }

    /// Adjoint of the forward Lifting Wavelet Transform.
    ///
    /// Computes the adjoint (transpose) of :func:`lwt`.  Together with
    /// :func:`ilwt_adj` this forms the adjoint pair used in iterative
    /// solvers and optimization problems.
    ///
    /// Parameters
    /// ----------
    /// wavelet : Wavelet
    ///     Wavelet family.
    /// x : numpy.ndarray
    ///     Input array.  Supported dtypes: ``float32``, ``float64``,
    ///     ``complex64``, ``complex128``.
    /// bc : BoundaryCondition, optional
    ///     Boundary extension mode.  Default is ``BoundaryCondition.Symmetric``.
    /// axes : int | sequence of int, optional
    ///     Axes along which to apply the adjoint.  ``None`` (default) means
    ///     all axes.
    /// level : int, optional
    ///     Number of decomposition levels.  ``0`` (default) selects the
    ///     maximum possible level.
    /// out : numpy.ndarray, optional
    ///     Pre-allocated output array.  If ``None`` (default) a new array
    ///     is allocated.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Result of the adjoint forward LWT.  Same shape and dtype as ``x``.
    #[pyfunction]
    #[pyo3(signature = (wavelet, x, *, bc=BoundaryCondition::Symmetric, axes=None, level=0, out=None), text_signature = "(wavelet, x, *, bc=BoundaryCondition.Symmetric, axes=None, level=0, out=None)")]
    fn lwt_adj(
        py: Python,
        wavelet: Wavelet,
        x: ReadArray,
        bc: BoundaryCondition,
        axes: Option<ValOrVec<isize>>,
        level: usize,
        out: Option<ReadWriteArray>,
    ) -> PyResult<Py<PyAny>> {
        match (x, out) {
            (ReadArray::Float32(x), Some(ReadWriteArray::Float32(y))) => {
                crate::lwt::lwt_adj(py, wavelet, x, Some(y), bc, axes, level)
            }
            (ReadArray::Float64(x), Some(ReadWriteArray::Float64(y))) => {
                crate::lwt::lwt_adj(py, wavelet, x, Some(y), bc, axes, level)
            }
            (ReadArray::Complex32(x), Some(ReadWriteArray::Complex32(y))) => {
                crate::lwt::lwt_adj(py, wavelet, x, Some(y), bc, axes, level)
            }
            (ReadArray::Complex64(x), Some(ReadWriteArray::Complex64(y))) => {
                crate::lwt::lwt_adj(py, wavelet, x, Some(y), bc, axes, level)
            }
            (ReadArray::Float32(x), None) => {
                crate::lwt::lwt_adj(py, wavelet, x, None, bc, axes, level)
            }
            (ReadArray::Float64(x), None) => {
                crate::lwt::lwt_adj(py, wavelet, x, None, bc, axes, level)
            }
            (ReadArray::Complex32(x), None) => {
                crate::lwt::lwt_adj(py, wavelet, x, None, bc, axes, level)
            }
            (ReadArray::Complex64(x), None) => {
                crate::lwt::lwt_adj(py, wavelet, x, None, bc, axes, level)
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "`input` and `output` arrays must be the same type",
            )),
        }
    }

    /// Adjoint of the inverse Lifting Wavelet Transform.
    ///
    /// Computes the adjoint (transpose) of :func:`ilwt`.  Together with
    /// :func:`lwt_adj` this forms the adjoint pair used in iterative
    /// solvers and optimization problems.
    ///
    /// Parameters
    /// ----------
    /// wavelet : Wavelet
    ///     Wavelet family.
    /// x : numpy.ndarray
    ///     Input array.  Supported dtypes: ``float32``, ``float64``,
    ///     ``complex64``, ``complex128``.
    /// bc : BoundaryCondition, optional
    ///     Boundary extension mode.  Default is ``BoundaryCondition.Symmetric``.
    /// axes : int | sequence of int, optional
    ///     Axes along which to apply the adjoint.  ``None`` (default) means
    ///     all axes.
    /// level : int, optional
    ///     Number of decomposition levels.  ``0`` (default) selects the
    ///     maximum possible level.
    /// out : numpy.ndarray, optional
    ///     Pre-allocated output array.  If ``None`` (default) a new array
    ///     is allocated.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Result of the adjoint inverse LWT.  Same shape and dtype as ``x``.
    #[pyfunction]
    #[pyo3(signature = (wavelet, x, *, bc=BoundaryCondition::Symmetric, axes=None, level=0, out=None), text_signature = "(wavelet, x, *, bc=BoundaryCondition.Symmetric, axes=None, level=0, out=None)")]
    fn ilwt_adj(
        py: Python,
        wavelet: Wavelet,
        x: ReadArray,
        bc: BoundaryCondition,
        axes: Option<ValOrVec<isize>>,
        level: usize,
        out: Option<ReadWriteArray>,
    ) -> PyResult<Py<PyAny>> {
        match (x, out) {
            (ReadArray::Float32(x), Some(ReadWriteArray::Float32(y))) => {
                crate::lwt::ilwt_adj(py, wavelet, x, Some(y), bc, axes, level)
            }
            (ReadArray::Float64(x), Some(ReadWriteArray::Float64(y))) => {
                crate::lwt::ilwt_adj(py, wavelet, x, Some(y), bc, axes, level)
            }
            (ReadArray::Complex32(x), Some(ReadWriteArray::Complex32(y))) => {
                crate::lwt::ilwt_adj(py, wavelet, x, Some(y), bc, axes, level)
            }
            (ReadArray::Complex64(x), Some(ReadWriteArray::Complex64(y))) => {
                crate::lwt::ilwt_adj(py, wavelet, x, Some(y), bc, axes, level)
            }
            (ReadArray::Float32(x), None) => {
                crate::lwt::ilwt_adj(py, wavelet, x, None, bc, axes, level)
            }
            (ReadArray::Float64(x), None) => {
                crate::lwt::ilwt_adj(py, wavelet, x, None, bc, axes, level)
            }
            (ReadArray::Complex32(x), None) => {
                crate::lwt::ilwt_adj(py, wavelet, x, None, bc, axes, level)
            }
            (ReadArray::Complex64(x), None) => {
                crate::lwt::ilwt_adj(py, wavelet, x, None, bc, axes, level)
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "`input` and `output` arrays must be the same type",
            )),
        }
    }

    /// Forward multi-level Discrete Wavelet Transform (DWT).
    ///
    /// The output shape is *larger* than the input; use :func:`get_dwt_shape`
    /// to compute the required dimensions in advance.
    ///
    /// Parameters
    /// ----------
    /// wavelet : Wavelet
    ///     Wavelet family.
    /// x : numpy.ndarray
    ///     Input array.  Supported dtypes: ``float32``, ``float64``,
    ///     ``complex64``, ``complex128``.
    /// bc : BoundaryCondition, optional
    ///     Boundary extension mode.  Default is ``BoundaryCondition.Symmetric``.
    /// axes : int | sequence of int, optional
    ///     Axes along which to transform.  ``None`` (default) means all axes.
    /// level : int, optional
    ///     Number of decomposition levels.  ``0`` (default) selects the
    ///     maximum possible level.
    /// out : numpy.ndarray, optional
    ///     Pre-allocated output array whose shape must equal
    ///     ``get_dwt_shape(wavelet, x.shape, axes=axes, level=level)``
    ///     and whose dtype must match ``x``.  If ``None`` (default) a new
    ///     array is allocated.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     DWT coefficient array.  Shape is given by :func:`get_dwt_shape`.
    #[pyfunction]
    #[pyo3(signature = (wavelet, x, *, bc=BoundaryCondition::Symmetric, axes=None, level=0, out=None), text_signature = "(wavelet, x, *, bc=BoundaryCondition.Symmetric, axes=None, level=0, out=None)")]
    fn dwt(
        py: Python,
        wavelet: Wavelet,
        x: ReadArray,
        bc: BoundaryCondition,
        axes: Option<ValOrVec<isize>>,
        level: usize,
        out: Option<ReadWriteArray>,
    ) -> PyResult<Py<PyAny>> {
        match (x, out) {
            (ReadArray::Float32(x), Some(ReadWriteArray::Float32(y))) => {
                crate::dwt::dwt(py, wavelet, x, Some(y), bc, axes, level)
            }
            (ReadArray::Float64(x), Some(ReadWriteArray::Float64(y))) => {
                crate::dwt::dwt(py, wavelet, x, Some(y), bc, axes, level)
            }
            (ReadArray::Complex32(x), Some(ReadWriteArray::Complex32(y))) => {
                crate::dwt::dwt(py, wavelet, x, Some(y), bc, axes, level)
            }
            (ReadArray::Complex64(x), Some(ReadWriteArray::Complex64(y))) => {
                crate::dwt::dwt(py, wavelet, x, Some(y), bc, axes, level)
            }
            (ReadArray::Float32(x), None) => crate::dwt::dwt(py, wavelet, x, None, bc, axes, level),
            (ReadArray::Float64(x), None) => crate::dwt::dwt(py, wavelet, x, None, bc, axes, level),
            (ReadArray::Complex32(x), None) => {
                crate::dwt::dwt(py, wavelet, x, None, bc, axes, level)
            }
            (ReadArray::Complex64(x), None) => {
                crate::dwt::dwt(py, wavelet, x, None, bc, axes, level)
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "`input` and `output` arrays must be the same type",
            )),
        }
    }

    /// Inverse multi-level Discrete Wavelet Transform (DWT).
    ///
    /// Reconstructs the original signal from its DWT representation.
    /// The input must have the expanded shape produced by :func:`dwt`.
    ///
    /// Parameters
    /// ----------
    /// wavelet : Wavelet
    ///     Wavelet family used for the forward transform.
    /// x : numpy.ndarray
    ///     DWT coefficient array with the shape produced by :func:`dwt`
    ///     for the same ``wavelet``, ``axes``, and ``level``.
    /// out : numpy.ndarray | sequence of int
    ///     Pre-allocated output array with the original signal shape, **or**
    ///     a sequence of ints specifying that shape (a new array is then
    ///     allocated).
    /// bc : BoundaryCondition, optional
    ///     Boundary extension mode.  Must match the forward transform.
    ///     Default is ``BoundaryCondition.Symmetric``.
    /// axes : int | sequence of int, optional
    ///     Axes to reconstruct.  ``None`` (default) means all axes.
    /// level : int, optional
    ///     Number of decomposition levels.  ``0`` (default) selects the
    ///     maximum possible level.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Reconstructed signal.
    ///
    /// Notes
    /// -----
    /// This function will always allocate an internal copy of the `x` array for its workspace.
    #[pyfunction]
    #[pyo3(signature = (wavelet, x, out,  *, bc=BoundaryCondition::Symmetric, axes=None, level=0), text_signature = "(wavelet, x, *, bc=BoundaryCondition.Symmetric, axes=None, level=0, out=None)")]
    fn idwt(
        py: Python,
        wavelet: Wavelet,
        x: ReadArray,
        out: ShapeOrOutArray,
        bc: BoundaryCondition,
        axes: Option<ValOrVec<isize>>,
        level: usize,
    ) -> PyResult<Py<PyAny>> {
        use crate::dwt;
        match (x, out) {
            (ReadArray::Float32(x), ShapeOrOutArray::Float32(y)) => crate::dwt::idwt(
                py,
                wavelet,
                x,
                dwt::ShapeOrOutArray::Out(y),
                bc,
                axes,
                level,
            ),
            (ReadArray::Float64(x), ShapeOrOutArray::Float64(y)) => crate::dwt::idwt(
                py,
                wavelet,
                x,
                dwt::ShapeOrOutArray::Out(y),
                bc,
                axes,
                level,
            ),
            (ReadArray::Complex32(x), ShapeOrOutArray::Complex32(y)) => crate::dwt::idwt(
                py,
                wavelet,
                x,
                dwt::ShapeOrOutArray::Out(y),
                bc,
                axes,
                level,
            ),
            (ReadArray::Complex64(x), ShapeOrOutArray::Complex64(y)) => crate::dwt::idwt(
                py,
                wavelet,
                x,
                dwt::ShapeOrOutArray::Out(y),
                bc,
                axes,
                level,
            ),
            (ReadArray::Float32(x), ShapeOrOutArray::Shape(shape)) => crate::dwt::idwt(
                py,
                wavelet,
                x,
                dwt::ShapeOrOutArray::Shape(shape),
                bc,
                axes,
                level,
            ),
            (ReadArray::Float64(x), ShapeOrOutArray::Shape(shape)) => crate::dwt::idwt(
                py,
                wavelet,
                x,
                dwt::ShapeOrOutArray::Shape(shape),
                bc,
                axes,
                level,
            ),
            (ReadArray::Complex32(x), ShapeOrOutArray::Shape(shape)) => crate::dwt::idwt(
                py,
                wavelet,
                x,
                dwt::ShapeOrOutArray::Shape(shape),
                bc,
                axes,
                level,
            ),
            (ReadArray::Complex64(x), ShapeOrOutArray::Shape(shape)) => crate::dwt::idwt(
                py,
                wavelet,
                x,
                dwt::ShapeOrOutArray::Shape(shape),
                bc,
                axes,
                level,
            ),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "`input` and `output` arrays must be the same type",
            )),
        }
    }

    /// Adjoint of the forward Discrete Wavelet Transform.
    ///
    /// Computes the adjoint (transpose) of :func:`dwt`.  The input has the
    /// expanded DWT shape; the output has the original signal shape.
    ///
    /// Parameters
    /// ----------
    /// wavelet : Wavelet
    ///     Wavelet family.
    /// x : numpy.ndarray
    ///     Input array with the expanded DWT shape.
    /// out : numpy.ndarray | sequence of int
    ///     Pre-allocated output array with the original signal shape, **or**
    ///     a sequence of ints specifying that shape (a new array is then
    ///     allocated).
    /// bc : BoundaryCondition, optional
    ///     Boundary extension mode.  Default is ``BoundaryCondition.Symmetric``.
    /// axes : int | sequence of int, optional
    ///     Axes along which to apply the adjoint.  ``None`` (default) means
    ///     all axes.
    /// level : int, optional
    ///     Number of decomposition levels.  ``0`` (default) selects the
    ///     maximum possible level.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Result of the adjoint forward DWT.
    ///
    /// Notes
    /// -----
    /// This function will always allocate an internal copy of the `x` array for its workspace.
    #[pyfunction]
    #[pyo3(signature = (wavelet, x, out,  *, bc=BoundaryCondition::Symmetric, axes=None, level=0), text_signature = "(wavelet, x, *, bc=BoundaryCondition.Symmetric, axes=None, level=0, out=None)")]
    fn dwt_adj(
        py: Python,
        wavelet: Wavelet,
        x: ReadArray,
        out: ShapeOrOutArray,
        bc: BoundaryCondition,
        axes: Option<ValOrVec<isize>>,
        level: usize,
    ) -> PyResult<Py<PyAny>> {
        use crate::dwt;
        match (x, out) {
            (ReadArray::Float32(x), ShapeOrOutArray::Float32(y)) => crate::dwt::dwt_adj(
                py,
                wavelet,
                x,
                dwt::ShapeOrOutArray::Out(y),
                bc,
                axes,
                level,
            ),
            (ReadArray::Float64(x), ShapeOrOutArray::Float64(y)) => crate::dwt::dwt_adj(
                py,
                wavelet,
                x,
                dwt::ShapeOrOutArray::Out(y),
                bc,
                axes,
                level,
            ),
            (ReadArray::Complex32(x), ShapeOrOutArray::Complex32(y)) => crate::dwt::dwt_adj(
                py,
                wavelet,
                x,
                dwt::ShapeOrOutArray::Out(y),
                bc,
                axes,
                level,
            ),
            (ReadArray::Complex64(x), ShapeOrOutArray::Complex64(y)) => crate::dwt::dwt_adj(
                py,
                wavelet,
                x,
                dwt::ShapeOrOutArray::Out(y),
                bc,
                axes,
                level,
            ),
            (ReadArray::Float32(x), ShapeOrOutArray::Shape(shape)) => crate::dwt::dwt_adj(
                py,
                wavelet,
                x,
                dwt::ShapeOrOutArray::Shape(shape),
                bc,
                axes,
                level,
            ),
            (ReadArray::Float64(x), ShapeOrOutArray::Shape(shape)) => crate::dwt::dwt_adj(
                py,
                wavelet,
                x,
                dwt::ShapeOrOutArray::Shape(shape),
                bc,
                axes,
                level,
            ),
            (ReadArray::Complex32(x), ShapeOrOutArray::Shape(shape)) => crate::dwt::dwt_adj(
                py,
                wavelet,
                x,
                dwt::ShapeOrOutArray::Shape(shape),
                bc,
                axes,
                level,
            ),
            (ReadArray::Complex64(x), ShapeOrOutArray::Shape(shape)) => crate::dwt::dwt_adj(
                py,
                wavelet,
                x,
                dwt::ShapeOrOutArray::Shape(shape),
                bc,
                axes,
                level,
            ),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "`input` and `output` arrays must be the same type",
            )),
        }
    }

    /// Adjoint of the inverse Discrete Wavelet Transform.
    ///
    /// Computes the adjoint (transpose) of :func:`idwt`.  The input has the
    /// original signal shape; the output shape is *larger*, matching
    /// :func:`get_dwt_shape`.
    ///
    /// Parameters
    /// ----------
    /// wavelet : Wavelet
    ///     Wavelet family.
    /// x : numpy.ndarray
    ///     Input array with the original signal shape.
    /// bc : BoundaryCondition, optional
    ///     Boundary extension mode.  Default is ``BoundaryCondition.Symmetric``.
    /// axes : int or sequence of int, optional
    ///     Axes along which to apply the adjoint.  ``None`` (default) means
    ///     all axes.
    /// level : int, optional
    ///     Number of decomposition levels.  ``0`` (default) selects the
    ///     maximum possible level.
    /// out : numpy.ndarray, optional
    ///     Pre-allocated output array whose shape must equal
    ///     ``get_dwt_shape(wavelet, x.shape, axes=axes, level=level)``.
    ///     If ``None`` (default) a new array is allocated.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Result of the adjoint inverse DWT.  Shape given by
    ///     :func:`get_dwt_shape`.
    #[pyfunction]
    #[pyo3(signature = (wavelet, x, *, bc=BoundaryCondition::Symmetric, axes=None, level=0, out=None), text_signature = "(wavelet, x, *, bc=BoundaryCondition.Symmetric, axes=None, level=0, out=None)")]
    fn idwt_adj(
        py: Python,
        wavelet: Wavelet,
        x: ReadArray,
        bc: BoundaryCondition,
        axes: Option<ValOrVec<isize>>,
        level: usize,
        out: Option<ReadWriteArray>,
    ) -> PyResult<Py<PyAny>> {
        match (x, out) {
            (ReadArray::Float32(x), Some(ReadWriteArray::Float32(y))) => {
                crate::dwt::idwt_adj(py, wavelet, x, Some(y), bc, axes, level)
            }
            (ReadArray::Float64(x), Some(ReadWriteArray::Float64(y))) => {
                crate::dwt::idwt_adj(py, wavelet, x, Some(y), bc, axes, level)
            }
            (ReadArray::Complex32(x), Some(ReadWriteArray::Complex32(y))) => {
                crate::dwt::idwt_adj(py, wavelet, x, Some(y), bc, axes, level)
            }
            (ReadArray::Complex64(x), Some(ReadWriteArray::Complex64(y))) => {
                crate::dwt::idwt_adj(py, wavelet, x, Some(y), bc, axes, level)
            }
            (ReadArray::Float32(x), None) => {
                crate::dwt::idwt_adj(py, wavelet, x, None, bc, axes, level)
            }
            (ReadArray::Float64(x), None) => {
                crate::dwt::idwt_adj(py, wavelet, x, None, bc, axes, level)
            }
            (ReadArray::Complex32(x), None) => {
                crate::dwt::idwt_adj(py, wavelet, x, None, bc, axes, level)
            }
            (ReadArray::Complex64(x), None) => {
                crate::dwt::idwt_adj(py, wavelet, x, None, bc, axes, level)
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "`input` and `output` arrays must be the same type",
            )),
        }
    }

    /// Forward multi-level periodic Discrete Wavelet Transform (DWT).
    ///
    /// Uses circular (periodic) boundary extension.  The output shape
    /// always equals the input shape.  Each transformed axis must have
    /// even length.
    ///
    /// Parameters
    /// ----------
    /// wavelet : Wavelet
    ///     Wavelet family.
    /// x : numpy.ndarray
    ///     Input array.  Supported dtypes: ``float32``, ``float64``,
    ///     ``complex64``, ``complex128``.
    /// axes : int | sequence of int, optional
    ///     Axes along which to transform.  ``None`` (default) means all axes.
    /// level : int, optional
    ///     Number of decomposition levels.  ``0`` (default) selects the
    ///     maximum possible level.
    /// out : numpy.ndarray, optional
    ///     Pre-allocated output array with the same shape and dtype as ``x``.
    ///     If ``None`` (default) a new array is allocated.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Periodic DWT coefficient array.  Same shape and dtype as ``x``.
    #[pyfunction]
    #[pyo3(signature = (wavelet, x, *, axes=None, level=0, out=None))]
    fn dwt_per(
        py: Python,
        wavelet: Wavelet,
        x: ReadArray,
        axes: Option<ValOrVec<isize>>,
        level: usize,
        out: Option<ReadWriteArray>,
    ) -> PyResult<Py<PyAny>> {
        match (x, out) {
            (ReadArray::Float32(x), Some(ReadWriteArray::Float32(y))) => {
                crate::dwt::dwt_per(py, wavelet, x, Some(y), axes, level)
            }
            (ReadArray::Float64(x), Some(ReadWriteArray::Float64(y))) => {
                crate::dwt::dwt_per(py, wavelet, x, Some(y), axes, level)
            }
            (ReadArray::Complex32(x), Some(ReadWriteArray::Complex32(y))) => {
                crate::dwt::dwt_per(py, wavelet, x, Some(y), axes, level)
            }
            (ReadArray::Complex64(x), Some(ReadWriteArray::Complex64(y))) => {
                crate::dwt::dwt_per(py, wavelet, x, Some(y), axes, level)
            }
            (ReadArray::Float32(x), None) => crate::dwt::dwt_per(py, wavelet, x, None, axes, level),
            (ReadArray::Float64(x), None) => crate::dwt::dwt_per(py, wavelet, x, None, axes, level),
            (ReadArray::Complex32(x), None) => {
                crate::dwt::dwt_per(py, wavelet, x, None, axes, level)
            }
            (ReadArray::Complex64(x), None) => {
                crate::dwt::dwt_per(py, wavelet, x, None, axes, level)
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "`input` and `output` arrays must be the same type",
            )),
        }
    }

    /// Inverse multi-level periodic Discrete Wavelet Transform (DWT).
    ///
    /// Reconstructs the original signal from its periodic DWT representation.
    /// The output shape equals the input shape.
    ///
    /// Parameters
    /// ----------
    /// wavelet : Wavelet
    ///     Wavelet family used for the forward transform.
    /// x : numpy.ndarray
    ///     Periodic DWT coefficient array.  Supported dtypes: ``float32``,
    ///     ``float64``, ``complex64``, ``complex128``.
    /// axes : int | sequence of int, optional
    ///     Axes to reconstruct.  ``None`` (default) means all axes.
    /// level : int, optional
    ///     Number of decomposition levels.  ``0`` (default) selects the
    ///     maximum possible level.
    /// out : numpy.ndarray, optional
    ///     Pre-allocated output array.  If ``None`` (default) a new array
    ///     is allocated.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Reconstructed signal.  Same shape and dtype as ``x``.
    #[pyfunction]
    #[pyo3(signature = (wavelet, x, *, axes=None, level=0, out=None))]
    fn idwt_per(
        py: Python,
        wavelet: Wavelet,
        x: ReadArray,
        axes: Option<ValOrVec<isize>>,
        level: usize,
        out: Option<ReadWriteArray>,
    ) -> PyResult<Py<PyAny>> {
        match (x, out) {
            (ReadArray::Float32(x), Some(ReadWriteArray::Float32(y))) => {
                crate::dwt::idwt_per(py, wavelet, x, Some(y), axes, level)
            }
            (ReadArray::Float64(x), Some(ReadWriteArray::Float64(y))) => {
                crate::dwt::idwt_per(py, wavelet, x, Some(y), axes, level)
            }
            (ReadArray::Complex32(x), Some(ReadWriteArray::Complex32(y))) => {
                crate::dwt::idwt_per(py, wavelet, x, Some(y), axes, level)
            }
            (ReadArray::Complex64(x), Some(ReadWriteArray::Complex64(y))) => {
                crate::dwt::idwt_per(py, wavelet, x, Some(y), axes, level)
            }
            (ReadArray::Float32(x), None) => {
                crate::dwt::idwt_per(py, wavelet, x, None, axes, level)
            }
            (ReadArray::Float64(x), None) => {
                crate::dwt::idwt_per(py, wavelet, x, None, axes, level)
            }
            (ReadArray::Complex32(x), None) => {
                crate::dwt::idwt_per(py, wavelet, x, None, axes, level)
            }
            (ReadArray::Complex64(x), None) => {
                crate::dwt::idwt_per(py, wavelet, x, None, axes, level)
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "`input` and `output` arrays must be the same type",
            )),
        }
    }

    /// Adjoint of the forward periodic Discrete Wavelet Transform.
    ///
    /// Computes the adjoint (transpose) of :func:`dwt_per`.  The output
    /// shape equals the input shape.
    ///
    /// Parameters
    /// ----------
    /// wavelet : Wavelet
    ///     Wavelet family.
    /// x : numpy.ndarray
    ///     Periodic DWT coefficient array.  Supported dtypes: ``float32``,
    ///     ``float64``, ``complex64``, ``complex128``.
    /// axes : int | sequence of int, optional
    ///     Axes along which to apply the adjoint.  ``None`` (default) means
    ///     all axes.
    /// level : int, optional
    ///     Number of decomposition levels.  ``0`` (default) selects the
    ///     maximum possible level.
    /// out : numpy.ndarray, optional
    ///     Pre-allocated output array.  If ``None`` (default) a new array
    ///     is allocated.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Result of the adjoint forward periodic DWT.  Same shape and
    ///     dtype as ``x``.
    #[pyfunction]
    #[pyo3(signature = (wavelet, x, *, axes=None, level=0, out=None))]
    fn dwt_per_adj(
        py: Python,
        wavelet: Wavelet,
        x: ReadArray,
        axes: Option<ValOrVec<isize>>,
        level: usize,
        out: Option<ReadWriteArray>,
    ) -> PyResult<Py<PyAny>> {
        match (x, out) {
            (ReadArray::Float32(x), Some(ReadWriteArray::Float32(y))) => {
                crate::dwt::dwt_per_adj(py, wavelet, x, Some(y), axes, level)
            }
            (ReadArray::Float64(x), Some(ReadWriteArray::Float64(y))) => {
                crate::dwt::dwt_per_adj(py, wavelet, x, Some(y), axes, level)
            }
            (ReadArray::Complex32(x), Some(ReadWriteArray::Complex32(y))) => {
                crate::dwt::dwt_per_adj(py, wavelet, x, Some(y), axes, level)
            }
            (ReadArray::Complex64(x), Some(ReadWriteArray::Complex64(y))) => {
                crate::dwt::dwt_per_adj(py, wavelet, x, Some(y), axes, level)
            }
            (ReadArray::Float32(x), None) => {
                crate::dwt::dwt_per_adj(py, wavelet, x, None, axes, level)
            }
            (ReadArray::Float64(x), None) => {
                crate::dwt::dwt_per_adj(py, wavelet, x, None, axes, level)
            }
            (ReadArray::Complex32(x), None) => {
                crate::dwt::dwt_per_adj(py, wavelet, x, None, axes, level)
            }
            (ReadArray::Complex64(x), None) => {
                crate::dwt::dwt_per_adj(py, wavelet, x, None, axes, level)
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "`input` and `output` arrays must be the same type",
            )),
        }
    }

    /// Adjoint of the inverse periodic Discrete Wavelet Transform.
    ///
    /// Computes the adjoint (transpose) of :func:`idwt_per`.  The output
    /// shape equals the input shape.
    ///
    /// Parameters
    /// ----------
    /// wavelet : Wavelet
    ///     Wavelet family.
    /// x : numpy.ndarray
    ///     Input array with the original signal shape.  Supported dtypes:
    ///     ``float32``, ``float64``, ``complex64``, ``complex128``.
    /// axes : int | sequence of int, optional
    ///     Axes along which to apply the adjoint.  ``None`` (default) means
    ///     all axes.
    /// level : int, optional
    ///     Number of decomposition levels.  ``0`` (default) selects the
    ///     maximum possible level.
    /// out : numpy.ndarray, optional
    ///     Pre-allocated output array.  If ``None`` (default) a new array
    ///     is allocated.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray
    ///     Result of the adjoint inverse periodic DWT.  Same shape and
    ///     dtype as ``x``.
    #[pyfunction]
    #[pyo3(signature = (wavelet, x, *, axes=None, level=0, out=None))]
    fn idwt_per_adj(
        py: Python,
        wavelet: Wavelet,
        x: ReadArray,
        axes: Option<ValOrVec<isize>>,
        level: usize,
        out: Option<ReadWriteArray>,
    ) -> PyResult<Py<PyAny>> {
        match (x, out) {
            (ReadArray::Float32(x), Some(ReadWriteArray::Float32(y))) => {
                crate::dwt::idwt_per_adj(py, wavelet, x, Some(y), axes, level)
            }
            (ReadArray::Float64(x), Some(ReadWriteArray::Float64(y))) => {
                crate::dwt::idwt_per_adj(py, wavelet, x, Some(y), axes, level)
            }
            (ReadArray::Complex32(x), Some(ReadWriteArray::Complex32(y))) => {
                crate::dwt::idwt_per_adj(py, wavelet, x, Some(y), axes, level)
            }
            (ReadArray::Complex64(x), Some(ReadWriteArray::Complex64(y))) => {
                crate::dwt::idwt_per_adj(py, wavelet, x, Some(y), axes, level)
            }
            (ReadArray::Float32(x), None) => {
                crate::dwt::idwt_per_adj(py, wavelet, x, None, axes, level)
            }
            (ReadArray::Float64(x), None) => {
                crate::dwt::idwt_per_adj(py, wavelet, x, None, axes, level)
            }
            (ReadArray::Complex32(x), None) => {
                crate::dwt::idwt_per_adj(py, wavelet, x, None, axes, level)
            }
            (ReadArray::Complex64(x), None) => {
                crate::dwt::idwt_per_adj(py, wavelet, x, None, axes, level)
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "`input` and `output` arrays must be the same type",
            )),
        }
    }
}
