use pyo3::prelude::{pyclass, pymethods, pymodule};

//mod dwt;
mod dwt;
mod lwt;
use ndwt::boundarys;
use ndwt_macros::generate_wavelet_enum;
use ndwt_macros::generate_wavelet_match_arms;
use num_complex::{Complex32 as c32, Complex64 as c64};
use numpy::{PyReadonlyArrayDyn, PyReadwriteArrayDyn};
use pyo3::FromPyObject;

generate_wavelet_enum! {
    Wavelets,
    (Clone, Copy, Debug, PartialEq, Eq, Hash),
    {#[pyclass]}
}

impl Wavelets {
    fn as_ndwt_wavelet(&self) -> ndwt::Wavelets {
        generate_wavelet_match_arms! {Self, self, {ndwt::Wavelets::#wvlt,}}
    }
}

#[pymethods]
impl Wavelets {
    fn width(&self) -> usize {
        self.as_ndwt_wavelet().width()
    }
}

#[pyclass]
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

#[pymodule]
mod _ndwt_ext {
    use pyo3::prelude::{pyfunction, Py, PyAny, PyErr, PyResult, Python};
    use pyo3::types::PyTuple;

    #[pymodule_export]
    use super::Wavelets;

    #[pymodule_export]
    use super::BoundaryCondition;

    use super::{ReadArray, ReadWriteArray, ShapeOrOutArray, ValOrVec};

    #[pyfunction]
    fn max_level(wavelet: Wavelets, n: usize) -> usize {
        wavelet.as_ndwt_wavelet().max_level(n)
    }

    #[pyfunction]
    fn max_level_nd(
        wavelet: Wavelets,
        shape: Vec<usize>,
        axes: Option<ValOrVec<isize>>,
    ) -> PyResult<usize> {
        let ndim = shape.len();
        let axes = normalize_axes!(axes, ndim);
        check_axes!(axes, ndim);
        let width = wavelet.as_ndwt_wavelet().width();
        Ok(ndwt::max_level_nd(width, &shape, &axes))
    }

    #[pyfunction]
    #[pyo3(signature = (wavelet, shape, *, axes=None, level=0))]
    fn get_dwt_shape<'py>(
        py: Python,
        wavelet: Wavelets,
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

    #[pyfunction]
    #[pyo3(signature = (wavelet, x, *, bc=BoundaryCondition::Symmetric, axes=None, level=0, out=None))]
    fn lwt<'py>(
        py: Python,
        wavelet: Wavelets,
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

    #[pyfunction]
    #[pyo3(signature = (wavelet, x, *, bc=BoundaryCondition::Symmetric, axes=None, level=0, out=None))]
    fn ilwt<'py>(
        py: Python,
        wavelet: Wavelets,
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

    #[pyfunction]
    #[pyo3(signature = (wavelet, x, *, bc=BoundaryCondition::Symmetric, axes=None, level=0, out=None))]
    fn lwt_adj<'py>(
        py: Python,
        wavelet: Wavelets,
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

    #[pyfunction]
    #[pyo3(signature = (wavelet, x, *, bc=BoundaryCondition::Symmetric, axes=None, level=0, out=None))]
    fn ilwt_adj<'py>(
        py: Python,
        wavelet: Wavelets,
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

    #[pyfunction]
    #[pyo3(signature = (wavelet, x, *, bc=BoundaryCondition::Symmetric, axes=None, level=0, out=None))]
    fn dwt<'py>(
        py: Python,
        wavelet: Wavelets,
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

    #[pyfunction]
    #[pyo3(signature = (wavelet, x, out,  *, bc=BoundaryCondition::Symmetric, axes=None, level=0))]
    fn idwt<'py>(
        py: Python,
        wavelet: Wavelets,
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

    #[pyfunction]
    #[pyo3(signature = (wavelet, x, out,  *, bc=BoundaryCondition::Symmetric, axes=None, level=0))]
    fn dwt_adj<'py>(
        py: Python,
        wavelet: Wavelets,
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

    #[pyfunction]
    #[pyo3(signature = (wavelet, x, *, bc=BoundaryCondition::Symmetric, axes=None, level=0, out=None))]
    fn idwt_adj<'py>(
        py: Python,
        wavelet: Wavelets,
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

    #[pyfunction]
    #[pyo3(signature = (wavelet, x, *, axes=None, level=0, out=None))]
    fn dwt_per<'py>(
        py: Python,
        wavelet: Wavelets,
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

    #[pyfunction]
    #[pyo3(signature = (wavelet, x, *, axes=None, level=0, out=None))]
    fn idwt_per<'py>(
        py: Python,
        wavelet: Wavelets,
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

    #[pyfunction]
    #[pyo3(signature = (wavelet, x, *, axes=None, level=0, out=None))]
    fn dwt_per_adj<'py>(
        py: Python,
        wavelet: Wavelets,
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

    #[pyfunction]
    #[pyo3(signature = (wavelet, x, *, axes=None, level=0, out=None))]
    fn idwt_per_adj<'py>(
        py: Python,
        wavelet: Wavelets,
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
