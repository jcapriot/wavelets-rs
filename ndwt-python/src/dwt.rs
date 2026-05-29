use ndwt::dwt::driver;
use ndwt::ChunkWidth;
use ndwt::Transformable;
use numpy::{PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyReadwriteArrayDyn};
use pyo3::prelude::{Py, PyAny, PyErr, PyResult, Python};

use super::{check_axes, normalize_axes, BoundaryCondition, ValOrVec, Wavelets};

pub(crate) enum ShapeOrOutArray<'py, T: numpy::Element> {
    Shape(Vec<usize>),
    Out(PyReadwriteArrayDyn<'py, T>),
}

// Generic-BC DWT: operations whose output is *larger* than the input
// (forward and adj-inverse). Input is read-only.
macro_rules! implement_expand_transform {
    ($(($name:ident, $trans_func:ident)),*) => {
        $(
        pub(crate) fn $name<'py, T, const N: usize>(
            py: Python<'py>,
            wavelet: Wavelets,
            x: PyReadonlyArrayDyn<'py, T>,
            y: Option<PyReadwriteArrayDyn<'py, T>>,
            bc: BoundaryCondition,
            axes: Option<ValOrVec<isize>>,
            level: usize,
        ) -> PyResult<Py<PyAny>>
        where
            T: Transformable + numpy::Element + num_traits::Zero + ChunkWidth<T, N> + 'py,
        {
            let x = x.as_array();
            let ndim = x.ndim();
            let axes = normalize_axes!(axes, ndim);
            check_axes!(axes, ndim);
            let level = if level > 0  {level} else {
                ndwt::max_level_nd(wavelet.width(), x.shape(), &axes)
            };

            let out_shape = driver::get_transform_shape(x.shape(), &axes, level, wavelet.width(), false);

            let mut out = y.unwrap_or_else(|| {
                PyArrayDyn::zeros(py, out_shape.clone(), false)
                    .try_readwrite()
                    .unwrap()
            });

            let mut y = out.as_array_mut();

            if y.shape() != &out_shape{
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Expected an output shape: {:?}, from transformed input shape: {:?}, got {:?}", out_shape, x.shape(), y.shape()),
                ));
            }
            py.detach(|| {
                let bc = bc.as_ndwt_boundary_condition();
                let wvlt = wavelet.as_ndwt_wavelet();
                let trans = driver::WaveletTransform::new(wvlt, bc);

                trans.$trans_func(&x, &mut y, &axes, level);
            });
            Ok(out.as_any().clone().unbind())
        })*
    };
}

implement_expand_transform! {
    (dwt, par_forward_ndarray_multilevel),
    (idwt_adj, par_adj_inverse_ndarray_multilevel)
}

macro_rules! implement_contract_transform {
    ($(($name:ident, $trans_func:ident)),*) => {
        $(
        pub(crate) fn $name<T, const N: usize>(
            py: Python,
            wavelet: Wavelets,
            x: PyReadonlyArrayDyn<T>,
            y: ShapeOrOutArray<T>,
            bc: BoundaryCondition,
            axes: Option<ValOrVec<isize>>,
            level: usize,
        ) -> PyResult<Py<PyAny>>
        where
            T: Transformable + numpy::Element + num_traits::Zero + ChunkWidth<T, N>,
        {
            let mut x = x.as_array().to_owned();
            let ndim = x.ndim();
            let axes = normalize_axes!(axes, ndim);
            check_axes!(axes, ndim);

            let mut out = match y{
                ShapeOrOutArray::Shape(out_shape) => {
                PyArrayDyn::zeros(py, out_shape, false)
                    .try_readwrite()
                    .unwrap()
                }
                ShapeOrOutArray::Out(arr) => arr
            };

            let mut y = out.as_array_mut();

            let level = if level > 0  {level} else {
                ndwt::max_level_nd(wavelet.width(), y.shape(), &axes)
            };

            let expected_input_shape = driver::get_transform_shape(y.shape(), &axes, level, wavelet.width(), false);

            if x.shape() != &expected_input_shape{
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Expected an input shape: {:?}, from transformed output shape: {:?}, got {:?}", expected_input_shape, y.shape(), x.shape()),
                ));
            }
            py.detach(|| {
                let bc = bc.as_ndwt_boundary_condition();
                let wvlt = wavelet.as_ndwt_wavelet();
                let trans = driver::WaveletTransform::new(wvlt, bc);

                trans.$trans_func(&mut x, &mut y, &axes, level);
            });
            Ok(out.as_any().clone().unbind())
        })*
    };
}

implement_contract_transform! {
    (idwt, par_inverse_ndarray_multilevel),
    (dwt_adj, par_adj_forward_ndarray_multilevel)
}

macro_rules! implement_per_transform {
    ($(($name:ident, $trans_func:ident)),*) => {
        $(
        pub(crate) fn $name<T, const N: usize>(
            py: Python,
            wavelet: Wavelets,
            x: PyReadonlyArrayDyn<T>,
            y: Option<PyReadwriteArrayDyn<T>>,
            axes: Option<ValOrVec<isize>>,
            level: usize,
        ) -> PyResult<Py<PyAny>>
        where
            T: Transformable + numpy::Element + num_traits::Zero + ChunkWidth<T, N>,
        {
            let x = x.as_array();

            let ndim = x.ndim();
            let axes = normalize_axes!(axes, ndim);
            check_axes!(axes, ndim);

            let mut out = y.unwrap_or_else(|| {
                PyArrayDyn::zeros(py, x.shape(), false)
                    .try_readwrite()
                    .unwrap()
            });

            let mut y = out.as_array_mut();
            if y.shape() != x.shape(){
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Output shape: {:?}, is not equal to input shape: {:?}", y.shape(), x.shape()),
                ));
            }
            py.detach(|| {
                let wvlt = wavelet.as_ndwt_wavelet();
                let trans = driver::WaveletTransformPer::new(wvlt);

                trans.$trans_func(&x, &mut y, &axes, level);
            });
            Ok(out.as_any().clone().unbind())
        })*
    };
}

implement_per_transform! {
    (dwt_per, par_forward_ndarray_multilevel),
    (idwt_per, par_inverse_ndarray_multilevel),
    (dwt_per_adj, par_adj_forward_ndarray_multilevel),
    (idwt_per_adj, par_adj_inverse_ndarray_multilevel)
}
