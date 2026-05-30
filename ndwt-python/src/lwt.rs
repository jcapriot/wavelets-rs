use ndwt::lwt::driver;
use ndwt::simd::SimdTransformable;
use ndwt::ChunkWidth;
use numpy::{PyArrayDyn, PyArrayMethods, PyReadonlyArrayDyn, PyReadwriteArrayDyn};
use pyo3::prelude::{Py, PyAny, PyErr, PyResult, Python};

use super::{check_axes, normalize_axes, BoundaryCondition, ValOrVec, Wavelets};

macro_rules! implement_transform {
    ($(($name:ident, $trans_func:ident)),*) => {
        $(
        pub(crate) fn $name<T, const N: usize>(
            py: Python,
            wavelet: Wavelets,
            x: PyReadonlyArrayDyn<T>,
            y: Option<PyReadwriteArrayDyn<T>>,
            bc: BoundaryCondition,
            axes: Option<ValOrVec<isize>>,
            level: usize,
        ) -> PyResult<Py<PyAny>>
        where
            T: SimdTransformable + numpy::Element + num_traits::Zero + ChunkWidth<T, N>
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
                let bc = bc.as_ndwt_boundary_condition();
                let wvlt = wavelet.as_ndwt_wavelet();
                let trans = driver::WaveletTransform::new(wvlt, bc);

                trans.$trans_func(&x, &mut y, &axes, level);
            });
            Ok(out.as_any().clone().unbind())
        })*
    };
}

implement_transform! {
    (lwt, par_forward_ndarray_multilevel),
    (ilwt, par_inverse_ndarray_multilevel),
    (lwt_adj, par_adj_forward_ndarray_multilevel),
    (ilwt_adj, par_adj_inverse_ndarray_multilevel)
}
