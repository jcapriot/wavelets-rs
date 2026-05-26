use pyo3::prelude::{pyclass, pymodule};

use ndwt;
use ndwt::boundarys;
use ndwt_macros::generate_wavelet_enum;
use ndwt_macros::generate_wavelet_match_arms;

generate_wavelet_enum! {
    Wavelets,
    (Clone, Copy, Debug, PartialEq, Eq, Hash),
    {#[pyclass]}
}

impl Wavelets {
    fn to_enum(&self) -> ndwt::Wavelets {
        generate_wavelet_match_arms! {Self, self, {ndwt::Wavelets::#wvlt,}}
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
    fn to_enum(&self) -> boundarys::BoundaryCondition {
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

#[pymodule]
mod _ndwt_ext {
    use ndwt::lwt::driver;
    use pyo3::prelude::{pyfunction, PyErr, PyResult, Python};

    use numpy::{PyReadonlyArrayDyn, PyReadwriteArrayDyn};

    #[pymodule_export]
    use super::Wavelets;

    #[pymodule_export]
    use super::BoundaryCondition;

    macro_rules! implement_transform {
        ($trans_func:ident, $py:ident, $wavelet:ident, $x:ident, $y:ident, $bc:ident, $axes:ident, $level:ident) => {
            let x = $x.as_array();
            let mut y = $y.as_array_mut();

            if x.shape() != y.shape() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "input and output arrays must have the same shape",
                ));
            }
            let ndim = x.ndim();
            let axes = $axes
                .map(|v| {
                    let v = v
                        .into_iter()
                        .map(|i| {
                            if i < 0 {
                                i.rem_euclid(ndim as isize) as usize
                            } else {
                                i as usize
                            }
                        })
                        .collect::<Vec<_>>();
                    v
                })
                .unwrap_or_else(|| (0..ndim).collect::<Vec<_>>());

            if axes.iter().any(|ax| *ax >= ndim) {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "request axis is beyond the dimensionality of the input.",
                ));
            }
            $py.detach(|| {
                let bc = $bc.unwrap_or(BoundaryCondition::Symmetric).to_enum();
                let wvlt = $wavelet.to_enum();

                let level = $level.unwrap_or_else(|| {
                    let min_n = axes.iter().map(|ax| x.shape()[*ax]).min().unwrap_or(0);
                    wvlt.max_level(min_n)
                });

                let trans = driver::WaveletTransform::new(wvlt, bc);

                trans.$trans_func(&x, &mut y, &axes, level);
            });

            Ok(())
        };
    }

    #[pyfunction]
    fn forward_transform_f64<'py>(
        py: Python<'py>,
        wavelet: Wavelets,
        x: PyReadonlyArrayDyn<f64>,
        mut y: PyReadwriteArrayDyn<f64>,
        bc: Option<BoundaryCondition>,
        axes: Option<Vec<isize>>,
        level: Option<usize>,
    ) -> PyResult<()> {
        implement_transform! {par_forward_ndarray_multilevel, py, wavelet, x, y, bc, axes, level}
    }

    #[pyfunction]
    fn inverse_transform_f64<'py>(
        py: Python<'py>,
        wavelet: Wavelets,
        x: PyReadonlyArrayDyn<f64>,
        mut y: PyReadwriteArrayDyn<f64>,
        bc: Option<BoundaryCondition>,
        axes: Option<Vec<isize>>,
        level: Option<usize>,
    ) -> PyResult<()> {
        implement_transform! {par_inverse_ndarray_multilevel, py, wavelet, x, y, bc, axes, level}
    }

    #[pyfunction]
    fn adj_forward_transform_f64<'py>(
        py: Python<'py>,
        wavelet: Wavelets,
        x: PyReadonlyArrayDyn<f64>,
        mut y: PyReadwriteArrayDyn<f64>,
        bc: Option<BoundaryCondition>,
        axes: Option<Vec<isize>>,
        level: Option<usize>,
    ) -> PyResult<()> {
        implement_transform! {par_adj_forward_ndarray_multilevel, py, wavelet, x, y, bc, axes, level}
    }

    #[pyfunction]
    fn adj_inverse_transform_f64<'py>(
        py: Python<'py>,
        wavelet: Wavelets,
        x: PyReadonlyArrayDyn<f64>,
        mut y: PyReadwriteArrayDyn<f64>,
        bc: Option<BoundaryCondition>,
        axes: Option<Vec<isize>>,
        level: Option<usize>,
    ) -> PyResult<()> {
        implement_transform! {par_adj_inverse_ndarray_multilevel, py, wavelet, x, y, bc, axes, level}
    }

    #[pyfunction]
    fn forward_transform_f32<'py>(
        py: Python<'py>,
        wavelet: Wavelets,
        x: PyReadonlyArrayDyn<f32>,
        mut y: PyReadwriteArrayDyn<f32>,
        bc: Option<BoundaryCondition>,
        axes: Option<Vec<isize>>,
        level: Option<usize>,
    ) -> PyResult<()> {
        implement_transform! {par_forward_ndarray_multilevel, py, wavelet, x, y, bc, axes, level}
    }

    #[pyfunction]
    fn inverse_transform_f32<'py>(
        py: Python<'py>,
        wavelet: Wavelets,
        x: PyReadonlyArrayDyn<f32>,
        mut y: PyReadwriteArrayDyn<f32>,
        bc: Option<BoundaryCondition>,
        axes: Option<Vec<isize>>,
        level: Option<usize>,
    ) -> PyResult<()> {
        implement_transform! {par_inverse_ndarray_multilevel, py, wavelet, x, y, bc, axes, level}
    }

    #[pyfunction]
    fn adj_forward_transform_f32<'py>(
        py: Python<'py>,
        wavelet: Wavelets,
        x: PyReadonlyArrayDyn<f32>,
        mut y: PyReadwriteArrayDyn<f32>,
        bc: Option<BoundaryCondition>,
        axes: Option<Vec<isize>>,
        level: Option<usize>,
    ) -> PyResult<()> {
        implement_transform! {par_adj_forward_ndarray_multilevel, py, wavelet, x, y, bc, axes, level}
    }

    #[pyfunction]
    fn adj_inverse_transform_f32<'py>(
        py: Python<'py>,
        wavelet: Wavelets,
        x: PyReadonlyArrayDyn<f32>,
        mut y: PyReadwriteArrayDyn<f32>,
        bc: Option<BoundaryCondition>,
        axes: Option<Vec<isize>>,
        level: Option<usize>,
    ) -> PyResult<()> {
        implement_transform! {par_adj_inverse_ndarray_multilevel, py, wavelet, x, y, bc, axes, level}
    }
}
