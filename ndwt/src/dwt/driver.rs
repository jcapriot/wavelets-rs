#[cfg(feature = "ndarray")]
use ndarray::{ArrayRef, Dimension};
use num_traits::Zero;
use std::collections::HashSet;

use aligned_vec::avec;

use crate::boundarys::BoundaryExtension;
use crate::dwt::{DiscreteTransform, get_outlen};
use crate::iter::LanesIterator;
use crate::{Wavelets, max_level_nd};

use crate::{ChunkWidth, Transformable};
use ndwt_macros::generate_wavelet_match_arms;

macro_rules! assert_slice_matches_shape {
    ($label:expr, $slice:expr, $shape:expr $(,)?) => {{
        let slice_len = $slice.len();

        let expected_len: usize = $shape.iter().product();

        assert!(
            slice_len == expected_len,
            "{}: slice length mismatch (got {}, expected {} from shape {:?})",
            $label,
            slice_len,
            expected_len,
            $shape
        );
    }};
}

/// Compute the output shape of a multi-level DWT applied along the given `axes`.
///
/// For each transformed axis the output length grows because the non-periodic DWT
/// pads sub-bands based on the filter `width`.  Each level appends the detail
/// sub-band length along that axis; the final level appends both approximation and
/// detail lengths.
///
/// When `per_mode` is `true` (periodic DWT) the output shape equals the input shape.
///
/// # Panics
///
/// Panics if any element of `axes` is `>= in_shape.len()`.
#[track_caller]
pub fn get_transform_shape<'a, IT: IntoIterator<Item = &'a usize>>(
    in_shape: &[usize],
    axes: IT,
    level: usize,
    width: usize,
    per_mode: bool,
) -> Vec<usize> {
    let ndim = in_shape.len();
    let axes: HashSet<usize> = HashSet::from_iter(axes.into_iter().cloned());
    assert!(axes.iter().all(|i| *i < ndim));
    let mut lvl_shape = in_shape.to_owned();
    if per_mode {
        // In per mode, the output shape is the same as the input shape, since odd length transforms are
        // handled by copying the last element to the end of the approximation coefficients.
        return lvl_shape;
    }
    // initialize as shape of input array to copy un-transformed axes
    let mut out_shape = in_shape.to_owned();
    // transformed axes will be replaced by approximation and detail coefficients, so we initialize them to 0 and add the lengths of the coefficients in the loop below.
    for &ax in axes.iter() {
        out_shape[ax] = 0;
    }

    for lvl in 0..level {
        for &ax in axes.iter() {
            let n_ax = lvl_shape[ax];
            let nds = get_outlen(width, n_ax);
            if lvl + 1 < level {
                out_shape[ax] += nds;
            } else {
                out_shape[ax] += 2 * nds;
            }
            lvl_shape[ax] = nds;
        }
    }
    out_shape
}

/// High-level Discrete Wavelet Transform driver.
///
/// `WaveletTransform` owns function pointers to the forward, inverse, and adjoint
/// kernels for a chosen wavelet and boundary condition.  They are resolved at
/// construction time so each transform call is a direct, non-virtual dispatch.
///
/// The const generic `N` ties this driver to appropriate cache sizes for the
/// current processor.
/// See [`ChunkWidth`] for the correct value per element type.
///
/// Unlike the LWT driver, the DWT output shape differs from the input shape for
/// non-periodic transforms.  Use [`get_transform_shape`] to compute the required
/// output buffer size.
pub struct WaveletTransform<T, BC, const N: usize>
where
    T: ChunkWidth<T, N>,
{
    dwt_forward: fn(&[T], &mut [T], &mut [T], &BC),
    dwt_inverse: fn(&[T], &[T], &mut [T]),
    dwt_adj_forward: fn(&[T], &[T], &mut [T], &BC),
    dwt_adj_inverse: fn(&[T], &mut [T], &mut [T]),
    bc: BC,
    width: usize,
}

impl<T, BC, const N: usize> WaveletTransform<T, BC, N>
where
    T: Transformable + Zero + ChunkWidth<T, N>,
    BC: BoundaryExtension,
{
    /// Construct a `WaveletTransform` for the given wavelet family `wvlt` and boundary
    /// condition `bc`.
    ///
    /// Function pointers to the correct DWT implementations are resolved at construction
    /// time so that every subsequent transform call is a direct (non-virtual) dispatch
    /// with no runtime branching on the wavelet type.
    pub fn new(wvlt: Wavelets, bc: BC) -> Self {
        use crate::dwt::bior::*;
        use crate::dwt::coiflet::*;
        use crate::dwt::daubechies::*;
        use crate::dwt::symlet::*;
        let dwt_forward: fn(&[T], &mut [T], &mut [T], &BC) = generate_wavelet_match_arms! {
            Wavelets,
            wvlt,
            {#wvlt::forward,}
        };
        let dwt_inverse: fn(&[T], &[T], &mut [T]) = generate_wavelet_match_arms! {
            Wavelets,
            wvlt,
            {#wvlt::inverse,}
        };
        let dwt_adj_forward: fn(&[T], &[T], &mut [T], &BC) = generate_wavelet_match_arms! {
            Wavelets,
            wvlt,
            {#wvlt::adjoint_forward,}
        };
        let dwt_adj_inverse: fn(&[T], &mut [T], &mut [T]) = generate_wavelet_match_arms! {
            Wavelets,
            wvlt,
            {#wvlt::adjoint_inverse,}
        };

        let width = wvlt.width();
        Self {
            dwt_forward,
            dwt_inverse,
            dwt_adj_forward,
            dwt_adj_inverse,
            bc,
            width,
        }
    }

    /// Single-level forward DWT of a 1-D signal.
    ///
    /// `s` and `d` must each have length `get_outlen(width, input.len())`.
    ///
    /// # Panics
    ///
    /// Panics if `s.len() != d.len()` or if either length differs from
    /// `get_outlen(width, input.len())`.
    #[track_caller]
    pub fn forward_1d(&self, input: &[T], s: &mut [T], d: &mut [T]) {
        (self.dwt_forward)(input, s, d, &self.bc);
    }

    /// Single-level inverse DWT of a 1-D signal.
    ///
    /// Reconstructs the signal from approximation `s` and detail `d`.
    ///
    /// # Panics
    ///
    /// Panics if `s.len() != d.len()` or if either length differs from
    /// `get_outlen(width, output.len())`.
    #[track_caller]
    pub fn inverse_1d(&self, s: &[T], d: &[T], output: &mut [T]) {
        (self.dwt_inverse)(s, d, output);
    }

    /// Adjoint of the forward 1-D DWT.
    ///
    /// Takes approximation `s` and detail `d` sub-bands and reconstructs a signal of
    /// the same length as the original input to the forward transform.  Has the same
    /// shape semantics as [`inverse_1d`](Self::inverse_1d).
    ///
    /// # Panics
    ///
    /// Panics if `s.len() != d.len()` or if either length differs from
    /// `get_outlen(width, output.len())`.
    #[track_caller]
    pub fn adj_forward_1d(&self, s: &[T], d: &[T], output: &mut [T]) {
        (self.dwt_adj_forward)(s, d, output, &self.bc);
    }

    /// Adjoint of the inverse 1-D DWT.
    ///
    /// # Panics
    ///
    /// Panics if `s.len() != d.len()` or if either length differs from
    /// `get_outlen(width, input.len())`.
    #[track_caller]
    pub fn adj_inverse_1d(&self, input: &[T], s: &mut [T], d: &mut [T]) {
        (self.dwt_adj_inverse)(input, s, d);
    }

    /// Single-level forward DWT applied along each axis in `axes` of an N-D array.
    ///
    /// `input` must be a flat slice whose logical shape is `shape`.  `output` must
    /// have the shape returned by `get_transform_shape(shape, axes, 1, width, false)`.
    ///
    /// # Panics
    ///
    /// See [`forward_multilevel_nd`](Self::forward_multilevel_nd).
    #[track_caller]
    pub fn forward_nd(&self, input: &[T], output: &mut [T], shape: &[usize], axes: &[usize]) {
        self.forward_multilevel_nd(input, output, shape, axes, 1);
    }

    /// Single-level inverse DWT on an N-D array.
    ///
    /// # Panics
    ///
    /// See [`inverse_multilevel_nd`](Self::inverse_multilevel_nd).
    #[track_caller]
    pub fn inverse_nd(&self, input: &mut [T], output: &mut [T], shape: &[usize], axes: &[usize]) {
        self.inverse_multilevel_nd(input, output, shape, axes, 1);
    }

    /// Single-level adjoint of the forward DWT on an N-D array.
    ///
    /// `shape` is the original signal shape; `input` must have the transform-expanded shape
    /// `get_transform_shape(shape, axes, 1, width, false)`.
    /// `input` is `&mut` for the same reason as [`adj_forward_multilevel_nd`](Self::adj_forward_multilevel_nd).
    ///
    /// # Panics
    ///
    /// See [`adj_forward_multilevel_nd`](Self::adj_forward_multilevel_nd).
    #[track_caller]
    pub fn adj_forward_nd(
        &self,
        input: &mut [T],
        output: &mut [T],
        shape: &[usize],
        axes: &[usize],
    ) {
        self.adj_forward_multilevel_nd(input, output, shape, axes, 1);
    }

    /// Single-level adjoint of the inverse DWT on an N-D array.
    ///
    /// # Panics
    ///
    /// See [`adj_inverse_multilevel_nd`](Self::adj_inverse_multilevel_nd).
    #[track_caller]
    pub fn adj_inverse_nd(&self, input: &[T], output: &mut [T], shape: &[usize], axes: &[usize]) {
        self.adj_inverse_multilevel_nd(input, output, shape, axes, 1);
    }

    /// Multi-level forward DWT on an N-D array.
    ///
    /// Applies `level` successive single-level forward transforms along each axis in
    /// `axes`, recursively decomposing the approximation sub-band.
    ///
    /// # Panics
    ///
    /// Panics if any element of `axes` is `>= in_shape.len()`, if `input.len()` does
    /// not equal `in_shape.iter().product()`, or if `output.len()` does not equal the
    /// product of `get_transform_shape(in_shape, axes, level, width, false)`.
    #[track_caller]
    pub fn forward_multilevel_nd(
        &self,
        input: &[T],
        output: &mut [T],
        in_shape: &[usize],
        axes: &[usize],
        level: usize,
    ) {
        let level = if level == 0 {
            max_level_nd(self.width, in_shape, axes)
        } else {
            level
        };
        let out_shape = get_transform_shape(in_shape, axes, level, self.width, false);
        assert_slice_matches_shape!("input", input, in_shape);
        assert_slice_matches_shape!("output", output, out_shape);
        general_nd_forward_multilevel(
            |x, s, d| (self.dwt_forward)(x, s, d, &self.bc),
            input,
            output,
            in_shape,
            &out_shape,
            TransformParams {
                axes,
                level,
                width: self.width,
            },
        );
    }

    /// Multi-level inverse DWT on an N-D array.
    ///
    /// `out_shape` is the shape of the *original* signal (before decomposition).
    /// The input must have the shape produced by `forward_multilevel_nd` with the
    /// same `level` and `axes`.
    ///
    /// # Panics
    ///
    /// Panics if any element of `axes` is `>= out_shape.len()`, if `output.len()` does
    /// not equal `out_shape.iter().product()`, or if `input.len()` does not equal the
    /// product of `get_transform_shape(out_shape, axes, level, width, false)`.
    #[track_caller]
    pub fn inverse_multilevel_nd(
        &self,
        input: &mut [T],
        output: &mut [T],
        out_shape: &[usize],
        axes: &[usize],
        level: usize,
    ) {
        let level = if level == 0 {
            max_level_nd(self.width, out_shape, axes)
        } else {
            level
        };
        let in_shape = get_transform_shape(out_shape, axes, level, self.width, false);
        assert_slice_matches_shape!("input", input, in_shape);
        assert_slice_matches_shape!("output", output, out_shape);
        general_nd_inverse_multilevel(
            |s, d, x| (self.dwt_inverse)(s, d, x),
            input,
            output,
            &in_shape,
            out_shape,
            TransformParams {
                axes,
                level,
                width: self.width,
            },
        );
    }

    /// Multi-level adjoint of the forward DWT on an N-D array.
    ///
    /// `out_shape` is the shape of the *original* signal (before decomposition).
    /// The input must have the transform-expanded shape returned by
    /// `get_transform_shape(out_shape, axes, level, width, false)`.
    /// Mirrors [`inverse_multilevel_nd`](Self::inverse_multilevel_nd).
    ///
    /// `input` is taken as `&mut` because the internal multilevel algorithm writes
    /// intermediate results back into the buffer during reconstruction.
    ///
    /// # Panics
    ///
    /// Same conditions as [`inverse_multilevel_nd`](Self::inverse_multilevel_nd).
    #[track_caller]
    pub fn adj_forward_multilevel_nd(
        &self,
        input: &mut [T],
        output: &mut [T],
        out_shape: &[usize],
        axes: &[usize],
        level: usize,
    ) {
        let level = if level == 0 {
            max_level_nd(self.width, out_shape, axes)
        } else {
            level
        };
        let in_shape = get_transform_shape(out_shape, axes, level, self.width, false);
        assert_slice_matches_shape!("input", input, in_shape);
        assert_slice_matches_shape!("output", output, out_shape);
        general_nd_inverse_multilevel(
            |s, d, x| (self.dwt_adj_forward)(s, d, x, &self.bc),
            input,
            output,
            &in_shape,
            out_shape,
            TransformParams {
                axes,
                level,
                width: self.width,
            },
        );
    }

    /// Multi-level adjoint of the inverse DWT on an N-D array.
    ///
    /// # Panics
    ///
    /// Same conditions as [`forward_multilevel_nd`](Self::forward_multilevel_nd).
    #[track_caller]
    pub fn adj_inverse_multilevel_nd(
        &self,
        input: &[T],
        output: &mut [T],
        in_shape: &[usize],
        axes: &[usize],
        level: usize,
    ) {
        let level = if level == 0 {
            max_level_nd(self.width, in_shape, axes)
        } else {
            level
        };
        let out_shape = get_transform_shape(in_shape, axes, level, self.width, false);
        assert_slice_matches_shape!("input", input, in_shape);
        assert_slice_matches_shape!("output", output, out_shape);
        general_nd_forward_multilevel(
            |x, s, d| (self.dwt_adj_inverse)(x, s, d),
            input,
            output,
            in_shape,
            &out_shape,
            TransformParams {
                axes,
                level,
                width: self.width,
            },
        );
    }
}

#[cfg(feature = "ndarray")]
impl<T, BC, const N: usize> WaveletTransform<T, BC, N>
where
    T: Transformable + Zero + ChunkWidth<T, N>,
    BC: BoundaryExtension,
{
    /// Forward DWT applied to an ndarray (multi-level).
    ///
    /// `output` must have the transform-expanded shape returned by [`get_transform_shape`].
    ///
    /// # Panics
    ///
    /// Panics if `output.shape()` does not match `get_transform_shape(input.shape(), axes,
    /// level, width, false)`, or if any element of `axes` is `>= input.ndim()`.
    #[track_caller]
    pub fn forward_ndarray_multilevel<D: Dimension>(
        &self,
        input: &ArrayRef<T, D>,
        output: &mut ArrayRef<T, D>,
        axes: &[usize],
        level: usize,
    ) {
        let in_shape = input.shape();
        let level = if level == 0 {
            max_level_nd(self.width, in_shape, axes)
        } else {
            level
        };
        let out_shape = get_transform_shape(in_shape, axes, level, self.width, false);
        assert_eq!(
            out_shape,
            output.shape(),
            "output shape is not consistent with transformed shape of the input shape."
        );

        general_nd_forward_multilevel(
            |x, s, d| (self.dwt_forward)(x, s, d, &self.bc),
            input,
            output,
            in_shape,
            &out_shape,
            TransformParams {
                axes,
                level,
                width: self.width,
            },
        );
    }

    /// Inverse DWT applied to an ndarray (multi-level).
    ///
    /// `input` must have the transform-expanded shape; `output` has the original signal shape.
    /// `input` is `&mut` because the multilevel algorithm writes intermediate results back.
    ///
    /// # Panics
    ///
    /// Panics if `input.shape()` does not match `get_transform_shape(output.shape(), axes,
    /// level, width, false)`, or if any element of `axes` is `>= output.ndim()`.
    #[track_caller]
    pub fn inverse_ndarray_multilevel<D: Dimension>(
        &self,
        input: &mut ArrayRef<T, D>,
        output: &mut ArrayRef<T, D>,
        axes: &[usize],
        level: usize,
    ) {
        let out_shape = output.shape().to_owned();
        let level = if level == 0 {
            max_level_nd(self.width, &out_shape, axes)
        } else {
            level
        };
        let in_shape = get_transform_shape(&out_shape, axes, level, self.width, false);
        assert_eq!(
            in_shape,
            input.shape(),
            "input shape is not consistent with transformed shape of the output shape."
        );

        general_nd_inverse_multilevel(
            |s, d, x| (self.dwt_inverse)(s, d, x),
            input,
            output,
            &in_shape,
            &out_shape,
            TransformParams {
                axes,
                level,
                width: self.width,
            },
        );
    }

    /// Adjoint of the forward DWT applied to an ndarray (multi-level).
    ///
    /// `input` must have the transform-expanded shape; `output` must have the original
    /// signal shape.  `input` is taken as `&mut` because the internal multilevel
    /// algorithm writes intermediate results back into the buffer.
    ///
    /// # Panics
    ///
    /// Same conditions as [`inverse_ndarray_multilevel`](Self::inverse_ndarray_multilevel).
    #[track_caller]
    pub fn adj_forward_ndarray_multilevel<D: Dimension>(
        &self,
        input: &mut ArrayRef<T, D>,
        output: &mut ArrayRef<T, D>,
        axes: &[usize],
        level: usize,
    ) {
        let out_shape = output.shape().to_owned();
        let level = if level == 0 {
            max_level_nd(self.width, &out_shape, axes)
        } else {
            level
        };
        let in_shape = get_transform_shape(&out_shape, axes, level, self.width, false);
        assert_eq!(
            in_shape,
            input.shape(),
            "input shape is not consistent with transformed shape of the output shape."
        );

        general_nd_inverse_multilevel(
            |s, d, x| (self.dwt_adj_forward)(s, d, x, &self.bc),
            input,
            output,
            &in_shape,
            &out_shape,
            TransformParams {
                axes,
                level,
                width: self.width,
            },
        );
    }

    /// Adjoint of the inverse DWT applied to an ndarray (multi-level).
    ///
    /// `output` must have the transform-expanded shape returned by [`get_transform_shape`].
    ///
    /// # Panics
    ///
    /// Same conditions as [`forward_ndarray_multilevel`](Self::forward_ndarray_multilevel).
    #[track_caller]
    pub fn adj_inverse_ndarray_multilevel<D: Dimension>(
        &self,
        input: &ArrayRef<T, D>,
        output: &mut ArrayRef<T, D>,
        axes: &[usize],
        level: usize,
    ) {
        let in_shape = input.shape();
        let level = if level == 0 {
            max_level_nd(self.width, in_shape, axes)
        } else {
            level
        };
        let out_shape = get_transform_shape(in_shape, axes, level, self.width, false);
        assert_eq!(
            out_shape,
            output.shape(),
            "output shape is not consistent with transformed shape of the input shape."
        );

        general_nd_forward_multilevel(
            |x, s, d| (self.dwt_adj_inverse)(x, s, d),
            input,
            output,
            in_shape,
            &out_shape,
            TransformParams {
                axes,
                level,
                width: self.width,
            },
        );
    }
}

/// Periodic-boundary DWT driver for 1-D and N-D transforms.
///
/// Uses circular (periodic) boundary extension: the signal wraps around at both ends, so
/// each sub-band has exactly half the length of the input.  Unlike [`WaveletTransform`],
/// this variant requires even-length signals along each transformed axis.
///
/// # Type parameters
/// * `T` - element type (e.g. `f32`, `f64`, `Complex<f32>`).
/// * `N` - SIMD lane width (use `1` to disable SIMD).
pub struct WaveletTransformPer<T, const N: usize>
where
    T: ChunkWidth<T, N>,
{
    dwt_forward: fn(&[T], &mut [T], &mut [T]),
    dwt_inverse: fn(&[T], &[T], &mut [T]),
    dwt_adj_forward: fn(&[T], &[T], &mut [T]),
    dwt_adj_inverse: fn(&[T], &mut [T], &mut [T]),
    width: usize,
}

impl<T, const N: usize> WaveletTransformPer<T, N>
where
    T: Transformable + Zero + ChunkWidth<T, N>,
{
    /// Construct a `WaveletTransformPer` for the given wavelet family `wvlt`.
    ///
    /// Function pointers are resolved at construction time; see [`WaveletTransform::new`]
    /// for details.  No boundary condition is stored because the periodic transform
    /// always wraps circularly.
    pub fn new(wvlt: Wavelets) -> Self {
        use crate::dwt::bior::*;
        use crate::dwt::coiflet::*;
        use crate::dwt::daubechies::*;
        use crate::dwt::symlet::*;
        let dwt_forward: fn(&[T], &mut [T], &mut [T]) = generate_wavelet_match_arms! {
            Wavelets,
            wvlt,
            {#wvlt::forward_per,}
        };
        let dwt_inverse: fn(&[T], &[T], &mut [T]) = generate_wavelet_match_arms! {
            Wavelets,
            wvlt,
            {#wvlt::inverse_per,}
        };
        let dwt_adj_forward: fn(&[T], &[T], &mut [T]) = generate_wavelet_match_arms! {
            Wavelets,
            wvlt,
            {#wvlt::adjoint_forward_per,}
        };
        let dwt_adj_inverse: fn(&[T], &mut [T], &mut [T]) = generate_wavelet_match_arms! {
            Wavelets,
            wvlt,
            {#wvlt::adjoint_inverse_per,}
        };
        let width = wvlt.width();
        Self {
            dwt_forward,
            dwt_inverse,
            dwt_adj_forward,
            dwt_adj_inverse,
            width,
        }
    }

    /// Forward periodic DWT: decompose `input` into approximation (`s`) and detail (`d`) sub-bands.
    ///
    /// # Panics
    ///
    /// Panics if `s.len() + d.len() != input.len()`, or if `s.len()` and `d.len()` are
    /// not related by `s.len() == d.len()` or `s.len() == d.len() + 1`.
    #[track_caller]
    pub fn forward_1d(&self, input: &[T], s: &mut [T], d: &mut [T]) {
        (self.dwt_forward)(input, s, d);
    }

    /// Inverse periodic DWT: reconstruct `output` from sub-bands `s` and `d`.
    ///
    /// # Panics
    ///
    /// Panics if `s.len() + d.len() != output.len()`, or if `s.len()` and `d.len()` are
    /// not related by `s.len() == d.len()` or `s.len() == d.len() + 1`.
    #[track_caller]
    pub fn inverse_1d(&self, s: &[T], d: &[T], output: &mut [T]) {
        (self.dwt_inverse)(s, d, output);
    }

    /// Adjoint of the forward periodic DWT (one level).
    ///
    /// # Panics
    ///
    /// Panics if `s.len() + d.len() != output.len()`, or if `s.len()` and `d.len()` are
    /// not related by `s.len() == d.len()` or `s.len() == d.len() + 1`.
    #[track_caller]
    pub fn adj_forward_1d(&self, s: &[T], d: &[T], output: &mut [T]) {
        (self.dwt_adj_forward)(s, d, output);
    }

    /// Adjoint of the inverse periodic DWT: split `input` into sub-bands `s` and `d`.
    ///
    /// # Panics
    ///
    /// Panics if `s.len() + d.len() != input.len()`, or if `s.len()` and `d.len()` are
    /// not related by `s.len() == d.len()` or `s.len() == d.len() + 1`.
    #[track_caller]
    pub fn adj_inverse_1d(&self, input: &[T], s: &mut [T], d: &mut [T]) {
        (self.dwt_adj_inverse)(input, s, d);
    }

    /// Single-level periodic forward DWT along the given `axes`.
    ///
    /// # Panics
    ///
    /// See [`forward_multilevel_nd`](Self::forward_multilevel_nd).
    #[track_caller]
    pub fn forward_nd(&self, input: &[T], output: &mut [T], shape: &[usize], axes: &[usize]) {
        self.forward_multilevel_nd(input, output, shape, axes, 1);
    }

    /// Single-level periodic inverse DWT along the given `axes`.
    ///
    /// # Panics
    ///
    /// See [`inverse_multilevel_nd`](Self::inverse_multilevel_nd).
    #[track_caller]
    pub fn inverse_nd(&self, input: &[T], output: &mut [T], shape: &[usize], axes: &[usize]) {
        self.inverse_multilevel_nd(input, output, shape, axes, 1);
    }

    /// Single-level periodic adjoint forward DWT along the given `axes`.
    ///
    /// # Panics
    ///
    /// See [`adj_forward_multilevel_nd`](Self::adj_forward_multilevel_nd).
    #[track_caller]
    pub fn adj_forward_nd(&self, input: &[T], output: &mut [T], shape: &[usize], axes: &[usize]) {
        self.adj_forward_multilevel_nd(input, output, shape, axes, 1);
    }

    /// Single-level periodic adjoint inverse DWT along the given `axes`.
    ///
    /// # Panics
    ///
    /// See [`adj_inverse_multilevel_nd`](Self::adj_inverse_multilevel_nd).
    #[track_caller]
    pub fn adj_inverse_nd(&self, input: &[T], output: &mut [T], shape: &[usize], axes: &[usize]) {
        self.adj_inverse_multilevel_nd(input, output, shape, axes, 1);
    }

    /// Multi-level periodic forward DWT along the given `axes`.
    ///
    /// # Panics
    ///
    /// Panics if any element of `axes` is `>= shape.len()`, or if `input.len()` or
    /// `output.len()` does not equal `shape.iter().product()`.
    #[track_caller]
    pub fn forward_multilevel_nd(
        &self,
        input: &[T],
        output: &mut [T],
        shape: &[usize],
        axes: &[usize],
        level: usize,
    ) {
        assert_slice_matches_shape!("input", input, shape);
        assert_slice_matches_shape!("output", output, shape);
        assert!(axes.iter().all(|i| *i < shape.len()));
        let level = if level == 0 {
            max_level_nd(self.width, shape, axes)
        } else {
            level
        };
        general_nd_per_forward_multilevel(
            |x, s, d| (self.dwt_forward)(x, s, d),
            input,
            output,
            shape,
            axes,
            level,
        );
    }

    /// Multi-level periodic inverse DWT along the given `axes`.
    ///
    /// # Panics
    ///
    /// Same conditions as [`forward_multilevel_nd`](Self::forward_multilevel_nd).
    #[track_caller]
    pub fn inverse_multilevel_nd(
        &self,
        input: &[T],
        output: &mut [T],
        shape: &[usize],
        axes: &[usize],
        level: usize,
    ) {
        assert_slice_matches_shape!("input", input, shape);
        assert_slice_matches_shape!("output", output, shape);
        assert!(axes.iter().all(|i| *i < shape.len()));
        let level = if level == 0 {
            max_level_nd(self.width, shape, axes)
        } else {
            level
        };

        general_nd_per_inverse_multilevel(
            |s, d, x| (self.dwt_inverse)(s, d, x),
            input,
            output,
            shape,
            axes,
            level,
        );
    }

    /// Multi-level periodic adjoint forward DWT along the given `axes`.
    ///
    /// # Panics
    ///
    /// Same conditions as [`forward_multilevel_nd`](Self::forward_multilevel_nd).
    #[track_caller]
    pub fn adj_forward_multilevel_nd(
        &self,
        input: &[T],
        output: &mut [T],
        shape: &[usize],
        axes: &[usize],
        level: usize,
    ) {
        assert_slice_matches_shape!("input", input, shape);
        assert_slice_matches_shape!("output", output, shape);
        assert!(axes.iter().all(|i| *i < shape.len()));

        let level = if level == 0 {
            max_level_nd(self.width, shape, axes)
        } else {
            level
        };
        general_nd_per_inverse_multilevel(
            |s, d, x| (self.dwt_adj_forward)(s, d, x),
            input,
            output,
            shape,
            axes,
            level,
        );
    }

    /// Multi-level periodic adjoint inverse DWT along the given `axes`.
    ///
    /// # Panics
    ///
    /// Same conditions as [`forward_multilevel_nd`](Self::forward_multilevel_nd).
    #[track_caller]
    pub fn adj_inverse_multilevel_nd(
        &self,
        input: &[T],
        output: &mut [T],
        shape: &[usize],
        axes: &[usize],
        level: usize,
    ) {
        assert_slice_matches_shape!("input", input, shape);
        assert_slice_matches_shape!("output", output, shape);
        assert!(axes.iter().all(|i| *i < shape.len()));

        let level = if level == 0 {
            max_level_nd(self.width, shape, axes)
        } else {
            level
        };
        general_nd_per_forward_multilevel(
            |x, s, d| (self.dwt_adj_inverse)(x, s, d),
            input,
            output,
            shape,
            axes,
            level,
        );
    }
}

#[cfg(feature = "ndarray")]
impl<T, const N: usize> WaveletTransformPer<T, N>
where
    T: Transformable + Zero + ChunkWidth<T, N>,
{
    /// Forward periodic DWT applied to an ndarray (multi-level).
    ///
    /// # Panics
    ///
    /// Panics if `input.shape() != output.shape()`, or if any element of `axes` is
    /// `>= input.ndim()`.
    #[track_caller]
    pub fn forward_ndarray_multilevel<D: Dimension>(
        &self,
        input: &ArrayRef<T, D>,
        output: &mut ArrayRef<T, D>,
        axes: &[usize],
        level: usize,
    ) {
        let shape = input.shape();
        assert_eq!(
            shape,
            output.shape(),
            "input and output shapes must be the same."
        );
        assert!(axes.iter().all(|i| *i < shape.len()));

        let level = if level == 0 {
            max_level_nd(self.width, shape, axes)
        } else {
            level
        };

        general_nd_per_forward_multilevel(
            |x, s, d| (self.dwt_forward)(x, s, d),
            input,
            output,
            shape,
            axes,
            level,
        );
    }

    /// Inverse periodic DWT applied to an ndarray (multi-level).
    ///
    /// # Panics
    ///
    /// Same conditions as [`forward_ndarray_multilevel`](Self::forward_ndarray_multilevel).
    #[track_caller]
    pub fn inverse_ndarray_multilevel<D: Dimension>(
        &self,
        input: &ArrayRef<T, D>,
        output: &mut ArrayRef<T, D>,
        axes: &[usize],
        level: usize,
    ) {
        let shape = input.shape();
        assert_eq!(
            shape,
            output.shape(),
            "input and output shapes must be the same."
        );
        assert!(axes.iter().all(|i| *i < shape.len()));

        let level = if level == 0 {
            max_level_nd(self.width, shape, axes)
        } else {
            level
        };

        general_nd_per_inverse_multilevel(
            |s, d, x| (self.dwt_inverse)(s, d, x),
            input,
            output,
            shape,
            axes,
            level,
        );
    }

    /// Adjoint of the forward periodic DWT applied to an ndarray (multi-level).
    ///
    /// # Panics
    ///
    /// Same conditions as [`forward_ndarray_multilevel`](Self::forward_ndarray_multilevel).
    #[track_caller]
    pub fn adj_forward_ndarray_multilevel<D: Dimension>(
        &self,
        input: &ArrayRef<T, D>,
        output: &mut ArrayRef<T, D>,
        axes: &[usize],
        level: usize,
    ) {
        let shape = input.shape();
        assert_eq!(
            shape,
            output.shape(),
            "input and output shapes must be the same."
        );
        assert!(axes.iter().all(|i| *i < shape.len()));

        let level = if level == 0 {
            max_level_nd(self.width, shape, axes)
        } else {
            level
        };

        general_nd_per_inverse_multilevel(
            |s, d, x| (self.dwt_adj_forward)(s, d, x),
            input,
            output,
            shape,
            axes,
            level,
        );
    }

    /// Adjoint of the inverse periodic DWT applied to an ndarray (multi-level).
    ///
    /// # Panics
    ///
    /// Same conditions as [`forward_ndarray_multilevel`](Self::forward_ndarray_multilevel).
    #[track_caller]
    pub fn adj_inverse_ndarray_multilevel<D: Dimension>(
        &self,
        input: &ArrayRef<T, D>,
        output: &mut ArrayRef<T, D>,
        axes: &[usize],
        level: usize,
    ) {
        let shape = input.shape();
        assert_eq!(
            shape,
            output.shape(),
            "input and output shapes must be the same."
        );
        assert!(axes.iter().all(|i| *i < shape.len()));

        let level = if level == 0 {
            max_level_nd(self.width, shape, axes)
        } else {
            level
        };

        general_nd_per_forward_multilevel(
            |x, s, d| (self.dwt_adj_inverse)(x, s, d),
            input,
            output,
            shape,
            axes,
            level,
        );
    }
}

struct TransformParams<'a> {
    axes: &'a [usize],
    level: usize,
    width: usize,
}

fn general_nd_forward_multilevel<F, T, L, const N: usize>(
    func: F,
    input: &L,
    output: &mut L,
    in_shape: &[usize],
    out_shape: &[usize],
    params: TransformParams,
) where
    F: Fn(&[T], &mut [T], &mut [T]),
    L: LanesIterator<Item = T> + ?Sized,
    T: Clone + Zero + ChunkWidth<T, N>,
{
    let TransformParams { axes, level, width } = params;
    let ndim = in_shape.len();
    let axes = HashSet::<_>::from_iter(axes.iter().cloned());
    debug_assert_eq!(
        in_shape.len(),
        out_shape.len(),
        "input and output shapes must have the same number of dimensions"
    );
    debug_assert!(axes.iter().all(|i| *i < ndim));
    // note that axes is a HashSet, so they are gauranteed to be different axes.

    let mut first = true;

    let mut in_sub_shape = in_shape.to_owned();
    let mut out_sub_shape = out_shape.to_owned();

    for _level in 0..level {
        let mut sub_shape = in_sub_shape.clone();
        for &ax in &axes {
            let n_ax = sub_shape[ax];
            let n_sd = get_outlen(width, n_ax);

            sub_shape[ax] = out_sub_shape[ax];

            // Note that everything does work for n_s == 1 (or 0 for that matter),
            // just that there really isn't anything useful to do.
            if n_sd > 1 {
                match first {
                    true => {
                        let (in_lanes, out_lanes) = if input.is_ax_contiguous(ax, in_shape)
                            || output.is_ax_contiguous(ax, out_shape)
                        {
                            (
                                input.iter_lanes(in_shape, ax),
                                output.iter_lanes_sub_mut(out_shape, &sub_shape, ax),
                            )
                        } else {
                            let (in_chunks, in_rem) = input.iter_lane_chunks::<N>(in_shape, ax);
                            let (out_chunks, out_rem) =
                                output.iter_lane_chunks_sub_mut::<N>(out_shape, &sub_shape, ax);

                            if in_chunks.len() > 0 {
                                let mut x = core::array::from_fn(|_| avec![T::zero(); n_ax]);
                                let mut s = core::array::from_fn(|_| avec![T::zero(); n_sd]);
                                let mut d = core::array::from_fn(|_| avec![T::zero(); n_sd]);
                                in_chunks
                                    .zip(out_chunks)
                                    .for_each(|(in_chunk, mut out_chunk)| {
                                        // copy strided chunks into the local storage
                                        in_chunk.pour_into(&mut x);
                                        x.iter().zip(s.iter_mut().zip(d.iter_mut())).for_each(
                                            |(x, (s, d))| {
                                                func(x, s, d);
                                            },
                                        );
                                        // clone local storage to the output
                                        out_chunk.stack(&s, &d);
                                    });
                            }
                            (in_rem, out_rem)
                        };
                        if in_lanes.len() > 0 {
                            let mut x = avec![T::zero(); n_ax];
                            let mut s = avec![T::zero(); n_sd];
                            let mut d = avec![T::zero(); n_sd];
                            in_lanes
                                .zip(out_lanes)
                                .for_each(|(in_slice, mut out_slice)| {
                                    // copy strided slice into local dimension storage
                                    in_slice.pour_into(&mut x);
                                    func(&x, &mut s, &mut d);
                                    // copy local back to output strided slice
                                    out_slice.stack(&s, &d);
                                });
                        }

                        first = false;
                    }
                    false => {
                        let lanes = if output.is_ax_contiguous(ax, out_shape) {
                            output.iter_lanes_sub_mut(out_shape, &sub_shape, ax)
                        } else {
                            let (chunks, rem) =
                                output.iter_lane_chunks_sub_mut::<N>(out_shape, &sub_shape, ax);

                            if chunks.len() > 0 {
                                let mut x = core::array::from_fn(|_| avec![T::zero(); n_ax]);
                                let mut s = core::array::from_fn(|_| avec![T::zero(); n_sd]);
                                let mut d = core::array::from_fn(|_| avec![T::zero(); n_sd]);
                                chunks.for_each(|mut chunk| {
                                    // copy (and deinterleave) strided chunks into the local storage
                                    chunk.pour_into(&mut x);
                                    x.iter().zip(s.iter_mut().zip(d.iter_mut())).for_each(
                                        |(x, (s, d))| {
                                            func(x, s, d);
                                        },
                                    );
                                    // clone local storage to the output
                                    chunk.stack(&s, &d);
                                });
                            }
                            rem
                        };
                        if lanes.len() > 0 {
                            let mut x = avec![T::zero(); n_ax];
                            let mut s = avec![T::zero(); n_sd];
                            let mut d = avec![T::zero(); n_sd];
                            lanes.for_each(|mut slc| {
                                // copy strided slice into local dimension storage
                                slc.pour_into(&mut x);
                                func(&x, &mut s, &mut d);
                                // copy local back to output strided slice
                                slc.stack(&s, &d);
                            });
                        }
                    }
                }
            }
        }

        // shrink shape for each axis we used.
        for &ax in &axes {
            let n_ax = in_sub_shape[ax];
            let n_sd = get_outlen(width, n_ax);
            if n_sd > 1 {
                out_sub_shape[ax] -= n_sd;
                in_sub_shape[ax] = n_sd;
            }
        }
    }
}

fn general_nd_inverse_multilevel<F, T, L, const N: usize>(
    func: F,
    inwork: &mut L,
    output: &mut L,
    in_shape: &[usize],
    out_shape: &[usize],
    params: TransformParams,
) where
    F: Fn(&[T], &[T], &mut [T]),
    L: LanesIterator<Item = T> + ?Sized,
    T: Clone + Zero + ChunkWidth<T, N>,
{
    let TransformParams { axes, level, width } = params;
    let ndim = in_shape.len();
    let axes = HashSet::<_>::from_iter(axes.iter().cloned());
    debug_assert_eq!(
        in_shape.len(),
        out_shape.len(),
        "input and output shapes must have the same number of dimensions"
    );
    debug_assert!(axes.iter().all(|i| *i < ndim));
    // note that axes is a HashSet, so they are gauranteed to be different axes.

    // make some lists to keep track of the shapes at each level, as we need to iterate in reverse order later.
    let mut ax_shapes = Vec::with_capacity(level);
    let mut out_shapes = Vec::with_capacity(level);
    let mut sd_shapes = Vec::with_capacity(level);

    out_shapes.push(in_shape.to_owned());
    ax_shapes.push(out_shape.to_owned());
    for _level in 0..level {
        // shrink shape for each axis that is used.
        let mut sd_shape = ax_shapes.last().unwrap().clone();
        let mut next_out_shape = out_shapes.last().unwrap().clone();
        for &ax in &axes {
            let n_ax = sd_shape[ax];
            let n_sd = get_outlen(width, n_ax);

            if n_sd > 1 {
                sd_shape[ax] = n_sd;
                next_out_shape[ax] -= n_sd;
            }
        }
        if _level + 1 < level {
            ax_shapes.push(sd_shape.clone());
            out_shapes.push(next_out_shape);
        }
        sd_shapes.push(sd_shape);
    }

    for level in (0..level).rev() {
        let mut sub_shape = out_shapes[level].clone();
        for &ax in &axes {
            let n_ax = ax_shapes[level][ax];
            let n_sd = sd_shapes[level][ax];

            // Note that everything does work for n_s == 1 (or 0 for that matter),
            // just that there really isn't anything useful to do.
            if n_sd > 1 {
                let lanes = if inwork.is_ax_contiguous(ax, in_shape) {
                    inwork.iter_lanes_sub_mut(in_shape, &sub_shape, ax)
                } else {
                    let (chunks, rem) =
                        inwork.iter_lane_chunks_sub_mut::<N>(in_shape, &sub_shape, ax);

                    if chunks.len() > 0 {
                        let mut x = core::array::from_fn(|_| avec![T::zero(); n_ax]);
                        let mut s = core::array::from_fn(|_| avec![T::zero(); n_sd]);
                        let mut d = core::array::from_fn(|_| avec![T::zero(); n_sd]);
                        chunks.for_each(|mut chunk| {
                            // split the chunk into the approximation and detail coefficients.
                            chunk.split(&mut s, &mut d);
                            x.iter_mut()
                                .zip(s.iter().zip(d.iter()))
                                .for_each(|(x, (s, d))| {
                                    func(s, d, x);
                                });
                            // clone local storage to the output
                            chunk.fill_from(&x);
                        });
                    }
                    rem
                };
                if lanes.len() > 0 {
                    let mut x = avec![T::zero(); n_ax];
                    let mut s = avec![T::zero(); n_sd];
                    let mut d = avec![T::zero(); n_sd];
                    lanes.for_each(|mut slc| {
                        // split the slice into the approximation and detail coefficients.
                        slc.split(&mut s, &mut d);
                        func(&s, &d, &mut x);
                        // copy local back to output strided slice
                        slc.fill_from(&x);
                    });
                }
                // the next passes sub shape along this dimension will have the size of n_ax
                sub_shape[ax] = n_ax;
            }
        }
    }

    // copy input into output
    let min_axis = output.min_stride_axis(out_shape);
    let (in_lanes, out_lanes) = if inwork.is_ax_contiguous(min_axis, in_shape)
        || output.is_ax_contiguous(min_axis, out_shape)
    {
        (
            inwork.iter_lanes_sub(in_shape, out_shape, min_axis),
            output.iter_lanes_mut(out_shape, min_axis),
        )
    } else {
        let (in_chunks, in_rem) = inwork.iter_lane_chunks_sub::<N>(in_shape, out_shape, min_axis);
        let (out_chunks, out_rem) = output.iter_lane_chunks_mut::<N>(out_shape, min_axis);

        out_chunks.zip(in_chunks).for_each(|(mut o, i)| {
            o.iter_mut().zip(i.iter()).for_each(|(o, i)| {
                o.into_iter()
                    .zip(i.into_iter().cloned())
                    .for_each(|(o, i)| {
                        *o = i;
                    });
            });
        });
        (in_rem, out_rem)
    };
    out_lanes.zip(in_lanes).for_each(|(mut o, i)| {
        o.iter_mut()
            .zip(i.iter().cloned())
            .for_each(|(o, i)| *o = i);
    });
}

fn general_nd_per_forward_multilevel<F, T, L, const N: usize>(
    func: F,
    input: &L,
    output: &mut L,
    shape: &[usize],
    axes: &[usize],
    level: usize,
) where
    F: Fn(&[T], &mut [T], &mut [T]),
    L: LanesIterator<Item = T> + ?Sized,
    T: Clone + Zero + ChunkWidth<T, N>,
{
    let ndim = shape.len();
    let axes = HashSet::<_>::from_iter(axes.iter().cloned());
    debug_assert!(axes.iter().all(|i| *i < ndim));
    // note that axes is a HashSet, so they are gauranteed to be different axes.

    let mut first = true;

    let mut in_sub_shape = shape.to_owned();
    let mut out_sub_shape = shape.to_owned();

    for _level in 0..level {
        let mut sub_shape = in_sub_shape.clone();
        for &ax in &axes {
            let n_ax = sub_shape[ax];
            let n_d = n_ax / 2;
            let n_s = n_ax - n_d;

            sub_shape[ax] = out_sub_shape[ax];

            // Note that everything does work for n_s == 1 (or 0 for that matter),
            // just that there really isn't anything useful to do.
            if n_s > 1 {
                match first {
                    true => {
                        let (in_lanes, out_lanes) = if input.is_ax_contiguous(ax, shape)
                            || output.is_ax_contiguous(ax, shape)
                        {
                            (
                                input.iter_lanes(shape, ax),
                                output.iter_lanes_sub_mut(shape, &sub_shape, ax),
                            )
                        } else {
                            let (in_chunks, in_rem) = input.iter_lane_chunks::<N>(shape, ax);
                            let (out_chunks, out_rem) =
                                output.iter_lane_chunks_sub_mut::<N>(shape, &sub_shape, ax);

                            if in_chunks.len() > 0 {
                                let mut x = core::array::from_fn(|_| avec![T::zero(); n_ax]);
                                let mut s = core::array::from_fn(|_| avec![T::zero(); n_s]);
                                let mut d = core::array::from_fn(|_| avec![T::zero(); n_d]);
                                in_chunks
                                    .zip(out_chunks)
                                    .for_each(|(in_chunk, mut out_chunk)| {
                                        // copy strided chunks into the local storage
                                        in_chunk.pour_into(&mut x);
                                        x.iter().zip(s.iter_mut().zip(d.iter_mut())).for_each(
                                            |(x, (s, d))| {
                                                func(x, s, d);
                                            },
                                        );
                                        // clone local storage to the output
                                        out_chunk.stack(&s, &d);
                                    });
                            }
                            (in_rem, out_rem)
                        };
                        if in_lanes.len() > 0 {
                            let mut x = avec![T::zero(); n_ax];
                            let mut s = avec![T::zero(); n_s];
                            let mut d = avec![T::zero(); n_d];
                            in_lanes
                                .zip(out_lanes)
                                .for_each(|(in_slice, mut out_slice)| {
                                    // copy strided slice into local dimension storage
                                    in_slice.pour_into(&mut x);
                                    func(&x, &mut s, &mut d);
                                    // copy local back to output strided slice
                                    out_slice.stack(&s, &d);
                                });
                        }

                        first = false;
                    }
                    false => {
                        let lanes = if output.is_ax_contiguous(ax, shape) {
                            output.iter_lanes_sub_mut(shape, &sub_shape, ax)
                        } else {
                            let (chunks, rem) =
                                output.iter_lane_chunks_sub_mut::<N>(shape, &sub_shape, ax);

                            if chunks.len() > 0 {
                                let mut x = core::array::from_fn(|_| avec![T::zero(); n_ax]);
                                let mut s = core::array::from_fn(|_| avec![T::zero(); n_s]);
                                let mut d = core::array::from_fn(|_| avec![T::zero(); n_d]);
                                chunks.for_each(|mut chunk| {
                                    // copy (and deinterleave) strided chunks into the local storage
                                    chunk.pour_into(&mut x);
                                    x.iter().zip(s.iter_mut().zip(d.iter_mut())).for_each(
                                        |(x, (s, d))| {
                                            func(x, s, d);
                                        },
                                    );
                                    // clone local storage to the output
                                    chunk.stack(&s, &d);
                                });
                            }
                            rem
                        };
                        if lanes.len() > 0 {
                            let mut x = avec![T::zero(); n_ax];
                            let mut s = avec![T::zero(); n_s];
                            let mut d = avec![T::zero(); n_d];
                            lanes.for_each(|mut slc| {
                                // copy strided slice into local dimension storage
                                slc.pour_into(&mut x);
                                func(&x, &mut s, &mut d);
                                // copy local back to output strided slice
                                slc.stack(&s, &d);
                            });
                        }
                    }
                }
            }
        }

        // shrink shape for each axis we used.
        for &ax in &axes {
            let n_ax = in_sub_shape[ax];
            let n_d = n_ax / 2;
            let n_s = n_ax - n_d;
            if n_s > 1 {
                out_sub_shape[ax] -= n_d;
                in_sub_shape[ax] = n_s;
            }
        }
    }
}

fn general_nd_per_inverse_multilevel<F, T, L, const N: usize>(
    func: F,
    input: &L,
    output: &mut L,
    shape: &[usize],
    axes: &[usize],
    level: usize,
) where
    F: Fn(&[T], &[T], &mut [T]),
    L: LanesIterator<Item = T> + ?Sized,
    T: Clone + Zero + ChunkWidth<T, N>,
{
    let ndim = shape.len();
    let axes = HashSet::<_>::from_iter(axes.iter().cloned());
    debug_assert!(axes.iter().all(|i| *i < ndim));
    // note that axes is a HashSet, so they are gauranteed to be different axes.

    // If the input shape and the output shape are the same, then we are in per mode

    // make some lists to keep track of the shapes at each level, as we need to iterate in reverse order later.
    let mut ax_shapes = Vec::with_capacity(level);
    let mut out_shapes = Vec::with_capacity(level);
    let mut approx_shapes = Vec::with_capacity(level);
    let mut detail_shapes = Vec::with_capacity(level);

    out_shapes.push(shape.to_owned());
    ax_shapes.push(shape.to_owned());
    for _level in 0..level {
        // shrink shape for each axis that is used.
        let mut approx_shape = ax_shapes.last().unwrap().clone();
        let mut detail_shape = approx_shape.clone();
        let mut next_out_shape = out_shapes.last().unwrap().clone();
        for &ax in &axes {
            let n_ax = approx_shape[ax];
            let n_d = n_ax / 2;
            let n_s = n_ax - n_d;
            if n_s > 1 {
                approx_shape[ax] = n_s;
                detail_shape[ax] = n_d;
                next_out_shape[ax] -= n_d;
            }
        }
        if _level + 1 < level {
            ax_shapes.push(approx_shape.clone());
            out_shapes.push(next_out_shape);
        }
        approx_shapes.push(approx_shape);
        detail_shapes.push(detail_shape);
    }

    // In per mode we can copy the input to the output right away and not modify the input array.
    let min_axis = output.min_stride_axis(shape);
    let (in_lanes, out_lanes) =
        if input.is_ax_contiguous(min_axis, shape) || output.is_ax_contiguous(min_axis, shape) {
            (
                input.iter_lanes(shape, min_axis),
                output.iter_lanes_mut(shape, min_axis),
            )
        } else {
            let (in_chunks, in_rem) = input.iter_lane_chunks::<N>(shape, min_axis);
            let (out_chunks, out_rem) = output.iter_lane_chunks_mut::<N>(shape, min_axis);

            out_chunks.zip(in_chunks).for_each(|(mut o, i)| {
                o.iter_mut().zip(i.iter()).for_each(|(o, i)| {
                    o.into_iter()
                        .zip(i.into_iter().cloned())
                        .for_each(|(o, i)| {
                            *o = i;
                        });
                });
            });
            (in_rem, out_rem)
        };
    out_lanes.zip(in_lanes).for_each(|(mut o, i)| {
        o.iter_mut()
            .zip(i.iter().cloned())
            .for_each(|(o, i)| *o = i);
    });

    for level in (0..level).rev() {
        let mut sub_shape = out_shapes[level].clone();
        for &ax in &axes {
            let n_ax = ax_shapes[level][ax];
            let n_s = approx_shapes[level][ax];
            let n_d = detail_shapes[level][ax];

            // Note that everything does work for n_s == 1 (or 0 for that matter),
            // just that there really isn't anything useful to do.
            if n_s > 1 {
                let lanes = if output.is_ax_contiguous(ax, shape) {
                    output.iter_lanes_sub_mut(shape, &sub_shape, ax)
                } else {
                    let (chunks, rem) = output.iter_lane_chunks_sub_mut::<N>(shape, &sub_shape, ax);

                    if chunks.len() > 0 {
                        let mut x = core::array::from_fn(|_| avec![T::zero(); n_ax]);
                        let mut s = core::array::from_fn(|_| avec![T::zero(); n_s]);
                        let mut d = core::array::from_fn(|_| avec![T::zero(); n_d]);
                        chunks.for_each(|mut chunk| {
                            // split the chunk into the approximation and detail coefficients.
                            chunk.split(&mut s, &mut d);
                            x.iter_mut()
                                .zip(s.iter().zip(d.iter()))
                                .for_each(|(x, (s, d))| {
                                    func(s, d, x);
                                });
                            // clone local storage to the output
                            chunk.fill_from(&x);
                        });
                    }
                    rem
                };
                if lanes.len() > 0 {
                    let mut x = avec![T::zero(); n_ax];
                    let mut s = avec![T::zero(); n_s];
                    let mut d = avec![T::zero(); n_d];
                    lanes.for_each(|mut slc| {
                        // split the slice into the approximation and detail coefficients.
                        slc.split(&mut s, &mut d);
                        func(&s, &d, &mut x);
                        // copy local back to output strided slice
                        slc.fill_from(&x);
                    });
                }
                // the next passes sub shape along this dimension will have the size of n_ax
                sub_shape[ax] = n_ax;
            }
        }
    }
}

#[cfg(feature = "rayon")]
/// Rayon-parallel DWT drivers.
///
/// Mirrors the sequential [`WaveletTransform`] and [`WaveletTransformPer`]
/// API but processes independent lanes on multiple threads via Rayon.
pub mod parallel {
    use super::*;

    use crate::iter::parallel::LanesParallelIterator;
    use rayon::iter::{IndexedParallelIterator, ParallelIterator};

    impl<T, BC, const N: usize> WaveletTransform<T, BC, N>
    where
        T: Transformable + Zero + ChunkWidth<T, N> + Sync + Send,
        BC: BoundaryExtension,
    {
        /// Single-level parallel forward DWT along the given `axes`.
        ///
        /// # Panics
        ///
        /// See [`par_forward_multilevel_nd`](WaveletTransform::par_forward_multilevel_nd).
        #[track_caller]
        pub fn par_forward_nd(
            &self,
            input: &[T],
            output: &mut [T],
            shape: &[usize],
            axes: &[usize],
        ) {
            self.par_forward_multilevel_nd(input, output, shape, axes, 1);
        }

        /// Single-level parallel inverse DWT along the given `axes`.
        ///
        /// # Panics
        ///
        /// See [`par_inverse_multilevel_nd`](WaveletTransform::par_inverse_multilevel_nd).
        #[track_caller]
        pub fn par_inverse_nd(
            &self,
            input: &mut [T],
            output: &mut [T],
            shape: &[usize],
            axes: &[usize],
        ) {
            self.par_inverse_multilevel_nd(input, output, shape, axes, 1);
        }

        /// Single-level parallel adjoint forward DWT along the given `axes`.
        ///
        /// `input` is taken as `&mut` because the internal multilevel algorithm writes
        /// intermediate results back into the buffer during reconstruction.
        ///
        /// # Panics
        ///
        /// See [`par_adj_forward_multilevel_nd`](WaveletTransform::par_adj_forward_multilevel_nd).
        #[track_caller]
        pub fn par_adj_forward_nd(
            &self,
            input: &mut [T],
            output: &mut [T],
            shape: &[usize],
            axes: &[usize],
        ) {
            self.par_adj_forward_multilevel_nd(input, output, shape, axes, 1);
        }

        /// Single-level parallel adjoint inverse DWT along the given `axes`.
        ///
        /// # Panics
        ///
        /// See [`par_adj_inverse_multilevel_nd`](WaveletTransform::par_adj_inverse_multilevel_nd).
        #[track_caller]
        pub fn par_adj_inverse_nd(
            &self,
            input: &[T],
            output: &mut [T],
            shape: &[usize],
            axes: &[usize],
        ) {
            self.par_adj_inverse_multilevel_nd(input, output, shape, axes, 1);
        }

        /// Multi-level parallel forward DWT along the given `axes`.
        ///
        /// # Panics
        ///
        /// Panics if any element of `axes` is `>= in_shape.len()`, if `input.len()` does
        /// not equal `in_shape.iter().product()`, or if `output.len()` does not equal the
        /// product of `get_transform_shape(in_shape, axes, level, width, false)`.
        #[track_caller]
        pub fn par_forward_multilevel_nd(
            &self,
            input: &[T],
            output: &mut [T],
            in_shape: &[usize],
            axes: &[usize],
            level: usize,
        ) {
            assert!(axes.iter().all(|i| *i < in_shape.len()));
            let level = if level == 0 {
                max_level_nd(self.width, in_shape, axes)
            } else {
                level
            };
            let out_shape = get_transform_shape(in_shape, axes, level, self.width, false);
            assert_slice_matches_shape!("input", input, in_shape);
            assert_slice_matches_shape!("output", output, out_shape);
            general_nd_forward_multilevel(
                |x, s, d| (self.dwt_forward)(x, s, d, &self.bc),
                input,
                output,
                in_shape,
                &out_shape,
                TransformParams {
                    axes,
                    level,
                    width: self.width,
                },
            );
        }

        /// Multi-level parallel inverse DWT along the given `axes`.
        ///
        /// `out_shape` is the shape of the original signal (before decomposition).
        /// The input must have the transform-expanded shape returned by
        /// `get_transform_shape(out_shape, axes, level, width, false)`.
        /// `input` is taken as `&mut` because the internal multilevel algorithm writes
        /// intermediate results back into the buffer during reconstruction.
        ///
        /// # Panics
        ///
        /// Panics if any element of `axes` is `>= out_shape.len()`, if `output.len()` does
        /// not equal `out_shape.iter().product()`, or if `input.len()` does not equal the
        /// product of `get_transform_shape(out_shape, axes, level, width, false)`.
        #[track_caller]
        pub fn par_inverse_multilevel_nd(
            &self,
            input: &mut [T],
            output: &mut [T],
            out_shape: &[usize],
            axes: &[usize],
            level: usize,
        ) {
            assert!(axes.iter().all(|i| *i < out_shape.len()));
            let level = if level == 0 {
                max_level_nd(self.width, out_shape, axes)
            } else {
                level
            };
            let in_shape = get_transform_shape(out_shape, axes, level, self.width, false);
            assert_slice_matches_shape!("input", input, in_shape);
            assert_slice_matches_shape!("output", output, out_shape);
            general_nd_inverse_multilevel(
                |s, d, x| (self.dwt_inverse)(s, d, x),
                input,
                output,
                &in_shape,
                out_shape,
                TransformParams {
                    axes,
                    level,
                    width: self.width,
                },
            );
        }

        /// Multi-level parallel adjoint forward DWT along the given `axes`.
        ///
        /// `out_shape` is the shape of the original signal (before decomposition).
        /// The input must have the transform-expanded shape returned by
        /// `get_transform_shape(out_shape, axes, level, width, false)`.
        /// `input` is taken as `&mut` because the internal multilevel algorithm writes
        /// intermediate results back into the buffer during reconstruction.
        ///
        /// # Panics
        ///
        /// Same conditions as [`par_inverse_multilevel_nd`](WaveletTransform::par_inverse_multilevel_nd).
        #[track_caller]
        pub fn par_adj_forward_multilevel_nd(
            &self,
            input: &mut [T],
            output: &mut [T],
            out_shape: &[usize],
            axes: &[usize],
            level: usize,
        ) {
            assert!(axes.iter().all(|i| *i < out_shape.len()));
            let level = if level == 0 {
                max_level_nd(self.width, out_shape, axes)
            } else {
                level
            };
            let in_shape = get_transform_shape(out_shape, axes, level, self.width, false);
            assert_slice_matches_shape!("input", input, in_shape);
            assert_slice_matches_shape!("output", output, out_shape);
            general_nd_inverse_multilevel(
                |s, d, x| (self.dwt_adj_forward)(s, d, x, &self.bc),
                input,
                output,
                &in_shape,
                out_shape,
                TransformParams {
                    axes,
                    level,
                    width: self.width,
                },
            );
        }

        /// Multi-level parallel adjoint inverse DWT along the given `axes`.
        ///
        /// # Panics
        ///
        /// Same conditions as [`par_forward_multilevel_nd`](WaveletTransform::par_forward_multilevel_nd).
        #[track_caller]
        pub fn par_adj_inverse_multilevel_nd(
            &self,
            input: &[T],
            output: &mut [T],
            in_shape: &[usize],
            axes: &[usize],
            level: usize,
        ) {
            assert!(axes.iter().all(|i| *i < in_shape.len()));
            let level = if level == 0 {
                max_level_nd(self.width, in_shape, axes)
            } else {
                level
            };
            let out_shape = get_transform_shape(in_shape, axes, level, self.width, false);
            assert_slice_matches_shape!("input", input, in_shape);
            assert_slice_matches_shape!("output", output, out_shape);
            general_nd_forward_multilevel(
                |x, s, d| (self.dwt_adj_inverse)(x, s, d),
                input,
                output,
                in_shape,
                &out_shape,
                TransformParams {
                    axes,
                    level,
                    width: self.width,
                },
            );
        }
    }

    #[cfg(feature = "ndarray")]
    impl<T, BC, const N: usize> WaveletTransform<T, BC, N>
    where
        T: Transformable + Zero + ChunkWidth<T, N> + Sync + Send,
        BC: BoundaryExtension,
    {
        /// Forward DWT applied to an ndarray (parallel, multi-level).
        ///
        /// # Panics
        ///
        /// Panics if `output.shape()` does not match `get_transform_shape(input.shape(), axes,
        /// level, width, false)`, or if any element of `axes` is `>= input.ndim()`.
        #[track_caller]
        pub fn par_forward_ndarray_multilevel<D: Dimension>(
            &self,
            input: &ArrayRef<T, D>,
            output: &mut ArrayRef<T, D>,
            axes: &[usize],
            level: usize,
        ) {
            let in_shape = input.shape();
            assert!(axes.iter().all(|i| *i < in_shape.len()));
            let level = if level == 0 {
                max_level_nd(self.width, in_shape, axes)
            } else {
                level
            };
            let out_shape = get_transform_shape(in_shape, axes, level, self.width, false);
            assert_eq!(
                out_shape,
                output.shape(),
                "output shape is not consistent with transformed shape of the input shape."
            );

            general_nd_forward_multilevel(
                |x, s, d| (self.dwt_forward)(x, s, d, &self.bc),
                input,
                output,
                in_shape,
                &out_shape,
                TransformParams {
                    axes,
                    level,
                    width: self.width,
                },
            );
        }

        /// Inverse DWT applied to an ndarray (parallel, multi-level).
        ///
        /// `input` is taken as `&mut` because the internal multilevel algorithm writes
        /// intermediate results back into the buffer during reconstruction.
        ///
        /// # Panics
        ///
        /// Panics if `input.shape()` does not match `get_transform_shape(output.shape(), axes,
        /// level, width, false)`, or if any element of `axes` is `>= output.ndim()`.
        #[track_caller]
        pub fn par_inverse_ndarray_multilevel<D: Dimension>(
            &self,
            input: &mut ArrayRef<T, D>,
            output: &mut ArrayRef<T, D>,
            axes: &[usize],
            level: usize,
        ) {
            let out_shape = output.shape().to_owned();
            assert!(axes.iter().all(|i| *i < out_shape.len()));
            let level = if level == 0 {
                max_level_nd(self.width, &out_shape, axes)
            } else {
                level
            };
            let in_shape = get_transform_shape(&out_shape, axes, level, self.width, false);
            assert_eq!(
                in_shape,
                input.shape(),
                "input shape is not consistent with transformed shape of the output shape."
            );

            general_nd_inverse_multilevel(
                |s, d, x| (self.dwt_inverse)(s, d, x),
                input,
                output,
                &in_shape,
                &out_shape,
                TransformParams {
                    axes,
                    level,
                    width: self.width,
                },
            );
        }

        /// Adjoint forward DWT applied to an ndarray (parallel, multi-level).
        ///
        /// `input` is taken as `&mut` because the internal multilevel algorithm writes
        /// intermediate results back into the buffer during reconstruction.
        ///
        /// # Panics
        ///
        /// Same conditions as [`par_inverse_ndarray_multilevel`](WaveletTransform::par_inverse_ndarray_multilevel).
        #[track_caller]
        pub fn par_adj_forward_ndarray_multilevel<D: Dimension>(
            &self,
            input: &mut ArrayRef<T, D>,
            output: &mut ArrayRef<T, D>,
            axes: &[usize],
            level: usize,
        ) {
            let out_shape = output.shape().to_owned();
            assert!(axes.iter().all(|i| *i < out_shape.len()));
            let level = if level == 0 {
                max_level_nd(self.width, &out_shape, axes)
            } else {
                level
            };
            let in_shape = get_transform_shape(&out_shape, axes, level, self.width, false);
            assert_eq!(
                in_shape,
                input.shape(),
                "input shape is not consistent with transformed shape of the output shape."
            );

            general_nd_inverse_multilevel(
                |s, d, x| (self.dwt_adj_forward)(s, d, x, &self.bc),
                input,
                output,
                &in_shape,
                &out_shape,
                TransformParams {
                    axes,
                    level,
                    width: self.width,
                },
            );
        }

        /// Adjoint inverse DWT applied to an ndarray (parallel, multi-level).
        ///
        /// # Panics
        ///
        /// Same conditions as [`par_forward_ndarray_multilevel`](WaveletTransform::par_forward_ndarray_multilevel).
        #[track_caller]
        pub fn par_adj_inverse_ndarray_multilevel<D: Dimension>(
            &self,
            input: &ArrayRef<T, D>,
            output: &mut ArrayRef<T, D>,
            axes: &[usize],
            level: usize,
        ) {
            let in_shape = input.shape();
            assert!(axes.iter().all(|i| *i < in_shape.len()));
            let level = if level == 0 {
                max_level_nd(self.width, in_shape, axes)
            } else {
                level
            };
            let out_shape = get_transform_shape(in_shape, axes, level, self.width, false);
            assert_eq!(
                out_shape,
                output.shape(),
                "output shape is not consistent with transformed shape of the input shape."
            );

            general_nd_forward_multilevel(
                |x, s, d| (self.dwt_adj_inverse)(x, s, d),
                input,
                output,
                in_shape,
                &out_shape,
                TransformParams {
                    axes,
                    level,
                    width: self.width,
                },
            );
        }
    }

    impl<T, const N: usize> WaveletTransformPer<T, N>
    where
        T: Transformable + Zero + ChunkWidth<T, N> + Sync + Send,
    {
        /// Single-level parallel periodic forward DWT along the given `axes`.
        ///
        /// # Panics
        ///
        /// See [`par_forward_multilevel_nd`](WaveletTransformPer::par_forward_multilevel_nd).
        #[track_caller]
        pub fn par_forward_nd(
            &self,
            input: &[T],
            output: &mut [T],
            shape: &[usize],
            axes: &[usize],
        ) {
            self.forward_multilevel_nd(input, output, shape, axes, 1);
        }

        /// Single-level parallel periodic inverse DWT along the given `axes`.
        ///
        /// # Panics
        ///
        /// See [`par_inverse_multilevel_nd`](WaveletTransformPer::par_inverse_multilevel_nd).
        #[track_caller]
        pub fn par_inverse_nd(
            &self,
            input: &[T],
            output: &mut [T],
            shape: &[usize],
            axes: &[usize],
        ) {
            self.inverse_multilevel_nd(input, output, shape, axes, 1);
        }

        /// Single-level parallel periodic adjoint forward DWT along the given `axes`.
        ///
        /// # Panics
        ///
        /// See [`par_adj_forward_multilevel_nd`](WaveletTransformPer::par_adj_forward_multilevel_nd).
        #[track_caller]
        pub fn par_adj_forward_nd(
            &self,
            input: &[T],
            output: &mut [T],
            shape: &[usize],
            axes: &[usize],
        ) {
            self.adj_forward_multilevel_nd(input, output, shape, axes, 1);
        }

        /// Single-level parallel periodic adjoint inverse DWT along the given `axes`.
        ///
        /// # Panics
        ///
        /// See [`par_adj_inverse_multilevel_nd`](WaveletTransformPer::par_adj_inverse_multilevel_nd).
        #[track_caller]
        pub fn par_adj_inverse_nd(
            &self,
            input: &[T],
            output: &mut [T],
            shape: &[usize],
            axes: &[usize],
        ) {
            self.adj_inverse_multilevel_nd(input, output, shape, axes, 1);
        }

        /// Multi-level parallel periodic forward DWT along the given `axes`.
        ///
        /// # Panics
        ///
        /// Panics if any element of `axes` is `>= shape.len()`, or if `input.len()` or
        /// `output.len()` does not equal `shape.iter().product()`.
        #[track_caller]
        pub fn par_forward_multilevel_nd(
            &self,
            input: &[T],
            output: &mut [T],
            shape: &[usize],
            axes: &[usize],
            level: usize,
        ) {
            assert_slice_matches_shape!("input", input, shape);
            assert_slice_matches_shape!("output", output, shape);
            assert!(axes.iter().all(|i| *i < shape.len()));
            let level = if level == 0 {
                max_level_nd(self.width, shape, axes)
            } else {
                level
            };
            general_nd_per_forward_multilevel(
                |x, s, d| (self.dwt_forward)(x, s, d),
                input,
                output,
                shape,
                axes,
                level,
            );
        }

        /// Multi-level parallel periodic inverse DWT along the given `axes`.
        ///
        /// # Panics
        ///
        /// Same conditions as [`par_forward_multilevel_nd`](WaveletTransformPer::par_forward_multilevel_nd).
        #[track_caller]
        pub fn par_inverse_multilevel_nd(
            &self,
            input: &[T],
            output: &mut [T],
            shape: &[usize],
            axes: &[usize],
            level: usize,
        ) {
            assert_slice_matches_shape!("input", input, shape);
            assert_slice_matches_shape!("output", output, shape);
            assert!(axes.iter().all(|i| *i < shape.len()));
            let level = if level == 0 {
                max_level_nd(self.width, shape, axes)
            } else {
                level
            };

            general_nd_per_inverse_multilevel(
                |s, d, x| (self.dwt_inverse)(s, d, x),
                input,
                output,
                shape,
                axes,
                level,
            );
        }

        /// Multi-level parallel periodic adjoint forward DWT along the given `axes`.
        ///
        /// # Panics
        ///
        /// Same conditions as [`par_forward_multilevel_nd`](WaveletTransformPer::par_forward_multilevel_nd).
        #[track_caller]
        pub fn par_adj_forward_multilevel_nd(
            &self,
            input: &[T],
            output: &mut [T],
            shape: &[usize],
            axes: &[usize],
            level: usize,
        ) {
            assert_slice_matches_shape!("input", input, shape);
            assert_slice_matches_shape!("output", output, shape);
            assert!(axes.iter().all(|i| *i < shape.len()));
            let level = if level == 0 {
                max_level_nd(self.width, shape, axes)
            } else {
                level
            };
            general_nd_per_inverse_multilevel(
                |s, d, x| (self.dwt_adj_forward)(s, d, x),
                input,
                output,
                shape,
                axes,
                level,
            );
        }

        /// Multi-level parallel periodic adjoint inverse DWT along the given `axes`.
        ///
        /// # Panics
        ///
        /// Same conditions as [`par_forward_multilevel_nd`](WaveletTransformPer::par_forward_multilevel_nd).
        #[track_caller]
        pub fn par_adj_inverse_multilevel_nd(
            &self,
            input: &[T],
            output: &mut [T],
            shape: &[usize],
            axes: &[usize],
            level: usize,
        ) {
            assert_slice_matches_shape!("input", input, shape);
            assert_slice_matches_shape!("output", output, shape);
            assert!(axes.iter().all(|i| *i < shape.len()));
            let level = if level == 0 {
                max_level_nd(self.width, shape, axes)
            } else {
                level
            };
            general_nd_per_forward_multilevel(
                |x, s, d| (self.dwt_adj_inverse)(x, s, d),
                input,
                output,
                shape,
                axes,
                level,
            );
        }
    }

    #[cfg(feature = "ndarray")]
    impl<T, const N: usize> WaveletTransformPer<T, N>
    where
        T: Transformable + Zero + ChunkWidth<T, N> + Sync + Send,
    {
        /// Forward periodic DWT applied to an ndarray (parallel, multi-level).
        ///
        /// # Panics
        ///
        /// Panics if `input.shape() != output.shape()`, or if any element of `axes` is
        /// `>= input.ndim()`.
        #[track_caller]
        pub fn par_forward_ndarray_multilevel<D: Dimension>(
            &self,
            input: &ArrayRef<T, D>,
            output: &mut ArrayRef<T, D>,
            axes: &[usize],
            level: usize,
        ) {
            let shape = input.shape();
            assert_eq!(
                shape,
                output.shape(),
                "input and output shapes must be the same."
            );
            assert!(axes.iter().all(|i| *i < shape.len()));

            let level = if level == 0 {
                max_level_nd(self.width, shape, axes)
            } else {
                level
            };

            general_nd_per_forward_multilevel(
                |x, s, d| (self.dwt_forward)(x, s, d),
                input,
                output,
                shape,
                axes,
                level,
            );
        }

        /// Inverse periodic DWT applied to an ndarray (parallel, multi-level).
        ///
        /// # Panics
        ///
        /// Same conditions as [`par_forward_ndarray_multilevel`](WaveletTransformPer::par_forward_ndarray_multilevel).
        #[track_caller]
        pub fn par_inverse_ndarray_multilevel<D: Dimension>(
            &self,
            input: &ArrayRef<T, D>,
            output: &mut ArrayRef<T, D>,
            axes: &[usize],
            level: usize,
        ) {
            let shape = input.shape();
            assert_eq!(
                shape,
                output.shape(),
                "input and output shapes must be the same."
            );
            assert!(axes.iter().all(|i| *i < shape.len()));

            let level = if level == 0 {
                max_level_nd(self.width, shape, axes)
            } else {
                level
            };

            general_nd_per_inverse_multilevel(
                |s, d, x| (self.dwt_inverse)(s, d, x),
                input,
                output,
                shape,
                axes,
                level,
            );
        }

        /// Adjoint of the forward periodic DWT applied to an ndarray (parallel, multi-level).
        ///
        /// # Panics
        ///
        /// Same conditions as [`par_forward_ndarray_multilevel`](WaveletTransformPer::par_forward_ndarray_multilevel).
        #[track_caller]
        pub fn par_adj_forward_ndarray_multilevel<D: Dimension>(
            &self,
            input: &ArrayRef<T, D>,
            output: &mut ArrayRef<T, D>,
            axes: &[usize],
            level: usize,
        ) {
            let shape = input.shape();
            assert_eq!(
                shape,
                output.shape(),
                "input and output shapes must be the same."
            );
            assert!(axes.iter().all(|i| *i < shape.len()));

            let level = if level == 0 {
                max_level_nd(self.width, shape, axes)
            } else {
                level
            };

            general_nd_per_inverse_multilevel(
                |s, d, x| (self.dwt_adj_forward)(s, d, x),
                input,
                output,
                shape,
                axes,
                level,
            );
        }

        /// Adjoint inverse periodic DWT applied to an ndarray (parallel, multi-level).
        ///
        /// # Panics
        ///
        /// Same conditions as [`par_forward_ndarray_multilevel`](WaveletTransformPer::par_forward_ndarray_multilevel).
        #[track_caller]
        pub fn par_adj_inverse_ndarray_multilevel<D: Dimension>(
            &self,
            input: &ArrayRef<T, D>,
            output: &mut ArrayRef<T, D>,
            axes: &[usize],
            level: usize,
        ) {
            let shape = input.shape();
            assert_eq!(
                shape,
                output.shape(),
                "input and output shapes must be the same."
            );
            assert!(axes.iter().all(|i| *i < shape.len()));

            let level = if level == 0 {
                max_level_nd(self.width, shape, axes)
            } else {
                level
            };

            general_nd_per_forward_multilevel(
                |x, s, d| (self.dwt_adj_inverse)(x, s, d),
                input,
                output,
                shape,
                axes,
                level,
            );
        }
    }

    fn general_nd_forward_multilevel<F, T, L, const N: usize>(
        func: F,
        input: &L,
        output: &mut L,
        in_shape: &[usize],
        out_shape: &[usize],
        params: TransformParams,
    ) where
        F: Fn(&[T], &mut [T], &mut [T]) + Sync,
        L: LanesParallelIterator<Item = T> + ?Sized,
        T: Clone + Zero + ChunkWidth<T, N> + Send + Sync,
    {
        let TransformParams { axes, level, width } = params;
        let ndim = in_shape.len();
        debug_assert_eq!(
            in_shape.len(),
            out_shape.len(),
            "input and output shapes must have the same number of dimensions"
        );
        debug_assert!(axes.iter().all(|i| *i < ndim));
        let axes = HashSet::<_>::from_iter(axes.iter().cloned());
        // note that axes is a HashSet, so they are gauranteed to be different axes.

        let mut first = true;

        let mut in_sub_shape = in_shape.to_owned();
        let mut out_sub_shape = out_shape.to_owned();

        for _level in 0..level {
            let mut sub_shape = in_sub_shape.clone();
            for &ax in &axes {
                let n_ax = sub_shape[ax];
                let n_sd = get_outlen(width, n_ax);

                sub_shape[ax] = out_sub_shape[ax];

                // Note that everything does work for n_s == 1 (or 0 for that matter),
                // just that there really isn't anything useful to do.
                if n_sd > 1 {
                    match first {
                        true => {
                            let (in_chunks, in_rem) = input.par_iter_lane_chunks::<N>(in_shape, ax);
                            let (out_chunks, out_rem) =
                                output.par_iter_lane_chunks_sub_mut::<N>(out_shape, &sub_shape, ax);

                            in_chunks.zip(out_chunks).for_each_init(
                                || {
                                    let x = core::array::from_fn(|_| avec![T::zero(); n_ax]);
                                    let s = core::array::from_fn(|_| avec![T::zero(); n_sd]);
                                    let d = core::array::from_fn(|_| avec![T::zero(); n_sd]);
                                    (x, s, d)
                                },
                                |(x, s, d), (in_chunk, mut out_chunk)| {
                                    // copy strided chunks into the local storage

                                    in_chunk.pour_into(x);
                                    x.iter().zip(s.iter_mut().zip(d.iter_mut())).for_each(
                                        |(x, (s, d))| {
                                            func(x, s, d);
                                        },
                                    );
                                    // clone local storage to the output
                                    out_chunk.stack(s, d);
                                },
                            );
                            in_rem.zip(out_rem).for_each_init(
                                || {
                                    let x = avec![T::zero(); n_ax];
                                    let s = avec![T::zero(); n_sd];
                                    let d = avec![T::zero(); n_sd];
                                    (x, s, d)
                                },
                                |(x, s, d), (in_slice, mut out_slice)| {
                                    // copy strided slice into local dimension storage
                                    in_slice.pour_into(x);
                                    func(x, s, d);
                                    // copy local back to output strided slice
                                    out_slice.stack(s, d);
                                },
                            );

                            first = false;
                        }
                        false => {
                            let (chunks, rem) =
                                output.par_iter_lane_chunks_sub_mut::<N>(out_shape, &sub_shape, ax);

                            if chunks.len() > 0 {
                                chunks.for_each_init(
                                    || {
                                        let x = core::array::from_fn(|_| avec![T::zero(); n_ax]);
                                        let s = core::array::from_fn(|_| avec![T::zero(); n_sd]);
                                        let d = core::array::from_fn(|_| avec![T::zero(); n_sd]);
                                        (x, s, d)
                                    },
                                    |(x, s, d), mut chunk| {
                                        // copy (and deinterleave) strided chunks into the local storage

                                        chunk.pour_into(x);
                                        x.iter().zip(s.iter_mut().zip(d.iter_mut())).for_each(
                                            |(x, (s, d))| {
                                                func(x, s, d);
                                            },
                                        );
                                        // clone local storage to the output
                                        chunk.stack(s, d);
                                    },
                                );
                            }
                            rem.for_each_init(
                                || {
                                    let x = avec![T::zero(); n_ax];
                                    let s = avec![T::zero(); n_sd];
                                    let d = avec![T::zero(); n_sd];
                                    (x, s, d)
                                },
                                |(x, s, d), mut slc| {
                                    // copy strided slice into local dimension storage
                                    slc.pour_into(x);
                                    func(x, s, d);
                                    // copy local back to output strided slice
                                    slc.stack(s, d);
                                },
                            );
                        }
                    }
                }
            }

            // shrink shape for each axis we used.
            for &ax in &axes {
                let n_ax = in_sub_shape[ax];
                let n_sd = get_outlen(width, n_ax);
                if n_sd > 1 {
                    out_sub_shape[ax] -= n_sd;
                    in_sub_shape[ax] = n_sd;
                }
            }
        }
    }

    fn general_nd_inverse_multilevel<F, T, L, const N: usize>(
        func: F,
        inwork: &mut L,
        output: &mut L,
        in_shape: &[usize],
        out_shape: &[usize],
        params: TransformParams,
    ) where
        F: Fn(&[T], &[T], &mut [T]) + Sync,
        L: LanesParallelIterator<Item = T> + ?Sized,
        T: Clone + Zero + ChunkWidth<T, N> + Send + Sync,
    {
        let TransformParams { axes, level, width } = params;
        let ndim = in_shape.len();
        debug_assert_eq!(
            in_shape.len(),
            out_shape.len(),
            "input and output shapes must have the same number of dimensions"
        );
        let axes = HashSet::<_>::from_iter(axes.iter().cloned());
        debug_assert!(axes.iter().all(|i| *i < ndim));
        // note that axes is a HashSet, so they are gauranteed to be different axes.

        // make some lists to keep track of the shapes at each level, as we need to iterate in reverse order later.
        let mut ax_shapes = Vec::with_capacity(level);
        let mut out_shapes = Vec::with_capacity(level);
        let mut sd_shapes = Vec::with_capacity(level);

        out_shapes.push(in_shape.to_owned());
        ax_shapes.push(out_shape.to_owned());
        for _level in 0..level {
            // shrink shape for each axis that is used.
            let mut sd_shape = ax_shapes.last().unwrap().clone();
            let mut next_out_shape = out_shapes.last().unwrap().clone();
            for &ax in &axes {
                let n_ax = sd_shape[ax];
                let n_sd = get_outlen(width, n_ax);

                if n_sd > 1 {
                    sd_shape[ax] = n_sd;
                    next_out_shape[ax] -= n_sd;
                }
            }
            if _level + 1 < level {
                ax_shapes.push(sd_shape.clone());
                out_shapes.push(next_out_shape);
            }
            sd_shapes.push(sd_shape);
        }

        for level in (0..level).rev() {
            let mut sub_shape = out_shapes[level].clone();
            for &ax in &axes {
                let n_ax = ax_shapes[level][ax];
                let n_sd = sd_shapes[level][ax];

                // Note that everything does work for n_s == 1 (or 0 for that matter),
                // just that there really isn't anything useful to do.
                if n_sd > 1 {
                    let (chunks, rem) =
                        inwork.par_iter_lane_chunks_sub_mut::<N>(in_shape, &sub_shape, ax);

                    if chunks.len() > 0 {
                        chunks.for_each_init(
                            || {
                                let x = core::array::from_fn(|_| avec![T::zero(); n_ax]);
                                let s = core::array::from_fn(|_| avec![T::zero(); n_sd]);
                                let d = core::array::from_fn(|_| avec![T::zero(); n_sd]);
                                (x, s, d)
                            },
                            |(x, s, d), mut chunk| {
                                // split the chunk into the approximation and detail coefficients.
                                chunk.split(s, d);
                                x.iter_mut()
                                    .zip(s.iter().zip(d.iter()))
                                    .for_each(|(x, (s, d))| {
                                        func(s, d, x);
                                    });
                                // clone local storage to the output
                                chunk.fill_from(x);
                            },
                        );
                    }
                    if rem.len() > 0 {
                        rem.for_each_init(
                            || {
                                let x = avec![T::zero(); n_ax];
                                let s = avec![T::zero(); n_sd];
                                let d = avec![T::zero(); n_sd];
                                (x, s, d)
                            },
                            |(x, s, d), mut slc| {
                                // split the slice into the approximation and detail coefficients.
                                slc.split(s, d);
                                func(s, d, x);
                                // copy local back to output strided slice
                                slc.fill_from(x);
                            },
                        );
                    }
                    // the next passes sub shape along this dimension will have the size of n_ax
                    sub_shape[ax] = n_ax;
                }
            }
        }

        // copy input into output
        let min_axis = output.min_stride_axis(out_shape);
        let (in_chunks, in_rem) =
            inwork.par_iter_lane_chunks_sub::<N>(in_shape, out_shape, min_axis);
        let (out_chunks, out_rem) = output.par_iter_lane_chunks_mut::<N>(out_shape, min_axis);

        out_chunks.zip(in_chunks).for_each(|(mut o, i)| {
            o.iter_mut().zip(i.iter()).for_each(|(o, i)| {
                o.into_iter()
                    .zip(i.into_iter().cloned())
                    .for_each(|(o, i)| {
                        *o = i;
                    });
            });
        });
        out_rem.zip(in_rem).for_each(|(mut o, i)| {
            o.iter_mut()
                .zip(i.iter().cloned())
                .for_each(|(o, i)| *o = i);
        });
    }

    fn general_nd_per_forward_multilevel<F, T, L, const N: usize>(
        func: F,
        input: &L,
        output: &mut L,
        shape: &[usize],
        axes: &[usize],
        level: usize,
    ) where
        F: Fn(&[T], &mut [T], &mut [T]) + Sync,
        L: LanesParallelIterator<Item = T> + ?Sized,
        T: Clone + Zero + ChunkWidth<T, N> + Send + Sync,
    {
        let ndim = shape.len();
        let axes = HashSet::<_>::from_iter(axes.iter().cloned());
        debug_assert!(axes.iter().all(|i| *i < ndim));
        // note that axes is a HashSet, so they are gauranteed to be different axes.

        let mut first = true;

        let mut in_sub_shape = shape.to_owned();
        let mut out_sub_shape = shape.to_owned();

        for _level in 0..level {
            let mut sub_shape = in_sub_shape.clone();
            for &ax in &axes {
                let n_ax = sub_shape[ax];
                let n_d = n_ax / 2;
                let n_s = n_ax - n_d;

                sub_shape[ax] = out_sub_shape[ax];

                // Note that everything does work for n_s == 1 (or 0 for that matter),
                // just that there really isn't anything useful to do.
                if n_s > 1 {
                    match first {
                        true => {
                            let (in_chunks, in_rem) = input.par_iter_lane_chunks::<N>(shape, ax);
                            let (out_chunks, out_rem) =
                                output.par_iter_lane_chunks_sub_mut::<N>(shape, &sub_shape, ax);

                            in_chunks.zip(out_chunks).for_each_init(
                                || {
                                    let x = core::array::from_fn(|_| avec![T::zero(); n_ax]);
                                    let s = core::array::from_fn(|_| avec![T::zero(); n_s]);
                                    let d = core::array::from_fn(|_| avec![T::zero(); n_d]);
                                    (x, s, d)
                                },
                                |(x, s, d), (in_chunk, mut out_chunk)| {
                                    // copy strided chunks into the local storage
                                    in_chunk.pour_into(x);
                                    x.iter().zip(s.iter_mut().zip(d.iter_mut())).for_each(
                                        |(x, (s, d))| {
                                            func(x, s, d);
                                        },
                                    );
                                    // clone local storage to the output
                                    out_chunk.stack(s, d);
                                },
                            );
                            in_rem.zip(out_rem).for_each_init(
                                || {
                                    let x = avec![T::zero(); n_ax];
                                    let s = avec![T::zero(); n_s];
                                    let d = avec![T::zero(); n_d];
                                    (x, s, d)
                                },
                                |(x, s, d), (in_slice, mut out_slice)| {
                                    // copy strided slice into local dimension storage
                                    in_slice.pour_into(x);
                                    func(x, s, d);
                                    // copy local back to output strided slice
                                    out_slice.stack(s, d);
                                },
                            );
                            first = false;
                        }
                        false => {
                            let (chunks, rem) =
                                output.par_iter_lane_chunks_sub_mut::<N>(shape, &sub_shape, ax);

                            if chunks.len() > 0 {
                                chunks.for_each_init(
                                    || {
                                        let x = core::array::from_fn(|_| avec![T::zero(); n_ax]);
                                        let s = core::array::from_fn(|_| avec![T::zero(); n_s]);
                                        let d = core::array::from_fn(|_| avec![T::zero(); n_d]);
                                        (x, s, d)
                                    },
                                    |(x, s, d), mut chunk| {
                                        // copy (and deinterleave) strided chunks into the local storage

                                        chunk.pour_into(x);
                                        x.iter().zip(s.iter_mut().zip(d.iter_mut())).for_each(
                                            |(x, (s, d))| {
                                                func(x, s, d);
                                            },
                                        );
                                        // clone local storage to the output
                                        chunk.stack(s, d);
                                    },
                                );
                            }
                            if rem.len() > 0 {
                                rem.for_each_init(
                                    || {
                                        let x = avec![T::zero(); n_ax];
                                        let s = avec![T::zero(); n_s];
                                        let d = avec![T::zero(); n_d];
                                        (x, s, d)
                                    },
                                    |(x, s, d), mut slc| {
                                        // copy strided slice into local dimension storage
                                        slc.pour_into(x);
                                        func(x, s, d);
                                        // copy local back to output strided slice
                                        slc.stack(s, d);
                                    },
                                );
                            }
                        }
                    }
                }
            }

            // shrink shape for each axis we used.
            for &ax in &axes {
                let n_ax = in_sub_shape[ax];
                let n_d = n_ax / 2;
                let n_s = n_ax - n_d;
                if n_s > 1 {
                    out_sub_shape[ax] -= n_d;
                    in_sub_shape[ax] = n_s;
                }
            }
        }
    }

    fn general_nd_per_inverse_multilevel<F, T, L, const N: usize>(
        func: F,
        input: &L,
        output: &mut L,
        shape: &[usize],
        axes: &[usize],
        level: usize,
    ) where
        F: Fn(&[T], &[T], &mut [T]) + Sync,
        L: LanesParallelIterator<Item = T> + ?Sized,
        T: Clone + Zero + ChunkWidth<T, N> + Send + Sync,
    {
        let ndim = shape.len();
        let axes = HashSet::<_>::from_iter(axes.iter().cloned());
        debug_assert!(axes.iter().all(|i| *i < ndim));
        // note that axes is a HashSet, so they are gauranteed to be different axes.

        // If the input shape and the output shape are the same, then we are in per mode

        // make some lists to keep track of the shapes at each level, as we need to iterate in reverse order later.
        let mut ax_shapes = Vec::with_capacity(level);
        let mut out_shapes = Vec::with_capacity(level);
        let mut approx_shapes = Vec::with_capacity(level);
        let mut detail_shapes = Vec::with_capacity(level);

        out_shapes.push(shape.to_owned());
        ax_shapes.push(shape.to_owned());
        for _level in 0..level {
            // shrink shape for each axis that is used.
            let mut approx_shape = ax_shapes.last().unwrap().clone();
            let mut detail_shape = approx_shape.clone();
            let mut next_out_shape = out_shapes.last().unwrap().clone();
            for &ax in &axes {
                let n_ax = approx_shape[ax];
                let n_d = n_ax / 2;
                let n_s = n_ax - n_d;
                if n_s > 1 {
                    approx_shape[ax] = n_s;
                    detail_shape[ax] = n_d;
                    next_out_shape[ax] -= n_d;
                }
            }
            if _level + 1 < level {
                ax_shapes.push(approx_shape.clone());
                out_shapes.push(next_out_shape);
            }
            approx_shapes.push(approx_shape);
            detail_shapes.push(detail_shape);
        }

        // In per mode we can copy the input to the output right away and not modify the input array.
        let min_axis = output.min_stride_axis(shape);
        let (in_chunks, in_rem) = input.par_iter_lane_chunks::<N>(shape, min_axis);
        let (out_chunks, out_rem) = output.par_iter_lane_chunks_mut::<N>(shape, min_axis);

        out_chunks.zip(in_chunks).for_each(|(mut o, i)| {
            o.iter_mut().zip(i.iter()).for_each(|(o, i)| {
                o.into_iter()
                    .zip(i.into_iter().cloned())
                    .for_each(|(o, i)| {
                        *o = i;
                    });
            });
        });
        out_rem.zip(in_rem).for_each(|(mut o, i)| {
            o.iter_mut()
                .zip(i.iter().cloned())
                .for_each(|(o, i)| *o = i);
        });

        for level in (0..level).rev() {
            let mut sub_shape = out_shapes[level].clone();
            for &ax in &axes {
                let n_ax = ax_shapes[level][ax];
                let n_s = approx_shapes[level][ax];
                let n_d = detail_shapes[level][ax];

                // Note that everything does work for n_s == 1 (or 0 for that matter),
                // just that there really isn't anything useful to do.
                if n_s > 1 {
                    let (chunks, rem) =
                        output.par_iter_lane_chunks_sub_mut::<N>(shape, &sub_shape, ax);

                    chunks.for_each_init(
                        || {
                            let x = core::array::from_fn(|_| avec![T::zero(); n_ax]);
                            let s = core::array::from_fn(|_| avec![T::zero(); n_s]);
                            let d = core::array::from_fn(|_| avec![T::zero(); n_d]);
                            (x, s, d)
                        },
                        |(x, s, d), mut chunk| {
                            // split the chunk into the approximation and detail coefficients.
                            chunk.split(s, d);
                            x.iter_mut()
                                .zip(s.iter().zip(d.iter()))
                                .for_each(|(x, (s, d))| {
                                    func(s, d, x);
                                });
                            // clone local storage to the output
                            chunk.fill_from(x);
                        },
                    );
                    if rem.len() > 0 {
                        rem.for_each_init(
                            || {
                                let x = avec![T::zero(); n_ax];
                                let s = avec![T::zero(); n_s];
                                let d = avec![T::zero(); n_d];
                                (x, s, d)
                            },
                            |(x, s, d), mut slc| {
                                // split the slice into the approximation and detail coefficients.
                                slc.split(s, d);
                                func(s, d, x);
                                // copy local back to output strided slice
                                slc.fill_from(x);
                            },
                        );
                    }
                    // the next passes sub shape along this dimension will have the size of n_ax
                    sub_shape[ax] = n_ax;
                }
            }
        }
    }
}
