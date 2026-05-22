#[cfg(feature = "ndarray")]
use ndarray::{ArrayRef, Dimension};
use num_traits::Zero;
use std::collections::HashSet;

use crate::Wavelets;
use crate::boundarys::BoundaryExtension;
use crate::iter::LanesIterator;

use aligned_vec::{AVec, avec};

use crate::utils::{avecs_to_mut_slices, avecs_to_slices, deinterleave, interleave};
use crate::{ChunkWidth, max_level_nd, simd::SimdTransformable};

use wavelets_macros::generate_wavelet_match_arms;

/// High-level Lifting Wavelet Transform driver.
///
/// `WaveletTransform` owns the lifting-step function pointers for a chosen wavelet
/// and boundary condition. The const generic `N` should be based on the processor's
/// cache size and the size of the type that is transformed. By default N is chosen
/// using the [`ChunkWidth`] impls to pick a reasonable value (e.g. `8` for `f32` and
/// `4` for `f64` on `x86_64` processors).s
///
/// # Example
///
/// ```rust,ignore
/// use wavelets::{Wavelets, boundarys::BoundaryCondition, lwt::driver::WaveletTransform};
///
/// let xfm: WaveletTransform =
///     WaveletTransform::new(Wavelets::Daubechies4, BoundaryCondition::Periodic);
///
/// let input = vec![1.0_f64; 64];
/// let mut output = vec![0.0_f64; 64];
/// xfm.forward_nd(&input, &mut output, &[64], &[0]);
/// ```
pub struct WaveletTransform<T, BC, const N: usize>
where
    T: ChunkWidth<T, N>,
{
    lwt_forward: fn(&mut [T], &mut [T], &BC),
    lwt_inverse: fn(&mut [T], &mut [T], &BC),
    lwt_adj_forward: fn(&mut [T], &mut [T], &BC),
    lwt_adj_inverse: fn(&mut [T], &mut [T], &BC),
    bc: BC,
    width: usize,
}

impl<T, BC, const N: usize> WaveletTransform<T, BC, N>
where
    T: SimdTransformable + Zero + ChunkWidth<T, N>,
    BC: BoundaryExtension,
{
    /// Construct a `WaveletTransform` for the given `wvlt` and `bc`.
    ///
    /// Function pointers to the correct lifting-step implementations are resolved at
    /// construction time so every subsequent transform call is a direct (non-virtual)
    /// dispatch with no runtime branching on the wavelet type.
    pub fn new(wvlt: Wavelets, bc: BC) -> Self {
        use crate::lwt::bior::*;
        use crate::lwt::coiflet::*;
        use crate::lwt::daubechies::*;
        use crate::lwt::symlet::*;
        let lwt_forward: fn(&mut [T], &mut [T], &BC) = generate_wavelet_match_arms! {
            Wavelets,
            wvlt,
            {#wvlt::forward,}
        };
        let lwt_inverse: fn(&mut [T], &mut [T], &BC) = generate_wavelet_match_arms! {
            Wavelets,
            wvlt,
            {#wvlt::inverse,}
        };
        let lwt_adj_forward: fn(&mut [T], &mut [T], &BC) = generate_wavelet_match_arms! {
            Wavelets,
            wvlt,
            {#wvlt::adjoint_forward,}
        };
        let lwt_adj_inverse: fn(&mut [T], &mut [T], &BC) = generate_wavelet_match_arms! {
            Wavelets,
            wvlt,
            {#wvlt::adjoint_inverse,}
        };

        let width = wvlt.width();
        Self {
            lwt_forward,
            lwt_inverse,
            lwt_adj_forward,
            lwt_adj_inverse,
            bc,
            width,
        }
    }

    /// Get the maximum recommended decomposition of the LWT forward transform.
    ///
    /// For a given shape and transformed axes, determine the maximum "useful" decomposition
    /// level for the transform.
    pub fn max_level_nd(&self, shape: &[usize], axes: &[usize]) -> usize {
        max_level_nd(self.width, shape, axes)
    }

    /// Single-level forward LWT of a 1-D signal.
    ///
    /// Splits `input` into even/odd samples, then applies the forward lifting steps
    /// in-place, writing approximation coefficients into `s` and detail coefficients
    /// into `d`.
    pub fn forward_1d(&self, input: &[T], s: &mut [T], d: &mut [T]) {
        deinterleave(input, s, d);
        (self.lwt_forward)(s, d, &self.bc);
    }

    /// Single-level inverse LWT of a 1-D signal.
    ///
    /// Applies the inverse lifting steps to a copy of `s` and `d`, then interleaves
    /// the result back into `output`.
    pub fn inverse_1d(&self, s: &[T], d: &[T], output: &mut [T]) {
        let (mut s, mut d) = (s.to_owned(), d.to_owned());
        (self.lwt_inverse)(&mut s, &mut d, &self.bc);
        interleave(&s, &d, output);
    }

    /// Adjoint of the forward 1-D LWT.
    pub fn adj_forward_1d(&self, s: &[T], d: &[T], output: &mut [T]) {
        let (mut s, mut d) = (s.to_owned(), d.to_owned());
        (self.lwt_adj_forward)(&mut s, &mut d, &self.bc);
        interleave(&s, &d, output);
    }

    /// Adjoint of the inverse 1-D LWT.
    pub fn adj_inverse_1d(&self, input: &[T], s: &mut [T], d: &mut [T]) {
        deinterleave(input, s, d);
        (self.lwt_adj_inverse)(s, d, &self.bc);
    }

    /// Single-level forward LWT applied to each axis in `axes` of an N-D array.
    ///
    /// `shape` describes the logical dimensions of the flat slice `input`/`output`.
    pub fn forward_nd(&self, input: &[T], output: &mut [T], shape: &[usize], axes: &[usize]) {
        self.forward_multilevel_nd(input, output, shape, axes, 1);
    }

    /// Single-level inverse LWT applied to each axis in `axes` of an N-D array.
    pub fn inverse_nd(&self, input: &[T], output: &mut [T], shape: &[usize], axes: &[usize]) {
        self.inverse_multilevel_nd(input, output, shape, axes, 1);
    }

    /// Single-level adjoint of the forward LWT on an N-D array.
    pub fn adj_forward_nd(&self, input: &[T], output: &mut [T], shape: &[usize], axes: &[usize]) {
        self.adj_forward_multilevel_nd(input, output, shape, axes, 1);
    }

    /// Single-level adjoint of the inverse LWT on an N-D array.
    pub fn adj_inverse_nd(&self, input: &[T], output: &mut [T], shape: &[usize], axes: &[usize]) {
        self.adj_inverse_multilevel_nd(input, output, shape, axes, 1);
    }

    /// Multi-level forward LWT on an N-D array.
    ///
    /// Applies `level` successive single-level forward transforms along each axis in
    /// `axes`.  The approximation sub-band is recursively decomposed at each level. If `level==0`
    /// then it will compute to the maximum level suggested by `max_level_nd`.
    pub fn forward_multilevel_nd(
        &self,
        input: &[T],
        output: &mut [T],
        shape: &[usize],
        axes: &[usize],
        level: usize,
    ) {
        let level = if level == 0 {
            max_level_nd(self.width, shape, axes)
        } else {
            level
        };
        general_nd_forward_multilevel(
            |s, d| (self.lwt_forward)(s, d, &self.bc),
            input,
            output,
            shape,
            axes,
            level,
        );
    }

    /// Multi-level inverse LWT on an N-D array.
    ///
    /// Reverses `forward_multilevel_nd` for the same `level` and `axes`.
    pub fn inverse_multilevel_nd(
        &self,
        input: &[T],
        output: &mut [T],
        shape: &[usize],
        axes: &[usize],
        level: usize,
    ) {
        let level = if level == 0 {
            max_level_nd(self.width, shape, axes)
        } else {
            level
        };
        general_nd_inverse_multilevel(
            |s, d| (self.lwt_inverse)(s, d, &self.bc),
            input,
            output,
            shape,
            axes,
            level,
        );
    }

    /// Multi-level adjoint of the forward LWT on an N-D array.
    pub fn adj_forward_multilevel_nd(
        &self,
        input: &[T],
        output: &mut [T],
        shape: &[usize],
        axes: &[usize],
        level: usize,
    ) {
        let level = if level == 0 {
            max_level_nd(self.width, shape, axes)
        } else {
            level
        };
        general_nd_inverse_multilevel(
            |s, d| (self.lwt_adj_forward)(s, d, &self.bc),
            input,
            output,
            shape,
            axes,
            level,
        );
    }

    /// Multi-level adjoint of the inverse LWT on an N-D array.
    pub fn adj_inverse_multilevel_nd(
        &self,
        input: &[T],
        output: &mut [T],
        shape: &[usize],
        axes: &[usize],
        level: usize,
    ) {
        let level = if level == 0 {
            max_level_nd(self.width, shape, axes)
        } else {
            level
        };
        general_nd_forward_multilevel(
            |s, d| (self.lwt_adj_inverse)(s, d, &self.bc),
            input,
            output,
            shape,
            axes,
            level,
        );
    }
}

#[cfg(feature = "ndarray")]
impl<T, BC, const N: usize> WaveletTransform<T, BC, N>
where
    T: SimdTransformable + Zero + ChunkWidth<T, N>,
    BC: BoundaryExtension,
{
    /// Forward LWT applied to an ndarray (multi-level).
    pub fn forward_ndarray_multilevel<D: Dimension>(
        &self,
        input: &ArrayRef<T, D>,
        output: &mut ArrayRef<T, D>,
        axes: &[usize],
        level: usize,
    ) {
        let shape = input.shape();
        let level = if level == 0 {
            max_level_nd(self.width, shape, axes)
        } else {
            level
        };
        assert_eq!(
            shape,
            output.shape(),
            "input shape and output shape must be equal."
        );

        general_nd_forward_multilevel(
            |s, d| (self.lwt_forward)(s, d, &self.bc),
            input,
            output,
            shape,
            axes,
            level,
        );
    }

    /// Inverse LWT applied to an ndarray (multi-level).
    pub fn inverse_ndarray_multilevel<D: Dimension>(
        &self,
        input: &ArrayRef<T, D>,
        output: &mut ArrayRef<T, D>,
        axes: &[usize],
        level: usize,
    ) {
        let shape = input.shape();
        let level = if level == 0 {
            max_level_nd(self.width, shape, axes)
        } else {
            level
        };
        assert_eq!(
            shape,
            output.shape(),
            "input shape and output shape must be equal."
        );

        general_nd_inverse_multilevel(
            |s, d| (self.lwt_inverse)(s, d, &self.bc),
            input,
            output,
            shape,
            axes,
            level,
        );
    }

    /// Adjoint forward LWT applied to an ndarray (multi-level).
    pub fn adj_forward_ndarray_multilevel<D: Dimension>(
        &self,
        input: &ArrayRef<T, D>,
        output: &mut ArrayRef<T, D>,
        axes: &[usize],
        level: usize,
    ) {
        let shape = input.shape();
        let level = if level == 0 {
            max_level_nd(self.width, shape, axes)
        } else {
            level
        };
        assert_eq!(
            shape,
            output.shape(),
            "input shape and output shape must be equal."
        );

        general_nd_inverse_multilevel(
            |s, d| (self.lwt_adj_forward)(s, d, &self.bc),
            input,
            output,
            shape,
            axes,
            level,
        );
    }

    /// Adjoint inverse LWT applied to an ndarray (multi-level).
    pub fn adj_inverse_ndarray_multilevel<D: Dimension>(
        &self,
        input: &ArrayRef<T, D>,
        output: &mut ArrayRef<T, D>,
        axes: &[usize],
        level: usize,
    ) {
        let shape = input.shape();
        let level = if level == 0 {
            max_level_nd(self.width, shape, axes)
        } else {
            level
        };
        assert_eq!(
            shape,
            output.shape(),
            "input shape and output shape must be equal."
        );

        general_nd_forward_multilevel(
            |s, d| (self.lwt_adj_inverse)(s, d, &self.bc),
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
    shape: &[usize],
    axes: &[usize],
    level: usize,
) where
    F: Fn(&mut [T], &mut [T]),
    L: LanesIterator<Item = T> + ?Sized,
    T: Clone + Zero + ChunkWidth<T, N>,
{
    let ndim = shape.len();
    let axes = HashSet::<_>::from_iter(axes.iter().cloned());
    assert!(axes.iter().all(|i| *i < ndim));
    // note that axes is a HashSet, so they are gauranteed to be different axes.

    let mut first = true;

    let mut sub_shape = shape.to_owned();
    for _level in 0..level {
        for &ax in &axes {
            let n_ax = sub_shape[ax];

            let n_d = n_ax / 2;
            let n_s = n_ax - n_d;

            // Note that everything does work for n_s == 1 (or 0 for that matter),
            // just that there really isn't anything useful to do.
            if n_s > 1 {
                match first {
                    true => {
                        let in_chunks = input.iter_lane_chunks_sub::<N>(shape, &sub_shape, ax);
                        let in_rem = in_chunks.remainder();
                        let out_chunks =
                            output.iter_lane_chunks_sub_mut::<N>(shape, &sub_shape, ax);
                        let out_rem = out_chunks.remainder();

                        if in_chunks.len() > 0 {
                            let mut s_w: [AVec<T>; N] =
                                core::array::from_fn(|_| avec![T::zero(); n_s]);
                            let mut d_w: [AVec<T>; N] =
                                core::array::from_fn(|_| avec![T::zero(); n_d]);
                            in_chunks
                                .zip(out_chunks)
                                .for_each(|(in_chunk, mut out_chunk)| {
                                    // copy (and deinterleave) strided chunks into the local storage
                                    let mut s = avecs_to_mut_slices(&mut s_w);
                                    let mut d = avecs_to_mut_slices(&mut d_w);
                                    in_chunk.deinterleave(&mut s, &mut d);

                                    s.iter_mut().zip(d.iter_mut()).for_each(|(s, d)| {
                                        func(s, d);
                                    });

                                    // clone local storage to the output
                                    let s = avecs_to_slices(&s_w);
                                    let d = avecs_to_slices(&d_w);
                                    out_chunk.stack(&s, &d);
                                });
                        }
                        if in_rem.len() > 0 {
                            let mut s = avec![T::zero(); n_s];
                            let mut d = avec![T::zero(); n_d];
                            in_rem.zip(out_rem).for_each(|(in_slice, mut out_slice)| {
                                // copy strided slice into local dimension storage
                                in_slice.deinterleave(&mut s, &mut d);
                                func(&mut s, &mut d);
                                // copy local back to output strided slice
                                out_slice.stack(&s, &d);
                            });
                        }

                        first = false;
                    }
                    false => {
                        let chunks = output.iter_lane_chunks_sub_mut::<N>(shape, &sub_shape, ax);
                        let rem = chunks.remainder();

                        if chunks.len() > 0 {
                            let mut s_w: [AVec<T>; N] =
                                core::array::from_fn(|_| avec![T::zero(); n_s]);
                            let mut d_w: [AVec<T>; N] =
                                core::array::from_fn(|_| avec![T::zero(); n_d]);
                            chunks.for_each(|mut chunk| {
                                // copy (and deinterleave) strided chunks into the local storage
                                let mut s = avecs_to_mut_slices(&mut s_w);
                                let mut d = avecs_to_mut_slices(&mut d_w);
                                chunk.deinterleave(&mut s, &mut d);
                                s.iter_mut().zip(d.iter_mut()).for_each(|(s, d)| {
                                    func(s, d);
                                });
                                // clone local storage to the output
                                let s = avecs_to_slices(&s_w);
                                let d = avecs_to_slices(&d_w);
                                chunk.stack(&s, &d);
                            });
                        }
                        if rem.len() > 0 {
                            let mut s = avec![T::zero(); n_s];
                            let mut d = avec![T::zero(); n_d];
                            rem.for_each(|mut slc| {
                                // copy strided slice into local dimension storage
                                slc.deinterleave(&mut s, &mut d);
                                func(&mut s, &mut d);
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
            if sub_shape[ax] > 1 {
                sub_shape[ax] = sub_shape[ax].div_ceil(2);
            }
        }
    }
}

fn general_nd_inverse_multilevel<F, T, L, const N: usize>(
    func: F,
    input: &L,
    output: &mut L,
    shape: &[usize],
    axes: &[usize],
    level: usize,
) where
    F: Fn(&mut [T], &mut [T]),
    L: LanesIterator<Item = T> + ?Sized,
    T: Clone + Zero + ChunkWidth<T, N>,
{
    let ndim = shape.len();
    let axes = HashSet::<_>::from_iter(axes.iter().cloned());
    assert!(axes.iter().all(|i| *i < ndim));
    // note that axes is a HashSet, so they are gauranteed to be different axes.

    // copy input into the output
    let min_axis = output.min_stride_axis(shape);

    let in_chunks = input.iter_lane_chunks::<N>(shape, min_axis);
    let in_rem = in_chunks.remainder();
    let out_chunks = output.iter_lane_chunks_mut::<N>(shape, min_axis);
    let out_rem = out_chunks.remainder();

    out_chunks.zip(in_chunks).for_each(|(mut o, i)| {
        o.iter_mut().zip(i.iter()).for_each(|(o, i)| {
            o.into_iter().zip(i).for_each(|(o, i)| {
                *o = i.clone();
            });
        });
    });
    out_rem.zip(in_rem).for_each(|(mut o, i)| {
        o.iter_mut().zip(i.iter()).for_each(|(o, i)| *o = i.clone());
    });

    let mut sub_shape = shape.to_owned();

    let shape_levels = (0..level)
        .map(|_| {
            let next_shape = sub_shape.clone();
            for &ax in &axes {
                if sub_shape[ax] > 1 {
                    sub_shape[ax] = sub_shape[ax].div_ceil(2);
                }
            }
            next_shape
        })
        .collect::<Vec<_>>();

    for lvl in (0..level).rev() {
        let sub_shape = &shape_levels[lvl];
        for &ax in &axes {
            let n_ax = sub_shape[ax];

            let n_d = n_ax / 2;
            let n_s = n_ax - n_d;
            if n_s > 1 {
                let chunks = output.iter_lane_chunks_sub_mut::<N>(shape, sub_shape, ax);
                let rem = chunks.remainder();

                if chunks.len() > 0 {
                    let mut s_w = core::array::from_fn(|_| avec![T::zero(); n_s]);
                    let mut d_w = core::array::from_fn(|_| avec![T::zero(); n_d]);
                    chunks.for_each(|mut chunk| {
                        let mut s = avecs_to_mut_slices(&mut s_w);
                        let mut d = avecs_to_mut_slices(&mut d_w);
                        chunk.split(&mut s, &mut d);
                        s.iter_mut().zip(d.iter_mut()).for_each(|(s, d)| {
                            func(s, d);
                        });

                        let s = avecs_to_slices(&s_w);
                        let d = avecs_to_slices(&d_w);
                        chunk.interleave(&s, &d);
                    });
                }
                if rem.len() > 0 {
                    let mut s = avec![T::zero(); n_s];
                    let mut d = avec![T::zero(); n_d];
                    rem.for_each(|mut slc| {
                        slc.split(&mut s, &mut d);
                        func(&mut s, &mut d);
                        slc.interleave(&s, &d);
                    })
                }
            }
        }
    }
}

#[cfg(feature = "rayon")]
/// Rayon-parallel LWT drivers.
///
/// Mirrors the sequential [`WaveletTransform`] API but processes independent lanes
/// on multiple threads via Rayon.
pub mod parallel {
    use super::*;

    use crate::iter::parallel::LanesParallelIterator;
    use rayon::iter::{IndexedParallelIterator, ParallelIterator};

    impl<T, BC, const N: usize> WaveletTransform<T, BC, N>
    where
        T: SimdTransformable + Zero + ChunkWidth<T, N> + Sync + Send,
        BC: BoundaryExtension,
    {
        /// Single-level parallel forward LWT along the given `axes`.
        pub fn par_forward_nd(
            &self,
            input: &[T],
            output: &mut [T],
            shape: &[usize],
            axes: &[usize],
        ) {
            self.par_forward_multilevel_nd(input, output, shape, axes, 1);
        }

        /// Single-level parallel inverse LWT along the given `axes`.
        pub fn par_inverse_nd(
            &self,
            input: &[T],
            output: &mut [T],
            shape: &[usize],
            axes: &[usize],
        ) {
            self.par_inverse_multilevel_nd(input, output, shape, axes, 1);
        }

        /// Single-level parallel adjoint forward LWT along the given `axes`.
        pub fn par_adj_forward_nd(
            &self,
            input: &[T],
            output: &mut [T],
            shape: &[usize],
            axes: &[usize],
        ) {
            self.par_adj_forward_multilevel_nd(input, output, shape, axes, 1);
        }

        /// Single-level parallel adjoint inverse LWT along the given `axes`.
        pub fn par_adj_inverse_nd(
            &self,
            input: &[T],
            output: &mut [T],
            shape: &[usize],
            axes: &[usize],
        ) {
            self.par_adj_inverse_multilevel_nd(input, output, shape, axes, 1);
        }

        /// Multi-level parallel forward LWT along the given `axes`.
        pub fn par_forward_multilevel_nd(
            &self,
            input: &[T],
            output: &mut [T],
            shape: &[usize],
            axes: &[usize],
            level: usize,
        ) {
            let level = if level == 0 {
                max_level_nd(self.width, shape, axes)
            } else {
                level
            };
            general_nd_forward_multilevel(
                |s, d| (self.lwt_forward)(s, d, &self.bc),
                input,
                output,
                shape,
                axes,
                level,
            );
        }

        /// Multi-level parallel inverse LWT along the given `axes`.
        pub fn par_inverse_multilevel_nd(
            &self,
            input: &[T],
            output: &mut [T],
            shape: &[usize],
            axes: &[usize],
            level: usize,
        ) {
            let level = if level == 0 {
                max_level_nd(self.width, shape, axes)
            } else {
                level
            };
            general_nd_inverse_multilevel(
                |s, d| (self.lwt_inverse)(s, d, &self.bc),
                input,
                output,
                shape,
                axes,
                level,
            );
        }

        /// Multi-level parallel adjoint forward LWT along the given `axes`.
        pub fn par_adj_forward_multilevel_nd(
            &self,
            input: &[T],
            output: &mut [T],
            shape: &[usize],
            axes: &[usize],
            level: usize,
        ) {
            let level = if level == 0 {
                max_level_nd(self.width, shape, axes)
            } else {
                level
            };
            general_nd_inverse_multilevel(
                |s, d| (self.lwt_adj_forward)(s, d, &self.bc),
                input,
                output,
                shape,
                axes,
                level,
            );
        }

        /// Multi-level parallel adjoint inverse LWT along the given `axes`.
        pub fn par_adj_inverse_multilevel_nd(
            &self,
            input: &[T],
            output: &mut [T],
            shape: &[usize],
            axes: &[usize],
            level: usize,
        ) {
            let level = if level == 0 {
                max_level_nd(self.width, shape, axes)
            } else {
                level
            };
            general_nd_forward_multilevel(
                |s, d| (self.lwt_adj_inverse)(s, d, &self.bc),
                input,
                output,
                shape,
                axes,
                level,
            );
        }
    }

    #[cfg(feature = "ndarray")]
    impl<T, BC, const N: usize> WaveletTransform<T, BC, N>
    where
        T: SimdTransformable + Zero + ChunkWidth<T, N> + Sync + Send,
        BC: BoundaryExtension,
    {
        /// Forward LWT applied to an ndarray (parallel, multi-level).
        pub fn par_forward_ndarray_multilevel<D: Dimension>(
            &self,
            input: &ArrayRef<T, D>,
            output: &mut ArrayRef<T, D>,
            axes: &[usize],
            level: usize,
        ) {
            let shape = input.shape();
            let level = if level == 0 {
                max_level_nd(self.width, shape, axes)
            } else {
                level
            };
            assert_eq!(
                shape,
                output.shape(),
                "input shape and output shape must be equal."
            );

            general_nd_forward_multilevel(
                |s, d| (self.lwt_forward)(s, d, &self.bc),
                input,
                output,
                shape,
                axes,
                level,
            );
        }

        /// Inverse LWT applied to an ndarray (parallel, multi-level).
        pub fn par_inverse_ndarray_multilevel<D: Dimension>(
            &self,
            input: &ArrayRef<T, D>,
            output: &mut ArrayRef<T, D>,
            axes: &[usize],
            level: usize,
        ) {
            let shape = input.shape();
            let level = if level == 0 {
                max_level_nd(self.width, shape, axes)
            } else {
                level
            };
            assert_eq!(
                shape,
                output.shape(),
                "input shape and output shape must be equal."
            );

            general_nd_inverse_multilevel(
                |s, d| (self.lwt_inverse)(s, d, &self.bc),
                input,
                output,
                shape,
                axes,
                level,
            );
        }
        /// Adjoint forward LWT applied to an ndarray (parallel, multi-level).
        pub fn par_adj_forward_ndarray_multilevel<D: Dimension>(
            &self,
            input: &ArrayRef<T, D>,
            output: &mut ArrayRef<T, D>,
            axes: &[usize],
            level: usize,
        ) {
            let shape = input.shape();
            let level = if level == 0 {
                max_level_nd(self.width, shape, axes)
            } else {
                level
            };
            assert_eq!(
                shape,
                output.shape(),
                "input shape and output shape must be equal."
            );

            general_nd_inverse_multilevel(
                |s, d| (self.lwt_adj_forward)(s, d, &self.bc),
                input,
                output,
                shape,
                axes,
                level,
            );
        }

        /// Adjoint inverse LWT applied to an ndarray (parallel, multi-level).
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
                "input shape and output shape must be equal."
            );

            general_nd_forward_multilevel(
                |s, d| (self.lwt_adj_inverse)(s, d, &self.bc),
                input,
                output,
                shape,
                axes,
                level,
            );
        }
    }

    pub(super) fn general_nd_forward_multilevel<F, T, L, const N: usize>(
        func: F,
        input: &L,
        output: &mut L,
        shape: &[usize],
        axes: &[usize],
        level: usize,
    ) where
        F: Fn(&mut [T], &mut [T]) + Sync,
        L: LanesParallelIterator<Item = T> + ?Sized,
        T: Clone + Zero + ChunkWidth<T, N> + Send + Sync,
    {
        let ndim = shape.len();
        let axes = HashSet::<_>::from_iter(axes.iter().cloned());
        assert!(axes.iter().all(|i| *i < ndim));
        // note that axes is a HashSet, so they are gauranteed to be different axes.

        let mut first = true;

        let mut sub_shape = shape.to_owned();
        for _level in 0..level {
            for &ax in &axes {
                let n_ax = sub_shape[ax];

                let n_d = n_ax / 2;
                let n_s = n_ax - n_d;

                if n_s > 1 {
                    match first {
                        true => {
                            let in_chunks =
                                input.par_iter_lane_chunks_sub::<N>(shape, &sub_shape, ax);
                            let in_rem = in_chunks.remainder();
                            let out_chunks =
                                output.par_iter_lane_chunks_sub_mut::<N>(shape, &sub_shape, ax);
                            let out_rem = out_chunks.remainder();

                            in_chunks.zip(out_chunks).for_each_init(
                                || {
                                    let s = core::array::from_fn(|_| avec![T::zero(); n_s]);
                                    let d = core::array::from_fn(|_| avec![T::zero(); n_d]);
                                    (s, d)
                                },
                                |(s, d), (in_chunk, mut out_chunk)| {
                                    // copy (and deinterleave) strided chunks into the local storage
                                    in_chunk.deinterleave(
                                        &mut avecs_to_mut_slices(s),
                                        &mut avecs_to_mut_slices(d),
                                    );
                                    s.iter_mut().zip(d.iter_mut()).for_each(|(s, d)| {
                                        func(s, d);
                                    });
                                    // clone local storage to the output
                                    out_chunk.stack(&avecs_to_slices(s), &avecs_to_slices(d));
                                },
                            );
                            in_rem.zip(out_rem).for_each_init(
                                || {
                                    let s = avec![T::zero(); n_s];
                                    let d = avec![T::zero(); n_d];
                                    (s, d)
                                },
                                |(s, d), (in_slice, mut out_slice)| {
                                    // copy strided slice into local dimension storage
                                    in_slice.deinterleave(s, d);
                                    func(s, d);
                                    // copy local back to output strided slice
                                    out_slice.stack(s, d);
                                },
                            );
                            first = false;
                        }
                        false => {
                            let chunks =
                                output.par_iter_lane_chunks_sub_mut::<N>(shape, &sub_shape, ax);
                            let rem = chunks.remainder();

                            chunks.for_each_init(
                                || {
                                    let s = core::array::from_fn(|_| avec![T::zero(); n_s]);
                                    let d = core::array::from_fn(|_| avec![T::zero(); n_d]);
                                    (s, d)
                                },
                                |(s, d), mut chunk| {
                                    chunk.deinterleave(
                                        &mut avecs_to_mut_slices(s),
                                        &mut avecs_to_mut_slices(d),
                                    );
                                    s.iter_mut().zip(d.iter_mut()).for_each(|(s, d)| {
                                        func(s, d);
                                    });
                                    // clone local storage to the output
                                    chunk.stack(&avecs_to_slices(s), &avecs_to_slices(d));
                                },
                            );
                            rem.for_each_init(
                                || {
                                    let s = avec![T::zero(); n_s];
                                    let d = avec![T::zero(); n_d];
                                    (s, d)
                                },
                                |(s, d), mut slc| {
                                    // copy strided slice into local dimension storage
                                    slc.deinterleave(s, d);
                                    func(s, d);
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
                if sub_shape[ax] > 1 {
                    sub_shape[ax] = sub_shape[ax].div_ceil(2);
                }
            }
        }
    }

    pub(super) fn general_nd_inverse_multilevel<F, T, L, const N: usize>(
        func: F,
        input: &L,
        output: &mut L,
        shape: &[usize],
        axes: &[usize],
        level: usize,
    ) where
        F: Fn(&mut [T], &mut [T]) + Sync,
        L: LanesParallelIterator<Item = T> + ?Sized,
        T: Clone + Zero + ChunkWidth<T, N> + Send + Sync,
    {
        let ndim = shape.len();
        let axes = HashSet::<_>::from_iter(axes.iter().cloned());
        assert!(axes.iter().all(|i| *i < ndim));
        // note that axes is a HashSet, so they are gauranteed to be different axes.

        // copy input into the output
        let min_axis = output.min_stride_axis(shape);

        let in_chunks = input.par_iter_lane_chunks::<N>(shape, min_axis);
        let in_rem = in_chunks.remainder();
        let out_chunks = output.par_iter_lane_chunks_mut::<N>(shape, min_axis);
        let out_rem = out_chunks.remainder();

        out_chunks.zip(in_chunks).for_each(|(mut o, i)| {
            o.iter_mut().zip(i.iter()).for_each(|(o, i)| {
                o.into_iter().zip(i).for_each(|(o, i)| {
                    *o = i.clone();
                });
            });
        });
        out_rem.zip(in_rem).for_each(|(mut o, i)| {
            o.iter_mut().zip(i.iter()).for_each(|(o, i)| *o = i.clone());
        });

        let mut sub_shape = shape.to_owned();

        let shape_levels = (0..level)
            .map(|_| {
                let next_shape = sub_shape.clone();
                for &ax in &axes {
                    if sub_shape[ax] > 1 {
                        sub_shape[ax] = sub_shape[ax].div_ceil(2);
                    }
                }
                next_shape
            })
            .collect::<Vec<_>>();

        for lvl in (0..level).rev() {
            let sub_shape = &shape_levels[lvl];
            for &ax in &axes {
                let n_ax = sub_shape[ax];

                let n_d = n_ax / 2;
                let n_s = n_ax - n_d;

                if n_s > 1 {
                    let chunks = output.par_iter_lane_chunks_sub_mut::<N>(shape, sub_shape, ax);
                    let rem = chunks.remainder();
                    if chunks.len() > 0 {
                        chunks.for_each_init(
                            || {
                                let s = core::array::from_fn(|_| avec![T::zero(); n_s]);
                                let d = core::array::from_fn(|_| avec![T::zero(); n_d]);
                                (s, d)
                            },
                            |(s, d), mut chunk| {
                                chunk.split(
                                    &mut avecs_to_mut_slices(s),
                                    &mut avecs_to_mut_slices(d),
                                );
                                s.iter_mut().zip(d.iter_mut()).for_each(|(s, d)| {
                                    func(s, d);
                                });
                                chunk.interleave(&avecs_to_slices(s), &avecs_to_slices(d));
                            },
                        );
                    }

                    rem.for_each_init(
                        || {
                            let s = avec![T::zero(); n_s];
                            let d = avec![T::zero(); n_d];
                            (s, d)
                        },
                        |(s, d), mut slc| {
                            slc.split(s, d);
                            func(s, d);
                            slc.interleave(s, d);
                        },
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use super::*;
    use rstest::rstest;

    #[rstest]
    fn test_roundtrip(
        #[values(16, 17)] n: usize,
        #[values(1, 2, 3, 4)] dim: usize,
        #[values(1, 2, 3)] level: usize,
    ) {
        let shape = vec![n; dim];

        let axes = (0..dim).collect_vec();
        let n_total = shape.iter().product();
        let v1 = (0..n_total).map(|i| i as f64).collect_vec();
        let mut v2 = vec![0.0; n_total];
        let mut v3 = vec![0.0; n_total];

        let v1 = v1.as_slice();
        let v2 = v2.as_mut_slice();
        let v3 = v3.as_mut_slice();

        general_nd_forward_multilevel(|_, _| {}, v1, v2, &shape, &axes, level);

        if level == 1 {
            let mut v2_ref = v1.to_owned();

            for ax in axes.iter().cloned() {
                let ns = (shape[ax] + 1) / 2;
                let nd = shape[ax] / 2;
                let mut work_e = vec![0.0; ns];
                let mut work_o = vec![0.0; nd];
                for mut slc in v2_ref.iter_lanes_mut(&shape, ax) {
                    slc.deinterleave(&mut work_e, &mut work_o);
                    slc.stack(&work_e, &work_o);
                }
            }
            assert_eq!(v2, v2_ref);
        }
        general_nd_inverse_multilevel(|_, _| {}, v2, v3, &shape, &axes, level);
        assert_eq!(v1, v3);
    }

    #[cfg(feature = "rayon")]
    #[rstest]
    fn test_par_roundtrip(
        #[values(16, 17)] n: usize,
        #[values(1, 2, 3, 4)] dim: usize,
        #[values(1, 2, 3)] level: usize,
    ) {
        let shape = vec![n; dim];

        let axes = (0..dim).collect_vec();
        let n_total = shape.iter().product();
        let v1 = (0..n_total).map(|i| i as f64).collect_vec();
        let mut v2 = vec![0.0; n_total];
        let mut v3 = vec![0.0; n_total];

        let v1 = v1.as_slice();
        let v2 = v2.as_mut_slice();
        let v3 = v3.as_mut_slice();

        parallel::general_nd_forward_multilevel(|_, _| {}, v1, v2, &shape, &axes, level);

        if level == 1 {
            let mut v2_ref = v1.to_owned();

            for ax in axes.iter().cloned() {
                let ns = (shape[ax] + 1) / 2;
                let nd = shape[ax] / 2;
                let mut work_e = vec![0.0; ns];
                let mut work_o = vec![0.0; nd];
                for mut slc in v2_ref.iter_lanes_mut(&shape, ax) {
                    slc.deinterleave(&mut work_e, &mut work_o);
                    slc.stack(&work_e, &work_o);
                }
            }
            assert_eq!(v2, v2_ref);
        }
        parallel::general_nd_inverse_multilevel(|_, _| {}, v2, v3, &shape, &axes, level);
        assert_eq!(v1, v3);
    }
}
