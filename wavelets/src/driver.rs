use num_traits::Zero;
use std::collections::HashSet;

use crate::Wavelets;
use crate::boundarys::BoundaryExtension;
use crate::boundarys::LiftedAdjointBoundary;
use crate::iter::LanesIterator;

use crate::utils::{
    deinterleave, deinterleave_strided, deinterleave_strided_chunk, stack_to_strided,
    stack_to_strided_chunk, stride_from_shape,
};
use crate::utils::{
    interleave, interleave_strided, interleave_strided_chunk, split_strided, split_strided_chunk,
};
use crate::{ChunkWidth, Transformable};

use wavelets_macros::generate_wavelet_match_arms;

pub struct Wavelet<T, BC, const N: usize>
where
    T: Transformable + Zero + ChunkWidth<T, N>,
    T::ScalarType: From<f64>,
    BC: BoundaryExtension + LiftedAdjointBoundary,
{
    lwt_forward: fn(&mut [T], &mut [T], &BC),
    lwt_inverse: fn(&mut [T], &mut [T], &BC),
    lwt_adj_forward: fn(&mut [T], &mut [T], &BC),
    lwt_adj_inverse: fn(&mut [T], &mut [T], &BC),
    bc: BC,
}

impl<T, BC, const N: usize> Wavelet<T, BC, N>
where
    T: Transformable + Zero + ChunkWidth<T, N>,
    T::ScalarType: From<f64>,
    BC: BoundaryExtension + LiftedAdjointBoundary + Sync,
{
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
        Self {
            lwt_forward,
            lwt_inverse,
            lwt_adj_forward,
            lwt_adj_inverse,
            bc,
        }
    }

    pub fn forward_1d(&self, input: &[T], s: &mut [T], d: &mut [T]) {
        deinterleave(input, s, d);
        (self.lwt_forward)(s, d, &self.bc);
    }

    pub fn inverse_1d(&self, s: &[T], d: &[T], output: &mut [T]) {
        let (mut s, mut d) = (s.to_owned(), d.to_owned());
        (self.lwt_inverse)(&mut s, &mut d, &self.bc);
        interleave(&s, &d, output);
    }

    pub fn adj_forward_1d(&self, s: &[T], d: &[T], output: &mut [T]) {
        let (mut s, mut d) = (s.to_owned(), d.to_owned());
        (self.lwt_adj_forward)(&mut s, &mut d, &self.bc);
        interleave(&s, &d, output);
    }

    pub fn adj_inverse_1d(&self, input: &[T], s: &mut [T], d: &mut [T]) {
        deinterleave(input, s, d);
        (self.lwt_adj_inverse)(s, d, &self.bc);
    }

    pub fn forward_nd(&self, input: &[T], output: &mut [T], shape: &[usize], axes: &[usize]) {
        let stride = stride_from_shape(shape);
        let axes = HashSet::from_iter(axes.iter().cloned());
        self.forward_strided_multilevel_nd(input, output, shape, &stride, &stride, &axes, 1);
    }

    pub fn inverse_nd(&self, input: &[T], output: &mut [T], shape: &[usize], axes: &[usize]) {
        let stride = stride_from_shape(shape);
        let axes = HashSet::from_iter(axes.iter().cloned());
        self.inverse_strided_multilevel_nd(input, output, shape, &stride, &stride, &axes, 1);
    }

    pub fn adj_forward_nd(&self, input: &[T], output: &mut [T], shape: &[usize], axes: &[usize]) {
        let stride = stride_from_shape(shape);
        let axes = HashSet::from_iter(axes.iter().cloned());
        self.adj_forward_strided_multilevel_nd(input, output, shape, &stride, &stride, &axes, 1);
    }

    pub fn adj_inverse_nd(&self, input: &[T], output: &mut [T], shape: &[usize], axes: &[usize]) {
        let stride = stride_from_shape(shape);
        let axes = HashSet::from_iter(axes.iter().cloned());
        self.adj_inverse_strided_multilevel_nd(input, output, shape, &stride, &stride, &axes, 1);
    }

    pub fn forward_strided_multilevel_nd(
        &self,
        input: &[T],
        output: &mut [T],
        shape: &[usize],
        stride_in: &[usize],
        stride_out: &[usize],
        axes: &HashSet<usize>,
        level: usize,
    ) {
        general_nd_forward_multilevel(
            |s, d| (self.lwt_forward)(s, d, &self.bc),
            input,
            output,
            shape,
            stride_in,
            stride_out,
            axes,
            level,
        );
    }

    pub fn inverse_strided_multilevel_nd(
        &self,
        input: &[T],
        output: &mut [T],
        shape: &[usize],
        stride_in: &[usize],
        stride_out: &[usize],
        axes: &HashSet<usize>,
        level: usize,
    ) {
        general_nd_inverse_multilevel(
            |s, d| (self.lwt_inverse)(s, d, &self.bc),
            input,
            output,
            shape,
            stride_in,
            stride_out,
            axes,
            level,
        );
    }

    pub fn adj_forward_strided_multilevel_nd(
        &self,
        input: &[T],
        output: &mut [T],
        shape: &[usize],
        stride_in: &[usize],
        stride_out: &[usize],
        axes: &HashSet<usize>,
        level: usize,
    ) {
        general_nd_inverse_multilevel(
            |s, d| (self.lwt_adj_forward)(s, d, &self.bc),
            input,
            output,
            shape,
            stride_in,
            stride_out,
            axes,
            level,
        );
    }

    pub fn adj_inverse_strided_multilevel_nd(
        &self,
        input: &[T],
        output: &mut [T],
        shape: &[usize],
        stride_in: &[usize],
        stride_out: &[usize],
        axes: &HashSet<usize>,
        level: usize,
    ) {
        general_nd_forward_multilevel(
            |s, d| (self.lwt_adj_inverse)(s, d, &self.bc),
            input,
            output,
            shape,
            stride_in,
            stride_out,
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
    stride_in: &[usize],
    stride_out: &[usize],
    axes: &HashSet<usize>,
    level: usize,
) where
    F: Fn(&mut [T], &mut [T]),
    L: LanesIterator<Item = T> + ?Sized,
    T: Transformable + Zero + ChunkWidth<T, N>,
    T::ScalarType: From<f64>,
{
    let ndim = shape.len();
    assert_eq!(ndim, stride_in.len());
    assert_eq!(ndim, stride_out.len());
    assert!(axes.iter().all(|i| *i < ndim));
    // note that axes is a HashSet, so they are gauranteed to be different axes.

    let mut first = true;

    let mut shape = shape.to_owned();
    for _level in 0..level {
        for &ax in axes {
            let n_ax = shape[ax];

            let n_d = n_ax / 2;
            let n_s = n_ax - n_d;

            match first {
                true => {
                    let in_chunks = input.iter_lane_chunks_strided::<N>(&shape, stride_in, ax);
                    let in_rem = in_chunks.remainder();
                    let out_chunks =
                        output.iter_lane_chunks_mut_strided::<N>(&shape, stride_out, ax);
                    let out_rem = out_chunks.remainder();

                    if in_chunks.len() > 0 {
                        let mut s = vec![T::zero(); n_s * N];
                        let mut d = vec![T::zero(); n_d * N];
                        in_chunks
                            .zip(out_chunks)
                            .for_each(|(in_chunk, mut out_chunk)| {
                                // copy (and deinterleave) strided chunks into the local storage
                                deinterleave_strided_chunk(&in_chunk, &mut s, &mut d);
                                s.chunks_exact_mut(n_s)
                                    .zip(d.chunks_exact_mut(n_d))
                                    .for_each(|(s, d)| {
                                        func(s, d);
                                    });
                                // clone local storage to the output
                                stack_to_strided_chunk(&s, &d, &mut out_chunk);
                            });
                    }
                    if in_rem.len() > 0 {
                        let mut s = vec![T::zero(); n_s];
                        let mut d = vec![T::zero(); n_d];
                        in_rem.zip(out_rem).for_each(|(in_slice, mut out_slice)| {
                            // copy strided slice into local dimension storage
                            deinterleave_strided(&in_slice, &mut s, &mut d);
                            func(&mut s, &mut d);
                            // copy local back to output strided slice
                            stack_to_strided(&s, &d, &mut out_slice);
                        });
                    }

                    first = false;
                }
                false => {
                    let chunks = output.iter_lane_chunks_mut_strided::<N>(&shape, stride_out, ax);
                    let rem = chunks.remainder();

                    if chunks.len() > 0 {
                        let mut s = vec![T::zero(); n_s * N];
                        let mut d = vec![T::zero(); n_d * N];
                        chunks.for_each(|mut chunk| {
                            // copy (and deinterleave) strided chunks into the local storage
                            deinterleave_strided_chunk(&chunk, &mut s, &mut d);
                            s.chunks_exact_mut(n_s)
                                .zip(d.chunks_exact_mut(n_d))
                                .for_each(|(s, d)| {
                                    func(s, d);
                                });
                            // clone local storage to the output
                            stack_to_strided_chunk(&s, &d, &mut chunk);
                        });
                    }
                    if rem.len() > 0 {
                        let mut s = vec![T::zero(); n_s];
                        let mut d = vec![T::zero(); n_d];
                        rem.for_each(|mut slc| {
                            // copy strided slice into local dimension storage
                            deinterleave_strided(&slc, &mut s, &mut d);
                            func(&mut s, &mut d);
                            // copy local back to output strided slice
                            stack_to_strided(&s, &d, &mut slc);
                        });
                    }
                }
            }
        }

        // shrink shape for each axis we used.
        for &ax in axes {
            shape[ax] = (shape[ax] + 1) / 2;
        }
    }
}

fn general_nd_inverse_multilevel<F, T, L, const N: usize>(
    func: F,
    input: &L,
    output: &mut L,
    shape: &[usize],
    stride_in: &[usize],
    stride_out: &[usize],
    axes: &HashSet<usize>,
    level: usize,
) where
    F: Fn(&mut [T], &mut [T]),
    L: LanesIterator<Item = T> + ?Sized,
    T: Transformable + Zero + ChunkWidth<T, N>,
    T::ScalarType: From<f64>,
{
    let ndim = shape.len();
    assert_eq!(ndim, stride_in.len());
    assert_eq!(ndim, stride_out.len());
    assert!(axes.iter().all(|i| *i < ndim));
    // note that axes is a HashSet, so they are gauranteed to be different axes.

    // copy input into the output
    let (mstride_in_axis, _) = stride_out
        .iter()
        .enumerate()
        .reduce(|acc, v| if v.1 < acc.1 { v } else { acc })
        .expect("dimensions should be greater thann 0.");

    let out_chunks = output.iter_lane_chunks_mut_strided::<N>(shape, stride_out, mstride_in_axis);
    let out_rem = out_chunks.remainder();
    let in_chunks = input.iter_lane_chunks_strided::<N>(shape, stride_in, mstride_in_axis);
    let in_rem = in_chunks.remainder();

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

    let mut shape = shape.to_owned();

    let shape_levels = (0..level)
        .map(|_| {
            let next_shape = shape.clone();
            for &ax in axes {
                shape[ax] = (shape[ax] + 1) / 2;
            }
            next_shape
        })
        .collect::<Vec<_>>();

    for lvl in (0..level).rev() {
        let shape = &shape_levels[lvl];
        for &ax in axes {
            let n_ax = shape[ax];

            let n_d = n_ax / 2;
            let n_s = n_ax - n_d;

            let chunks = output.iter_lane_chunks_mut_strided::<N>(shape, stride_out, ax);
            let rem = chunks.remainder();

            if chunks.len() > 0 {
                let mut s = vec![T::zero(); n_s * N];
                let mut d = vec![T::zero(); n_d * N];
                chunks.for_each(|mut chunk| {
                    split_strided_chunk(&chunk, &mut s, &mut d);
                    s.chunks_exact_mut(n_s)
                        .zip(d.chunks_exact_mut(n_d))
                        .for_each(|(s, d)| {
                            func(s, d);
                        });
                    interleave_strided_chunk(&s, &d, &mut chunk);
                });
            }
            if rem.len() > 0 {
                let mut s = vec![T::zero(); n_s];
                let mut d = vec![T::zero(); n_d];
                rem.for_each(|mut slc| {
                    split_strided(&slc, &mut s, &mut d);
                    func(&mut s, &mut d);
                    interleave_strided(&s, &d, &mut slc);
                })
            }
        }
    }
}

#[cfg(feature = "rayon")]
pub mod parallel {
    use super::*;

    use crate::iter::parallel::ParallelLanesIterator;
    use rayon::iter::{IndexedParallelIterator, ParallelIterator};

    pub struct Wavelet<T, BC, const N: usize>
    where
        T: Transformable + Zero + ChunkWidth<T, N> + Sync + Send,
        T::ScalarType: From<f64>,
        BC: BoundaryExtension + LiftedAdjointBoundary + Sync,
    {
        lwt_forward: fn(&mut [T], &mut [T], &BC),
        lwt_inverse: fn(&mut [T], &mut [T], &BC),
        lwt_adj_forward: fn(&mut [T], &mut [T], &BC),
        lwt_adj_inverse: fn(&mut [T], &mut [T], &BC),
        bc: BC,
    }

    impl<T, BC, const N: usize> Wavelet<T, BC, N>
    where
        T: Transformable + Zero + ChunkWidth<T, N> + Sync + Send,
        T::ScalarType: From<f64>,
        BC: BoundaryExtension + LiftedAdjointBoundary + Sync,
    {
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
            Self {
                lwt_forward,
                lwt_inverse,
                lwt_adj_forward,
                lwt_adj_inverse,
                bc,
            }
        }

        pub fn forward_1d(&self, input: &[T], s: &mut [T], d: &mut [T]) {
            deinterleave(input, s, d);
            (self.lwt_forward)(s, d, &self.bc);
        }

        pub fn inverse_1d(&self, s: &[T], d: &[T], output: &mut [T]) {
            let (mut s, mut d) = (s.to_owned(), d.to_owned());
            (self.lwt_inverse)(&mut s, &mut d, &self.bc);
            interleave(&s, &d, output);
        }

        pub fn adj_forward_1d(&self, s: &[T], d: &[T], output: &mut [T]) {
            let (mut s, mut d) = (s.to_owned(), d.to_owned());
            (self.lwt_adj_forward)(&mut s, &mut d, &self.bc);
            interleave(&s, &d, output);
        }

        pub fn adj_inverse_1d(&self, input: &[T], s: &mut [T], d: &mut [T]) {
            deinterleave(input, s, d);
            (self.lwt_adj_inverse)(s, d, &self.bc);
        }

        pub fn forward_nd(&self, input: &[T], output: &mut [T], shape: &[usize], axes: &[usize]) {
            let stride = stride_from_shape(shape);
            let axes = HashSet::from_iter(axes.iter().cloned());
            self.forward_strided_multilevel_nd(input, output, shape, &stride, &stride, &axes, 1);
        }

        pub fn inverse_nd(&self, input: &[T], output: &mut [T], shape: &[usize], axes: &[usize]) {
            let stride = stride_from_shape(shape);
            let axes = HashSet::from_iter(axes.iter().cloned());
            self.inverse_strided_multilevel_nd(input, output, shape, &stride, &stride, &axes, 1);
        }

        pub fn adj_forward_nd(
            &self,
            input: &[T],
            output: &mut [T],
            shape: &[usize],
            axes: &[usize],
        ) {
            let stride = stride_from_shape(shape);
            let axes = HashSet::from_iter(axes.iter().cloned());
            self.adj_forward_strided_multilevel_nd(
                input, output, shape, &stride, &stride, &axes, 1,
            );
        }

        pub fn adj_inverse_nd(
            &self,
            input: &[T],
            output: &mut [T],
            shape: &[usize],
            axes: &[usize],
        ) {
            let stride = stride_from_shape(shape);
            let axes = HashSet::from_iter(axes.iter().cloned());
            self.adj_inverse_strided_multilevel_nd(
                input, output, shape, &stride, &stride, &axes, 1,
            );
        }

        pub fn forward_strided_multilevel_nd(
            &self,
            input: &[T],
            output: &mut [T],
            shape: &[usize],
            stride_in: &[usize],
            stride_out: &[usize],
            axes: &HashSet<usize>,
            level: usize,
        ) {
            general_nd_forward_multilevel(
                |s, d| (self.lwt_forward)(s, d, &self.bc),
                input,
                output,
                shape,
                stride_in,
                stride_out,
                axes,
                level,
            );
        }

        pub fn inverse_strided_multilevel_nd(
            &self,
            input: &[T],
            output: &mut [T],
            shape: &[usize],
            stride_in: &[usize],
            stride_out: &[usize],
            axes: &HashSet<usize>,
            level: usize,
        ) {
            general_nd_inverse_multilevel(
                |s, d| (self.lwt_inverse)(s, d, &self.bc),
                input,
                output,
                shape,
                stride_in,
                stride_out,
                axes,
                level,
            );
        }

        pub fn adj_forward_strided_multilevel_nd(
            &self,
            input: &[T],
            output: &mut [T],
            shape: &[usize],
            stride_in: &[usize],
            stride_out: &[usize],
            axes: &HashSet<usize>,
            level: usize,
        ) {
            general_nd_inverse_multilevel(
                |s, d| (self.lwt_adj_forward)(s, d, &self.bc),
                input,
                output,
                shape,
                stride_in,
                stride_out,
                axes,
                level,
            );
        }

        pub fn adj_inverse_strided_multilevel_nd(
            &self,
            input: &[T],
            output: &mut [T],
            shape: &[usize],
            stride_in: &[usize],
            stride_out: &[usize],
            axes: &HashSet<usize>,
            level: usize,
        ) {
            general_nd_forward_multilevel(
                |s, d| (self.lwt_adj_inverse)(s, d, &self.bc),
                input,
                output,
                shape,
                stride_in,
                stride_out,
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
        stride_in: &[usize],
        stride_out: &[usize],
        axes: &HashSet<usize>,
        level: usize,
    ) where
        F: Fn(&mut [T], &mut [T]) + Sync,
        L: ParallelLanesIterator<Item = T> + ?Sized,
        T: Transformable + Zero + ChunkWidth<T, N> + Sync + Send,
        T::ScalarType: From<f64>,
    {
        let ndim = shape.len();
        assert_eq!(ndim, stride_in.len());
        assert_eq!(ndim, stride_out.len());
        assert!(axes.iter().all(|i| *i < ndim));
        // note that axes is a HashSet, so they are gauranteed to be different axes.

        let mut first = true;

        let mut shape = shape.to_owned();
        for _level in 0..level {
            for &ax in axes {
                let n_ax = shape[ax];

                let n_d = n_ax / 2;
                let n_s = n_ax - n_d;

                match first {
                    true => {
                        let in_chunks = input.iter_lane_chunks_strided::<N>(&shape, stride_in, ax);
                        let in_rem = in_chunks.remainder();
                        let out_chunks =
                            output.iter_lane_chunks_mut_strided::<N>(&shape, stride_out, ax);
                        let out_rem = out_chunks.remainder();

                        in_chunks.zip(out_chunks).for_each_init(
                            || {
                                let s = vec![T::zero(); n_s * N];
                                let d = vec![T::zero(); n_d * N];
                                (s, d)
                            },
                            |(s, d), (in_chunk, mut out_chunk)| {
                                // copy (and deinterleave) strided chunks into the local storage
                                deinterleave_strided_chunk(&in_chunk, s, d);
                                s.chunks_exact_mut(n_s)
                                    .zip(d.chunks_exact_mut(n_d))
                                    .for_each(|(s, d)| {
                                        func(s, d);
                                    });
                                // clone local storage to the output
                                stack_to_strided_chunk(s, d, &mut out_chunk);
                            },
                        );
                        in_rem.zip(out_rem).for_each_init(
                            || {
                                let s = vec![T::zero(); n_s];
                                let d = vec![T::zero(); n_d];
                                (s, d)
                            },
                            |(s, d), (in_slice, mut out_slice)| {
                                // copy strided slice into local dimension storage
                                deinterleave_strided(&in_slice, s, d);
                                func(s, d);
                                // copy local back to output strided slice
                                stack_to_strided(s, d, &mut out_slice);
                            },
                        );
                        first = false;
                    }
                    false => {
                        let chunks =
                            output.iter_lane_chunks_mut_strided::<N>(&shape, stride_out, ax);
                        let rem = chunks.remainder();

                        chunks.for_each_init(
                            || {
                                let s = vec![T::zero(); n_s * N];
                                let d = vec![T::zero(); n_d * N];
                                (s, d)
                            },
                            |(s, d), mut chunk| {
                                // copy (and deinterleave) strided chunks into the local storage
                                deinterleave_strided_chunk(&chunk, s, d);
                                s.chunks_exact_mut(n_s)
                                    .zip(d.chunks_exact_mut(n_d))
                                    .for_each(|(s, d)| {
                                        func(s, d);
                                    });
                                // clone local storage to the output
                                stack_to_strided_chunk(s, d, &mut chunk);
                            },
                        );
                        rem.for_each_init(
                            || {
                                let s = vec![T::zero(); n_s];
                                let d = vec![T::zero(); n_d];
                                (s, d)
                            },
                            |(s, d), mut slc| {
                                // copy strided slice into local dimension storage
                                deinterleave_strided(&slc, s, d);
                                func(s, d);
                                // copy local back to output strided slice
                                stack_to_strided(s, d, &mut slc);
                            },
                        );
                    }
                }
            }
            // shrink shape for each axis we used.
            for &ax in axes {
                shape[ax] = (shape[ax] + 1) / 2;
            }
        }
    }

    fn general_nd_inverse_multilevel<F, T, L, const N: usize>(
        func: F,
        input: &L,
        output: &mut L,
        shape: &[usize],
        stride_in: &[usize],
        stride_out: &[usize],
        axes: &HashSet<usize>,
        level: usize,
    ) where
        F: Fn(&mut [T], &mut [T]) + Sync,
        L: ParallelLanesIterator<Item = T> + ?Sized,
        T: Transformable + Zero + ChunkWidth<T, N> + Send + Sync,
        T::ScalarType: From<f64>,
    {
        let ndim = shape.len();
        assert_eq!(ndim, stride_in.len());
        assert_eq!(ndim, stride_out.len());
        assert!(axes.iter().all(|i| *i < ndim));
        // note that axes is a HashSet, so they are gauranteed to be different axes.

        // copy input into the output
        let (mstride_in_axis, _) = stride_out
            .iter()
            .enumerate()
            .reduce(|acc, v| if v.1 < acc.1 { v } else { acc })
            .expect("dimensions should be greater than 0.");

        let in_chunks = input.iter_lane_chunks_strided::<N>(shape, stride_in, mstride_in_axis);
        let in_rem = in_chunks.remainder();
        let out_chunks =
            output.iter_lane_chunks_mut_strided::<N>(shape, stride_out, mstride_in_axis);
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

        let mut shape = shape.to_owned();

        let shape_levels = (0..level)
            .map(|_| {
                let next_shape = shape.clone();
                for &ax in axes {
                    shape[ax] = (shape[ax] + 1) / 2;
                }
                next_shape
            })
            .collect::<Vec<_>>();

        for lvl in (0..level).rev() {
            let shape = &shape_levels[lvl];
            for &ax in axes {
                let n_ax = shape[ax];

                let n_d = n_ax / 2;
                let n_s = n_ax - n_d;

                let chunks = output.iter_lane_chunks_mut_strided::<N>(shape, stride_out, ax);
                let rem = chunks.remainder();
                if chunks.len() > 0 {
                    chunks.for_each_init(
                        || {
                            let s = vec![T::zero(); n_s * N];
                            let d = vec![T::zero(); n_d * N];
                            (s, d)
                        },
                        |(s, d), mut chunk| {
                            split_strided_chunk(&chunk, s, d);
                            s.chunks_exact_mut(n_s)
                                .zip(d.chunks_exact_mut(n_d))
                                .for_each(|(s, d)| {
                                    func(s, d);
                                });
                            interleave_strided_chunk(s, d, &mut chunk);
                        },
                    );
                }

                rem.for_each_init(
                    || {
                        let s = vec![T::zero(); n_s];
                        let d = vec![T::zero(); n_d];
                        (s, d)
                    },
                    |(s, d), mut slc| {
                        split_strided(&slc, s, d);
                        func(s, d);
                        interleave_strided(s, d, &mut slc);
                    },
                );
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
    fn test_roundtrip(#[values(5, 6)] n: usize, #[values(2)] dim: usize) {
        let shape = vec![n; dim];
        let stride = stride_from_shape(&shape);

        let axes = HashSet::from_iter(0..dim);
        let n_total = shape.iter().product();
        let v1 = (0..n_total).map(|i| i as f64).collect_vec();
        let mut v2_ref = (0..n_total).map(|i| i as f64).collect_vec();
        let mut v2 = vec![0.0; n_total];
        let mut v3 = vec![0.0; n_total];

        let v1 = v1.as_slice();
        let v2 = v2.as_mut_slice();
        let v3 = v3.as_mut_slice();

        for ax in axes.iter().cloned() {
            let ns = (shape[ax] + 1) / 2;
            let nd = shape[ax] / 2;
            let mut work_e = vec![0.0; ns];
            let mut work_o = vec![0.0; nd];
            for mut slc in v2_ref.iter_lanes_mut(&shape, ax) {
                deinterleave_strided(&slc, &mut work_e, &mut work_o);
                stack_to_strided(&work_e, &work_o, &mut slc);
            }
        }

        general_nd_forward_multilevel(|_, _| {}, v1, v2, &shape, &stride, &stride, &axes, 1);

        assert_eq!(v2, v2_ref);
        general_nd_inverse_multilevel(|_, _| {}, v2, v3, &shape, &stride, &stride, &axes, 1);
        assert_eq!(v1, v3);
    }
}
