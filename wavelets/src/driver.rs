use num_traits::Zero;
use std::collections::HashSet;

use crate::Wavelets;
use crate::boundarys::BoundaryExtension;
use crate::boundarys::LiftedAdjointBoundary;
use crate::utils::{
    deinterleave, deinterleave_strided, deinterleave_strided_chunk, stack_to_strided,
    stack_to_strided_chunk, stride_from_shape,
};
use crate::utils::{
    interleave, interleave_strided, interleave_strided_chunk, split_strided, split_strided_chunk,
};
use crate::{ChunkWidth, Transformable};

pub struct Wavelet<T, BC, const N: usize>
where
    T: Transformable + Zero + ChunkWidth<T, N>,
    T::ScalarType: From<f64>,
    BC: BoundaryExtension + LiftedAdjointBoundary + std::marker::Sync,
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
    BC: BoundaryExtension + LiftedAdjointBoundary + std::marker::Sync,
{
    pub fn new(wvlt: Wavelets, bc: BC) -> Self {
        use crate::lwt::bior::*;
        use crate::lwt::daubechies::*;
        let lwt_forward: fn(&mut [T], &mut [T], &BC) = match wvlt {
            Wavelets::Daubechies1 => Daubechies1::forward,
            Wavelets::Daubechies2 => Daubechies2::forward,
            Wavelets::Daubechies3 => Daubechies3::forward,
            Wavelets::Daubechies4 => Daubechies4::forward,
            Wavelets::Daubechies5 => Daubechies5::forward,
            Wavelets::Daubechies6 => Daubechies6::forward,
            _ => Daubechies2::forward,
        };
        let lwt_inverse: fn(&mut [T], &mut [T], &BC) = match wvlt {
            Wavelets::Daubechies1 => Daubechies1::inverse,
            Wavelets::Daubechies2 => Daubechies2::inverse,
            Wavelets::Daubechies3 => Daubechies3::inverse,
            Wavelets::Daubechies4 => Daubechies4::inverse,
            Wavelets::Daubechies5 => Daubechies5::inverse,
            Wavelets::Daubechies6 => Daubechies6::inverse,
            _ => Daubechies2::inverse,
        };
        let lwt_adj_forward: fn(&mut [T], &mut [T], &BC) = match wvlt {
            Wavelets::Daubechies1 => Daubechies1::adjoint_forward,
            _ => Daubechies2::adjoint_forward,
        };
        let lwt_adj_inverse: fn(&mut [T], &mut [T], &BC) = match wvlt {
            Wavelets::Daubechies1 => Daubechies1::adjoint_inverse,
            _ => Daubechies2::adjoint_inverse,
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

fn general_nd_forward_multilevel<F, T, const N: usize>(
    func: F,
    input: &[T],
    output: &mut [T],
    shape: &[usize],
    stride_in: &[usize],
    stride_out: &[usize],
    axes: &HashSet<usize>,
    level: usize,
) where
    F: Fn(&mut [T], &mut [T]) + std::marker::Sync,
    T: Transformable + Zero + ChunkWidth<T, N>,
    T::ScalarType: From<f64>,
{
    #[cfg(not(feature = "rayon"))]
    use crate::iter::slice::LanesIterator;
    #[cfg(feature = "rayon")]
    use crate::iter::slice::parallel::{
        IndexedParallelIterator, ParallelIterator, ParallelLanesIterator,
    };

    let ndim = shape.len();
    assert_eq!(ndim, stride_in.len());
    assert_eq!(ndim, stride_out.len());
    assert!(axes.iter().all(|i| *i < ndim));
    // note that axes is a HashSet, so they are gauranteed to be different axes.

    let max_in_offset: usize = shape
        .iter()
        .zip(stride_in)
        .map(|(n, step)| (n - 1) * step)
        .sum();
    assert!(max_in_offset < input.len());
    let max_out_offset: usize = shape
        .iter()
        .zip(stride_out)
        .map(|(n, step)| (n - 1) * step)
        .sum();
    assert!(max_out_offset < output.len());

    let mut first = true;

    let mut shape = shape.to_owned();
    for _level in 0..level {
        for &ax in axes {
            let n_ax = shape[ax];

            let n_d = n_ax / 2;
            let n_s = n_ax - n_d;

            let (input, stride_in) = match first {
                false => {
                    // create a clone of the output to read from
                    // we are not reading from and writing to output during the same function
                    // it is always copied to a temporary array in between.
                    // so there is no aliasing.
                    (
                        unsafe { std::slice::from_raw_parts(output.as_ptr(), output.len()) },
                        stride_out,
                    )
                }
                true => {
                    first = false;
                    (input, stride_in)
                }
            };

            #[cfg(feature = "rayon")]
            {
                let (iter_in_chunks, iter_in_rem) =
                    input.par_iter_lane_chunks_strided::<N>(&shape, stride_in, ax);
                let (iter_out_chunks, iter_out_rem) =
                    output.par_iter_lane_chunks_mut_strided::<N>(&shape, stride_out, ax);

                if iter_in_chunks.len() > 0 {
                    iter_in_chunks.zip(iter_out_chunks).for_each_init(
                        || {
                            let s = vec![T::zero(); n_s * N];
                            let d = vec![T::zero(); n_d * N];
                            (s, d)
                        },
                        |(s, d), (in_chunk, out_chunk)| {
                            // copy (and deinterleave) strided chunks into the local storage
                            deinterleave_strided_chunk(in_chunk, s, d);
                            s.chunks_exact_mut(n_s)
                                .zip(d.chunks_exact_mut(n_d))
                                .for_each(|(s, d)| {
                                    func(s, d);
                                });
                            // clone local storage to the output
                            stack_to_strided_chunk(s, d, out_chunk);
                        },
                    );
                }
                iter_in_rem.zip(iter_out_rem).for_each_init(
                    || {
                        let s = vec![T::zero(); n_s];
                        let d = vec![T::zero(); n_d];
                        (s, d)
                    },
                    |(s, d), (in_slice, out_slice)| {
                        // copy strided slice into local dimension storage
                        deinterleave_strided(in_slice, s, d);
                        func(s, d);
                        // copy local back to output strided slice
                        stack_to_strided(s, d, out_slice);
                    },
                );
            }

            #[cfg(not(feature = "rayon"))]
            {
                let (iter_in_chunks, iter_in_rem) =
                    input.iter_lane_chunks_strided::<N>(&shape, stride_in, ax);
                let (iter_out_chunks, iter_out_rem) =
                    output.iter_lane_chunks_mut_strided::<N>(&shape, stride_out, ax);

                if iter_in_chunks.len() > 0 {
                    let mut s = vec![T::zero(); n_s * N];
                    let mut d = vec![T::zero(); n_d * N];
                    iter_in_chunks
                        .zip(iter_out_chunks)
                        .for_each(|(in_chunk, out_chunk)| {
                            // copy (and deinterleave) strided chunks into the local storage
                            deinterleave_strided_chunk(in_chunk, &mut s, &mut d);
                            s.chunks_exact_mut(n_s)
                                .zip(d.chunks_exact_mut(n_d))
                                .for_each(|(s, d)| {
                                    func(s, d);
                                });
                            // clone local storage to the output
                            stack_to_strided_chunk(&s, &d, out_chunk);
                        });
                }
                if iter_in_rem.len() > 0 {
                    let mut s = vec![T::zero(); n_s];
                    let mut d = vec![T::zero(); n_d];
                    iter_in_rem
                        .zip(iter_out_rem)
                        .for_each(|(in_slice, out_slice)| {
                            // copy strided slice into local dimension storage
                            deinterleave_strided(in_slice, &mut s, &mut d);
                            func(&mut s, &mut d);
                            // copy local back to output strided slice
                            stack_to_strided(&s, &d, out_slice);
                        });
                }
            }
        }

        // shrink shape for each axis we used.
        for &ax in axes {
            shape[ax] = (shape[ax] + 1) / 2;
        }
    }
}

fn general_nd_inverse_multilevel<F, T, const N: usize>(
    func: F,
    input: &[T],
    output: &mut [T],
    shape: &[usize],
    stride_in: &[usize],
    stride_out: &[usize],
    axes: &HashSet<usize>,
    level: usize,
) where
    F: Fn(&mut [T], &mut [T]) + std::marker::Sync,
    T: Transformable + Zero + ChunkWidth<T, N>,
    T::ScalarType: From<f64>,
{
    #[cfg(not(feature = "rayon"))]
    use crate::iter::slice::LanesIterator;
    #[cfg(feature = "rayon")]
    use crate::iter::slice::parallel::{
        IndexedParallelIterator, ParallelIterator, ParallelLanesIterator,
    };

    let ndim = shape.len();
    assert_eq!(ndim, stride_in.len());
    assert_eq!(ndim, stride_out.len());
    assert!(axes.iter().all(|i| *i < ndim));
    // note that axes is a HashSet, so they are gauranteed to be different axes.

    let max_in_offset: usize = shape
        .iter()
        .zip(stride_in)
        .map(|(n, step)| (n - 1) * step)
        .sum();
    assert!(max_in_offset < input.len());
    let max_out_offset: usize = shape
        .iter()
        .zip(stride_out)
        .map(|(n, step)| (n - 1) * step)
        .sum();
    assert!(max_out_offset < output.len());

    // copy input into the output
    let (mstride_in_axis, _) = stride_out
        .iter()
        .enumerate()
        .reduce(|acc, v| if v.1 < acc.1 { v } else { acc })
        .expect("dimensions should be greater thann 0.");

    #[cfg(feature = "rayon")]
    {
        let (out_chunks, out_rem) =
            output.par_iter_lane_chunks_mut_strided::<N>(shape, stride_out, mstride_in_axis);

        let (in_chunks, in_rem) =
            input.par_iter_lane_chunks_strided::<N>(shape, stride_in, mstride_in_axis);

        out_chunks.zip(in_chunks).for_each(|(o, i)| {
            o.iter_mut().zip(i.iter()).for_each(|(o, i)| {
                o.into_iter().zip(i).for_each(|(o, i)| {
                    *o = i.clone();
                });
            });
        });
        out_rem.zip(in_rem).for_each(|(o, i)| {
            o.iter_mut().zip(i.iter()).for_each(|(o, i)| *o = i.clone());
        });
    }
    #[cfg(not(feature = "rayon"))]
    {
        let (out_chunks, out_rem) =
            output.iter_lane_chunks_mut_strided::<N>(shape, stride_out, mstride_in_axis);
        let (in_chunks, in_rem) =
            input.iter_lane_chunks_strided::<N>(shape, stride_in, mstride_in_axis);

        out_chunks.zip(in_chunks).for_each(|(o, i)| {
            o.iter_mut().zip(i.iter()).for_each(|(o, i)| {
                o.into_iter().zip(i).for_each(|(o, i)| {
                    *o = i.clone();
                });
            });
        });
        out_rem.zip(in_rem).for_each(|(o, i)| {
            o.iter_mut().zip(i.iter()).for_each(|(o, i)| *o = i.clone());
        });
    }

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

            #[cfg(feature = "rayon")]
            {
                let (chunks, rem) =
                    output.par_iter_lane_chunks_mut_strided::<N>(shape, stride_out, ax);
                if chunks.len() > 0 {
                    #[cfg(feature = "rayon")]
                    chunks.for_each_init(
                        || {
                            let s = vec![T::zero(); n_s * N];
                            let d = vec![T::zero(); n_d * N];
                            (s, d)
                        },
                        |(s, d), chunk| {
                            split_strided_chunk(chunk.clone().into(), s, d);
                            s.chunks_exact_mut(n_s)
                                .zip(d.chunks_exact_mut(n_d))
                                .for_each(|(s, d)| {
                                    func(s, d);
                                });
                            interleave_strided_chunk(s, d, chunk);
                        },
                    );
                }

                rem.for_each_init(
                    || {
                        let s = vec![T::zero(); n_s];
                        let d = vec![T::zero(); n_d];
                        (s, d)
                    },
                    |(s, d), slc| {
                        split_strided(slc.clone().into(), s, d);
                        func(s, d);
                        interleave_strided(s, d, slc);
                    },
                );
            }
            #[cfg(not(feature = "rayon"))]
            {
                let (chunks, rem) = output.iter_lane_chunks_mut_strided::<N>(shape, stride_out, ax);

                if chunks.len() > 0 {
                    let mut s = vec![T::zero(); n_s * N];
                    let mut d = vec![T::zero(); n_d * N];
                    chunks.for_each(|chunk| {
                        split_strided_chunk(chunk.clone().into(), &mut s, &mut d);
                        s.chunks_exact_mut(n_s)
                            .zip(d.chunks_exact_mut(n_d))
                            .for_each(|(s, d)| {
                                func(s, d);
                            });
                        interleave_strided_chunk(&s, &d, chunk);
                    });
                }
                if rem.len() > 0 {
                    let mut s = vec![T::zero(); n_s];
                    let mut d = vec![T::zero(); n_d];
                    rem.for_each(|slc| {
                        split_strided(slc.clone().into(), &mut s, &mut d);
                        func(&mut s, &mut d);
                        interleave_strided(&s, &d, slc);
                    })
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;

    use super::*;

    #[test]
    fn test_roundtrip() {
        for n in [12, 13] {
            for dim in [1, 2, 3, 4] {
                let shape = vec![n; dim];
                let stride = stride_from_shape(&shape);
                let axes = HashSet::from_iter(0..dim);
                let n_total = shape.iter().product();
                let v1 = (0..n_total).map(|i| i as f64).collect_vec();
                let mut v2 = vec![0.0; n_total];
                let mut v3 = vec![0.0; n_total];
                general_nd_forward_multilevel(
                    |_, _| {},
                    &v1,
                    &mut v2,
                    &shape,
                    &stride,
                    &stride,
                    &axes,
                    1,
                );
                general_nd_inverse_multilevel(
                    |_, _| {},
                    &v2,
                    &mut v3,
                    &shape,
                    &stride,
                    &stride,
                    &axes,
                    1,
                );
                assert_eq!(v1, v3);
            }
        }
    }
}
