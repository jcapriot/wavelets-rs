use num_traits::{Num, NumAssignOps};
use std::ops::Neg;

use crate::Wavelets;
use crate::boundarys::BoundaryExtension;
use crate::boundarys::LiftedAdjointBoundary;
use crate::utils::{
    deinterleave, deinterleave_strided, deinterleave_strided_chunk, stack_to_strided,
    stack_to_strided_chunk,
};
use crate::utils::{
    interleave, interleave_strided, interleave_strided_chunk, split_strided, split_strided_chunk,
};

const N: usize = 4;

pub struct Wavelet<
    T: Num + NumAssignOps + Clone + From<f64> + Neg<Output = T>,
    BC: BoundaryExtension + LiftedAdjointBoundary,
> {
    lwt_forward: fn(&mut [T], &mut [T], &BC),
    lwt_inverse: fn(&mut [T], &mut [T], &BC),
    lwt_adj_forward: fn(&mut [T], &mut [T], &BC),
    lwt_adj_inverse: fn(&mut [T], &mut [T], &BC),
    bc: BC,
}

impl<
    T: Num + NumAssignOps + Clone + From<f64> + Neg<Output = T>,
    BC: BoundaryExtension + LiftedAdjointBoundary + Clone,
> Wavelet<T, BC>
{
    pub fn new(wvlt: Wavelets, bc: BC) -> Self {
        use crate::lwt::bior::*;
        use crate::lwt::daubechies::*;
        let lwt_forward: fn(&mut [T], &mut [T], &BC) = match wvlt {
            Wavelets::Daubechies1 => Daubechies1::forward,
            _ => Daubechies2::forward,
        };
        let lwt_inverse: fn(&mut [T], &mut [T], &BC) = match wvlt {
            Wavelets::Daubechies1 => Daubechies1::inverse,
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
        general_nd_forward(
            |s, d| (self.lwt_forward)(s, d, &self.bc),
            input,
            output,
            shape,
            axes,
        );
    }

    pub fn inverse_nd(&self, input: &[T], output: &mut [T], shape: &[usize], axes: &[usize]) {
        general_nd_inverse(
            |s, d| (self.lwt_inverse)(s, d, &self.bc),
            input,
            output,
            shape,
            axes,
        );
    }

    pub fn adj_forward_nd(&self, input: &[T], output: &mut [T], shape: &[usize], axes: &[usize]) {
        general_nd_inverse(
            |s, d| (self.lwt_adj_forward)(s, d, &self.bc),
            input,
            output,
            shape,
            axes,
        );
    }

    pub fn adj_inverse_nd(&self, input: &[T], output: &mut [T], shape: &[usize], axes: &[usize]) {
        general_nd_forward(
            |s, d| (self.lwt_adj_inverse)(s, d, &self.bc),
            input,
            output,
            shape,
            axes,
        );
    }
}

pub fn general_nd_forward<T>(
    func: impl Fn(&mut [T], &mut [T]),
    input: &[T],
    output: &mut [T],
    shape: &[usize],
    axes: &[usize],
) where
    T: Num + NumAssignOps + Clone + From<f64>,
{
    use crate::iter::slice::LanesIterator;

    assert_eq!(input.len(), output.len());
    assert_eq!(input.len(), shape.iter().product());
    let mut first = true;

    for &ax in axes {
        let n_ax = shape[ax];

        let n_d = n_ax / 2;
        let n_s = n_ax - n_d;

        let input = match first {
            false => {
                // create a clone of the output to read from
                // we are not reading from and writing to output during the same function
                // it is always copied to a temporary array in between.
                // so there is no aliasing.
                unsafe { std::slice::from_raw_parts(output.as_ptr(), output.len()) }
            }
            true => {
                first = false;
                input
            }
        };

        let (iter_in_chunks, iter_in_rem) = input.iter_lane_chunks::<N>(shape, ax);
        let (iter_out_chunks, iter_out_rem) = output.iter_lane_chunks_mut::<N>(shape, ax);

        if iter_in_chunks.len() > 0 {
            let mut s = vec![T::zero(); n_s * N];
            let mut d = vec![T::zero(); n_d * N];
            for (in_chunk, mut out_chunk) in iter_in_chunks.zip(iter_out_chunks) {
                deinterleave_strided_chunk(&in_chunk, &mut s, &mut d);
                s.chunks_exact_mut(n_s)
                    .zip(d.chunks_exact_mut(n_d))
                    .for_each(|(mut s, mut d)| {
                        func(&mut s, &mut d);
                    });
                stack_to_strided_chunk(&s, &d, &mut out_chunk);
            }
        }
        let mut s = vec![T::zero(); n_s];
        let mut d = vec![T::zero(); n_d];
        iter_in_rem
            .zip(iter_out_rem)
            .for_each(|(in_slice, mut out_slice)| {
                // copy strided slice into local dimension storage
                deinterleave_strided(&in_slice, &mut s, &mut d);
                func(&mut s, &mut d);
                // copy local back to output strided slice
                stack_to_strided(&s, &d, &mut out_slice);
            });
    }
}

pub fn general_nd_inverse<T>(
    func: impl Fn(&mut [T], &mut [T]),
    input: &[T],
    output: &mut [T],
    shape: &[usize],
    axes: &[usize],
) where
    T: Num + NumAssignOps + Clone + From<f64>,
{
    use crate::iter::slice::LanesIterator;

    assert_eq!(input.len(), output.len());
    assert_eq!(input.len(), shape.iter().product());
    let mut first = true;

    for &ax in axes {
        let n_ax = shape[ax];

        let n_d = n_ax / 2;
        let n_s = n_ax - n_d;

        let input = match first {
            false => {
                // create a clone of the output to read from
                // we are not reading from and writing to output during the same function
                // it is always copied to a temporary array in between.
                // so there is no aliasing.
                unsafe { std::slice::from_raw_parts(output.as_ptr(), output.len()) }
            }
            true => {
                first = false;
                input
            }
        };

        let (iter_in_chunks, iter_in_rem) = input.iter_lane_chunks::<N>(shape, ax);
        let (iter_out_chunks, iter_out_rem) = output.iter_lane_chunks_mut::<N>(shape, ax);

        if iter_in_chunks.len() > 0 {
            let mut s = vec![T::zero(); n_s * N];
            let mut d = vec![T::zero(); n_d * N];
            for (in_chunk, mut out_chunk) in iter_in_chunks.zip(iter_out_chunks) {
                split_strided_chunk(&in_chunk, &mut s, &mut d);
                s.chunks_exact_mut(n_s)
                    .zip(d.chunks_exact_mut(n_d))
                    .for_each(|(mut s, mut d)| {
                        func(&mut s, &mut d);
                    });
                interleave_strided_chunk(&s, &d, &mut out_chunk);
            }
        }
        let mut s = vec![T::zero(); n_s];
        let mut d = vec![T::zero(); n_d];
        iter_in_rem
            .zip(iter_out_rem)
            .for_each(|(in_slice, mut out_slice)| {
                // copy strided slice into local dimension storage
                split_strided(&in_slice, &mut s, &mut d);
                func(&mut s, &mut d);
                // copy local back to output strided slice
                interleave_strided(&s, &d, &mut out_slice);
            });
    }
}

pub mod parallel {
    use super::*;
    use rayon::iter::IndexedParallelIterator;
    use rayon::iter::ParallelIterator;

    pub fn general_nd_forward<T>(
        func: impl Fn(&mut [T], &mut [T]) + Sync,
        input: &[T],
        output: &mut [T],
        shape: &[usize],
        axes: &[usize],
    ) where
        T: Num + NumAssignOps + Clone + From<f64> + Sync + Send,
    {
        use crate::iter::slice::parallel::ParallelLanesIterator;
        use crate::utils::{
            deinterleave_strided, deinterleave_strided_chunk, stack_to_strided,
            stack_to_strided_chunk,
        };

        assert_eq!(input.len(), output.len());
        assert_eq!(input.len(), shape.iter().product());
        let mut first = true;

        for &ax in axes {
            let n_ax = shape[ax];

            let n_d = n_ax / 2;
            let n_s = n_ax - n_d;

            let input = match first {
                false => {
                    // create a clone of the output to read from
                    // we are not reading from and writing to output during the same function
                    // it is always copied to a temporary array in between.
                    // so there is no aliasing.
                    unsafe { std::slice::from_raw_parts(output.as_ptr(), output.len()) }
                }
                true => {
                    first = false;
                    input
                }
            };

            let (iter_in_chunks, iter_in_rem) = input.par_iter_lane_chunks::<N>(shape, ax);
            let (iter_out_chunks, iter_out_rem) = output.par_iter_lane_chunks_mut::<N>(shape, ax);

            let n_threads = rayon::current_num_threads();
            let min_len = std::cmp::max(1, iter_in_chunks.len() / (n_threads + 1));

            if iter_in_chunks.len() > 0 {
                iter_in_chunks
                    .zip(iter_out_chunks)
                    .with_min_len(min_len)
                    .for_each_init(
                        || (vec![T::zero(); n_s * N], vec![T::zero(); n_d * N]),
                        |(s, d), (in_chunk, mut out_chunk)| {
                            deinterleave_strided_chunk(&in_chunk, s, d);
                            s.chunks_exact_mut(n_s)
                                .zip(d.chunks_exact_mut(n_d))
                                .for_each(|(mut s, mut d)| {
                                    func(&mut s, &mut d);
                                });
                            stack_to_strided_chunk(&s, &d, &mut out_chunk);
                        },
                    );
            }
            iter_in_rem.zip(iter_out_rem).with_min_len(N).for_each_init(
                || (vec![T::zero(); n_s], vec![T::zero(); n_d]),
                |(s, d), (in_slice, mut out_slice)| {
                    // copy strided slice into local dimension storage
                    deinterleave_strided(&in_slice, s, d);
                    func(s, d);
                    // copy local back to output strided slice
                    stack_to_strided(&s, &d, &mut out_slice);
                },
            );
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
                let axes = (0..dim).collect_vec();
                let n_total = shape.iter().product();
                let v1 = (0..n_total).map(|i| i as f64).collect_vec();
                let mut v2 = vec![0.0; n_total];
                let mut v3 = vec![0.0; n_total];
                general_nd_forward(|_, _| {}, &v1, &mut v2, &shape, &axes);
                general_nd_inverse(|_, _| {}, &v2, &mut v3, &shape, &axes);
                assert_eq!(v1, v3);
            }
        }
    }
}
