pub mod daubechies;

use itertools::{Itertools, izip};
use num_traits::{Num, NumAssignOps};

use crate::boundarys::BoundaryExtension;
use crate::iter::slice::{ChunkStridedSlice, MutChunkStridedSlice, MutStridedSlice, StridedSlice};

pub trait LiftingTransform {
    fn forward<T: NumAssignOps + Num + From<f64> + Clone, BC: BoundaryExtension>(
        s: &mut [T],
        d: &mut [T],
        bc: &BC,
    );
    fn inverse<T: NumAssignOps + Num + From<f64> + Clone, BC: BoundaryExtension>(
        s: &mut [T],
        d: &mut [T],
        bc: &BC,
    );
}

pub fn deinterleave<T: Clone>(x: &[T], evens: &mut [T], odds: &mut [T]) {
    let nx = x.len();
    let n_e = evens.len();
    let n_o = odds.len();

    assert_eq!(nx / 2, n_o);
    assert_eq!((nx + 1) / 2, n_e);

    x.iter()
        .zip(evens.iter_mut().interleave(odds.iter_mut()))
        .for_each(|(x, v)| *v = x.clone());
}

pub fn deinterleave_strided<T: Clone>(x: &StridedSlice<T>, evens: &mut [T], odds: &mut [T]) {
    let nx = x.len();
    let n_e = evens.len();
    let n_o = odds.len();

    assert_eq!(nx / 2, n_o);
    assert_eq!((nx + 1) / 2, n_e);

    x.iter()
        .zip(evens.iter_mut().interleave(odds.iter_mut()))
        .for_each(|(v, ou)| {
            *ou = v.clone();
        });
}

pub fn stack_to_strided<'a, T: Clone>(
    first: &[T],
    second: &[T],
    out: &'a mut MutStridedSlice<'a, T>,
) {
    assert_eq!(first.len() + second.len(), out.len());
    first
        .iter()
        .chain(second.iter())
        .zip(out.iter_mut())
        .for_each(|(v_in, v_out)| *v_out = v_in.clone());
}

pub fn interleave<T: Clone>(evens: &[T], odds: &[T], x: &mut [T]) {
    let nx = x.len();
    let n_e = evens.len();
    let n_o = odds.len();

    assert_eq!(nx / 2, n_o);
    assert_eq!((nx + 1) / 2, n_e);

    let (chunks, rem) = x.as_chunks_mut::<2>();
    let mut ev_iter = evens.iter();
    izip!(chunks.iter_mut(), ev_iter.by_ref(), odds.iter()).for_each(|(xc, even, odd)| {
        xc[0] = even.clone();
        xc[1] = odd.clone();
    });
    rem.iter_mut()
        .zip(ev_iter)
        .for_each(|(x, ev)| *x = ev.clone());
}

const N: usize = 4;

pub fn deinterleave_strided_chunk<T: Clone, const N: usize>(
    x: &ChunkStridedSlice<T, N>,
    evens: &mut [T],
    odds: &mut [T],
) {
    assert_ne!(N, 0);

    let nx = x.len();

    let n_e = evens.len() / N;
    let n_o = odds.len() / N;

    assert_eq!(nx / 2, n_o);
    assert_eq!(nx - n_o, n_e);

    let mut e_chunks = evens.chunks_exact_mut(n_e).collect::<Vec<_>>();
    let mut o_chunks = odds.chunks_exact_mut(n_o).collect::<Vec<_>>();
    assert_eq!(e_chunks.len(), N);
    assert_eq!(o_chunks.len(), N);

    x.iter().enumerate().for_each(|(i_row, x)| {
        let ind_io = i_row / 2;
        x.enumerate().for_each(|(i, x)| unsafe {
            if i_row % 2 == 0 {
                *e_chunks[i].get_unchecked_mut(ind_io) = x.clone();
            } else {
                *o_chunks[i].get_unchecked_mut(ind_io) = x.clone();
            }
        });
    });
}

pub fn stack_to_strided_chunk<'a, T: Clone, const N: usize>(
    first: &[T],
    second: &[T],
    out: &'a mut MutChunkStridedSlice<'a, T, N>,
) {
    assert_ne!(N, 0);

    let nx = out.len();

    let n_first = first.len() / N;
    let n_second = second.len() / N;

    let f_chunks = first.chunks_exact(n_first).collect::<Vec<_>>();
    let s_chunks = second.chunks_exact(n_second).collect::<Vec<_>>();
    assert_eq!(f_chunks.len(), N);
    assert_eq!(s_chunks.len(), N);

    assert_eq!(n_first + n_second, nx);

    out.iter_mut().enumerate().for_each(|(i_row, out)| {
        out.enumerate().for_each(|(i, out)| unsafe {
            if i_row < n_first {
                *out = f_chunks[i].get_unchecked(i_row).clone();
            } else {
                *out = s_chunks[i].get_unchecked(i_row - n_first).clone();
            }
        });
    });
}

pub fn general_nd_forward<T>(
    func: fn(&mut [T], &mut [T]),
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

pub mod parallel {
    use super::*;
    use rayon::iter::IndexedParallelIterator;
    use rayon::iter::ParallelIterator;

    pub fn general_nd_forward<T>(
        func: fn(&mut [T], &mut [T]),
        input: &[T],
        output: &mut [T],
        shape: &[usize],
        axes: &[usize],
    ) where
        T: Num + NumAssignOps + Clone + From<f64> + Sync + Send,
    {
        use crate::iter::slice::parallel::ParallelLanesIterator;

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
    use super::*;

    use crate::tests::test_approx_equal;
    use wavelets_macros::implement_lifting_scheme;

    use crate::boundarys::ZeroBoundary;

    const RTOL: f64 = 1E-6;
    const ATOL: f64 = 1E-14;

    pub struct TestWavelet;

    implement_lifting_scheme!(
        TestWavelet,
        UpdateD(-1, [1.0, 2.0]),
        UpdateS(1, [-1.0, 2.0]),
        Scale(0.5)
    );

    #[test]
    fn test_multisteps_forward_inverse() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let mut s = [0.0; 8];
        let mut d = [0.0; 8];

        let (chunks, _rem) = input.as_chunks::<2>();
        chunks.iter().enumerate().for_each(|(i, xc)| {
            s[i] = xc[0];
            d[i] = xc[1];
        });

        let bc = ZeroBoundary {};

        TestWavelet::forward(&mut s, &mut d, &bc);

        TestWavelet::inverse(&mut s, &mut d, &bc);

        let mut output = [0.0; 8];
        let (chunks, _rem) = output.as_chunks_mut::<2>();
        chunks.iter_mut().enumerate().for_each(|(i, out)| {
            out[0] = s[i];
            out[1] = d[i];
        });

        test_approx_equal(&input, &output, RTOL, ATOL);
    }
}
