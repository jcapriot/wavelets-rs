mod steps;
pub mod daubechies;

use num_traits::{Num, NumAssignOps, Zero};
use itertools::{Itertools, izip};
use std::ops::{MulAssign, Neg, Mul};

use crate::{boundarys::BoundaryExtension, iter::slice::LanesIterator};
use crate::iter::slice::{StridedSlice, MutStridedSlice, ChunkStridedSlice, MutChunkStridedSlice};
use crate::vector::{Vector, simd_lanes::VectorType};


pub trait LiftedTransform<W>
where
    W: Copy + Clone + Neg<Output = W>,
{
    type StepListType;

    fn get_steps(&self) -> &Self::StepListType;
    fn forward<SD, BC: BoundaryExtension>(&self, s: &mut [SD], d: &mut[SD], bc: &BC)
    where
        SD: Num + NumAssignOps + Copy + Mul<W, Output=SD> + MulAssign<W>;
    fn inverse<SD, BC: BoundaryExtension>(&self, s: &mut [SD], d: &mut[SD], bc: &BC)
    where
        SD: Num + NumAssignOps + Copy + Mul<W, Output=SD> + MulAssign<W>;
}

pub fn deinterleave<T: Copy>(x: &[T], evens: &mut [T], odds: &mut [T]){
    let nx = x.len();
    let n_e = evens.len();
    let n_o = odds.len();

    assert_eq!(nx / 2, n_o);
    assert_eq!((nx + 1) / 2, n_e);

    x.iter()
        .zip(evens.iter_mut().interleave(odds.iter_mut()))
        .for_each(|(x, v)| *v = *x);
}

pub fn deinterleave_strided<T: Copy>(x: &StridedSlice<T>, evens: &mut [T], odds: &mut [T]){
    let nx = x.len();
    let n_e = evens.len();
    let n_o = odds.len();

    assert_eq!(nx / 2, n_o);
    assert_eq!((nx + 1) / 2, n_e);

    x.iter()
        .zip(evens.iter_mut().interleave(odds.iter_mut()))
        .for_each(|(v, ou)|{
            *ou = *v;
        });
}

pub fn deinterleave_strided_chunk<T: Copy, const N: usize>(x: &ChunkStridedSlice<T, N>, evens: &mut [Vector<T, N>], odds: &mut [Vector<T, N>]){
    assert_ne!(N, 0);

    let nx = x.len();

    let n_e = evens.len();
    let n_o = odds.len();

    assert_eq!(nx / 2, n_o);
    assert_eq!(nx - n_o, n_e);

    x.iter()
        .zip(evens.iter_mut().interleave(odds.iter_mut()))
        .for_each(|(v_v, o_v)|{
            v_v.zip(o_v.iter_mut())
                .for_each(|(v, ou)|{ *ou = * v});
        });
}

pub fn stack_to_strided<'a, T: Copy>(first: &[T], second: &[T], out: &'a mut MutStridedSlice<'a, T>){
    assert_eq!(first.len() + second.len(), out.len());
    first.iter()
        .chain(second.iter())
        .zip(out.iter_mut())
        .for_each(|(v_in, v_out)|*v_out = *v_in);
}

pub fn stack_to_strided_chunk<'a, T: Copy, const N: usize>(first: &[Vector<T, N>], second: &[Vector<T, N>], out: &'a mut MutChunkStridedSlice<'a, T, N>){
    assert_ne!(N, 0);

    let nx = out.len();

    assert_eq!(first.len() + second.len(), nx);

    out.iter_mut().zip(first.iter().chain(second.iter())).for_each(|(out_row, in_row)|{
        out_row.zip(in_row.iter()).for_each(|(o, f)|{*o = *f});
    });
}

pub fn interleave<T: Copy>(evens: &[T], odds: &[T], x: &mut [T]){
    let nx = x.len();
    let n_e = evens.len();
    let n_o = odds.len();

    assert_eq!(nx / 2, n_o);
    assert_eq!((nx + 1) / 2, n_e);

    let (chunks, rem) = x.as_chunks_mut::<2>();
    let mut ev_iter = evens.iter();
    izip!(chunks.iter_mut(), ev_iter.by_ref(), odds.iter())
        .for_each(|(xc, even, odd)|{
            xc[0] = *even;
            xc[1] = *odd;
        });
    rem.iter_mut().zip(ev_iter).for_each(|(x, ev)| *x = *ev);
}

const N: usize = 1;

pub fn forward_transform<W, T, BC>(wvlt: &W, input: &[T], output: &mut[T], shape: &[usize], axes: &[usize], bc: &BC)
where
    W: LiftedTransform<T>,
    T: Num + NumAssignOps + Copy + Mul<T, Output=T> + MulAssign<T> + Neg<Output=T> + VectorType,
    BC: BoundaryExtension,
{

    assert_eq!(input.len(), output.len());
    assert_eq!(input.len(), shape.iter().product());
    let mut first = true;

    for &ax in axes{

        let n_ax = shape[ax];

        let n_d = n_ax / 2;
        let n_s = n_ax - n_d;

        let input = match first{
                false => {
                    // create a clone of the output to read from
                    // we are not reading from and writing to output during the same function
                    // it is always copied to a temporary array in between.
                    // so there is no aliasing.
                    unsafe{
                        std::slice::from_raw_parts(output.as_ptr(), output.len())
                    }
                },
                true => {first = false; input}
            };

        let iter_in_rem;
        let iter_out_rem;
        if ax < shape.len() - 1{

            let iter_in_chunks;
            let iter_out_chunks;
            (iter_in_chunks, iter_in_rem) = input.iter_lane_chunks(shape, ax);
            (iter_out_chunks, iter_out_rem) = output.iter_lane_chunks_mut(shape, ax);

            if iter_in_chunks.len() > 0 {
                let mut s = vec![Vector::<T, N>::zero(); n_s];
                let mut d = vec![Vector::<T, N>::zero(); n_d];
                for (in_chunk, mut out_chunk) in iter_in_chunks.zip(iter_out_chunks){
                    deinterleave_strided_chunk(&in_chunk, &mut s, &mut d);
                    wvlt.forward(&mut s, &mut d, bc);
                    stack_to_strided_chunk(&s, &d, &mut out_chunk);
                }
            }
        } else {
            iter_in_rem = input.iter_lanes(shape, ax);
            iter_out_rem = output.iter_lanes_mut(shape, ax);
        }

        let mut s = vec![T::zero(); n_s];
        let mut d = vec![T::zero(); n_d];
        iter_in_rem.zip(iter_out_rem).for_each(|(in_slice, mut out_slice)|{
            // copy strided slice into local dimension storage
            deinterleave_strided(&in_slice, &mut s, &mut d);
            wvlt.forward(&mut s, &mut d, bc);
            // copy local back to output strided slice
            stack_to_strided(&s, &d, &mut out_slice);
        });
    }

}

// pub mod parallel{
//     use super::*;
//     use crate::{iter::slice::parallel::ParallelLanesIterator};
//     use rayon::prelude::*;

//     pub fn forward_transform<W, T, BC>(wvlt: &W, input: &[T], output: &mut[T], shape: &[usize], axes: &[usize], bc: &BC)
//     where
//         W: LiftedTransform<T> + Sync,
//         T: Num + NumAssignOps + Copy + Mul<T, Output=T> + MulAssign<T> + Neg<Output=T> + Sync + Send,
//         BC: BoundaryExtension + Sync,
//     {

//         assert_eq!(input.len(), output.len());
//         assert_eq!(input.len(), shape.iter().product());
//         let mut first = true;
//         for &ax in axes{
//             let iter_in = match first{
//                 false => {
//                     // create a clone of the output to read from
//                     // we are not reading from and writing to output curind the same function
//                     // it is always copied to a temporary array in between.
//                     unsafe{
//                         std::slice::from_raw_parts(output.as_ptr(), output.len())
//                     }.par_iter_lanes(shape, ax)
//                 },
//                 true => {first = false; input.par_iter_lanes(shape, ax)}
//             };
//             let iter_out = output.par_iter_lanes_mut(shape, ax);

//             let n_ax = shape[ax];

//             let n_d = n_ax / 2;
//             let n_s = n_ax - n_d;
//             if (ax != shape.len() - 1) && (N > 1){
//                 iter_in.chunks(N)
//                     .zip(iter_out.chunks(N))
//                     .for_each_with((
//                             vec![Vector::<T, N>::zero(); n_s],
//                             vec![Vector::<T, N>::zero(); n_d],
//                         ),
//                         |(s, d), (in_chunk, out_chunk)|{
//                             if in_chunk.len() == N && out_chunk.len() == N{
//                                 let in_chunk: [_; N] = unsafe{in_chunk.try_into().unwrap_unchecked()};
//                                 let out_chunk: [_; N] = unsafe{out_chunk.try_into().unwrap_unchecked()};

//                                 deinterleave_strided_chunk(in_chunk, s, d);
//                                 wvlt.forward(s, d, bc);
//                                 stack_to_strided_chunk(s , d, out_chunk);
//                             }else{
//                                 in_chunk.into_iter().zip(out_chunk.into_iter()).for_each(
//                                     |(inp, out)|{

//                                         let mut s = vec![T::zero(); n_s];
//                                         let mut d = vec![T::zero(); n_d];
//                                         deinterleave_strided(inp, &mut s, &mut d);
//                                         wvlt.forward(&mut s, &mut d, bc);
//                                         stack_to_strided(&s, &d, out);
//                                     }
//                                 );
//                             }
//                         }
//                     );
//                 } else {
//                     iter_in.zip(iter_out).for_each_with(
//                         (vec![T::zero(); n_s], vec![T::zero(); n_d]),
//                         |(s, d), (in_lane, out_lane)|{
//                             // copy strided slice into local dimension storage
//                             deinterleave_strided(in_lane, s, d);
//                             wvlt.forward(s, d, bc);
//                             // copy local back to output strided slice
//                             stack_to_strided(s, d, out_lane);
//                         }
//                     );
//             }
//         }
//     }
// }


pub mod alt{

    use super::*;

    pub fn deinterleave_strided_chunk2<T: Copy, const N: usize>(x: &ChunkStridedSlice<T, N>, evens: &mut [T], odds: &mut [T]){
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

        x.iter().enumerate().for_each(|(i_row, x)|{
            let ind_io = i_row / 2;
            x.enumerate().for_each(|(i, x)|{
                unsafe{
                    if i_row % 2 == 0{
                        *e_chunks[i].get_unchecked_mut(ind_io) = *x;
                    }else{
                        *o_chunks[i].get_unchecked_mut(ind_io) = *x;
                    }
                }
            });
        });
    }


    pub fn stack_to_strided_chunk2<'a, T: Copy, const N: usize>(first: &[T], second: &[T], out: &'a mut MutChunkStridedSlice<'a, T, N>){
        assert_ne!(N, 0);

        let nx = out.len();

        let n_first = first.len() / N;
        let n_second = second.len() / N;

        let f_chunks = first.chunks_exact(n_first).collect::<Vec<_>>();
        let s_chunks = second.chunks_exact(n_second).collect::<Vec<_>>();
        assert_eq!(f_chunks.len(), N);
        assert_eq!(s_chunks.len(), N);

        assert_eq!(n_first + n_second, nx);

        out.iter_mut().enumerate().for_each(|(i_row, out)|{
            out.enumerate().for_each(|(i, out)|{
                unsafe{
                    if i_row < n_first{
                        *out = *f_chunks[i].get_unchecked(i_row);
                    }else{
                        *out = *s_chunks[i].get_unchecked(i_row - n_first);
                    }
                }
            });
        });
    }

    pub fn forward_transform<W, T, BC>(wvlt: &W, input: &[T], output: &mut[T], shape: &[usize], axes: &[usize], bc: &BC)
where
    W: LiftedTransform<T>,
    T: Num + NumAssignOps + Copy + Mul<T, Output=T> + MulAssign<T> + Neg<Output=T> + VectorType,
    BC: BoundaryExtension,
{

    assert_eq!(input.len(), output.len());
    assert_eq!(input.len(), shape.iter().product());
    let mut first = true;

    for &ax in axes{

        let n_ax = shape[ax];

        let n_d = n_ax / 2;
        let n_s = n_ax - n_d;

        let input = match first{
                false => {
                    // create a clone of the output to read from
                    // we are not reading from and writing to output during the same function
                    // it is always copied to a temporary array in between.
                    // so there is no aliasing.
                    unsafe{
                        std::slice::from_raw_parts(output.as_ptr(), output.len())
                    }
                },
                true => {first = false; input}
            };

        let (iter_in_chunks, iter_in_rem) = input.iter_lane_chunks::<N>(shape, ax);
        let (iter_out_chunks, iter_out_rem) = output.iter_lane_chunks_mut::<N>(shape, ax);

        if iter_in_chunks.len() > 0 {
            let mut s = vec![T::zero(); n_s * N];
            let mut d = vec![T::zero(); n_d * N];
            for (in_chunk, mut out_chunk) in iter_in_chunks.zip(iter_out_chunks){
                deinterleave_strided_chunk2(&in_chunk, &mut s, &mut d);
                s.chunks_exact_mut(n_s)
                    .zip(d.chunks_exact_mut(n_d))
                    .for_each(|(mut s, mut d)|{
                        wvlt.forward(&mut s, &mut d, bc);
                    });
                stack_to_strided_chunk2(&s, &d, &mut out_chunk);
            }
        }
        let mut s = vec![T::zero(); n_s];
        let mut d = vec![T::zero(); n_d];
        iter_in_rem.zip(iter_out_rem).for_each(|(in_slice, mut out_slice)|{
            // copy strided slice into local dimension storage
            deinterleave_strided(&in_slice, &mut s, &mut d);
            wvlt.forward(&mut s, &mut d, bc);
            // copy local back to output strided slice
            stack_to_strided(&s, &d, &mut out_slice);
        });
    }

}
}


#[cfg(test)]
mod tests{
    use super::*;

    use crate::lwt::steps::{UpdateD, UpdateS, ScaleStep};
    use wavelets_macros::LiftedTransform;
    use crate::test_approx_equal;

    use crate::boundarys::ZeroBoundary;


    const RTOL: f64 = 1E-6;
    const ATOL: f64 = 1E-14;

    #[derive(LiftedTransform)]
    struct TestWavelet<T: Copy + Num + Neg<Output=T>>{
        steps: (
            UpdateD<T, 2>,
            UpdateS<T, 2>,
            ScaleStep<T>
        ),
    }
    impl TestWavelet<f64>{
        pub fn new() -> Self{
            Self{steps:(
                UpdateD{offset: -1, coefs:[1.0, 2.0]},
                UpdateS{offset: 1, coefs:[-1.0, -2.0]},
                ScaleStep{scale: 0.5}
            )}
        }
    }


    #[test]
    fn test_multisteps_forward_inverse(){

        let wvlt = TestWavelet::new();

        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let mut s = [0.0; 8];
        let mut d = [0.0; 8];

        let (chunks, _rem) = input.as_chunks::<2>();
        chunks.iter().enumerate().for_each(|(i, xc)|{
            s[i] = xc[0];
            d[i] = xc[1];
        });

        let bc = ZeroBoundary{};

        wvlt.forward(&mut s, &mut d, &bc);

        wvlt.inverse(&mut s, &mut d, &bc);

        let mut output = [0.0; 8];
        let (chunks, _rem) = output.as_chunks_mut::<2>();
        chunks.iter_mut().enumerate().for_each(|(i, out)|{
            out[0] = s[i];
            out[1] = d[i];
        });

        test_approx_equal!(&input, &output, RTOL, ATOL);

    }
}