pub mod bior;
pub mod daubechies;

use num_traits::{Num, NumAssignOps};

use crate::boundarys::BoundaryExtension;
use crate::boundarys::LiftedAdjointBoundary;

const N: usize = 4;

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
    fn adjoint_forward<T: NumAssignOps + Num + From<f64> + Clone, BC: LiftedAdjointBoundary>(
        s: &mut [T],
        d: &mut [T],
        bc: &BC,
    );
    fn adjoint_inverse<T: NumAssignOps + Num + From<f64> + Clone, BC: LiftedAdjointBoundary>(
        s: &mut [T],
        d: &mut [T],
        bc: &BC,
    );
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
    use crate::utils::{
        deinterleave_strided, deinterleave_strided_chunk, stack_to_strided, stack_to_strided_chunk,
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
    use super::*;

    use crate::tests::test_approx_equal;
    use itertools::Itertools;
    use wavelets_macros::implement_lifting_scheme;

    use crate::boundarys::BoundaryCondition;

    const RTOL: f64 = 1E-6;
    const ATOL: f64 = 1E-14;

    pub struct TestWavelet;

    implement_lifting_scheme!(
        TestWavelet,
        UpdateD(-1, [1.0, 2.0, 3.0, 4.0]),
        UpdateS(-2, [-1.0, 2.0, -3.0, 4.0, -5.0]),
        Scale(0.5)
    );

    #[test]
    fn test_multisteps_forward_inverse() {
        let bcs = [
            BoundaryCondition::Zero,
            BoundaryCondition::Periodic,
            BoundaryCondition::Constant,
            BoundaryCondition::Reflect,
            BoundaryCondition::Symmetric,
        ];
        for n in [32, 31] {
            let ns = (n + 1) / 2;
            let input = (0..n).map(|i| (i + 1) as f64).collect_vec();

            for bc in bcs.iter() {
                let mut s = input[..ns].to_vec();
                let mut d = input[ns..].to_vec();

                TestWavelet::forward(&mut s, &mut d, bc);

                TestWavelet::inverse(&mut s, &mut d, bc);

                let output = s.iter().chain(d.iter()).cloned().collect_vec();

                test_approx_equal(&output, &input, RTOL, ATOL);
            }
        }
    }

    #[test]
    fn test_multisteps_adjoint_forward_inverse() {
        let bcs = [
            BoundaryCondition::Zero,
            BoundaryCondition::Periodic,
            BoundaryCondition::Constant,
            BoundaryCondition::Reflect,
            BoundaryCondition::Symmetric,
        ];
        for n in [32, 31] {
            let ns = (n + 1) / 2;
            let input = (0..n).map(|i| (i + 1) as f64).collect_vec();

            for bc in bcs.iter() {
                let mut s = input[..ns].to_vec();
                let mut d = input[ns..].to_vec();

                TestWavelet::adjoint_forward(&mut s, &mut d, bc);

                TestWavelet::adjoint_inverse(&mut s, &mut d, bc);

                let output = s.iter().chain(d.iter()).cloned().collect_vec();

                test_approx_equal(&output, &input, RTOL, ATOL);
            }
        }
    }

    #[test]
    fn test_multisteps_forward_adjoint() {
        let bcs = [
            BoundaryCondition::Zero,
            BoundaryCondition::Periodic,
            BoundaryCondition::Constant,
            BoundaryCondition::Reflect,
            BoundaryCondition::Symmetric,
        ];

        let n = 32;
        let ns = (n + 1) / 2;

        for bc in bcs.iter() {
            let u = (0..n).map(|v| v as f64).collect::<Vec<_>>();
            let v = (0..n).map(|v| -((v + 500) as f64)).collect::<Vec<_>>();

            let mut s_u = u[..ns].iter().cloned().collect::<Vec<_>>();
            let mut d_u = u[ns..n].iter().cloned().collect::<Vec<_>>();

            TestWavelet::forward(&mut s_u, &mut d_u, bc);

            let left: f64 = s_u
                .iter()
                .chain(d_u.iter())
                .zip(v.iter())
                .map(|(v1, v2)| v1 * v2)
                .sum();

            let mut s_v = v[..ns].iter().cloned().collect::<Vec<_>>();
            let mut d_v = v[ns..n].iter().cloned().collect::<Vec<_>>();

            TestWavelet::adjoint_forward(&mut s_v, &mut d_v, bc);

            let right: f64 = s_v
                .iter()
                .chain(d_v.iter())
                .zip(u.iter())
                .map(|(v1, v2)| v1 * v2)
                .sum();

            assert_eq!(left, right)
        }
    }

    #[test]
    fn test_multisteps_inverse_adjoint() {
        let bcs = [
            BoundaryCondition::Zero,
            BoundaryCondition::Periodic,
            BoundaryCondition::Constant,
            BoundaryCondition::Reflect,
            BoundaryCondition::Symmetric,
        ];

        let n = 32;
        let ns = (n + 1) / 2;

        for bc in bcs.iter() {
            let u = (0..n).map(|v| v as f64).collect::<Vec<_>>();
            let v = (0..n).map(|v| -((v + 500) as f64)).collect::<Vec<_>>();

            let mut s_u = u[..ns].iter().cloned().collect::<Vec<_>>();
            let mut d_u = u[ns..n].iter().cloned().collect::<Vec<_>>();

            TestWavelet::inverse(&mut s_u, &mut d_u, bc);

            let left: f64 = s_u
                .iter()
                .chain(d_u.iter())
                .zip(v.iter())
                .map(|(v1, v2)| v1 * v2)
                .sum();

            let mut s_v = v[..ns].iter().cloned().collect::<Vec<_>>();
            let mut d_v = v[ns..n].iter().cloned().collect::<Vec<_>>();

            TestWavelet::adjoint_inverse(&mut s_v, &mut d_v, bc);

            let right: f64 = s_v
                .iter()
                .chain(d_v.iter())
                .zip(u.iter())
                .map(|(v1, v2)| v1 * v2)
                .sum();

            assert_eq!(left, right)
        }
    }
}
