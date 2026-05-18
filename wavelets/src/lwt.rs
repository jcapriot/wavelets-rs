//! Lifting Wavelet Transform (LWT).
//!
//! The LWT is an in-place factorisation of the DWT into a sequence of simple
//! *predict* and *update* steps (the lifting scheme).  It operates directly on
//! pre-split approximation `s` and detail `d` sub-arrays (even/odd samples of the
//! original signal) and is generally faster and more memory-efficient than the
//! convolution-based DWT.
//!
//! # Relationship to DWT
//!
//! The LWT and DWT compute the same mathematical transform — they differ only in
//! the implementation strategy.  The LWT operates in-place on split sub-bands,
//! while the DWT convolves the full signal and subsamples.
//!
//! # Sub-modules
//!
//! - [`driver`] — high-level [`driver::WaveletTransform`] for 1-D and N-D transforms.
//! - [`daubechies`], [`symlet`], [`coiflet`], [`bior`] — per-family lifting steps.

pub mod bior;
pub mod coiflet;
pub mod daubechies;
pub mod driver;
pub mod symlet;

use crate::boundarys::BoundaryExtension;
use crate::{SimdTransformable, Transformable};

/// Lifting-scheme transform for a specific wavelet.
///
/// All methods operate **in-place** on pre-split sub-arrays:
/// - `s` contains the even-indexed samples (approximation channel).
/// - `d` contains the odd-indexed samples (detail channel).
///
/// Before calling `forward`, split the signal with
/// [`crate::utils::deinterleave`]; after calling `inverse`, merge with
/// [`crate::utils::interleave`].  The high-level [`driver::WaveletTransform`]
/// handles this automatically.
pub trait LiftingTransform {
    /// Forward lifting transform: update `s` and `d` in-place.
    fn forward<T: SimdTransformable, BC: BoundaryExtension>(s: &mut [T], d: &mut [T], bc: &BC);

    /// Forward lifting transform using explicit chunk size for SIMD/parallel dispatch.
    fn forward_chunk<T: Transformable, BC: BoundaryExtension>(
        s: &mut [T],
        d: &mut [T],
        chunk_size: usize,
        bc: &BC,
    );

    /// Inverse lifting transform: undo `forward` in-place.
    fn inverse<T: SimdTransformable, BC: BoundaryExtension>(s: &mut [T], d: &mut [T], bc: &BC);

    /// Adjoint (transpose) of the forward lifting transform.
    fn adjoint_forward<T: SimdTransformable, BC: BoundaryExtension>(
        s: &mut [T],
        d: &mut [T],
        bc: &BC,
    );

    /// Adjoint (transpose) of the inverse lifting transform.
    fn adjoint_inverse<T: SimdTransformable, BC: BoundaryExtension>(
        s: &mut [T],
        d: &mut [T],
        bc: &BC,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;

    use crate::tests::test_approx_equal;
    use itertools::Itertools;
    use wavelets_macros::implement_lifting_scheme;

    use crate::boundarys::BoundaryCondition;

    const RTOL: f64 = 1E-6;
    const ATOL: f64 = 1E-14;
    pub struct TestWavelet;

    implement_lifting_scheme! {
        TestWavelet,
        //UpdateS(-2, [2.0, 3.0]),
        UpdateD(-1, [1.0, 2.0, 3.0, 4.0]),
        UpdateS(-2, [-1.0, 2.0, -3.0, 4.0, -5.0]),
        Scale(0.5)
    }

    #[rstest]
    fn test_multisteps_forward_inverse(
        #[values(
            BoundaryCondition::Zero,
            BoundaryCondition::Periodic,
            BoundaryCondition::Constant,
            BoundaryCondition::Reflect,
            BoundaryCondition::Symmetric,
            BoundaryCondition::Antisymmetric,
            BoundaryCondition::Smooth,
            BoundaryCondition::Antireflect
        )]
        bc: BoundaryCondition,
        #[values(1, 2, 3, 4, 31, 32, 1000, 1001)] n: usize, // testing for very small sizes to ensure the code doesn't crash or panic.
    ) {
        let ns = (n + 1) / 2;
        let input = (0..n).map(|i| (i + 1) as f64).collect_vec();

        let mut s = input[..ns].to_vec();
        let mut d = input[ns..].to_vec();

        TestWavelet::forward(&mut s, &mut d, &bc);

        TestWavelet::inverse(&mut s, &mut d, &bc);

        let output = s.iter().chain(d.iter()).cloned().collect_vec();

        test_approx_equal(&output, &input, RTOL, ATOL);
    }

    #[rstest]
    fn test_multisteps_adjoint_forward_inverse(
        #[values(
            BoundaryCondition::Zero,
            BoundaryCondition::Periodic,
            BoundaryCondition::Constant,
            BoundaryCondition::Reflect,
            BoundaryCondition::Symmetric,
            BoundaryCondition::Antisymmetric,
            BoundaryCondition::Smooth,
            BoundaryCondition::Antireflect
        )]
        bc: BoundaryCondition,
        #[values(1, 2, 3, 4, 31, 32, 1000, 1001)] n: usize, // testing for very small sizes to ensure the code doesn't crash or panic.
    ) {
        let ns = (n + 1) / 2;
        let input = (0..n).map(|i| (i + 1) as f64).collect_vec();

        let mut s = input[..ns].to_vec();
        let mut d = input[ns..].to_vec();

        TestWavelet::adjoint_forward(&mut s, &mut d, &bc);

        TestWavelet::adjoint_inverse(&mut s, &mut d, &bc);

        let output = s.iter().chain(d.iter()).cloned().collect_vec();

        test_approx_equal(&output, &input, RTOL, ATOL);
    }

    #[rstest]
    fn test_multisteps_forward_adjoint(
        #[values(
            BoundaryCondition::Zero,
            BoundaryCondition::Periodic,
            BoundaryCondition::Constant,
            BoundaryCondition::Reflect,
            BoundaryCondition::Symmetric,
            BoundaryCondition::Antisymmetric,
            BoundaryCondition::Smooth,
            BoundaryCondition::Antireflect
        )]
        bc: BoundaryCondition,
        #[values(1, 2, 3, 4, 31, 32, 1000, 1001)] n: usize, // testing for very small sizes to ensure the code doesn't crash or panic.
    ) {
        let ns = (n + 1) / 2;
        let u = (0..n).map(|v| v as f64).collect::<Vec<_>>();
        let v = (0..n).map(|v| -((v + 500) as f64)).collect::<Vec<_>>();

        let mut s_u = u[..ns].iter().cloned().collect::<Vec<_>>();
        let mut d_u = u[ns..n].iter().cloned().collect::<Vec<_>>();

        TestWavelet::forward(&mut s_u, &mut d_u, &bc);

        let left: f64 = s_u
            .iter()
            .chain(d_u.iter())
            .zip(v.iter())
            .map(|(v1, v2)| v1 * v2)
            .sum();

        let mut s_v = v[..ns].iter().cloned().collect::<Vec<_>>();
        let mut d_v = v[ns..n].iter().cloned().collect::<Vec<_>>();

        TestWavelet::adjoint_forward(&mut s_v, &mut d_v, &bc);

        let right: f64 = s_v
            .iter()
            .chain(d_v.iter())
            .zip(u.iter())
            .map(|(v1, v2)| v1 * v2)
            .sum();

        assert_eq!(left, right)
    }

    #[rstest]
    fn test_multisteps_inverse_adjoint(
        #[values(
            BoundaryCondition::Zero,
            BoundaryCondition::Periodic,
            BoundaryCondition::Constant,
            BoundaryCondition::Reflect,
            BoundaryCondition::Symmetric,
            BoundaryCondition::Antisymmetric,
            BoundaryCondition::Smooth,
            BoundaryCondition::Antireflect
        )]
        bc: BoundaryCondition,
        #[values(1, 2, 3, 4, 31, 32, 1000, 1001)] n: usize, // testing for very small sizes to ensure the code doesn't crash or panic.
    ) {
        let ns = (n + 1) / 2;
        let u = (0..n).map(|v| v as f64).collect::<Vec<_>>();
        let v = (0..n).map(|v| -((v + 500) as f64)).collect::<Vec<_>>();

        let mut s_u = u[..ns].iter().cloned().collect::<Vec<_>>();
        let mut d_u = u[ns..n].iter().cloned().collect::<Vec<_>>();

        TestWavelet::inverse(&mut s_u, &mut d_u, &bc);

        let left: f64 = s_u
            .iter()
            .chain(d_u.iter())
            .zip(v.iter())
            .map(|(v1, v2)| v1 * v2)
            .sum();

        let mut s_v = v[..ns].iter().cloned().collect::<Vec<_>>();
        let mut d_v = v[ns..n].iter().cloned().collect::<Vec<_>>();

        TestWavelet::adjoint_inverse(&mut s_v, &mut d_v, &bc);

        let right: f64 = s_v
            .iter()
            .chain(d_v.iter())
            .zip(u.iter())
            .map(|(v1, v2)| v1 * v2)
            .sum();

        assert_eq!(left, right)
    }

    #[rstest]
    fn test_multisteps_forward_chunk(
        #[values(
            BoundaryCondition::Zero,
            BoundaryCondition::Periodic,
            BoundaryCondition::Constant,
            BoundaryCondition::Reflect,
            BoundaryCondition::Symmetric,
            BoundaryCondition::Antisymmetric,
            BoundaryCondition::Smooth,
            BoundaryCondition::Antireflect
        )]
        bc: BoundaryCondition,
        #[values(32, 31)] n: usize,
    ) {
        let ns = (n + 1) / 2;
        let input = (0..n).map(|i| (i + 1) as f64).collect_vec();

        let mut s = input[..ns].to_vec();
        let mut d = input[ns..].to_vec();

        TestWavelet::forward(&mut s, &mut d, &bc);

        let output1 = s.iter().chain(d.iter()).cloned().collect_vec();

        let mut sc = input[..ns].to_vec();
        let mut dc = input[ns..].to_vec();

        TestWavelet::forward_chunk(&mut sc, &mut dc, 1, &bc);

        let output2 = sc.iter().chain(dc.iter()).cloned().collect_vec();

        test_approx_equal(&output2, &output1, RTOL, ATOL);
    }
}
