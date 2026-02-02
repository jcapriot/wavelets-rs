pub mod bior;
pub mod daubechies;

use crate::Transformable;
use crate::boundarys::BoundaryExtension;
use crate::boundarys::LiftedAdjointBoundary;

pub trait LiftingTransform {
    fn forward<T: Transformable + From<f64>, BC: BoundaryExtension>(
        s: &mut [T],
        d: &mut [T],
        bc: &BC,
    );
    fn inverse<T: Transformable + From<f64>, BC: BoundaryExtension>(
        s: &mut [T],
        d: &mut [T],
        bc: &BC,
    );
    fn adjoint_forward<T: Transformable + From<f64>, BC: LiftedAdjointBoundary>(
        s: &mut [T],
        d: &mut [T],
        bc: &BC,
    );
    fn adjoint_inverse<T: Transformable + From<f64>, BC: LiftedAdjointBoundary>(
        s: &mut [T],
        d: &mut [T],
        bc: &BC,
    );
}

pub fn broadcasted_db2<'a, T: Transformable + From<f64>>(s: &'a mut [T], d: &'a mut [T], n: usize) {
    assert_eq!(s.len() % n, 0);
    let ns = s.len() / n;

    assert_eq!(d.len() % n, 0);
    let nd = d.len() / n;

    assert!(ns == nd || ns == nd + 1);

    let v = T::from(-1.73205080756887729352744634150587236694280525381038062805581_f64);
    d.chunks_exact_mut(n)
        .zip(s.chunks_exact(n))
        .for_each(|(d, s)| {
            d.iter_mut()
                .zip(s.iter())
                .for_each(|(d, s)| *d += v.clone() * s.clone());
        });

    let c = (
        T::from(0.433012701892219323381861585376468091735701313452595157013952),
        T::from(-0.0669872981077806766181384146235319082642986865474048429860483),
    );

    let mut s_chunks = s.chunks_exact_mut(n).enumerate();
    s_chunks
        .by_ref()
        .zip(d.chunks_exact(n).zip(d[n..].chunks_exact(n)))
        .for_each(|((_i, s), d)| {
            s.iter_mut()
                .zip(d.0.iter().zip(d.1.iter()))
                .for_each(|(s, d)| {
                    *s += c.0.clone() * d.0.clone() + c.1.clone() * d.1.clone();
                })
        });
    for (i, s) in s_chunks {
        if i < nd {
            s.iter_mut()
                .zip(d[i * n..(i + 1) * n].iter())
                .for_each(|(s, d)| *s += c.0.clone() * d.clone());
        }
    }

    let v = T::from(1.0);
    d.chunks_exact_mut(n)
        .zip(s.chunks_exact(n))
        .for_each(|(d, s)| {
            d.iter_mut()
                .zip(s.iter())
                .for_each(|(d, s)| *d += v.clone() * s.clone());
        });

    let scale = T::from(1.93185165257813657349948639945779473526780967801680910080469);
    s.iter_mut().for_each(|v| *v *= scale.clone());
    d.iter_mut().for_each(|v| *v /= scale.clone());
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

    implement_lifting_scheme!(
        TestWavelet,
        UpdateD(-1, [1.0, 2.0, 3.0, 4.0]),
        UpdateS(-2, [-1.0, 2.0, -3.0, 4.0, -5.0]),
        Scale(0.5)
    );

    #[rstest]
    fn test_multisteps_forward_inverse(
        #[values(
            BoundaryCondition::Zero,
            BoundaryCondition::Periodic,
            BoundaryCondition::Constant,
            BoundaryCondition::Reflect,
            BoundaryCondition::Symmetric,
            BoundaryCondition::Antisymmetric
        )]
        bc: BoundaryCondition,
        #[values(32, 31)] n: usize,
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
            BoundaryCondition::Antisymmetric
        )]
        bc: BoundaryCondition,
        #[values(32, 31)] n: usize,
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
            BoundaryCondition::Antisymmetric
        )]
        bc: BoundaryCondition,
        #[values(32, 31)] n: usize,
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
            BoundaryCondition::Antisymmetric
        )]
        bc: BoundaryCondition,
        #[values(32, 31)] n: usize,
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
}
