pub use crate::dwt::DiscreteTransform;
pub use crate::wavelets::bior::*;
use wavelets_macros::implement_dwt_biorthogonal;

implement_dwt_biorthogonal! {
    Bior3_1,
    [
        -0.3535533905932738,
        1.0606601717798212,
        1.0606601717798212,
        -0.3535533905932738
    ],
    [
        -0.1767766952966369,
        0.5303300858899106,
        -0.5303300858899106,
        0.1767766952966369
    ]
}

#[cfg(test)]
mod test {
    use crate::{boundarys::BoundaryCondition, tests::test_approx_equal};

    use super::Bior3_1;
    use crate::dwt::DiscreteTransform;

    #[test]
    fn bior3_1() {
        let n = 32;
        type WVLT = Bior3_1;
        let x: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();

        let nsd = WVLT::get_outlen(n);

        let mut s = vec![0.0; nsd];
        let mut d = vec![0.0; nsd];

        let bcs = [
            BoundaryCondition::ZeroBoundary,
            BoundaryCondition::PeriodicBoundary,
        ];

        for bc in bcs {
            WVLT::forward(&x, &mut s, &mut d, &bc);
            let mut x2 = vec![0.0; n];
            WVLT::inverse(&s, &d, &mut x2);

            test_approx_equal(&x2, &x, 1E-14, 0.0);
        }
    }
}
