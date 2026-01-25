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

        let bcs = [BoundaryCondition::Zero, BoundaryCondition::Periodic];

        for bc in bcs {
            WVLT::forward(&x, &mut s, &mut d, &bc);
            let mut x2 = vec![0.0; n];
            WVLT::inverse(&s, &d, &mut x2);

            test_approx_equal(&x2, &x, 1E-14, 0.0);
        }
    }

    #[test]
    fn test_bior3_1_per() {
        let n = 32;
        type WVLT = Bior3_1;
        let x: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let mut x2 = vec![0.0; n];
        let nd = n / 2;
        let ns = (n + 1) / 2;

        let mut s = vec![0.0; ns];
        let mut d = vec![0.0; nd];

        WVLT::forward_per(&x, &mut s, &mut d);

        dbg!(&s);
        dbg!(&d);

        WVLT::inverse_per(&s, &d, &mut x2);

        test_approx_equal(&x2, &x, 1E-14, 0.0);
    }

    #[test]
    fn test_bior3_1_adj() {
        let n = 32;
        type WVLT = Bior3_1;
        let x: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let mut x2 = vec![0.0; n];
        let nd = n / 2;
        let ns = (n + 1) / 2;

        let mut s = vec![0.0; ns];
        let mut d = vec![0.0; nd];

        WVLT::adjoint_inverse_per(&x, &mut s, &mut d);

        WVLT::adjoint_forward_per(&s, &d, &mut x2);

        test_approx_equal(&x2, &x, 1E-14, 0.0);

        let u = (0..n as isize).map(|i| (i + 1) as f64).collect::<Vec<_>>();
        let v = (0..n as isize).map(|i| (5 - i) as f64).collect::<Vec<_>>();
        let (vs, vd) = v.split_at(ns);

        WVLT::forward_per(&u, &mut s, &mut d);
        let v_dot_f_u = vs.iter().zip(s.iter()).map(|(v1, v2)| v1 * v2).sum::<f64>()
            + vd.iter().zip(d.iter()).map(|(v1, v2)| v1 * v2).sum::<f64>();

        WVLT::adjoint_forward_per(&vs, &vd, &mut x2);
        let v_f_t_dot_u = x2.iter().zip(u.iter()).map(|(a, b)| a * b).sum::<f64>();

        assert_eq!(v_dot_f_u, v_f_t_dot_u);
    }
}
