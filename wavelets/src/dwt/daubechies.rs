pub use crate::dwt::DiscreteTransform;
pub use crate::wavelets::daubechies::*;
use wavelets_macros::implement_dwt_orthogonal;

implement_dwt_orthogonal! {
    Daubechies1,
    [
        7.071067811865475244008443621048490392848359376884740365883398e-01,
        7.071067811865475244008443621048490392848359376884740365883398e-01,
    ]
}

implement_dwt_orthogonal! {
    Daubechies2,
    [
        -1.294095225512603811744494188120241641745344506599652569070016e-01,
        2.241438680420133810259727622404003554678835181842717613871683e-01,
        8.365163037378079055752937809168732034593703883484392934953414e-01,
        4.829629131445341433748715998644486838169524195042022752011715e-01,
    ]
}

implement_dwt_orthogonal! {
    Daubechies3,
    [
        0.03522629188570953,
        -0.08544127388202666,
        -0.13501102001025458,
        0.45987750211849154,
        0.8068915093110925,
        0.33267055295008263,
    ]
}

#[cfg(test)]
mod test {
    use crate::{
        boundarys::{PeriodicBoundary, ZeroBoundary},
        tests::test_approx_equal,
    };

    use super::{Daubechies1, Daubechies2, Daubechies3};
    use crate::dwt::DiscreteTransform;

    #[test]
    fn test_db1() {
        let n = 32;
        type WVLT = Daubechies1;
        let x: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();

        let nsd = WVLT::get_outlen(n);

        let mut s = vec![0.0; nsd];
        let mut d = vec![0.0; nsd];

        let bc = PeriodicBoundary {};

        WVLT::forward(&x, &mut s, &mut d, &bc);
        let mut x2 = vec![0.0; n];
        WVLT::inverse(&s, &d, &mut x2);

        test_approx_equal(&x2, &x, 1E-14, 0.0);
    }

    #[test]
    fn test_db2() {
        let n = 32;
        type WVLT = Daubechies2;
        let x: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();

        let nsd = WVLT::get_outlen(n);

        let mut s = vec![0.0; nsd];
        let mut d = vec![0.0; nsd];

        let bc = PeriodicBoundary {};

        //let mut x = vec![0.0; n];
        //x[31] = 1.0;

        WVLT::forward(&x, &mut s, &mut d, &bc);
        dbg!(&s);
        dbg!(&d);
        let mut x2 = vec![0.0; n];
        WVLT::inverse(&s, &d, &mut x2);

        test_approx_equal(&x2, &x, 1E-14, 0.0);
    }

    #[test]
    fn test_db2_zero() {
        let n = 32;
        type WVLT = Daubechies2;
        let x: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();

        let nsd = WVLT::get_outlen(n);

        let mut s = vec![0.0; nsd];
        let mut d = vec![0.0; nsd];

        let bc = ZeroBoundary {};

        WVLT::forward(&x, &mut s, &mut d, &bc);
        let mut x2 = vec![0.0; n];
        WVLT::inverse(&s, &d, &mut x2);

        dbg!(&x2);
        test_approx_equal(&x2, &x, 1E-14, 0.0);
    }

    #[test]
    fn test_db3() {
        let n = 32;
        type WVLT = Daubechies3;
        let x: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();

        let nsd = WVLT::get_outlen(n);

        let mut s = vec![0.0; nsd];
        let mut d = vec![0.0; nsd];

        let bc = PeriodicBoundary {};

        WVLT::forward(&x, &mut s, &mut d, &bc);
        let mut x2 = vec![0.0; n];
        WVLT::inverse(&s, &d, &mut x2);

        test_approx_equal(&x2, &x, 1E-14, 0.0);
    }
}
