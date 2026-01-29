use itertools::Itertools;
use wavelets::Wavelets;
use wavelets::boundarys::ZeroBoundary;
use wavelets::driver::Wavelet;
use wavelets::iter::slice::LanesIterator;
use wavelets::lwt::{self, LiftingTransform};
use wavelets::utils::{deinterleave_2d, deinterleave_strided, stack_to_strided};

#[test]
pub fn test_broadcasted_db2() {
    let shape = [55, 55];
    let n_total = shape.iter().product();
    let x = (0..n_total).map(|i| i as f64).collect_vec();

    let wvlt = Wavelets::Daubechies2;
    let trans = Wavelet::new(wvlt, ZeroBoundary {});

    let mut sd = vec![0.0; n_total];
    let axes = [0];

    trans.forward_nd(&x, &mut sd, &shape, &axes);

    // deinterleave x into the s and d chunks of axis 0;

    let mut x2 = vec![0.0; n_total];
    let ne = (shape[0] + 1) / 2;
    let no = shape[0] / 2;
    let mut e = vec![0.0; ne];
    let mut s = vec![0.0; no];
    for (in_lane, mut out_lane) in x.iter_lanes(&shape, 0).zip(x2.iter_lanes_mut(&shape, 0)) {
        deinterleave_strided(&in_lane, &mut e, &mut s);
        stack_to_strided(&e, &s, &mut out_lane);
    }

    let (s, d) = x2.split_at_mut(ne * shape[1]);
    lwt::broadcasted_db2(s, d, shape[1]);

    let sd_ref = sd.split_at(ne * shape[1]);

    wavelets::tests::test_approx_equal(s, sd_ref.0, 1E-15, 0.0);

    wavelets::tests::test_approx_equal(&x2, &sd, 1E-15, 0.0);
}

#[test]
pub fn test_broadcasted_db2_full() {
    let shape = [6, 6];
    let n_total = shape.iter().product();
    let x = (0..n_total).map(|i| i as f64).collect_vec();

    let wvlt = Wavelets::Daubechies2;
    let trans = Wavelet::new(wvlt, ZeroBoundary {});

    let mut sd = vec![0.0; n_total];
    let axes = [1, 0];

    trans.forward_nd(&x, &mut sd, &shape, &axes);

    // deinterleave x into the s and d chunks of axis 0;

    let mut x2 = vec![0.0; n_total];

    deinterleave_2d(&x, &mut x2, &shape);

    let ne = (shape[0] + 1) / 2;
    let (s, d) = x2.split_at_mut(ne * shape[1]);
    lwt::broadcasted_db2(s, d, shape[1]);

    let ne = (shape[1] + 1) / 2;
    let bc = ZeroBoundary;
    for slc in x2.chunks_exact_mut(shape[1]) {
        let (s, d) = slc.split_at_mut(ne);
        wavelets::lwt::daubechies::Daubechies2::forward(s, d, &bc);
    }

    wavelets::tests::test_approx_equal(&x2, &sd, 1E-15, 1E-13);
}
