use itertools::Itertools;
use ndwt::Wavelets;
use ndwt::boundarys::ZeroBoundary;
use ndwt::iter::LanesIterator;
use ndwt::lwt::driver::WaveletTransform;
use ndwt::lwt::{self, LiftingTransform};
use ndwt::utils::deinterleave_2d;

#[test]
pub fn test_broadcasted_db2() {
    let shape = [30, 1];
    let n_total = shape.iter().product();
    let x = (0..n_total).map(|i| i as f64).collect_vec();

    let wvlt = Wavelets::Daubechies2;
    let trans = WaveletTransform::new(wvlt, ZeroBoundary {});

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
        in_lane.deinterleave(&mut e, &mut s);
        out_lane.stack(&e, &s);
    }

    let (s, d) = x2.split_at_mut(ne * shape[1]);

    lwt::daubechies::Daubechies2::forward_chunk(s, d, shape[1], &ZeroBoundary {});
    //lwt::broadcasted_db2(s, d, shape[1]);

    //    let sd_ref = sd.split_at(ne * shape[1]);

    // dbg!("s");
    // ndwt::tests::test_approx_equal(s, sd_ref.0, 1E-15, 0.0);

    dbg!("x");
    ndwt::tests::test_approx_equal(&x2, &sd, 1E-15, 1e-13);
}

#[test]
pub fn test_broadcasted_db2_full() {
    let shape = [6, 6];
    let n_total = shape.iter().product();
    let x = (0..n_total).map(|i| i as f64).collect_vec();

    let wvlt = Wavelets::Daubechies2;
    let trans = WaveletTransform::new(wvlt, ZeroBoundary {});

    let mut sd = vec![0.0; n_total];
    let axes = [1, 0];

    trans.forward_nd(&x, &mut sd, &shape, &axes);

    // deinterleave x into the s and d chunks of axis 0;

    let mut x2 = vec![0.0; n_total];

    deinterleave_2d(&x, &mut x2, &shape);

    let ne = (shape[0] + 1) / 2;
    let (s, d) = x2.split_at_mut(ne * shape[1]);
    lwt::daubechies::Daubechies2::forward_chunk(s, d, shape[1], &ZeroBoundary {});

    let ne = (shape[1] + 1) / 2;
    let bc = ZeroBoundary;
    for slc in x2.chunks_exact_mut(shape[1]) {
        let (s, d) = slc.split_at_mut(ne);
        ndwt::lwt::daubechies::Daubechies2::forward(s, d, &bc);
    }

    ndwt::tests::test_approx_equal(&x2, &sd, 1E-15, 1E-13);
}
