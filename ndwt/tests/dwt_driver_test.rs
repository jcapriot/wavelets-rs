use ndwt::Wavelet;
use ndwt::boundarys::{PeriodicBoundary, ZeroBoundary};
use ndwt::dwt::driver::{WaveletTransform, WaveletTransformPer, get_transform_shape};
use ndwt::dwt::{DiscreteTransform, get_outlen};
use ndwt::iter::LanesIterator;
use num_complex::{Complex32, Complex64};
use rstest::rstest;

#[test]
pub fn test_dwt_driver_db2_single_level_1d_along() {
    let shape = [30, 35];
    let axes = [1];
    let level = 1;
    let wvlt = Wavelet::Daubechies2;

    let n_ax: usize = shape[axes[0]];
    let n_sd = get_outlen(wvlt.width(), n_ax);

    let out_shape = get_transform_shape(&shape, &axes, level, wvlt.width(), false);

    assert_eq!(out_shape[axes[0]], 2 * n_sd);

    let n_in_total = shape.iter().product();

    let bc = ZeroBoundary {};
    let trans = WaveletTransform::new(wvlt, bc);

    let n_out_total = out_shape.iter().product();

    let x = (0..n_in_total).map(|i| i as f64).collect::<Vec<_>>();
    let mut sd = vec![0.0; n_out_total];

    trans.forward_nd(&x, &mut sd, &shape, &axes);

    let mut sd2 = vec![0.0; n_out_total];

    x.chunks_exact(shape[1])
        .zip(sd2.chunks_exact_mut(out_shape[1]))
        .for_each(|(x, sd)| {
            let (s, d) = sd.split_at_mut(n_sd);
            debug_assert_eq!(s.len(), n_sd);
            debug_assert_eq!(d.len(), n_sd);
            ndwt::dwt::daubechies::Daubechies2::forward(x, s, d, &bc);
        });

    ndwt::tests::test_approx_equal(&sd, &sd2, 1E-15, 0.0);
}

#[test]
pub fn test_dwt_driver_db2_single_level_1d_across() {
    let shape = [30, 35];
    let axes = [0];
    let level = 1;
    let wvlt = Wavelet::Daubechies2;

    let n_ax: usize = shape[axes[0]];
    let n_sd = get_outlen(wvlt.width(), n_ax);

    let out_shape = get_transform_shape(&shape, &axes, level, wvlt.width(), false);

    assert_eq!(out_shape[axes[0]], 2 * n_sd);

    let n_in_total = shape.iter().product();

    let bc = ZeroBoundary {};
    let trans = WaveletTransform::new(wvlt, bc);

    let n_out_total = out_shape.iter().product();

    let x = (0..n_in_total).map(|i| i as f64).collect::<Vec<_>>();
    let mut sd = vec![0.0; n_out_total];

    trans.forward_nd(&x, &mut sd, &shape, &axes);

    let mut sd2 = vec![0.0; n_out_total];

    let mut x_w = vec![0.0; n_ax];
    let mut s_w = vec![0.0; n_sd];
    let mut d_w = vec![0.0; n_sd];

    x.iter_lanes(&shape, axes[0])
        .zip(sd2.iter_lanes_mut(&out_shape, axes[0]))
        .for_each(|(x, mut sd)| {
            x.pour_into(&mut x_w);
            ndwt::dwt::daubechies::Daubechies2::forward(&x_w, &mut s_w, &mut d_w, &bc);
            sd.stack(&s_w, &d_w);
        });

    ndwt::tests::test_approx_equal(&sd, &sd2, 1E-15, 0.0);
}

#[test]
pub fn test_dwt_driver_db2_single_2d() {
    let shape = [30, 35];
    let axes = [0, 1];
    let level = 1;
    let wvlt = Wavelet::Daubechies2;
    let bc = ZeroBoundary {};

    let inter_shape = get_transform_shape(&shape, &[axes[0]], level, wvlt.width(), false);
    let out_shape = get_transform_shape(&shape, &axes, level, wvlt.width(), false);

    for &ax in axes.iter() {
        let n_ax: usize = shape[ax];
        let n_sd = get_outlen(wvlt.width(), n_ax);
        assert_eq!(out_shape[ax], 2 * n_sd);
    }

    let n_in_total = shape.iter().product();
    let n_inter_total = inter_shape.iter().product();
    let n_out_total = out_shape.iter().product();

    let trans = WaveletTransform::new(wvlt, bc);

    let x = (0..n_in_total)
        .map(|i| (i * i) as f64 / 3.0)
        .collect::<Vec<_>>();
    let mut x_w = vec![0.0; n_out_total];

    trans.forward_nd(&x, &mut x_w, &shape, &axes);

    let mut x_inter = vec![0.0; n_inter_total];
    let mut x_w2 = vec![0.0; n_out_total];
    trans.forward_nd(&x, &mut x_inter, &shape, &[axes[0]]);
    trans.forward_nd(&x_inter, &mut x_w2, &inter_shape, &[axes[1]]);

    ndwt::tests::test_approx_equal(&x_w, &x_w2, 1E-12, 1E-10);
}

#[test]
pub fn test_dwt_driver_db2_multi_level_1d() {
    let shape = [30, 35];
    let axes = [1];
    let level = 3;
    let wvlt = Wavelet::Daubechies2;

    dbg!(&shape, &axes, level, wvlt.width());

    let n_ax = shape[axes[0]];
    let n_sd_0 = get_outlen(wvlt.width(), n_ax);
    let n_sd_1 = get_outlen(wvlt.width(), n_sd_0);
    let n_sd_2 = get_outlen(wvlt.width(), n_sd_1);

    let n_sd_total = n_sd_2 * 2 + n_sd_1 + n_sd_0;

    let out_shape = get_transform_shape(&shape, &axes, level, wvlt.width(), false);
    dbg!(&out_shape);

    assert_eq!(out_shape[axes[0]], n_sd_total);

    let n_in_total = shape.iter().product();

    let bc = ZeroBoundary {};
    let trans = WaveletTransform::new(wvlt, bc);

    let n_out_total = out_shape.iter().product();

    let x = (0..n_in_total).map(|i| i as f64).collect::<Vec<_>>();
    let mut sd = vec![0.0; n_out_total];

    trans.forward_multilevel_nd(&x, &mut sd, &shape, &axes, level);

    let mut sd2 = vec![0.0; n_out_total];

    x.chunks_exact(shape[1])
        .zip(sd2.chunks_exact_mut(out_shape[1]))
        .for_each(|(x, sd)| {
            let (s, d0) = sd.split_at_mut(n_sd_2 * 2 + n_sd_1);
            debug_assert_eq!(d0.len(), n_sd_0);
            let mut s0 = vec![0.0; n_sd_0];
            ndwt::dwt::daubechies::Daubechies2::forward(x, &mut s0, d0, &bc);

            let (s, d1) = s.split_at_mut(n_sd_2 * 2);
            debug_assert_eq!(d1.len(), n_sd_1);
            let mut s1 = vec![0.0; n_sd_1];
            ndwt::dwt::daubechies::Daubechies2::forward(&s0, &mut s1, d1, &bc);

            let (s2, d2) = s.split_at_mut(n_sd_2);
            debug_assert_eq!(d2.len(), n_sd_2);
            ndwt::dwt::daubechies::Daubechies2::forward(&s1, s2, d2, &bc);
        });

    ndwt::tests::test_approx_equal(&sd, &sd2, 1E-15, 0.0);
}

#[test]
pub fn test_dwt_driver_db2_multi_level_2d() {
    let shape = dbg!([30, 35]);
    let axes = [1, 0];
    let level = 3;
    let wvlt = Wavelet::Daubechies2;

    let out_shape = dbg!(get_transform_shape(
        &shape,
        &axes,
        level,
        wvlt.width(),
        false
    ));

    for ax in axes {
        let n_ax = shape[ax];
        let n_sd_0 = get_outlen(wvlt.width(), n_ax);
        let n_sd_1 = get_outlen(wvlt.width(), n_sd_0);
        let n_sd_2 = get_outlen(wvlt.width(), n_sd_1);

        let n_sd_total = n_sd_2 * 2 + n_sd_1 + n_sd_0;

        assert_eq!(out_shape[ax], n_sd_total);
    }

    let n_in_total = shape.iter().product();

    let bc = ZeroBoundary {};
    let trans = WaveletTransform::new(wvlt, bc);

    let n_out_total = out_shape.iter().product();

    let x = (0..n_in_total).map(|i| (i * i) as f64).collect::<Vec<_>>();

    let mut sd = vec![0.0; n_out_total];

    trans.forward_multilevel_nd(&x, &mut sd, &shape, &axes, level);

    let shape_inter1 = dbg!(get_transform_shape(&shape, &axes, 1, wvlt.width(), false));
    let n_inter1 = shape_inter1.iter().product();

    let mut sd2 = vec![0.0; n_out_total];
    let mut sd_i1 = vec![0.0; n_inter1];

    trans.forward_nd(&x, &mut sd_i1, &shape, &axes);

    let shape2 = dbg!(shape_inter1.iter().map(|v| *v / 2).collect::<Vec<_>>());
    let n_total2 = shape2.iter().product();
    let shape_inter2 = dbg!(get_transform_shape(&shape2, &axes, 1, wvlt.width(), false));
    let n_inter2 = shape_inter2.iter().product();

    let mut s = vec![0.0; n_total2];
    let mut sd_i2 = vec![0.0; n_inter2];

    s.chunks_exact_mut(shape2[1])
        .zip(sd_i1.chunks_exact(shape_inter1[1]))
        .for_each(|(a, b)| {
            a.iter_mut().zip(b).for_each(|(a, b)| *a = *b);
        });

    trans.forward_nd(&s, &mut sd_i2, &shape2, &axes);

    let shape3 = dbg!(shape_inter2.iter().map(|v| *v / 2).collect::<Vec<_>>());
    let n_total3 = shape3.iter().product();
    let shape_inter3 = dbg!(get_transform_shape(&shape3, &axes, 1, wvlt.width(), false));
    let n_inter3 = shape_inter3.iter().product();

    let mut s = vec![0.0; n_total3];
    let mut sd_i3 = vec![0.0; n_inter3];
    s.chunks_exact_mut(shape3[1])
        .zip(sd_i2.chunks_exact(shape_inter2[1]))
        .for_each(|(a, b)| {
            a.iter_mut().zip(b).for_each(|(a, b)| *a = *b);
        });

    trans.forward_nd(&s, &mut sd_i3, &shape3, &axes);

    let nsd1 = [shape_inter1[0] / 2, shape_inter1[1] / 2];
    let nsd2 = [shape_inter2[0] / 2, shape_inter2[1] / 2];

    let nf0: usize = out_shape[0] - nsd1[0];
    let nf1 = out_shape[1] - nsd1[1];

    // copy the detail coefficients to the output
    sd2.chunks_exact_mut(out_shape[1])
        .zip(sd_i1.chunks_exact(shape_inter1[1]))
        .take(nsd1[0])
        .for_each(|(a, b)| {
            let (_, b) = b.split_at(nsd1[1]);
            a[nf1..].iter_mut().zip(b).for_each(|(a, b)| *a = *b);
        });

    sd2.chunks_exact_mut(out_shape[1])
        .skip(nf0)
        .zip(sd_i1.chunks_exact(shape_inter1[1]).skip(nsd1[0]))
        .for_each(|(a, b)| {
            let (f, s) = b.split_at(nsd1[1]);
            a.iter_mut().zip(f).for_each(|(a, b)| *a = *b);
            a[nf1..].iter_mut().zip(s).for_each(|(a, b)| *a = *b);
        });

    let nf0 = nf0 - nsd2[0];
    let nf1 = nf1 - nsd2[1];

    // copy the detail coefficients to the output
    sd2.chunks_exact_mut(out_shape[1])
        .zip(sd_i2.chunks_exact(shape_inter2[1]))
        .take(nsd2[0])
        .for_each(|(a, b)| {
            let (_, b) = b.split_at(nsd2[1]);
            a[nf1..].iter_mut().zip(b).for_each(|(a, b)| *a = *b);
        });

    sd2.chunks_exact_mut(out_shape[1])
        .skip(nf0)
        .zip(sd_i2.chunks_exact(shape_inter2[1]).skip(nsd2[0]))
        .for_each(|(a, b)| {
            let (f, s) = b.split_at(nsd2[1]);
            a.iter_mut().zip(f).for_each(|(a, b)| *a = *b);
            a[nf1..].iter_mut().zip(s).for_each(|(a, b)| *a = *b);
        });

    // Copy the whole array into first corner of sd2.
    sd2.chunks_exact_mut(out_shape[1])
        .zip(sd_i3.chunks_exact(shape_inter3[1]))
        .for_each(|(a, b)| {
            a.iter_mut().zip(b).for_each(|(a, b)| *a = *b);
        });

    ndwt::tests::test_approx_equal(&sd, &sd2, 1E-10, 1E-9);
}

#[test]
pub fn test_dwt_driver_inv_db2_single_level_1d_along() {
    let shape = [30, 35];
    let axes = [1];
    let level = 1;
    let wvlt = Wavelet::Daubechies2;

    let n_ax: usize = shape[axes[0]];
    let n_sd = get_outlen(wvlt.width(), n_ax);

    let out_shape = get_transform_shape(&shape, &axes, level, wvlt.width(), false);

    assert_eq!(out_shape[axes[0]], 2 * n_sd);

    let n_in_total = shape.iter().product();

    let bc = ZeroBoundary {};
    let trans = WaveletTransform::new(wvlt, bc);

    let n_out_total = out_shape.iter().product();

    let x = (0..n_in_total).map(|i| (i + 1) as f64).collect::<Vec<_>>();
    let mut sd = vec![0.0; n_out_total];

    trans.forward_nd(&x, &mut sd, &shape, &axes);

    let mut x2 = vec![0.0; n_in_total];

    trans.inverse_nd(&mut sd, &mut x2, &shape, &axes);

    ndwt::tests::test_approx_equal(&x2, &x, 1E-15, 0.0);
}

#[test]
pub fn test_dwt_driver_inv_db2_single_level_1d_across() {
    let shape = [30, 35];
    let axes = [0];
    let level = 1;
    let wvlt = Wavelet::Daubechies2;

    let n_ax: usize = shape[axes[0]];
    let n_sd = get_outlen(wvlt.width(), n_ax);

    let out_shape = get_transform_shape(&shape, &axes, level, wvlt.width(), false);

    assert_eq!(out_shape[axes[0]], 2 * n_sd);

    let n_in_total = shape.iter().product();

    let bc = ZeroBoundary {};
    let trans = WaveletTransform::new(wvlt, bc);

    let n_out_total = out_shape.iter().product();

    let x = (0..n_in_total).map(|i| (i + 1) as f64).collect::<Vec<_>>();
    let mut sd = vec![0.0; n_out_total];

    trans.forward_nd(&x, &mut sd, &shape, &axes);

    let mut x2 = vec![0.0; n_in_total];

    trans.inverse_nd(&mut sd, &mut x2, &shape, &axes);

    ndwt::tests::test_approx_equal(&x2, &x, 1E-15, 0.0);
}

#[test]
pub fn test_dwt_driver_inv_db2_single_level_2d() {
    let shape = [30, 35];
    let axes = [0, 1];
    let level = 1;
    let wvlt = Wavelet::Daubechies2;

    let out_shape = get_transform_shape(&shape, &axes, level, wvlt.width(), false);

    let n_in_total = shape.iter().product();

    let bc = ZeroBoundary {};
    let trans = WaveletTransform::new(wvlt, bc);

    let n_out_total = out_shape.iter().product();

    let x = (0..n_in_total).map(|i| (i + 1) as f64).collect::<Vec<_>>();
    let mut sd = vec![0.0; n_out_total];

    trans.forward_nd(&x, &mut sd, &shape, &axes);

    let mut x2 = vec![0.0; n_in_total];

    trans.inverse_nd(&mut sd, &mut x2, &shape, &axes);

    ndwt::tests::test_approx_equal(&x2, &x, 1E-13, 1E-14);
}

#[test]
pub fn test_dwt_driver_inv_db2_multi_level_1d() {
    let shape = [30, 35];
    let axes = [1];
    let level = 3;
    let wvlt = Wavelet::Daubechies2;

    let out_shape = get_transform_shape(&shape, &axes, level, wvlt.width(), false);

    let n_in_total = shape.iter().product();

    let bc = ZeroBoundary {};
    let trans = WaveletTransform::new(wvlt, bc);

    let n_out_total = out_shape.iter().product();

    let x = (0..n_in_total).map(|i| (i + 1) as f64).collect::<Vec<_>>();
    let mut sd = vec![0.0; n_out_total];

    trans.forward_multilevel_nd(&x, &mut sd, &shape, &axes, level);

    let mut x2 = vec![0.0; n_in_total];

    trans.inverse_multilevel_nd(&mut sd, &mut x2, &shape, &axes, level);

    ndwt::tests::test_approx_equal(&x2, &x, 1E-15, 0.0);
}

#[test]
pub fn test_dwt_driver_inv_db2_multi_level_2d() {
    let shape = [30, 35];
    let axes = [0, 1];
    let level = 3;
    let wvlt = Wavelet::Daubechies2;

    let out_shape = get_transform_shape(&shape, &axes, level, wvlt.width(), false);

    let n_in_total = shape.iter().product();

    let bc = ZeroBoundary {};
    let trans = WaveletTransform::new(wvlt, bc);

    let n_out_total = out_shape.iter().product();

    let x = (0..n_in_total).map(|i| (i + 1) as f64).collect::<Vec<_>>();
    let mut sd = vec![0.0; n_out_total];

    trans.forward_multilevel_nd(&x, &mut sd, &shape, &axes, level);

    let mut x2 = vec![0.0; n_in_total];

    trans.inverse_multilevel_nd(&mut sd, &mut x2, &shape, &axes, level);

    ndwt::tests::test_approx_equal(&x2, &x, 1E-13, 0.0);
}

#[test]
pub fn test_dwt_driver_db2_per_single_level_1d_along() {
    let shape = [30, 35];
    let axes = [1];
    let level = 1;
    let wvlt = Wavelet::Daubechies2;

    let out_shape = get_transform_shape(&shape, &axes, level, wvlt.width(), false);

    let n_in_total = shape.iter().product();

    let bc = PeriodicBoundary {};
    let trans = WaveletTransform::new(wvlt, bc);
    let trans_per = WaveletTransformPer::new(wvlt);

    let n_out_total = out_shape.iter().product();

    let x = (0..n_in_total).map(|i| i as f64).collect::<Vec<_>>();
    let mut sd_per = vec![0.0; n_in_total];
    let mut sd_bc = vec![0.0; n_out_total];

    trans_per.forward_nd(&x, &mut sd_per, &shape, &axes);
    trans.forward_nd(&x, &mut sd_bc, &shape, &axes);

    let n_ax: usize = shape[axes[0]];
    let n_ax_out = out_shape[axes[0]];
    let n_sd = get_outlen(wvlt.width(), n_ax);
    let ns = (n_ax + 1) / 2;
    let nd = n_ax / 2;
    sd_per
        .chunks_exact(n_ax)
        .zip(sd_bc.chunks_exact(n_ax_out))
        .for_each(|(sd_per, sd_bc)| {
            let (s, d) = sd_per.split_at(ns);
            let (s2, d2) = sd_bc.split_at(n_sd);

            ndwt::tests::test_approx_equal(&s[1..nd - 1], &s2[2..nd], 1E-15, 0.0);
            ndwt::tests::test_approx_equal(&d[1..nd - 1], &d2[2..nd], 1E-15, 0.0);
        })
}

#[test]
pub fn test_dwt_driver_db2_per_single_level_1d_across() {
    let shape = [30, 35];
    let axes = [0];
    let level = 1;
    let wvlt = Wavelet::Daubechies2;

    let out_shape = get_transform_shape(&shape, &axes, level, wvlt.width(), false);

    let n_in_total = shape.iter().product();

    let bc = PeriodicBoundary {};
    let trans = WaveletTransform::new(wvlt, bc);
    let trans_per = WaveletTransformPer::new(wvlt);

    let n_out_total = out_shape.iter().product();

    let x = (0..n_in_total).map(|i| i as f64).collect::<Vec<_>>();
    let mut sd_per = vec![0.0; n_in_total];
    let mut sd_bc = vec![0.0; n_out_total];

    trans_per.forward_nd(&x, &mut sd_per, &shape, &axes);
    trans.forward_nd(&x, &mut sd_bc, &shape, &axes);

    let n_ax: usize = shape[axes[0]];
    let n_ax_out = out_shape[axes[0]];
    let n_sd = get_outlen(wvlt.width(), n_ax);
    let ns = (n_ax + 1) / 2;

    let mut sd1 = vec![0.0; n_ax];
    let mut sd2 = vec![0.0; n_ax_out];
    sd_per
        .iter_lanes(&shape, axes[0])
        .zip(sd_bc.iter_lanes(&out_shape, axes[0]))
        .for_each(|(sd_per, sd_bc)| {
            sd_per.pour_into(&mut sd1);
            sd_bc.pour_into(&mut sd2);
            let (s, d) = sd1.split_at(ns);
            let (s2, d2) = sd2.split_at(n_sd);

            ndwt::tests::test_approx_equal(&s[..ns], &s2[1..ns + 1], 1E-15, 0.0);
            ndwt::tests::test_approx_equal(&d[..ns], &d2[1..ns + 1], 1E-15, 0.0);
        })
}

#[test]
pub fn test_dwt_driver_db2_per_single_2d() {
    let shape = [30, 35];
    let axes = [0, 1];
    let wvlt = Wavelet::Daubechies2;

    let n_total = shape.iter().product();

    let trans = WaveletTransformPer::new(wvlt);

    let x = (0..n_total)
        .map(|i| ((i + 1) * (i + 1)) as f64 / 3.0)
        .collect::<Vec<_>>();
    let mut x_w = vec![0.0; n_total];

    trans.forward_nd(&x, &mut x_w, &shape, &axes);

    let mut x_inter = vec![0.0; n_total];
    let mut x_w2 = vec![0.0; n_total];
    trans.forward_nd(&x, &mut x_inter, &shape, &[axes[0]]);
    trans.forward_nd(&x_inter, &mut x_w2, &shape, &[axes[1]]);

    ndwt::tests::test_approx_equal(&x_w, &x_w2, 1E-15, 1E-10);
}

#[test]
pub fn test_dwt_driver_db2_per_multi_level_1d() {
    let shape = [30, 35];
    let axes = [1];
    let level = 3;
    let wvlt = Wavelet::Daubechies2;

    let n_total = shape.iter().product();

    let trans = WaveletTransformPer::new(wvlt);

    let x = (1..n_total + 1).map(|i| (i * i) as f64).collect::<Vec<_>>();
    let mut sd = vec![0.0; n_total];

    trans.forward_multilevel_nd(&x, &mut sd, &shape, &axes, level);

    let mut sd2 = vec![0.0; n_total];

    let n_ax = shape[1];
    x.chunks_exact(n_ax)
        .zip(sd2.chunks_exact_mut(n_ax))
        .for_each(|(x, sd)| {
            let ns = (n_ax + 1) / 2;
            let nd = n_ax / 2;
            let (s, d) = sd.split_at_mut(ns);
            assert_eq!(s.len(), ns);
            assert_eq!(d.len(), nd);
            ndwt::dwt::daubechies::Daubechies2::forward_per(x, s, d);

            let n_ax = ns;
            let ns = (n_ax + 1) / 2;
            let nd = n_ax / 2;
            let x = s.to_owned();

            let (s, d) = s.split_at_mut(ns);
            assert_eq!(s.len(), ns);
            assert_eq!(d.len(), nd);
            ndwt::dwt::daubechies::Daubechies2::forward_per(&x, s, d);

            let n_ax = ns;
            let ns = (n_ax + 1) / 2;
            let nd = n_ax / 2;
            let x = s.to_owned();

            let (s, d) = s.split_at_mut(ns);
            assert_eq!(s.len(), ns);
            assert_eq!(d.len(), nd);
            ndwt::dwt::daubechies::Daubechies2::forward_per(&x, s, d);
        });

    ndwt::tests::test_approx_equal(&sd, &sd2, 1E-12, 2E-10);
}

#[test]
pub fn test_dwt_driver_db2_per_multi_level_2d() {
    let shape = [30, 35];
    let axes = [1, 0];
    let level = 3;
    let wvlt = Wavelet::Daubechies2;

    let n_total = shape.iter().product();
    let trans = WaveletTransformPer::new(wvlt);

    let x = (0..n_total)
        .map(|i| ((i + 1) * (i + 1)) as f64)
        .collect::<Vec<_>>();

    let mut sd = vec![0.0; n_total];

    trans.forward_multilevel_nd(&x, &mut sd, &shape, &axes, level);

    let mut sd_p1 = vec![0.0; n_total];
    trans.forward_nd(&x, &mut sd_p1, &shape, &axes);

    let ns0 = (shape[0] + 1) / 2;
    let ns1 = (shape[1] + 1) / 2;
    let s = sd_p1
        .chunks_exact(shape[1])
        .take(ns0)
        .map(|sd| &sd[..ns1])
        .flatten()
        .cloned()
        .collect::<Vec<_>>();

    let shape2 = [ns0, ns1];
    let n_total2 = shape2.iter().product();
    let mut sd_p2 = vec![0.0; n_total2];

    trans.forward_nd(&s, &mut sd_p2, &shape2, &axes);

    let ns0 = (shape2[0] + 1) / 2;
    let ns1 = (shape2[1] + 1) / 2;
    let s = sd_p2
        .chunks_exact(shape2[1])
        .take(ns0)
        .map(|sd| &sd[..ns1])
        .flatten()
        .cloned()
        .collect::<Vec<_>>();

    let shape3 = [ns0, ns1];
    let n_total3 = shape3.iter().product();
    let mut sd_p3 = vec![0.0; n_total3];

    trans.forward_nd(&s, &mut sd_p3, &shape3, &axes);

    let mut sd2 = sd_p1.clone();

    sd2.chunks_exact_mut(shape[1])
        .zip(sd_p2.chunks_exact(shape2[1]))
        .for_each(|(a, b)| {
            a.iter_mut().zip(b).for_each(|(a, b)| *a = *b);
        });

    sd2.chunks_exact_mut(shape[1])
        .zip(sd_p3.chunks_exact(shape3[1]))
        .for_each(|(a, b)| {
            a.iter_mut().zip(b).for_each(|(a, b)| *a = *b);
        });

    ndwt::tests::test_approx_equal(&sd, &sd2, 1E-12, 1E-9);
}

#[test]
pub fn test_dwt_driver_inv_db2_per_single_level_1d_along() {
    let shape = [30, 35];
    let axes = [1];
    let wvlt = Wavelet::Daubechies2;

    let n_total = shape.iter().product();

    let trans = WaveletTransformPer::new(wvlt);

    let x = (0..n_total).map(|i| (i + 1) as f64).collect::<Vec<_>>();
    let mut sd = vec![0.0; n_total];

    trans.forward_nd(&x, &mut sd, &shape, &axes);

    let mut x2 = vec![0.0; n_total];

    trans.inverse_nd(&mut sd, &mut x2, &shape, &axes);

    ndwt::tests::test_approx_equal(&x2, &x, 1E-15, 0.0);
}

#[test]
pub fn test_dwt_driver_inv_db2_per_single_level_1d_across() {
    let shape = [30, 35];
    let axes = [0];
    let wvlt = Wavelet::Daubechies2;

    let n_total = shape.iter().product();

    let trans = WaveletTransformPer::new(wvlt);

    let x = (0..n_total).map(|i| (i + 1) as f64).collect::<Vec<_>>();
    let mut sd = vec![0.0; n_total];

    trans.forward_nd(&x, &mut sd, &shape, &axes);

    let mut x2 = vec![0.0; n_total];

    trans.inverse_nd(&mut sd, &mut x2, &shape, &axes);

    ndwt::tests::test_approx_equal(&x2, &x, 1E-12, 1E-13);
}

#[test]
pub fn test_dwt_driver_inv_db2_per_single_level_2d() {
    let shape = [30, 35];
    let axes = [0, 1];
    let wvlt = Wavelet::Daubechies2;

    let n_total = shape.iter().product();

    let trans = WaveletTransformPer::new(wvlt);

    let x = (0..n_total).map(|i| (i + 1) as f64).collect::<Vec<_>>();
    let mut sd = vec![0.0; n_total];

    trans.forward_nd(&x, &mut sd, &shape, &axes);

    let mut x2 = vec![0.0; n_total];

    trans.inverse_nd(&mut sd, &mut x2, &shape, &axes);

    ndwt::tests::test_approx_equal(&x2, &x, 1E-13, 1E-14);
}

#[test]
pub fn test_dwt_driver_inv_db2_per_multi_level_1d() {
    let shape = [30, 35];
    let axes = [1];
    let level = 3;
    let wvlt = Wavelet::Daubechies2;

    let n_total = shape.iter().product();

    let trans = WaveletTransformPer::new(wvlt);

    let x = (0..n_total).map(|i| (i + 1) as f64).collect::<Vec<_>>();
    let mut sd = vec![0.0; n_total];

    trans.forward_multilevel_nd(&x, &mut sd, &shape, &axes, level);

    let mut x2 = vec![0.0; n_total];

    trans.inverse_multilevel_nd(&mut sd, &mut x2, &shape, &axes, level);

    ndwt::tests::test_approx_equal(&x2, &x, 1E-15, 0.0);
}

#[test]
pub fn test_dwt_driver_inv_db2_per_multi_level_2d() {
    let shape = [30, 35];
    let axes = [0, 1];
    let level = 3;
    let wvlt = Wavelet::Daubechies2;

    let n_total = shape.iter().product();

    let trans = WaveletTransformPer::new(wvlt);

    let x = (0..n_total).map(|i| (i + 1) as f64).collect::<Vec<_>>();
    let mut sd = vec![0.0; n_total];

    trans.forward_multilevel_nd(&x, &mut sd, &shape, &axes, level);

    let mut x2 = vec![0.0; n_total];

    trans.inverse_multilevel_nd(&mut sd, &mut x2, &shape, &axes, level);

    ndwt::tests::test_approx_equal(&x2, &x, 1E-12, 1E-10);
}

// ── Parametrised wavelet-family tests ──────────────────────────────────────────

/// Forward + inverse round-trip for the non-periodic DWT.
///
/// For all wavelet families and levels 1 and 3, checks that
/// `inverse(forward(x)) ≈ x` on a 2-D signal with the default zero boundary.
#[rstest]
#[case(Wavelet::Daubechies4, 1)]
#[case(Wavelet::Daubechies4, 3)]
#[case(Wavelet::Symlet4, 1)]
#[case(Wavelet::Symlet4, 3)]
#[case(Wavelet::Coiflet2, 1)]
#[case(Wavelet::Coiflet2, 3)]
#[case(Wavelet::Bior2_2, 1)]
#[case(Wavelet::Bior2_2, 3)]
#[case(Wavelet::CDF9_7, 1)]
#[case(Wavelet::CDF9_7, 3)]
pub fn test_round_trip_wavelet_families(#[case] wvlt: Wavelet, #[case] level: usize) {
    let shape = [30, 35];
    let axes = [1];
    let bc = ZeroBoundary {};
    let trans = WaveletTransform::new(wvlt, bc);

    let out_shape = get_transform_shape(&shape, &axes, level, wvlt.width(), false);
    let n_in = shape.iter().product::<usize>();
    let n_out = out_shape.iter().product::<usize>();

    let x: Vec<f64> = (0..n_in).map(|i| (i as f64 + 1.0) * 0.37).collect();
    let mut sd = vec![0.0f64; n_out];
    trans.forward_multilevel_nd(&x, &mut sd, &shape, &axes, level);

    let mut x2 = vec![0.0f64; n_in];
    trans.inverse_multilevel_nd(&mut sd, &mut x2, &shape, &axes, level);

    ndwt::tests::test_approx_equal(&x2, &x, 1e-12, 1e-10);
}

/// Adjoint property: `<v, T(u)> ≈ <T*(v), u>` where `T` is the forward DWT and
/// `T*` is `adj_forward`.
#[rstest]
#[case(Wavelet::Daubechies4, 1)]
#[case(Wavelet::Daubechies4, 3)]
#[case(Wavelet::Symlet4, 1)]
#[case(Wavelet::Coiflet2, 1)]
#[case(Wavelet::Bior2_2, 1)]
#[case(Wavelet::CDF9_7, 1)]
pub fn test_adj_forward_wavelet_families(#[case] wvlt: Wavelet, #[case] level: usize) {
    let n_signal: usize = 64;
    let shape = [n_signal];
    let axes = [0];
    let bc = ZeroBoundary {};
    let trans = WaveletTransform::new(wvlt, bc);

    let out_shape = get_transform_shape(&shape, &axes, level, wvlt.width(), false);
    let n_in = n_signal;
    let n_out = out_shape.iter().product::<usize>();

    let u: Vec<f64> = (0..n_in).map(|i| (i as f64 * 0.31 + 1.0).sin()).collect();
    let v: Vec<f64> = (0..n_out).map(|i| (i as f64 * 0.17 + 0.5).cos()).collect();

    ndwt::tests::test_approx_adjoint(
        |u, fu| trans.forward_multilevel_nd(u, fu, &shape, &axes, level),
        |v, atv| {
            let mut tmp = v.to_owned();
            trans.adj_forward_multilevel_nd(&mut tmp, atv, &shape, &axes, level)
        },
        &u,
        &v,
        1e-12,
        1e-10,
    );
}

/// Round-trip test using `Complex32` element type.
#[test]
pub fn test_round_trip_complex32() {
    let wvlt = Wavelet::Daubechies4;
    let shape = [32];
    let axes = [0];
    let level = 2;
    let bc = ZeroBoundary {};

    let out_shape = get_transform_shape(&shape, &axes, level, wvlt.width(), false);
    let n_in = shape.iter().product::<usize>();
    let n_out = out_shape.iter().product::<usize>();

    let x: Vec<Complex32> = (0..n_in)
        .map(|i| Complex32::new(i as f32 * 0.5, -(i as f32) * 0.3))
        .collect();
    let mut sd = vec![Complex32::new(0.0, 0.0); n_out];

    // Let the compiler infer the correct SIMD chunk width N for Complex32.
    let trans = WaveletTransform::new(wvlt, bc);
    trans.forward_multilevel_nd(&x, &mut sd, &shape, &axes, level);

    let mut x2 = vec![Complex32::new(0.0, 0.0); n_in];
    trans.inverse_multilevel_nd(&mut sd, &mut x2, &shape, &axes, level);

    let x_re: Vec<f32> = x.iter().map(|c| c.re).collect();
    let x2_re: Vec<f32> = x2.iter().map(|c| c.re).collect();
    let x_im: Vec<f32> = x.iter().map(|c| c.im).collect();
    let x2_im: Vec<f32> = x2.iter().map(|c| c.im).collect();
    ndwt::tests::test_approx_equal(&x2_re, &x_re, 1e-5, 1e-4);
    ndwt::tests::test_approx_equal(&x2_im, &x_im, 1e-5, 1e-4);
}

/// Round-trip test using `Complex64` element type.
#[test]
pub fn test_round_trip_complex64() {
    let wvlt = Wavelet::Daubechies4;
    let shape = [32];
    let axes = [0];
    let level = 2;
    let bc = ZeroBoundary {};

    let out_shape = get_transform_shape(&shape, &axes, level, wvlt.width(), false);
    let n_in = shape.iter().product::<usize>();
    let n_out = out_shape.iter().product::<usize>();

    let x: Vec<Complex64> = (0..n_in)
        .map(|i| Complex64::new(i as f64 * 0.5, -(i as f64) * 0.3))
        .collect();
    let mut sd = vec![Complex64::new(0.0, 0.0); n_out];

    // Let the compiler infer the correct SIMD chunk width N for Complex64.
    let trans = WaveletTransform::new(wvlt, bc);
    trans.forward_multilevel_nd(&x, &mut sd, &shape, &axes, level);

    let mut x2 = vec![Complex64::new(0.0, 0.0); n_in];
    trans.inverse_multilevel_nd(&mut sd, &mut x2, &shape, &axes, level);

    let x_re: Vec<f64> = x.iter().map(|c| c.re).collect();
    let x2_re: Vec<f64> = x2.iter().map(|c| c.re).collect();
    let x_im: Vec<f64> = x.iter().map(|c| c.im).collect();
    let x2_im: Vec<f64> = x2.iter().map(|c| c.im).collect();
    ndwt::tests::test_approx_equal(&x2_re, &x_re, 1e-12, 1e-10);
    ndwt::tests::test_approx_equal(&x2_im, &x_im, 1e-12, 1e-10);
}
