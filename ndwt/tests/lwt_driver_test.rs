use ndwt::Wavelet;
use ndwt::boundarys::BoundaryCondition;
use ndwt::iter::LanesIterator;
use ndwt::lwt::driver::WaveletTransform;
use num_complex::{Complex32, Complex64};
use rstest::rstest;

// ── Single-level forward: compare against hand-rolled 1-D transforms ───────────

/// Transform along axis 1 of a 2-D array and verify each row matches a
/// direct `forward_1d` call.
#[test]
pub fn test_lwt_driver_db2_single_level_1d_along() {
    let shape = [30, 35];
    let axes = [1];
    let wvlt = Wavelet::Daubechies2;
    let bc = BoundaryCondition::Periodic;
    let trans = WaveletTransform::new(wvlt, bc);

    let n_ax = shape[axes[0]]; // 35
    let ns = (n_ax + 1) / 2; // 18
    let nd = n_ax / 2; // 17

    let n_total = shape.iter().product();
    let x: Vec<f64> = (0..n_total).map(|i| i as f64).collect();

    let mut sd = vec![0.0f64; n_total];
    trans.forward_nd(&x, &mut sd, &shape, &axes);

    // Hand-roll: apply forward_1d to each row.
    let mut sd2 = vec![0.0f64; n_total];
    x.chunks_exact(shape[1])
        .zip(sd2.chunks_exact_mut(shape[1]))
        .for_each(|(row, out_row)| {
            let (s_out, d_out) = out_row.split_at_mut(ns);
            trans.forward_1d(row, s_out, d_out);
        });

    assert_eq!(ns + nd, shape[1]);
    ndwt::tests::test_approx_equal(&sd, &sd2, 1e-12, 1e-11);
}

/// Transform along axis 0 of a 2-D array and verify each column matches a
/// direct `forward_1d` call.
#[test]
pub fn test_lwt_driver_db2_single_level_1d_across() {
    let shape = [30, 35];
    let axes = [0];
    let wvlt = Wavelet::Daubechies2;
    let bc = BoundaryCondition::Periodic;
    let trans = WaveletTransform::new(wvlt, bc);

    let n_ax = shape[axes[0]]; // 30
    let ns = (n_ax + 1) / 2; // 15
    let nd = n_ax / 2; // 15

    let n_total = shape.iter().product();
    let x: Vec<f64> = (0..n_total).map(|i| i as f64).collect();

    let mut sd = vec![0.0f64; n_total];
    trans.forward_nd(&x, &mut sd, &shape, &axes);

    // Hand-roll: extract each column (strided), apply forward_1d, write back.
    let mut sd2 = vec![0.0f64; n_total];
    let mut x_w = vec![0.0f64; n_ax];
    let mut s_w = vec![0.0f64; ns];
    let mut d_w = vec![0.0f64; nd];
    x.iter_lanes(&shape, axes[0])
        .zip(sd2.iter_lanes_mut(&shape, axes[0]))
        .for_each(|(x_lane, mut sd_lane)| {
            x_lane.pour_into(&mut x_w);
            trans.forward_1d(&x_w, &mut s_w, &mut d_w);
            sd_lane.stack(&s_w, &d_w);
        });

    ndwt::tests::test_approx_equal(&sd, &sd2, 1e-12, 1e-11);
}

/// Two-axis single-level transform equals two sequential single-axis transforms.
#[test]
pub fn test_lwt_driver_db2_single_2d() {
    let shape = [30, 35];
    let axes = [0, 1];
    let wvlt = Wavelet::Daubechies2;
    let bc = BoundaryCondition::Periodic;
    let trans = WaveletTransform::new(wvlt, bc);

    let n_total = shape.iter().product();
    let x: Vec<f64> = (0..n_total).map(|i| (i * i) as f64 / 3.0).collect();

    let mut sd = vec![0.0f64; n_total];
    trans.forward_nd(&x, &mut sd, &shape, &axes);

    // Sequential: axis 0 then axis 1.
    let mut inter = vec![0.0f64; n_total];
    let mut sd2 = vec![0.0f64; n_total];
    trans.forward_nd(&x, &mut inter, &shape, &[axes[0]]);
    trans.forward_nd(&inter, &mut sd2, &shape, &[axes[1]]);

    ndwt::tests::test_approx_equal(&sd, &sd2, 1e-12, 1e-10);
}

/// Three-level forward matches three successive single-level calls.
#[test]
pub fn test_lwt_driver_db2_multi_level_1d() {
    let shape = [30, 35];
    let axes = [1];
    let level = 3;
    let wvlt = Wavelet::Daubechies2;
    let bc = BoundaryCondition::Periodic;
    let trans = WaveletTransform::new(wvlt, bc);

    let n_ax = shape[axes[0]]; // 35
    let ns1 = (n_ax + 1) / 2; // 18
    let nd1 = n_ax / 2; // 17
    let ns2 = (ns1 + 1) / 2; // 9
    let nd2 = ns1 / 2; // 9
    let ns3 = (ns2 + 1) / 2; // 5
    let nd3 = ns2 / 2; // 4

    let n_total = shape.iter().product();
    let x: Vec<f64> = (0..n_total).map(|i| i as f64).collect();

    let mut sd = vec![0.0f64; n_total];
    trans.forward_multilevel_nd(&x, &mut sd, &shape, &axes, level);

    // Hand-roll three nested forward_1d calls per row, then assemble the
    // packed layout [s3 | d3 | d2 | d1].
    let mut sd2 = vec![0.0f64; n_total];
    x.chunks_exact(shape[1])
        .zip(sd2.chunks_exact_mut(shape[1]))
        .for_each(|(row, out)| {
            let mut s1 = vec![0.0f64; ns1];
            let mut d1 = vec![0.0f64; nd1];
            trans.forward_1d(row, &mut s1, &mut d1);

            let mut s2 = vec![0.0f64; ns2];
            let mut d2 = vec![0.0f64; nd2];
            trans.forward_1d(&s1, &mut s2, &mut d2);

            let mut s3 = vec![0.0f64; ns3];
            let mut d3 = vec![0.0f64; nd3];
            trans.forward_1d(&s2, &mut s3, &mut d3);

            // Packed layout: [s3 | d3 | d2 | d1]
            out[..ns3].copy_from_slice(&s3);
            out[ns3..ns3 + nd3].copy_from_slice(&d3);
            out[ns3 + nd3..ns3 + nd3 + nd2].copy_from_slice(&d2);
            out[ns3 + nd3 + nd2..].copy_from_slice(&d1);
        });

    ndwt::tests::test_approx_equal(&sd, &sd2, 1e-12, 1e-11);
}

// ── Inverse round-trip tests ───────────────────────────────────────────────────

#[test]
pub fn test_lwt_driver_inv_db2_single_level_1d_along() {
    let shape = [30, 35];
    let axes = [1];
    let wvlt = Wavelet::Daubechies2;
    let bc = BoundaryCondition::Periodic;
    let trans = WaveletTransform::new(wvlt, bc);

    let n_total = shape.iter().product();
    let x: Vec<f64> = (0..n_total).map(|i| (i + 1) as f64).collect();

    let mut sd = vec![0.0f64; n_total];
    trans.forward_nd(&x, &mut sd, &shape, &axes);

    let mut x2 = vec![0.0f64; n_total];
    trans.inverse_nd(&sd, &mut x2, &shape, &axes);

    ndwt::tests::test_approx_equal(&x2, &x, 1e-12, 1e-11);
}

#[test]
pub fn test_lwt_driver_inv_db2_single_level_1d_across() {
    let shape = [30, 35];
    let axes = [0];
    let wvlt = Wavelet::Daubechies2;
    let bc = BoundaryCondition::Periodic;
    let trans = WaveletTransform::new(wvlt, bc);

    let n_total = shape.iter().product();
    let x: Vec<f64> = (0..n_total).map(|i| (i + 1) as f64).collect();

    let mut sd = vec![0.0f64; n_total];
    trans.forward_nd(&x, &mut sd, &shape, &axes);

    let mut x2 = vec![0.0f64; n_total];
    trans.inverse_nd(&sd, &mut x2, &shape, &axes);

    ndwt::tests::test_approx_equal(&x2, &x, 1e-12, 1e-11);
}

#[test]
pub fn test_lwt_driver_inv_db2_single_level_2d() {
    let shape = [30, 35];
    let axes = [0, 1];
    let wvlt = Wavelet::Daubechies2;
    let bc = BoundaryCondition::Periodic;
    let trans = WaveletTransform::new(wvlt, bc);

    let n_total = shape.iter().product();
    let x: Vec<f64> = (0..n_total).map(|i| (i + 1) as f64).collect();

    let mut sd = vec![0.0f64; n_total];
    trans.forward_nd(&x, &mut sd, &shape, &axes);

    let mut x2 = vec![0.0f64; n_total];
    trans.inverse_nd(&sd, &mut x2, &shape, &axes);

    ndwt::tests::test_approx_equal(&x2, &x, 1e-12, 1e-11);
}

#[test]
pub fn test_lwt_driver_inv_db2_multi_level_1d() {
    let shape = [30, 35];
    let axes = [1];
    let level = 3;
    let wvlt = Wavelet::Daubechies2;
    let bc = BoundaryCondition::Periodic;
    let trans = WaveletTransform::new(wvlt, bc);

    let n_total = shape.iter().product();
    let x: Vec<f64> = (0..n_total).map(|i| (i + 1) as f64).collect();

    let mut sd = vec![0.0f64; n_total];
    trans.forward_multilevel_nd(&x, &mut sd, &shape, &axes, level);

    let mut x2 = vec![0.0f64; n_total];
    trans.inverse_multilevel_nd(&sd, &mut x2, &shape, &axes, level);

    ndwt::tests::test_approx_equal(&x2, &x, 1e-12, 1e-11);
}

#[test]
pub fn test_lwt_driver_inv_db2_multi_level_2d() {
    let shape = [30, 35];
    let axes = [0, 1];
    let level = 3;
    let wvlt = Wavelet::Daubechies2;
    let bc = BoundaryCondition::Periodic;
    let trans = WaveletTransform::new(wvlt, bc);

    let n_total = shape.iter().product();
    let x: Vec<f64> = (0..n_total).map(|i| (i + 1) as f64).collect();

    let mut sd = vec![0.0f64; n_total];
    trans.forward_multilevel_nd(&x, &mut sd, &shape, &axes, level);

    let mut x2 = vec![0.0f64; n_total];
    trans.inverse_multilevel_nd(&sd, &mut x2, &shape, &axes, level);

    ndwt::tests::test_approx_equal(&x2, &x, 1e-12, 1e-11);
}

// ── Zero boundary condition round-trips ───────────────────────────────────────

#[test]
pub fn test_lwt_driver_inv_db2_zero_bc_multi_level_1d() {
    use ndwt::boundarys::ZeroBoundary;

    let shape = [30, 35];
    let axes = [1];
    let level = 3;
    let wvlt = Wavelet::Daubechies2;
    let trans = WaveletTransform::new(wvlt, ZeroBoundary {});

    let n_total = shape.iter().product();
    let x: Vec<f64> = (0..n_total).map(|i| (i + 1) as f64).collect();

    let mut sd = vec![0.0f64; n_total];
    trans.forward_multilevel_nd(&x, &mut sd, &shape, &axes, level);

    let mut x2 = vec![0.0f64; n_total];
    trans.inverse_multilevel_nd(&sd, &mut x2, &shape, &axes, level);

    ndwt::tests::test_approx_equal(&x2, &x, 1e-12, 1e-11);
}

// ── Parametrised wavelet-family tests ─────────────────────────────────────────

/// Forward + inverse round-trip for a variety of wavelet families and levels.
///
/// Uses periodic boundary so that all wavelets have an exact inverse regardless
/// of signal length.
#[rstest]
#[case(Wavelet::Daubechies2, 1)]
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
    let bc = BoundaryCondition::Periodic;
    let trans = WaveletTransform::new(wvlt, bc);

    let n_total = shape.iter().product();
    let x: Vec<f64> = (0..n_total).map(|i| (i as f64 + 1.0) * 0.37).collect();

    let mut sd = vec![0.0f64; n_total];
    trans.forward_multilevel_nd(&x, &mut sd, &shape, &axes, level);

    let mut x2 = vec![0.0f64; n_total];
    trans.inverse_multilevel_nd(&sd, &mut x2, &shape, &axes, level);

    ndwt::tests::test_approx_equal(&x2, &x, 1e-12, 1e-10);
}

/// Adjoint property: `<v, T(u)> = <T*(v), u>` where `T` is the forward LWT
/// and `T*` is `adj_forward`.
///
/// Both u and v live in the same space (LWT is shape-preserving).
#[rstest]
#[case(Wavelet::Daubechies2, 1)]
#[case(Wavelet::Daubechies4, 1)]
#[case(Wavelet::Daubechies4, 3)]
#[case(Wavelet::Symlet4, 1)]
#[case(Wavelet::Coiflet2, 1)]
#[case(Wavelet::Bior2_2, 1)]
#[case(Wavelet::CDF9_7, 1)]
pub fn test_adj_forward_wavelet_families(#[case] wvlt: Wavelet, #[case] level: usize) {
    let n: usize = 64;
    let shape = [n];
    let axes = [0];
    let bc = BoundaryCondition::Periodic;
    let trans = WaveletTransform::new(wvlt, bc);

    let u: Vec<f64> = (0..n).map(|i| (i as f64 * 0.31 + 1.0).sin()).collect();
    let v: Vec<f64> = (0..n).map(|i| (i as f64 * 0.17 + 0.5).cos()).collect();

    ndwt::tests::test_approx_adjoint(
        |u, fu| trans.forward_multilevel_nd(u, fu, &shape, &axes, level),
        |v, atv| trans.adj_forward_multilevel_nd(v, atv, &shape, &axes, level),
        &u,
        &v,
        1e-12,
        1e-10,
    );
}

/// Adjoint property: `<v, T^{-1}(u)> = <(T^{-1})*(v), u>` where `T^{-1}` is
/// the inverse LWT and `(T^{-1})*` is `adj_inverse`.
#[rstest]
#[case(Wavelet::Daubechies2, 1)]
#[case(Wavelet::Daubechies4, 1)]
#[case(Wavelet::Daubechies4, 3)]
#[case(Wavelet::Symlet4, 1)]
#[case(Wavelet::Coiflet2, 1)]
#[case(Wavelet::Bior2_2, 1)]
#[case(Wavelet::CDF9_7, 1)]
pub fn test_adj_inverse_wavelet_families(#[case] wvlt: Wavelet, #[case] level: usize) {
    let n: usize = 64;
    let shape = [n];
    let axes = [0];
    let bc = BoundaryCondition::Periodic;
    let trans = WaveletTransform::new(wvlt, bc);

    let u: Vec<f64> = (0..n).map(|i| (i as f64 * 0.31 + 1.0).sin()).collect();
    let v: Vec<f64> = (0..n).map(|i| (i as f64 * 0.17 + 0.5).cos()).collect();

    ndwt::tests::test_approx_adjoint(
        |u, fu| trans.inverse_multilevel_nd(u, fu, &shape, &axes, level),
        |v, atv| trans.adj_inverse_multilevel_nd(v, atv, &shape, &axes, level),
        &u,
        &v,
        1e-12,
        1e-10,
    );
}

// ── Complex element-type round-trips ──────────────────────────────────────────
// LWT uses SimdTransformable which handles complex types by treating f32*C32
// and f64*C64 — the scalar multiplier is broadcast via f32s/f64s SIMD vectors,
// and the complex data rides in c32s/c64s SIMD vectors of the same byte width.

/// Round-trip test using `Complex32` element type.
#[test]
pub fn test_round_trip_complex32() {
    let wvlt = Wavelet::Daubechies4;
    let shape = [32];
    let axes = [0];
    let level = 2;
    let bc = BoundaryCondition::Periodic;

    let n = shape.iter().product::<usize>();
    let x: Vec<Complex32> = (0..n)
        .map(|i| Complex32::new(i as f32 * 0.5, -(i as f32) * 0.3))
        .collect();
    let mut sd = vec![Complex32::new(0.0, 0.0); n];

    // Let the compiler infer T=Complex32 and N=N_C32 from usage.
    let trans = WaveletTransform::new(wvlt, bc);
    trans.forward_multilevel_nd(&x, &mut sd, &shape, &axes, level);

    let mut x2 = vec![Complex32::new(0.0, 0.0); n];
    trans.inverse_multilevel_nd(&sd, &mut x2, &shape, &axes, level);

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
    let bc = BoundaryCondition::Periodic;

    let n = shape.iter().product::<usize>();
    let x: Vec<Complex64> = (0..n)
        .map(|i| Complex64::new(i as f64 * 0.5, -(i as f64) * 0.3))
        .collect();
    let mut sd = vec![Complex64::new(0.0, 0.0); n];

    // Let the compiler infer T=Complex64 and N=N_C64 from usage.
    let trans = WaveletTransform::new(wvlt, bc);
    trans.forward_multilevel_nd(&x, &mut sd, &shape, &axes, level);

    let mut x2 = vec![Complex64::new(0.0, 0.0); n];
    trans.inverse_multilevel_nd(&sd, &mut x2, &shape, &axes, level);

    let x_re: Vec<f64> = x.iter().map(|c| c.re).collect();
    let x2_re: Vec<f64> = x2.iter().map(|c| c.re).collect();
    let x_im: Vec<f64> = x.iter().map(|c| c.im).collect();
    let x2_im: Vec<f64> = x2.iter().map(|c| c.im).collect();
    ndwt::tests::test_approx_equal(&x2_re, &x_re, 1e-12, 1e-10);
    ndwt::tests::test_approx_equal(&x2_im, &x_im, 1e-12, 1e-10);
}
