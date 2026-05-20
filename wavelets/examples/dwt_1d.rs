//! Forward + inverse non-periodic DWT of a 128-sample signal.
//!
//! Run with: `cargo run --example dwt_1d`

use wavelets::Wavelets;
use wavelets::boundarys::ZeroBoundary;
use wavelets::dwt::driver::{WaveletTransform, get_transform_shape};

fn main() {
    let wvlt = Wavelets::Daubechies4;
    let level = 3;
    let shape = [128_usize];
    let axes = [0_usize];
    let bc = ZeroBoundary {};

    // Construct the driver (N is inferred from the f64 ChunkWidth impl).
    let trans = WaveletTransform::new(wvlt, bc);

    // A simple ramp signal.
    let x: Vec<f64> = (0..128).map(|i| i as f64 / 127.0).collect();

    // Forward transform — output is larger than input because DWT pads each level.
    let out_shape = get_transform_shape(&shape, &axes, level, wvlt.width(), false);
    let mut coeffs = vec![0.0_f64; out_shape.iter().product()];
    trans.forward_multilevel_nd(&x, &mut coeffs, &shape, &axes, level);

    // The approximation sub-band after `level` levels sits at the front.
    // Its length shrinks by roughly half each level.
    let approx_len = out_shape[0] - shape[0]; // detail coefficients fill the rest
    println!("Signal length   : {}", shape[0]);
    println!("Coefficient len : {}", out_shape[0]);
    println!("Approx sub-band : first {} coefficients", approx_len);

    // Inverse transform — reconstruct the original signal.
    let mut x2 = vec![0.0_f64; shape.iter().product()];
    trans.inverse_multilevel_nd(&mut coeffs, &mut x2, &shape, &axes, level);

    let max_err = x
        .iter()
        .zip(&x2)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!("Max round-trip error: {:.2e}", max_err);
    assert!(max_err < 1e-10, "round-trip error too large: {max_err}");
    println!("Round-trip OK.");
}
