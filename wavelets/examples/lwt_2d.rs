//! Three-level multi-axis LWT on a 64×64 array using Daubechies4 with periodic boundary.
//!
//! Run with: `cargo run --example lwt_2d`

use wavelets::Wavelets;
use wavelets::boundarys::BoundaryCondition;
use wavelets::lwt::driver::WaveletTransform;

fn main() {
    let wvlt = Wavelets::Daubechies4;
    let level = 3;
    let rows = 64_usize;
    let cols = 64_usize;
    let shape = [rows, cols];
    let axes = [0_usize, 1]; // transform along both axes
    let bc = BoundaryCondition::Periodic;

    // Construct the LWT driver (N is inferred from the f64 ChunkWidth impl).
    let trans = WaveletTransform::new(wvlt, bc);

    // A 2-D test image stored as a row-major flat vec.
    let n = rows * cols;
    let x: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.17).sin()).collect();

    // The LWT is length-preserving: output has the same shape as input.
    let mut coeffs = vec![0.0_f64; n];
    trans.forward_multilevel_nd(&x, &mut coeffs, &shape, &axes, level);

    println!("Input shape  : {}×{}", rows, cols);
    println!(
        "Output shape : {}×{} (LWT is length-preserving)",
        rows, cols
    );
    println!("First 8 coefficients: {:?}", &coeffs[..8]);

    // Inverse transform.
    let mut x2 = vec![0.0_f64; n];
    trans.inverse_multilevel_nd(&coeffs, &mut x2, &shape, &axes, level);

    let max_err = x
        .iter()
        .zip(&x2)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!("Max round-trip error: {:.2e}", max_err);
    assert!(max_err < 1e-10, "round-trip error too large: {max_err}");
    println!("Round-trip OK.");
}
