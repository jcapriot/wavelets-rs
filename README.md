# ndwt

[![crates.io](https://img.shields.io/crates/v/ndwt.svg)](https://crates.io/crates/ndwt)
[![docs.rs](https://docs.rs/ndwt/badge.svg)](https://docs.rs/ndwt)
[![CI](https://github.com/jcapriot/wavelets-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/jcapriot/wavelets-rs/actions/workflows/ci.yml)

N-Dimensional Wavelet transforms for real and complex signals in Rust.

Provides two transform families:

- **DWT** — the classic Discrete Wavelet Transform via convolution and subsampling.
- **LWT** — the Lifting Wavelet Transform, an in-place factorisation of the DWT that is
  length-preserving and cache-friendly.

Both families support 1-D and N-D transforms on arbitrary axes, multi-level decomposition,
SIMD acceleration via [`pulp`](https://crates.io/crates/pulp), optional Rayon parallelism,
and real or complex element types. Adjoint (transpose) operations are provided for every
forward and inverse transform, respecting the chosen boundary extension, which makes the
transforms suitable for use in optimization and inverse problems.

## Installation

```toml
[dependencies]
ndwt = "0.1"
```

Default features enable `rayon` (multi-threading), `ndarray` integration, and `x86-v3`
(AVX2/FMA) SIMD paths. See [Feature flags](#feature-flags) for details.

## Quick start

### DWT — 1-D signal

```rust
use ndwt::Wavelet;
use ndwt::boundarys::ZeroBoundary;
use ndwt::dwt::driver::{WaveletTransform, get_transform_shape};

let wvlt = Wavelet::Daubechies4;
let level = 3;
let shape = [128_usize];
let axes = [0_usize];

let trans = WaveletTransform::new(wvlt, ZeroBoundary);

let x: Vec<f64> = (0..128).map(|i| i as f64 / 127.0).collect();

// Output is larger than input because non-periodic DWT pads at each level.
let out_shape = get_transform_shape(&shape, &axes, level, wvlt.width(), false);
let mut coeffs = vec![0.0_f64; out_shape.iter().product()];
trans.forward_multilevel_nd(&x, &mut coeffs, &shape, &axes, level);

// Reconstruct.
let mut x2 = vec![0.0_f64; 128];
trans.inverse_multilevel_nd(&mut coeffs, &mut x2, &shape, &axes, level);
```

### LWT — 2-D image (periodic boundary)

```rust
use ndwt::Wavelet;
use ndwt::boundarys::BoundaryCondition;
use ndwt::lwt::driver::WaveletTransform;

let wvlt = Wavelet::Daubechies4;
let level = 3;
let shape = [64_usize, 64];
let axes = [0_usize, 1]; // both axes

let trans = WaveletTransform::new(wvlt, BoundaryCondition::Periodic);

let x: Vec<f64> = (0..64 * 64).map(|i| ((i as f64) * 0.17).sin()).collect();

// LWT is length-preserving: output has the same shape as input.
let mut coeffs = vec![0.0_f64; 64 * 64];
trans.forward_multilevel_nd(&x, &mut coeffs, &shape, &axes, level);

let mut x2 = vec![0.0_f64; 64 * 64];
trans.inverse_multilevel_nd(&coeffs, &mut x2, &shape, &axes, level);
```

## Supported wavelets

| Family | Types | Module |
|--------|-------|--------|
| Daubechies | 1–10 | `ndwt::daubechies` |
| Symlet | 4–6 | `ndwt::symlet` |
| Coiflet | 1–3 | `ndwt::coiflet` |
| Biorthogonal | Bior1\_3 … Bior6\_8 | `ndwt::bior` |
| CDF (JPEG 2000) | CDF5\_3, CDF9\_7 | `ndwt::bior` |

Each wavelet type is a zero-size marker struct carrying filter coefficients as compile-time
constants. The [`Wavelet`](https://docs.rs/ndwt/latest/ndwt/enum.Wavelet.html) enum
lets you select a wavelet at runtime without generics.

## Boundary conditions

| Mode | Behaviour |
|------|-----------|
| `Zero` | Pads with zeros |
| `Periodic` | Wraps the signal |
| `Constant` | Extends the edge value |
| `Symmetric` | Mirror reflection at the boundary (edge repeated) |
| `Reflect` | Mirror reflection without repeating the edge sample |
| `Antisymmetric` | Antisymmetric reflection |
| `Smooth` | Linear extrapolation from the two edge samples |
| `Antireflect` | Antisymmetric reflect |

Compile-time boundary types (`ZeroBoundary`, `PeriodicBoundary`) are also available for
zero-cost dispatch. The `BoundaryCondition` enum enables runtime selection.

## Feature flags

| Flag | Default | Effect |
|------|---------|--------|
| `rayon` | enabled | Multi-threaded N-D transforms via Rayon |
| `ndarray` | enabled | `ndarray` integration |
| `x86-v3` | enabled | AVX2 / FMA SIMD paths (requires x86-64-v3 CPU) |
| `x86-v4` | — | AVX-512 SIMD paths (requires x86-64-v4 CPU) |

Disable default features to get a minimal build with no SIMD or parallelism:

```toml
ndwt = { version = "0.1", default-features = false }
```

## License

MIT — see [LICENSE.md](LICENSE.md).
