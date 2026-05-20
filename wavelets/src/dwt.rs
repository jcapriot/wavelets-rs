//! Discrete Wavelet Transform (DWT) via direct convolution and subsampling.
//!
//! The DWT decomposes a 1-D signal `x` of length `n` into two sub-bands of roughly
//! half the length:
//!
//! - **approximation coefficients** `s` (low-pass) — computed via the analysis
//!   scaling filter `G`.
//! - **detail coefficients** `d` (high-pass) — computed via the analysis wavelet
//!   filter `H`.
//!
//! The inverse transform reconstructs `x` from `s` and `d` using the synthesis
//! filters `GI` and `HI`.
//!
//! # Output length
//!
//! For a non-periodic transform the output length depends on the filter width via
//! [`get_outlen`].  For the periodic transform (suffix `_per`) the signal must have
//! even length and the two sub-bands together equal the input length.
//!
//! # Sub-modules
//!
//! - [`driver`] — high-level [`driver::WaveletTransform`] for 1-D and N-D transforms.
//! - [`daubechies`], [`symlet`], [`coiflet`], [`bior`] — per-family coefficient tables.

use itertools::Itertools;
use num_traits::Zero;

use crate::Transformable;
use crate::boundarys::{BoundaryExtension, PeriodicBoundary, ZeroBoundary};

/// Biorthogonal wavelet coefficient tables.
pub mod bior;
/// Coiflet wavelet coefficient tables.
pub mod coiflet;
/// Daubechies wavelet coefficient tables.
pub mod daubechies;
/// High-level DWT driver: [`driver::WaveletTransform`] and [`driver::WaveletTransformPer`].
pub mod driver;
/// Symlet wavelet coefficient tables.
pub mod symlet;

/// Compile-time filter coefficients and default transform methods for a specific wavelet.
///
/// Implementors supply four coefficient arrays:
///
/// - `G` / `H` — analysis (forward) low-pass and high-pass filters.
/// - `GI` / `HI` — synthesis (inverse) low-pass and high-pass filters.
///
/// All default method implementations delegate to the free functions in this module
/// ([`dwt_forward`], [`dwt_inverse`], etc.) so implementors only need to supply the
/// four arrays.
pub trait DiscreteTransform<const N: usize, const NH: usize> {
    /// Analysis low-pass (scaling) filter coefficients.
    const G: [f64; N];
    /// Analysis high-pass (wavelet) filter coefficients.
    const H: [f64; N];
    /// Synthesis low-pass filter coefficients.
    const GI: [f64; N];
    /// Synthesis high-pass filter coefficients.
    const HI: [f64; N];

    /// Forward DWT: decompose `x` into approximation `s` and detail `d`.
    #[inline]
    fn forward<T: Transformable + Zero, BC: BoundaryExtension>(
        x: &[T],
        s: &mut [T],
        d: &mut [T],
        bc: &BC,
    ) {
        dwt_forward(&Self::G, &Self::H, x, s, d, bc);
    }

    /// Inverse DWT: reconstruct `x` from approximation `s` and detail `d`.
    #[inline]
    fn inverse<T: Transformable + Zero>(s: &[T], d: &[T], x: &mut [T]) {
        dwt_inverse::<_, _, NH>(&Self::GI, &Self::HI, s, d, x);
    }

    /// Adjoint (transpose) of the forward DWT.
    ///
    /// Uses [`dwt_adjoint_forward`] with the analysis filters and the supplied boundary
    /// condition, giving the exact mathematical transpose of [`forward`](Self::forward)
    /// for any `BC`.
    #[inline]
    fn adjoint_forward<T: Transformable + Zero, BC: BoundaryExtension>(
        s: &[T],
        d: &[T],
        x: &mut [T],
        bc: &BC,
    ) {
        dwt_adjoint_forward(&Self::G, &Self::H, s, d, x, bc);
    }

    /// Adjoint (transpose) of the inverse DWT.
    #[inline]
    fn adjoint_inverse<T: Transformable + Zero>(x: &[T], s: &mut [T], d: &mut [T]) {
        let ga: [_; N] = Self::GI.into_iter().rev().collect_array().unwrap();
        let ha: [_; N] = Self::HI.into_iter().rev().collect_array().unwrap();
        dwt_forward(&ga, &ha, x, s, d, &ZeroBoundary {});
    }

    /// Periodic forward DWT: decompose `x` with circular boundary conditions.
    #[inline]
    fn forward_per<T: Transformable + Zero>(x: &[T], s: &mut [T], d: &mut [T]) {
        dwt_per_forward(&Self::G, &Self::H, x, s, d);
    }

    /// Adjoint of the periodic forward DWT.
    #[inline]
    fn adjoint_forward_per<T: Transformable + Zero>(s: &[T], d: &[T], x: &mut [T]) {
        let ga: [_; N] = Self::G.into_iter().rev().collect_array().unwrap();
        let ha: [_; N] = Self::H.into_iter().rev().collect_array().unwrap();
        dwt_per_inverse::<_, _, NH>(&ga, &ha, s, d, x);
    }

    /// Periodic inverse DWT: reconstruct `x` with circular boundary conditions.
    #[inline]
    fn inverse_per<T: Transformable + Zero>(s: &[T], d: &[T], x: &mut [T]) {
        dwt_per_inverse::<_, _, NH>(&Self::GI, &Self::HI, s, d, x);
    }

    /// Adjoint of the periodic inverse DWT.
    #[inline]
    fn adjoint_inverse_per<T: Transformable + Zero>(x: &[T], s: &mut [T], d: &mut [T]) {
        let gia: [_; N] = Self::GI.into_iter().rev().collect_array().unwrap();
        let hia: [_; N] = Self::HI.into_iter().rev().collect_array().unwrap();
        dwt_per_forward(&gia, &hia, x, s, d);
    }
}

/// Compute the sub-band length for a non-periodic DWT of a signal with `n` samples
/// using a filter of the given `width`.
///
/// Both the approximation and detail arrays must have this length.
#[inline(always)]
pub fn get_outlen(width: usize, n: usize) -> usize {
    let offset = (width - 2) / 2;
    let n_ds = n.div_ceil(2) + 2 * (width / 4);
    if (offset % 2 == 1) && (n % 2 == 1) {
        n_ds - 1
    } else {
        n_ds
    }
}

/// Compile-time assertion that `N` is a valid filter length (≥ 2 and even).
///
/// Evaluate `CheckCoefLen::<N>::VALID` in a const context to trigger the assert.
struct CheckCoefLen<const N: usize>();
impl<const N: usize> CheckCoefLen<N> {
    /// Asserts at compile time that `N >= 2` and `N % 2 == 0`.
    const VALID: () = {
        assert!(N >= 2, "Coefficient length must be 2 or more.");
        assert!(N.is_multiple_of(2), "Coefficient length must be even.");
    };
}

struct CheckHalfCoefLen<const N: usize, const NH: usize>();
impl<const N: usize, const NH: usize> CheckHalfCoefLen<N, NH> {
    /// Asserts at compile time that `N >= 2` and `N % 2 == 0`.
    const VALID: () = {
        assert!(N >= 2, "Coefficient length must be 2 or more.");
        assert!(
            NH * 2 == N,
            "Twice coefficient half length must be equal to coefficient length."
        )
    };
}

/// Assert at compile time that a wavelet coefficient length `N` is valid (even and ≥ 2).
///
/// Emits a compile error if `N` is odd or less than 2.  Call this inside `const` blocks that
/// accept a coefficient-length type parameter to get a clearer error message.
macro_rules! static_assert_valid_coefficient_length {
    ($N: ty) => {
        let _ = $crate::dwt::CheckCoefLen::<$N>::VALID;
    };
    ($N: ty, $NH: ty) => {
        let _ = $crate::dwt::CheckHalfCoefLen::<$N, $NH>::VALID;
    };
}

/// Filter offset used to centre the convolution window for a filter of width `n`.
const fn get_offset(n: usize) -> usize {
    (n - 2) / 2
}

/// Low-level forward DWT with explicit filter arrays and boundary condition.
///
/// Convolves `x` with the analysis filters `g` (low-pass) and `h` (high-pass),
/// downsampling by 2 to produce the sub-band outputs `s` and `d`.
///
/// # Panics
///
/// Panics if `s.len() != d.len()` or if either length is inconsistent with
/// `get_outlen(N, x.len())`.
pub fn dwt_forward<T: Transformable + Zero, const N: usize, BC: BoundaryExtension>(
    g: &[f64; N],
    h: &[f64; N],
    x: &[T],
    s: &mut [T],
    d: &mut [T],
    bc: &BC,
) {
    static_assert_valid_coefficient_length!(N);
    let (nx, ns, nd) = (x.len(), s.len(), d.len());

    assert_eq!(ns, nd, "'d.len()' must be equal to 's.len()'");

    assert_eq!(
        get_outlen(N, nx),
        ns,
        "'s.len()` and `d.len()' are inconsistent with 'x.len()'"
    );

    let offset = const { get_offset(N) };

    let gh: [_; N] = core::array::from_fn(|i| {
        [
            T::scalar_type_from_f64(g[N - (i + 1)]),
            T::scalar_type_from_f64(h[N - (i + 1)]),
        ]
    });

    // front boundary:
    let n_bcs = const { N / 4 };
    //let mut sd_iter = (-n_bcs..(ns as isize - n_bcs)).zip(s.iter_mut().zip(d.iter_mut()));

    let first_x = const { get_offset(N) % 2 };

    // calculate the break points of the front, main, and back loops.
    let n1 = std::cmp::min(2 * n_bcs, ns);
    // N - 2 is safe because N >= 2;
    let nx_steps = nx.saturating_sub(N - 2 + first_x) / 2;
    let n2 = std::cmp::min(n1 + nx_steps, ns);

    // split s and d into the front, main, and back loops (*_f, *_m, *_b)
    // split off the back parts
    let (s, s_b) = s.split_at_mut(n2);
    let (d, d_b) = d.split_at_mut(n2);
    // split off the front parts
    let (s_f, s_m) = s.split_at_mut(n1);
    let (d_f, d_m) = d.split_at_mut(n1);

    (-(n_bcs as isize)..n1 as isize - n_bcs as isize)
        .zip(s_f.iter_mut().zip(d_f))
        .for_each(|(i, (s, d))| {
            let ix = 2 * i - offset as isize;
            *s = T::zero();
            *d = T::zero();
            (ix..ix + N as isize)
                .zip(gh.iter())
                .for_each(|(j, [g, h])| {
                    if let Some(xo) = bc.get_bc(x, j) {
                        *s += xo.clone() * *g;
                        *d += xo * *h;
                    }
                })
        });
    // x[first_x..].array_windows::<N>().step_by(2);
    let x_iter = x[first_x..].array_windows::<N>().step_by(2);

    debug_assert_eq!(x_iter.len(), nx_steps); // double check in debug that nx_steps is correct
    debug_assert_eq!(x_iter.len(), s_m.len());
    debug_assert_eq!(x_iter.len(), d_m.len());

    x_iter
        .zip(s_m.iter_mut().zip(d_m))
        .for_each(|(xs, (s, d))| {
            *s = T::zero();
            *d = T::zero();
            gh.iter().zip(xs).for_each(|([g, h], x)| {
                *s += x.clone() * *g;
                *d += x.clone() * *h;
            });
        });

    (n2 as isize..ns as isize)
        .zip(s_b.iter_mut().zip(d_b))
        .for_each(|(i, (s, d))| {
            *s = T::zero();
            *d = T::zero();
            let ix = 2 * (i - n_bcs as isize) - offset as isize;
            (ix..ix + N as isize)
                .zip(gh.iter())
                .for_each(|(j, [g, h])| {
                    if let Some(xo) = bc.get_bc(x, j) {
                        *s += xo.clone() * *g;
                        *d += xo * *h;
                    }
                })
        });
}

/// Low-level inverse DWT with explicit synthesis filter arrays.
///
/// Upsamples `s` and `d` and convolves with the synthesis filters `gi` and `hi`
/// to reconstruct `x`.
///
/// # Panics
///
/// Panics if `s.len() != d.len()` or if the lengths are inconsistent with `x.len()`.
pub fn dwt_inverse<T: Transformable + Zero, const N: usize, const NH: usize>(
    gi: &[f64; N],
    hi: &[f64; N],
    s: &[T],
    d: &[T],
    x: &mut [T],
) {
    static_assert_valid_coefficient_length!(N, NH);
    let (nx, ns, nd) = (x.len(), s.len(), d.len());

    assert_eq!(ns, nd, "'d.len()' must be equal to 's.len()'");

    assert_eq!(
        get_outlen(N, nx),
        ns,
        "'s.len()` and `d.len()' are inconsistent with 'x.len()'"
    );

    let gh: [_; N] = core::array::from_fn(|i| {
        [
            T::scalar_type_from_f64(gi[N - (i + 1)]),
            T::scalar_type_from_f64(hi[N - (i + 1)]),
        ]
    });
    let gh_chunks = gh.as_chunks::<2>().0; // no remainder as N is even.

    let pair_shift = const { get_offset(N) % 2 };

    let (x_f, x) = x.split_at_mut(pair_shift);
    let (x_chunks, x_b) = x.as_chunks_mut::<2>();

    if let Some(x1) = x_f.get_mut(0)
        && let Some(s) = s.get(..NH)
        && let Some(d) = d.get(..NH)
    {
        *x1 = T::zero();

        gh_chunks
            .iter()
            .zip(s.iter().zip(d.iter()))
            .for_each(|([[g0, h0], _], (s, d))| {
                *x1 += s.clone() * *g0 + d.clone() * *h0
            });
    }

    x_chunks
        .iter_mut()
        .zip(
            s[pair_shift..]
                .array_windows::<NH>()
                .zip(d[pair_shift..].array_windows::<NH>()),
        )
        .for_each(|([x0, x1], (s, d))| {
            *x0 = T::zero();
            *x1 = T::zero();
            gh_chunks.iter().zip(s.iter().zip(d.iter())).for_each(
                |([[g0, h0], [g1, h1]], (s, d))| {
                    *x0 += s.clone() * *g1 + d.clone() * *h1;
                    *x1 += s.clone() * *g0 + d.clone() * *h0;
                },
            );
        });

    let last_sd = ns.checked_sub(NH).unwrap_or(ns);
    if let Some(x0) = x_b.get_mut(0)
        && let Some(s) = s.get(last_sd..)
        && let Some(d) = d.get(last_sd..)
    {
        *x0 = T::zero();

        gh_chunks
            .iter()
            .zip(s.iter().zip(d))
            .for_each(|([_, [g1, h1]], (s, d))| {
                *x0 += s.clone() * *g1 + d.clone() * *h1;
            });
    }
}

/// Low-level adjoint of the forward DWT with explicit filter arrays and boundary condition.
///
/// Computes the exact mathematical transpose of [`dwt_forward`] for any `BC`.  Where
/// `dwt_forward` gathers signal values through the boundary extension, this function
/// *scatters* sub-band contributions back using [`BoundaryExtension::get_parts`].
///
/// # Panics
///
/// Same length constraints as [`dwt_forward`].
pub fn dwt_adjoint_forward<T: Transformable + Zero, const N: usize, BC: BoundaryExtension>(
    g: &[f64; N],
    h: &[f64; N],
    s: &[T],
    d: &[T],
    x: &mut [T],
    bc: &BC,
) {
    static_assert_valid_coefficient_length!(N);
    let (nx, ns, nd) = (x.len(), s.len(), d.len());

    assert_eq!(ns, nd, "'d.len()' must be equal to 's.len()'");

    assert_eq!(
        get_outlen(N, nx),
        ns,
        "'s.len()` and `d.len()' are inconsistent with 'x.len()'"
    );

    x.iter_mut().for_each(|v| *v = T::zero());

    let offset = const { get_offset(N) };
    let n_bcs = const { N / 4 };
    let first_x = const { get_offset(N) % 2 };

    let gh: [_; N] = core::array::from_fn(|i| {
        [
            T::scalar_type_from_f64(g[N - (i + 1)]),
            T::scalar_type_from_f64(h[N - (i + 1)]),
        ]
    });

    // Mirror the forward's three-region split.
    let n1 = std::cmp::min(2 * n_bcs, ns);
    let nx_steps = nx.saturating_sub(N - 2 + first_x) / 2;
    let n2 = std::cmp::min(n1 + nx_steps, ns);

    // Front boundary: window may extend past the left edge.
    s[..n1]
        .iter()
        .zip(d[..n1].iter())
        .enumerate()
        .for_each(|(pos, (sv, dv))| {
            let ix = 2 * (pos as isize - n_bcs as isize) - offset as isize;
            gh.iter().enumerate().for_each(|(k, [gk, hk])| {
                let contrib = sv.clone() * *gk + dv.clone() * *hk;
                for (scale, j_real) in bc.get_parts::<T>(nx, ix + k as isize) {
                    match scale {
                        None => x[j_real] += contrib.clone(),
                        Some(sc) => x[j_real] += contrib.clone() * sc,
                    }
                }
            });
        });

    // Main region: every window index is in [0, nx), no BC needed.
    s[n1..n2]
        .iter()
        .zip(d[n1..n2].iter())
        .enumerate()
        .for_each(|(m, (sv, dv))| {
            let ix = first_x + 2 * m;
            gh.iter().zip(&mut x[ix..ix + N]).for_each(|([gk, hk], x)| {
                *x += sv.clone() * *gk + dv.clone() * *hk;
            });
        });

    // Back boundary: window may extend past the right edge.
    s[n2..]
        .iter()
        .zip(d[n2..].iter())
        .enumerate()
        .for_each(|(m, (sv, dv))| {
            let pos = m + n2;
            let ix = 2 * (pos as isize - n_bcs as isize) - offset as isize;
            gh.iter().enumerate().for_each(|(k, [gk, hk])| {
                let contrib = sv.clone() * *gk + dv.clone() * *hk;
                for (scale, j_real) in bc.get_parts::<T>(nx, ix + k as isize) {
                    match scale {
                        None => x[j_real] += contrib.clone(),
                        Some(sc) => x[j_real] += contrib.clone() * sc,
                    }
                }
            });
        });
}

/// Low-level periodic forward DWT.
///
/// Like [`dwt_forward`] but uses circular (periodic) boundary conditions so that
/// `s.len() + d.len() == x.len()`.  Odd-length signals are handled by copying the
/// last element to the end of `s`.
///
/// # Panics
///
/// Panics if `s.len() + d.len() != x.len()` or if the relative lengths of `s` and
/// `d` are inconsistent (they must satisfy `s.len() == d.len()` or `s.len() == d.len() + 1`).
pub fn dwt_per_forward<T: Transformable + Zero, const N: usize>(
    g: &[f64; N],
    h: &[f64; N],
    x: &[T],
    s: &mut [T],
    d: &mut [T],
) {
    static_assert_valid_coefficient_length!(N);
    let (nx, ns, nd) = (x.len(), s.len(), d.len());

    assert!(
        (ns == nd) || (ns == nd + 1),
        "'d.len()' must be equal to or 1 less than 's.len()'"
    );

    assert_eq!(
        nx,
        ns + nd,
        "'s.len()` + `d.len()' must be equal to `x.len()'"
    );
    let (x, s) = if ns > nd {
        // for the odd length transform, the last x value just becomes the last approximation coefficient.
        // Then shorten x and s by one element.
        if let Some(sl) = s.last_mut()
            && let Some(xl) = x.last()
        {
            *sl = xl.clone();
        }
        // if ns == 1 (or 0), there is nothing to do.
        if nd == 0 {
            return;
        }
        (&x[0..nx - 1], &mut s[0..nd])
    } else {
        (x, s)
    };

    let offset = const { get_offset(N) };
    let gh: [_; N] = core::array::from_fn(|i| {
        [
            T::scalar_type_from_f64(g[N - (i + 1)]),
            T::scalar_type_from_f64(h[N - (i + 1)]),
        ]
    });

    // front boundary:
    let n_bcs = const { N / 4 };

    let per_bc = PeriodicBoundary {};

    let first_x = const { get_offset(N) % 2 };

    // calculate the break points of the front, main, and back loops.
    let n1 = std::cmp::min(n_bcs, nd);
    // N - 2 is safe because N >= 2;
    let nx_steps = x.len().saturating_sub(N - 2 + first_x) / 2;
    let n2 = std::cmp::min(n1 + nx_steps, nd);

    // split s and d into the front, main, and back loops (*_f, *_m, *_b)
    // split off the back parts
    let (s, s_b) = s.split_at_mut(n2);
    let (d, d_b) = d.split_at_mut(n2);
    // split off the front parts
    let (s_f, s_m) = s.split_at_mut(n1);
    let (d_f, d_m) = d.split_at_mut(n1);

    (0..n1 as isize)
        .zip(s_f.iter_mut().zip(d_f))
        .for_each(|(i, (s, d))| {
            let ix = 2 * i - offset as isize;
            *s = T::zero();
            *d = T::zero();
            (ix..ix + N as isize)
                .zip(gh.iter())
                .for_each(|(j, [g, h])| {
                    if let Some(xo) = per_bc.get_bc(x, j) {
                        *s += xo.clone() * *g;
                        *d += xo * *h;
                    }
                })
        });
    // x[first_x..].array_windows::<N>().step_by(2);
    let x_iter = x[first_x..].array_windows::<N>().step_by(2);

    debug_assert_eq!(x_iter.len(), nx_steps); // double check in debug that nx_steps is correct
    debug_assert_eq!(x_iter.len(), s_m.len());
    debug_assert_eq!(x_iter.len(), d_m.len());

    x_iter
        .zip(s_m.iter_mut().zip(d_m))
        .for_each(|(xs, (s, d))| {
            *s = T::zero();
            *d = T::zero();
            gh.iter().zip(xs).for_each(|([g, h], x)| {
                *s += x.clone() * *g;
                *d += x.clone() * *h;
            });
        });

    (n2 as isize..nd as isize)
        .zip(s_b.iter_mut().zip(d_b))
        .for_each(|(i, (s, d))| {
            *s = T::zero();
            *d = T::zero();
            let ix = 2 * i - offset as isize;
            (ix..ix + N as isize)
                .zip(gh.iter())
                .for_each(|(j, [g, h])| {
                    if let Some(xo) = per_bc.get_bc(x, j) {
                        *s += xo.clone() * *g;
                        *d += xo * *h;
                    }
                })
        });
}

/// Low-level periodic inverse DWT.
///
/// Reconstructs `x` from sub-bands `s` and `d` using circular boundary conditions.
/// Mirror of [`dwt_per_forward`].
///
/// # Panics
///
/// Same length constraints as [`dwt_per_forward`].
pub fn dwt_per_inverse<T: Transformable + Zero, const N: usize, const NH: usize>(
    gi: &[f64; N],
    hi: &[f64; N],
    s: &[T],
    d: &[T],
    x: &mut [T],
) {
    static_assert_valid_coefficient_length!(N, NH);
    let (nx, ns, nd) = (x.len(), s.len(), d.len());

    assert!(
        (ns == nd) || (ns == nd + 1),
        "'d.len()' must be equal to or 1 less than 's.len()'"
    );

    assert_eq!(
        nx,
        ns + nd,
        "'s.len()` + `d.len()' must be equal to `x.len()'"
    );
    let (x, s) = if ns > nd {
        // for the odd length inverse transform, the last smooth coefficient just becomes the last x coefficient.
        // Then shorten x and s by one element.
        if let Some(sl) = s.last()
            && let Some(xl) = x.last_mut()
        {
            *xl = sl.clone();
        }
        // if ns == 1 (or 0), there is nothing to do.
        if nd == 0 {
            return;
        }
        (&mut x[0..nx - 1], &s[0..nd])
    } else {
        (x, s)
    };

    let gh: [_; N] = core::array::from_fn(|i| {
        [
            T::scalar_type_from_f64(gi[N - (i + 1)]),
            T::scalar_type_from_f64(hi[N - (i + 1)]),
        ]
    });
    let gh_chunks = gh.as_chunks::<2>().0; // no remainder as N is even.

    let pair_shift = const { get_offset(N) % 2 };

    let n_bcs = { N as isize / 4 };

    let (x_f, x) = x.split_at_mut(pair_shift);

    let per_bc = PeriodicBoundary {};
    if pair_shift > 0
        && let Some(x1) = x_f.get_mut(0)
    {
        let i_sd = -n_bcs;

        *x1 = T::zero();
        (i_sd..i_sd + N as isize / 2)
            .zip(gh_chunks)
            .for_each(|(j, [[g0, h0], _])| {
                if let Some(s) = per_bc.get_bc(s, j)
                    && let Some(d) = per_bc.get_bc(d, j)
                {
                    *x1 += s * *g0 + d * *h0
                }
            });
    }
    // s and d have lengths equal to N / 2
    // gh_iter is an N/2 length iterator that produces items of length 2
    // need to do for each x0 =

    // front boundarys

    // note here that x_b will only have an entry if pair_shift == 1;
    // as at this point x would start with an even length, and the first
    // would only be pealed off if there was a pair_shift.
    let (x_chunks, x_b) = x.as_chunks_mut::<2>();

    // now count how many x_chunks we need to handle at the front boundary:

    // n_bcs - pair_shift (if there was one)
    let nx_chunks = x_chunks.len();
    let n_wrap = const { N / 4 - get_offset(N) % 2 };
    let n1 = std::cmp::min(n_wrap, nx_chunks);

    // then the number of steps we can completely do to the x_chunks
    let nx_steps = s.len().saturating_sub(N / 2 - 1);

    debug_assert_eq!(nx_steps, s.array_windows::<NH>().len());
    debug_assert_eq!(nx_steps, d.array_windows::<NH>().len());

    // added to the first boundary...
    let n2 = std::cmp::min(n1 + nx_steps, nx_chunks);

    let (x_chunks, x_chunks_b) = x_chunks.split_at_mut(n2);
    let (x_chunks_f, x_chunks) = x_chunks.split_at_mut(n1);

    // let mut x_iter =
    //     (pair_shift as isize - n_bcs..nd as isize - n_bcs).zip(x[pair_shift..].chunks_exact_mut(2));

    (-(n_wrap as isize)..0)
        .zip(x_chunks_f)
        .for_each(|(i_sd, [x0, x1])| {
            *x0 = T::zero();
            *x1 = T::zero();
            (i_sd..i_sd + NH as isize)
                .zip(gh_chunks)
                .for_each(|(j, [[g0, h0], [g1, h1]])| {
                    if let Some(s) = per_bc.get_bc(s, j)
                        && let Some(d) = per_bc.get_bc(d, j)
                    {
                        *x0 += s.clone() * *g1 + d.clone() * *h1;
                        *x1 += s * *g0 + d * *h0;
                    }
                });
        });

    x_chunks
        .iter_mut()
        .zip(s.array_windows::<NH>().zip(d.array_windows::<NH>()))
        .for_each(|([x0, x1], (s, d))| {
            *x0 = T::zero();
            *x1 = T::zero();
            gh_chunks.iter().zip(s.iter().zip(d.iter())).for_each(
                |([[g0, h0], [g1, h1]], (s, d))| {
                    *x0 += s.clone() * *g1 + d.clone() * *h1;
                    *x1 += s.clone() * *g0 + d.clone() * *h0;
                },
            );
        });

    (n2 as isize - n_wrap as isize..nx_chunks as isize - n_wrap as isize)
        .zip(x_chunks_b)
        .for_each(|(i_sd, [x0, x1])| {
            *x0 = T::zero();
            *x1 = T::zero();
            (i_sd..i_sd + NH as isize)
                .zip(gh_chunks)
                .for_each(|(j, [[g0, h0], [g1, h1]])| {
                    if let Some(s) = per_bc.get_bc(s, j)
                        && let Some(d) = per_bc.get_bc(d, j)
                    {
                        *x0 += s.clone() * *g1 + d.clone() * *h1;
                        *x1 += s * *g0 + d * *h0;
                    }
                });
        });
    if pair_shift > 0
        && let Some(x0) = x_b.get_mut(0)
    {
        let i_sd = nd as isize - n_bcs;
        *x0 = T::zero();
        (i_sd..i_sd + NH as isize)
            .zip(gh_chunks)
            .for_each(|(j, [_, [g1, h1]])| {
                if let Some(s) = per_bc.get_bc(s, j)
                    && let Some(d) = per_bc.get_bc(d, j)
                {
                    *x0 += s * *g1 + d * *h1
                }
            });
    }
}

#[cfg(test)]
mod test {
    use crate::boundarys::ZeroBoundary;

    use super::*;

    #[test]
    fn test_simple() {
        const N: usize = 4;
        const NH: usize = 2;
        let g = [1.0; N];
        let h = core::array::from_fn(|i| (-1 * (i as isize % 2)) as f64 * 1.0);

        let bc = ZeroBoundary {};

        let nx = 33;
        let x = (0..nx).map(|i| (i + 1) as f64).collect::<Vec<_>>();
        let nsd = dbg!(get_outlen(N, nx));

        // let ns = (nx + 1) / 2;
        // let nd = nx / 2;

        let mut s = vec![0.0; nsd];
        let mut d = vec![0.0; nsd];

        dwt_forward(&g, &h, &x, &mut s, &mut d, &bc);

        let mut x = vec![0.0; nx];
        dwt_inverse::<_, _, NH>(&g, &h, &s, &d, &mut x);
    }
}
