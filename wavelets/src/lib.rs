#![deny(missing_docs)]
//! Wavelet transforms for real and complex signals.
//!
//! This crate provides two families of wavelet transform:
//!
//! - **DWT** ([`dwt`]) — the classic Discrete Wavelet Transform via convolution/subsampling.
//! - **LWT** ([`lwt`]) — the Lifting Wavelet Transform, an in-place factorisation of the DWT.
//!
//! Both families support 1-D and N-D transforms, multi-level decomposition, periodic and
//! general boundary conditions, SIMD acceleration (via [`pulp`]), and optional Rayon
//! parallelism (feature `rayon`).
//!
//! # Wavelet families
//!
//! All concrete wavelet types live in the sub-modules below.  Each type is a zero-size marker
//! struct that carries the filter coefficients as compile-time constants.
//!
//! | Module | Wavelets |
//! |--------|----------|
//! | [`daubechies`] | Daubechies 1–10 |
//! | [`symlet`] | Symlets 4–6 |
//! | [`coiflet`] | Coiflets 1–3 |
//! | [`bior`] | Biorthogonal & CDF variants |
//!
//! The [`Wavelets`] enum lets you select a wavelet at runtime without generics.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use wavelets::{Wavelets, boundarys::BoundaryCondition, dwt::driver::WaveletTransform};
//!
//! // Build a DWT driver for Daubechies-4 with zero boundary padding.
//! let xfm: WaveletTransform<f64, _, 64> =
//!     WaveletTransform::new(Wavelets::Daubechies4, BoundaryCondition::Zero);
//!
//! let input = vec![1.0_f64; 128];
//! let nsd = wavelets::dwt::get_outlen(8, input.len());
//! let mut approx = vec![0.0; nsd];
//! let mut detail = vec![0.0; nsd];
//! xfm.forward_1d(&input, &mut approx, &mut detail);
//! ```

pub mod boundarys;
pub mod dwt;
pub mod iter;
pub mod lwt;
/// SIMD extension traits and CPU-dispatch infrastructure used by the wavelet kernels.
pub mod simd;
pub mod utils;

use num_traits::{FromPrimitive, MulAdd, Num, NumAssignOps, NumOps};
use std::{fmt::Debug, ops::Neg};
use wavelets_macros::{generate_wavelet_enum, generate_wavelet_match_arms};

macro_rules! gen_wavelet_struct {
    (
        $( ($name:ident, $width:expr) ),* $(,)?
    ) => {
        $(
            /// Zero-size marker struct representing a specific wavelet.
            ///
            /// `WIDTH` is the filter length (number of coefficients).  Use [`crate::Wavelets`] to
            /// select a wavelet dynamically.
            pub struct $name;
            impl $name{
                /// Number of filter coefficients for this wavelet.
                pub const WIDTH: usize = $width;

                /// Construct the wavelet marker (zero-cost; no heap allocation).
                pub fn new() -> Self{ Self{}}
            }
            impl Default for $name {
                fn default() -> Self { Self::new() }
            }
        )*
    };
}

/// Daubechies orthogonal wavelets (orders 1–10).
///
/// `Daubechies1` is the Haar wavelet (filter width 2).  Higher orders have
/// increasing regularity and support width `2 * order`.
pub mod daubechies {

    gen_wavelet_struct!(
        (Daubechies1, 2),
        (Daubechies2, 4),
        (Daubechies3, 6),
        (Daubechies4, 8),
        (Daubechies5, 10),
        (Daubechies6, 12),
        (Daubechies7, 14),
        (Daubechies8, 16),
        (Daubechies9, 18),
        (Daubechies10, 20),
    );
}

/// Symlet near-symmetric wavelets (orders 4–6).
///
/// Symlets are least-asymmetric modifications of Daubechies wavelets with the same
/// number of vanishing moments.
pub mod symlet {

    gen_wavelet_struct!((Symlet4, 8), (Symlet5, 10), (Symlet6, 12),);
}

/// Coiflet wavelets (orders 1–3).
///
/// Coiflets have vanishing moments for both the wavelet and scaling functions, making
/// them useful when both analysis and synthesis need to be smooth.
pub mod coiflet {

    gen_wavelet_struct!((Coiflet1, 6), (Coiflet2, 12), (Coiflet3, 18),);
}

/// Biorthogonal and CDF wavelet families.
///
/// Biorthogonal wavelets use separate analysis and synthesis filters, enabling exact
/// linear-phase responses.  The naming convention `BiorA_B` refers to the order of
/// the synthesis/analysis filter pair.  The [`CDF5_3`] and [`CDF9_7`] variants are
/// the Cohen–Daubechies–Feauveau wavelets used in the JPEG 2000 standard.
pub mod bior {

    gen_wavelet_struct!((Bior1_3, 6));
    gen_wavelet_struct!((Bior1_5, 10));
    gen_wavelet_struct!((Bior2_2, 6));
    gen_wavelet_struct!((Bior2_4, 10));
    gen_wavelet_struct!((Bior2_6, 14));
    gen_wavelet_struct!((Bior2_8, 18));
    gen_wavelet_struct!((Bior3_1, 4));
    gen_wavelet_struct!((Bior3_3, 8));
    gen_wavelet_struct!((Bior3_5, 12));
    gen_wavelet_struct!((Bior3_7, 16));
    gen_wavelet_struct!((Bior3_9, 20));
    gen_wavelet_struct!((Bior4_2, 8));
    gen_wavelet_struct!((Bior4_4, 12));
    gen_wavelet_struct!((Bior4_6, 16));
    gen_wavelet_struct!((Bior5_5, 14));
    gen_wavelet_struct!((Bior6_8, 22));
    gen_wavelet_struct!((CDF5_3, 6));
    gen_wavelet_struct!((CDF9_7, 10));
}

/// Compute the maximum number of decomposition levels for a filter of width `N` applied
/// to a signal of length `n`.
///
/// Returns 0 when the signal is too short for even a single level.
#[inline]
pub fn max_level(width: usize, n: usize) -> usize {
    if width == 0 {
        return 0;
    }
    if n < width - 1 {
        return 0;
    }
    let mut lvl = 0;
    let mut n = n;
    while n >= 2 * (width - 1) {
        lvl += 1;
        n = (n + 1) / 2;
    }
    lvl
}

// `Wavelets` is generated by a proc-macro so rustdoc cannot attach the doc comment
// here.  See the crate-level docs for a description of this enum.
generate_wavelet_enum!(Wavelets, (Clone, Copy, Debug, PartialEq, Eq, Hash));

impl Wavelets {
    /// Maximum decomposition levels for a signal of length `n`.
    pub fn max_level(&self, n: usize) -> usize {
        max_level(self.width(), n)
        // use bior::*;
        // use coiflet::*;
        // use daubechies::*;
        // use symlet::*;
        // generate_wavelet_match_arms! {Self, self, { max_level::<{#wvlt::WIDTH}>(n),}}
    }

    /// Number of filter coefficients for this wavelet.
    pub fn width(&self) -> usize {
        use bior::*;
        use coiflet::*;
        use daubechies::*;
        use symlet::*;
        generate_wavelet_match_arms! {Self, self, { #wvlt::WIDTH,}}
    }
}

/// Fused multiply-add: `self * a + b`.
///
/// This mirrors [`num_traits::MulAdd`] but allows heterogeneous operand types, which is
/// needed for complex types where the scalar multiplier is real.
pub trait MulScalarAdd<A = Self, B = Self> {
    /// The result type of `self * a + b`.
    type Output;

    /// Compute `self * a + b` (fused, where hardware supports it).
    fn mul_add(self, a: A, b: B) -> Self::Output;
}

impl<T: num_traits::MulAdd<T, T, Output = T>> MulScalarAdd<T, T> for T {
    type Output = T;

    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self::Output {
        <Self as num_traits::MulAdd>::mul_add(self, a, b)
    }
}

/// Element type that can participate in wavelet filter convolutions.
///
/// Implemented for all integer and floating-point primitives as well as
/// `num_complex::Complex<T>` where `T` itself is transformable.
///
/// The associated `Scalar` type is the coefficient type: for real types it equals
/// `Self`; for complex types it is the underlying real type so that filter
/// coefficients remain real-valued.
pub trait Transformable:
    NumOps
    + NumOps<Self::Scalar>
    + Clone
    + Neg<Output = Self>
    + NumAssignOps
    + NumAssignOps<Self::Scalar>
    + MulScalarAdd<Self::Scalar, Self, Output = Self>
{
    /// The scalar coefficient type.  For real types this is `Self`; for complex types
    /// this is the underlying real type.
    type Scalar: FromPrimitive + Copy + NumOps + std::fmt::Debug;

    /// Compute `self * b + c`.
    #[inline(always)]
    fn mul_add_op(self, b: Self::Scalar, c: Self) -> Self {
        self.mul_add(b, c)
    }

    /// Compute `(-self) * b + c`.
    #[inline(always)]
    fn neg_mul_add_op(self, b: Self::Scalar, c: Self) -> Self {
        (-self).mul_add(b, c)
    }

    /// Convert an `isize` into `Self::Scalar`.  Panics if the conversion fails.
    #[inline(always)]
    fn scalar_type_from_isize(x: isize) -> Self::Scalar {
        Self::Scalar::from_isize(x).unwrap()
    }

    /// Convert an `f64` into `Self::Scalar`.  Panics if the conversion fails.
    #[inline(always)]
    fn scalar_type_from_f64(x: f64) -> Self::Scalar {
        Self::Scalar::from_f64(x).unwrap()
    }
}

macro_rules! impl_transformable {
    ($T:ty) => {
        impl Transformable for $T {
            type Scalar = Self;
        }
    };
}
impl_transformable!(i8);
impl_transformable!(i16);
impl_transformable!(i32);
impl_transformable!(i64);
impl_transformable!(i128);
impl_transformable!(isize);
impl_transformable!(f32);
impl_transformable!(f64);

impl<T: MulAdd<Output = T> + Clone> MulScalarAdd<T, num_complex::Complex<T>>
    for num_complex::Complex<T>
{
    type Output = Self;

    #[inline(always)]
    fn mul_add(self, a: T, b: Self) -> Self::Output {
        Self::Output {
            re: T::mul_add(self.re, a.clone(), b.re),
            im: T::mul_add(self.im, a, b.im),
        }
    }
}

impl<T: Num + Copy + Debug + FromPrimitive + MulAdd<Output = T> + Neg<Output = T> + NumAssignOps>
    Transformable for num_complex::Complex<T>
{
    type Scalar = T;
}

const N_BITS: usize = 512;
const N_I8: usize = N_BITS / 8;
const N_I16: usize = N_BITS / 16;
const N_I32: usize = N_BITS / 32;
const N_I64: usize = N_BITS / 64;
const N_I128: usize = N_BITS / 128;
const N_ISIZE: usize = N_BITS / isize::BITS as usize;
const N_F32: usize = N_BITS / 32;
const N_F64: usize = N_BITS / 64;
const N_C32: usize = N_BITS / 64;
const N_C64: usize = N_BITS / 128;

/// Marker trait asserting that `N` is the correct SIMD chunk width for type `T`.
///
/// This is a sealed compile-time assertion used to tie the const generic `N` in
/// driver structs to the actual SIMD lane count for `T`, preventing mismatched chunk
/// sizes from compiling.
pub trait ChunkWidth<T, const N: usize> {}
impl ChunkWidth<i8, N_I8> for i8 {}
impl ChunkWidth<i16, N_I16> for i16 {}
impl ChunkWidth<i32, N_I32> for i32 {}
impl ChunkWidth<i64, N_I64> for i64 {}
impl ChunkWidth<i128, N_I128> for i128 {}
impl ChunkWidth<isize, N_ISIZE> for isize {}
impl ChunkWidth<f32, N_F32> for f32 {}
impl ChunkWidth<f64, N_F64> for f64 {}
impl ChunkWidth<num_complex::Complex32, N_C32> for num_complex::Complex32 {}
impl ChunkWidth<num_complex::Complex64, N_C64> for num_complex::Complex64 {}

/// Test helpers.  Hidden from rustdoc; not intended for library consumers.
#[doc(hidden)]
pub mod tests {

    #[track_caller]
    pub fn test_approx_equal<T>(actual: &[T], desired: &[T], rtol: T, atol: T)
    where
        T: num_traits::Float + std::fmt::Debug,
    {
        let n_a = actual.len();
        let n_d = desired.len();
        assert_eq!(
            n_a, n_d,
            "Slice length mismatch:\n actual: {n_a}\n desired: {n_d}"
        );
        let mut mismatch = None;
        let mut max_adiff = None;
        let mut max_rdiff = None;
        actual.iter().zip(desired.iter()).for_each(|(a, d)| {
            let abs_diff = (*a - *d).abs();
            if abs_diff > rtol * d.abs() + atol {
                mismatch = Some(mismatch.unwrap_or(0) + 1);
                max_adiff = Some(max_adiff.unwrap_or(T::zero()).max(abs_diff));
                let r_diff = if d.abs() == T::zero() {
                    T::infinity()
                } else {
                    abs_diff / d.abs()
                };
                max_rdiff = Some(max_rdiff.unwrap_or(T::zero()).max(r_diff));
            }
        });

        if let (Some(mismatch), Some(max_adiff), Some(max_rdiff)) = (mismatch, max_adiff, max_rdiff)
        {
            panic!(
                "{}/{} mismatched elements:\n Maximum differences: absolute={:?}, relative={:?}\n actual:\n{:?}\n desired:\n{:?}",
                mismatch, n_a, max_adiff, max_rdiff, actual, desired
            );
        }
    }

    #[track_caller]
    pub fn test_approx_adjoint<F, FA, T>(f: F, f_adj: FA, u: &[T], v: &[T], rtol: T, atol: T)
    where
        F: Fn(&[T], &mut [T]),
        FA: Fn(&[T], &mut [T]),
        T: num_traits::Float + std::fmt::Debug,
    {
        let n_u = u.len();
        let n_v = v.len();

        let mut f_u = vec![T::zero(); n_v];
        let mut f_adj_v = vec![T::zero(); n_u];

        // inner product of <v, f(u)>
        f(u, &mut f_u);
        let v1 = std::iter::zip(f_u, v.iter().cloned()).fold(T::zero(), |acc, (x, y)| acc + x * y);

        // inner product of <f_adj(v), u>
        f_adj(v, &mut f_adj_v);
        let v2 =
            std::iter::zip(f_adj_v, u.iter().cloned()).fold(T::zero(), |acc, (x, y)| acc + x * y);

        let abs_diff = (v1 - v2).abs();
        let thresh = rtol * v1.abs() + atol;

        assert!(
            abs_diff <= thresh,
            "{v1:?} and {v2:?} are not equal to tolerance rtol={rtol:?}, atol={atol:?}
            Absolute difference: {:?}
            Relative difference: {:?}
            ",
            abs_diff,
            abs_diff / v1.abs()
        );
    }
}
