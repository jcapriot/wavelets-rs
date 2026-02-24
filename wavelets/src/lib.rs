pub mod boundarys;
pub mod dwt;
pub mod iter;
pub mod lwt;
pub mod utils;

use num_traits::{Float, FromPrimitive, MulAdd, Num, NumAssignOps, NumAssignRef, NumOps, NumRef};
use pulp::{Simd, cast};
use std::{fmt::Debug, ops::Neg, sync::LazyLock};
use wavelets_macros::{generate_wavelet_enum, generate_wavelet_match_arms};

macro_rules! gen_wavelet_struct {
    (
        $( ($name:ident, $width:expr) ),* $(,)?
    ) => {
        $(
            pub struct $name;
            impl $name{
                pub const WIDTH: usize = $width;

                pub fn new() -> Self{ Self{}}
            }
        )*
    };
}
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

pub mod symlet {

    gen_wavelet_struct!((Symlet4, 8), (Symlet5, 10), (Symlet6, 12),);
}

pub mod coiflet {

    gen_wavelet_struct!((Coiflet1, 6), (Coiflet2, 12), (Coiflet3, 18),);
}

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

#[inline]
pub fn max_level<const N: usize>(n: usize) -> usize {
    if N == 0 {
        return 0;
    }
    if n < N - 1 {
        return 0;
    }
    let mut lvl = 0;
    let mut n = n;
    while n >= 2 * (N - 1) {
        lvl += 1;
        n = (n + 1) / 2;
    }
    lvl
}

generate_wavelet_enum!(Wavelets, (Clone, Copy, Debug, PartialEq, Eq, Hash));

pub trait FloatOps: Float + NumAssignRef + NumRef + MulAdd {}
impl<T: Float + NumAssignRef + NumRef + MulAdd> FloatOps for T {}

impl Wavelets {
    pub fn max_level(&self, n: usize) -> usize {
        use bior::*;
        use coiflet::*;
        use daubechies::*;
        use symlet::*;
        generate_wavelet_match_arms! {Self, self, { max_level::<{#wvlt::WIDTH}>(n),}}
    }
}

pub trait MulScalarAdd<A = Self, B = Self> {
    type Output;

    fn mul_add(self, a: A, b: B) -> Self::Output;
}

impl<T: num_traits::MulAdd<T, T, Output = T>> MulScalarAdd<T, T> for T {
    type Output = T;

    #[inline(always)]
    fn mul_add(self, a: Self, b: Self) -> Self::Output {
        <Self as num_traits::MulAdd>::mul_add(self, a, b)
    }
}

pub trait Transformable:
    NumOps
    + NumOps<Self::Scalar>
    + Clone
    + Neg<Output = Self>
    + NumAssignOps
    + NumAssignOps<Self::Scalar>
    + MulScalarAdd<Self::Scalar, Self, Output = Self>
{
    type Scalar: FromPrimitive + Copy + NumOps + std::fmt::Debug;

    #[inline(always)]
    fn mul_add_op(self, b: Self::Scalar, c: Self) -> Self {
        self.mul_add(b, c)
    }

    #[inline(always)]
    fn neg_mul_add_op(self, b: Self::Scalar, c: Self) -> Self {
        (-self).mul_add(b, c)
    }

    #[inline(always)]
    fn scalar_type_from_isize(x: isize) -> Self::Scalar {
        Self::Scalar::from_isize(x).unwrap()
    }
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

pub trait SimdTransformable: Sized + Transformable {
    type Vector<S: Simd>: Copy + std::fmt::Debug;
    type SplatVector<S: Simd>: Copy + std::fmt::Debug;
    //type SplatScalar: FromPrimitive + Transformable;

    fn simd_lanes<S: Simd>(simd: S) -> usize;

    fn as_simd<S: Simd>(simd: S, x: &[Self]) -> (&[Self::Vector<S>], &[Self]);

    fn as_mut_simd<S: Simd>(simd: S, x: &mut [Self]) -> (&mut [Self::Vector<S>], &mut [Self]);

    fn simd_splat<S: Simd>(simd: S, v: Self::Scalar) -> Self::SplatVector<S>;

    fn simd_mul_add<S: Simd>(
        simd: S,
        a: Self::Vector<S>,
        b: Self::SplatVector<S>,
        c: Self::Vector<S>,
    ) -> Self::Vector<S>;
    fn simd_negate_mul_add<S: Simd>(
        simd: S,
        a: Self::Vector<S>,
        b: Self::SplatVector<S>,
        c: Self::Vector<S>,
    ) -> Self::Vector<S>;
    fn simd_add<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Vector<S>;
    fn simd_sub<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Vector<S>;
    fn simd_mul<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::SplatVector<S>) -> Self::Vector<S>;
    fn simd_div<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::SplatVector<S>) -> Self::Vector<S>;
}

impl SimdTransformable for f32 {
    type Vector<S: Simd> = S::f32s;
    type SplatVector<S: Simd> = Self::Vector<S>;

    #[inline(always)]
    fn simd_lanes<S: Simd>(_: S) -> usize {
        S::F32_LANES
    }

    #[inline(always)]
    fn as_simd<S: Simd>(_: S, slice: &[Self]) -> (&[Self::Vector<S>], &[Self]) {
        S::as_simd_f32s(slice)
    }

    #[inline(always)]
    fn as_mut_simd<S: Simd>(_: S, slice: &mut [Self]) -> (&mut [Self::Vector<S>], &mut [Self]) {
        S::as_mut_simd_f32s(slice)
    }

    #[inline(always)]
    fn simd_splat<S: Simd>(simd: S, v: Self) -> Self::SplatVector<S> {
        simd.splat_f32s(v)
    }

    #[inline(always)]
    fn simd_mul_add<S: Simd>(
        simd: S,
        a: Self::Vector<S>,
        b: Self::Vector<S>,
        c: Self::Vector<S>,
    ) -> Self::Vector<S> {
        simd.mul_add_f32s(a, b, c)
    }

    #[inline(always)]
    fn simd_negate_mul_add<S: Simd>(
        simd: S,
        a: Self::Vector<S>,
        b: Self::Vector<S>,
        c: Self::Vector<S>,
    ) -> Self::Vector<S> {
        let neg_a = simd.neg_f32s(a);
        simd.mul_add_f32s(neg_a, b, c)
    }

    #[inline(always)]
    fn simd_add<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Vector<S> {
        simd.add_f32s(a, b)
    }

    #[inline(always)]
    fn simd_sub<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Vector<S> {
        simd.sub_f32s(a, b)
    }

    #[inline(always)]
    fn simd_mul<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Vector<S> {
        simd.mul_f32s(a, b)
    }

    #[inline(always)]
    fn simd_div<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Vector<S> {
        simd.div_f32s(a, b)
    }
}

impl SimdTransformable for f64 {
    type Vector<S: Simd> = S::f64s;
    type SplatVector<S: Simd> = Self::Vector<S>;

    #[inline(always)]
    fn simd_lanes<S: Simd>(_: S) -> usize {
        S::F64_LANES
    }

    #[inline(always)]
    fn as_simd<S: Simd>(_: S, slice: &[Self]) -> (&[Self::Vector<S>], &[Self]) {
        S::as_simd_f64s(slice)
    }

    #[inline(always)]
    fn as_mut_simd<S: Simd>(_: S, slice: &mut [Self]) -> (&mut [Self::Vector<S>], &mut [Self]) {
        S::as_mut_simd_f64s(slice)
    }

    #[inline(always)]
    fn simd_splat<S: Simd>(simd: S, v: Self) -> Self::Vector<S> {
        simd.splat_f64s(v)
    }

    #[inline(always)]
    fn simd_mul_add<S: Simd>(
        simd: S,
        a: Self::Vector<S>,
        b: Self::Vector<S>,
        c: Self::Vector<S>,
    ) -> Self::Vector<S> {
        simd.mul_add_f64s(a, b, c)
    }

    #[inline(always)]
    fn simd_negate_mul_add<S: Simd>(
        simd: S,
        a: Self::Vector<S>,
        b: Self::Vector<S>,
        c: Self::Vector<S>,
    ) -> Self::Vector<S> {
        let neg_a = simd.neg_f64s(a);
        simd.mul_add_f64s(neg_a, b, c)
    }

    #[inline(always)]
    fn simd_add<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Vector<S> {
        simd.add_f64s(a, b)
    }

    #[inline(always)]
    fn simd_sub<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Vector<S> {
        simd.sub_f64s(a, b)
    }

    #[inline(always)]
    fn simd_mul<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Vector<S> {
        simd.mul_f64s(a, b)
    }

    #[inline(always)]
    fn simd_div<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Vector<S> {
        simd.div_f64s(a, b)
    }
}

impl SimdTransformable for num_complex::Complex32 {
    type Vector<S: Simd> = S::c32s;
    type SplatVector<S: Simd> = S::f32s;

    #[inline(always)]
    fn simd_lanes<S: Simd>(_: S) -> usize {
        S::C32_LANES
    }

    #[inline(always)]
    fn as_simd<S: Simd>(_: S, slice: &[Self]) -> (&[Self::Vector<S>], &[Self]) {
        S::as_simd_c32s(slice)
    }

    #[inline(always)]
    fn as_mut_simd<S: Simd>(_: S, slice: &mut [Self]) -> (&mut [Self::Vector<S>], &mut [Self]) {
        S::as_mut_simd_c32s(slice)
    }

    #[inline(always)]
    fn simd_splat<S: Simd>(simd: S, v: Self::Scalar) -> Self::SplatVector<S> {
        simd.splat_f32s(v)
    }

    #[inline(always)]
    fn simd_mul_add<S: Simd>(
        simd: S,
        a: Self::Vector<S>,
        b: Self::SplatVector<S>,
        c: Self::Vector<S>,
    ) -> Self::Vector<S> {
        cast(simd.mul_add_f32s(cast(a), b, cast(c)))
    }

    #[inline(always)]
    fn simd_negate_mul_add<S: Simd>(
        simd: S,
        a: Self::Vector<S>,
        b: Self::SplatVector<S>,
        c: Self::Vector<S>,
    ) -> Self::Vector<S> {
        let neg_a = simd.neg_c32s(a);
        cast(simd.mul_add_f32s(cast(neg_a), b, cast(c)))
    }

    #[inline(always)]
    fn simd_add<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Vector<S> {
        simd.add_c32s(a, b)
    }

    #[inline(always)]
    fn simd_sub<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Vector<S> {
        simd.sub_c32s(a, b)
    }

    #[inline(always)]
    fn simd_mul<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::SplatVector<S>) -> Self::Vector<S> {
        cast(simd.mul_f32s(cast(a), b))
    }

    #[inline(always)]
    fn simd_div<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::SplatVector<S>) -> Self::Vector<S> {
        cast(simd.div_f32s(cast(a), b))
    }
}

impl SimdTransformable for num_complex::Complex64 {
    type Vector<S: Simd> = S::c64s;
    type SplatVector<S: Simd> = S::f64s;

    #[inline(always)]
    fn simd_lanes<S: Simd>(_: S) -> usize {
        S::C64_LANES
    }

    #[inline(always)]
    fn as_simd<S: Simd>(_: S, slice: &[Self]) -> (&[Self::Vector<S>], &[Self]) {
        S::as_simd_c64s(slice)
    }

    #[inline(always)]
    fn as_mut_simd<S: Simd>(_: S, slice: &mut [Self]) -> (&mut [Self::Vector<S>], &mut [Self]) {
        S::as_mut_simd_c64s(slice)
    }

    #[inline(always)]
    fn simd_splat<S: Simd>(simd: S, v: Self::Scalar) -> Self::SplatVector<S> {
        simd.splat_f64s(v)
    }

    #[inline(always)]
    fn simd_mul_add<S: Simd>(
        simd: S,
        a: Self::Vector<S>,
        b: Self::SplatVector<S>,
        c: Self::Vector<S>,
    ) -> Self::Vector<S> {
        cast(simd.mul_add_f64s(cast(a), b, cast(c)))
    }

    #[inline(always)]
    fn simd_negate_mul_add<S: Simd>(
        simd: S,
        a: Self::Vector<S>,
        b: Self::SplatVector<S>,
        c: Self::Vector<S>,
    ) -> Self::Vector<S> {
        let neg_a = simd.neg_c64s(a);
        cast(simd.mul_add_f64s(cast(neg_a), b, cast(c)))
    }

    #[inline(always)]
    fn simd_add<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Vector<S> {
        simd.add_c64s(a, b)
    }

    #[inline(always)]
    fn simd_sub<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Vector<S> {
        simd.sub_c64s(a, b)
    }

    #[inline(always)]
    fn simd_mul<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::SplatVector<S>) -> Self::Vector<S> {
        cast(simd.mul_f64s(cast(a), b))
    }

    #[inline(always)]
    fn simd_div<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::SplatVector<S>) -> Self::Vector<S> {
        cast(simd.div_f64s(cast(a), b))
    }
}

pub static ARCH: LazyLock<pulp::Arch> = LazyLock::new(|| pulp::Arch::new());

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

pub mod tests {
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
        actual.iter().zip(desired.iter()).for_each(|(a, d)| {
            let abs_diff = (*a - *d).abs();
            if abs_diff > rtol * d.abs() + atol {
                mismatch = Some(mismatch.unwrap_or(0) + 1);
            }
        });

        if let Some(mismatch) = mismatch {
            panic!(
                "{} mismatched elements: \n  actual: {:?}\n desired: {:?}",
                mismatch, actual, desired
            );
        }
    }

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
