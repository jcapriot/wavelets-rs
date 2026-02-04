pub mod boundarys;
pub mod driver;
pub mod dwt;
pub mod iter;
pub mod lwt;
pub mod utils;

use num_traits::{FromPrimitive, NumAssignOps, NumOps};
use std::ops::Neg;

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
        (Bior3_1, 4)
    );
}
pub mod bior {

    gen_wavelet_struct!((Bior3_1, 4));
}

macro_rules! for_each_wavelet {
    ($m:ident) => {
        $m!(Daubechies1);
        $m!(Daubechies2);
        $m!(Daubechies3);
        $m!(Daubechies4);
        $m!(Daubechies5);
        $m!(Daubechies6);
        $m!(Bior3_1);
    };
}
pub(crate) use for_each_wavelet;

pub enum Wavelets {
    Daubechies1,
    Daubechies2,
    Daubechies3,
    Daubechies4,
    Daubechies5,
    Daubechies6,
    Bior3_1,
}

#[derive(Clone, Copy, Debug)]
pub enum Direction {
    Forward,
    Inverse,
}

#[derive(Clone, Copy, Debug)]
pub enum Operation {
    Normal,
    Adjoint,
}

pub trait Transformable:
    NumOps
    + Clone
    + Neg<Output = Self>
    + NumAssignOps
    + std::ops::Mul<Self::ScalarType, Output = Self>
    + std::ops::MulAssign<Self::ScalarType>
    + std::ops::DivAssign<Self::ScalarType>
{
    type ScalarType: FromPrimitive
        + Clone
        + std::ops::Mul<Self::ScalarType, Output = Self::ScalarType>
        + std::fmt::Debug;
}

macro_rules! impl_transformable {
    ($T:ty) => {
        impl Transformable for $T {
            type ScalarType = Self;
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

impl Transformable for num_complex::Complex32 {
    type ScalarType = f32;
}
impl Transformable for num_complex::Complex64 {
    type ScalarType = f64;
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
}
