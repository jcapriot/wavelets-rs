pub mod boundarys;
pub mod dwt;
pub mod lwt;
pub mod wavelets;
//pub mod vector;
pub mod driver;
pub mod iter;
pub mod utils;

use num_traits::{NumAssignOps, NumOps};
use std::ops::Neg;

#[derive(Clone, Copy, Debug)]
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

pub trait Transformable: NumOps + Clone + Neg<Output = Self> + NumAssignOps {}

impl<T: NumOps + Clone + Neg<Output = T> + NumAssignOps> Transformable for T {}

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
