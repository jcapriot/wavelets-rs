#[cfg(feature = "ndarray")]
use ndarray::{ArrayRef, Dimension};
use num_traits::Zero;
use std::collections::HashSet;

use crate::Wavelets;
use crate::boundarys::BoundaryExtension;
use crate::dwt::DiscreteTransform;
use crate::iter::LanesIterator;

use crate::{ChunkWidth, Transformable};

use wavelets_macros::generate_wavelet_match_arms;
pub struct WaveletTransform<T, BC, const N: usize>
where
    T: Transformable + Zero + ChunkWidth<T, N>,
    BC: BoundaryExtension,
{
    dwt_forward: fn(&[T], &mut [T], &mut [T], &BC),
    dwt_inverse: fn(&[T], &[T], &mut [T]),
    dwt_adj_inverse: fn(&[T], &mut [T], &mut [T]),
    dwt_per_forward: fn(&[T], &mut [T], &mut [T]),
    dwt_per_inverse: fn(&[T], &[T], &mut [T]),
    dwt_per_adj_forward: fn(&[T], &[T], &mut [T]),
    dwt_per_adj_inverse: fn(&[T], &mut [T], &mut [T]),
    bc: BC,
}

impl<T, BC, const N: usize> WaveletTransform<T, BC, N>
where
    T: Transformable + Zero + ChunkWidth<T, N>,
    BC: BoundaryExtension,
{
    pub fn new(wvlt: Wavelets, bc: BC) -> Self {
        use crate::dwt::bior::*;
        use crate::dwt::coiflet::*;
        use crate::dwt::daubechies::*;
        use crate::dwt::symlet::*;
        let dwt_forward: fn(&[T], &mut [T], &mut [T], &BC) = generate_wavelet_match_arms! {
            Wavelets,
            wvlt,
            {#wvlt::forward,}
        };
        let dwt_inverse: fn(&[T], &[T], &mut [T]) = generate_wavelet_match_arms! {
            Wavelets,
            wvlt,
            {#wvlt::inverse,}
        };
        let dwt_adj_inverse: fn(&[T], &mut [T], &mut [T]) = generate_wavelet_match_arms! {
            Wavelets,
            wvlt,
            {#wvlt::adjoint_inverse,}
        };
        let dwt_per_forward: fn(&[T], &mut [T], &mut [T]) = generate_wavelet_match_arms! {
            Wavelets,
            wvlt,
            {#wvlt::forward_per,}
        };
        let dwt_per_inverse: fn(&[T], &[T], &mut [T]) = generate_wavelet_match_arms! {
            Wavelets,
            wvlt,
            {#wvlt::inverse_per,}
        };
        let dwt_per_adj_forward: fn(&[T], &[T], &mut [T]) = generate_wavelet_match_arms! {
            Wavelets,
            wvlt,
            {#wvlt::adjoint_forward_per,}
        };
        let dwt_per_adj_inverse: fn(&[T], &mut [T], &mut [T]) = generate_wavelet_match_arms! {
            Wavelets,
            wvlt,
            {#wvlt::adjoint_inverse_per,}
        };
        Self {
            dwt_forward,
            dwt_inverse,
            dwt_adj_inverse,
            dwt_per_forward,
            dwt_per_inverse,
            dwt_per_adj_forward,
            dwt_per_adj_inverse,
            bc,
        }
    }
}
