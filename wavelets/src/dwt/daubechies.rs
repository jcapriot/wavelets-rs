use crate::boundarys::BoundaryExtension;
pub use crate::wavelets::daubechies::*;
pub use crate::dwt::DiscreteTransform;

impl DiscreteTransform for Daubechies1{
    type FilterType = [f64; 2];
    const G: Self::FilterType = [
        7.071067811865475244008443621048490392848359376884740365883398e-01,
        7.071067811865475244008443621048490392848359376884740365883398e-01
    ];
    const H: Self::FilterType = [
        -7.071067811865475244008443621048490392848359376884740365883398e-01,
        7.071067811865475244008443621048490392848359376884740365883398e-01
    ];

    fn forward<T: From<f64>, BC: BoundaryExtension>(x: &[T], s: &mut [T], d:&mut[T], bc: &BC) {unimplemented!()}
    fn inverse<T: From<f64>, BC: BoundaryExtension>(s: &[T], d: &[T], x: &mut[T], bc: &BC) {unimplemented!()}
}