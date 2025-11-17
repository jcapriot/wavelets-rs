mod steps;
pub mod daubechies;

use num_traits::{Num, NumAssignOps};
use itertools::{Itertools, izip};
use std::ops::{MulAssign, Neg, Mul};

use crate::{boundarys::BoundaryExtension};

pub trait LiftedTransform<W>
where 
    W: Copy + Clone + Neg<Output = W>,
{
    type StepListType;

    fn get_steps(&self) -> &Self::StepListType;
    fn forward<SD, BC: BoundaryExtension>(&self, s: &mut [SD], d: &mut[SD], bc: &BC)
    where 
        SD: Num + NumAssignOps + Copy + Mul<W, Output=SD> + MulAssign<W>;
    fn inverse<SD, BC: BoundaryExtension>(&self, s: &mut [SD], d: &mut[SD], bc: &BC)
    where 
        SD: Num + NumAssignOps + Copy + Mul<W, Output=SD> + MulAssign<W>;
}

pub fn deinterleave<T: Copy>(x: &[T], evens: &mut [T], odds: &mut [T]){
    let nx = x.len();
    let n_e = evens.len();
    let n_o = odds.len();

    assert_eq!(nx / 2, n_o);
    assert_eq!((nx + 1) / 2, n_e);

    let (chunks, rem) = x.as_chunks::<2>();
    let mut ev_iter = evens.iter_mut();
    izip!(chunks.iter(), ev_iter.by_ref(), odds.iter_mut())
        .for_each(|(xc, ev, od)|{
            *ev = xc[0];
            *od = xc[1];
        });
    rem.iter().zip(ev_iter).for_each(|(x, ev)| *ev = *x);
}

pub fn interleave<T: Copy>(evens: &[T], odds: &[T], x: &mut [T]){
    let nx = x.len();
    let n_e = evens.len();
    let n_o = odds.len();

    assert_eq!(nx / 2, n_o);
    assert_eq!((nx + 1) / 2, n_e);

    let (chunks, rem) = x.as_chunks_mut::<2>();
    let mut ev_iter = evens.iter();
    izip!(chunks.iter_mut(), ev_iter.by_ref(), odds.iter())
        .for_each(|(xc, even, odd)|{
            xc[0] = *even;
            xc[1] = *odd;
        });
    rem.iter_mut().zip(ev_iter).for_each(|(x, ev)| *x = *ev);
}


pub fn forward_transform<W, T, BC, WT>(wvlt: W, input: &[T], output: &mut[T], shape: &[usize], axes: &[usize])
where
    W: LiftedTransform<WT>,
    T: Num + NumAssignOps + Copy + Mul<WT, Output=T> + MulAssign<T>,
    BC: BoundaryExtension,
    WT: Copy + Clone + Neg<Output = WT>,
{
    for ax in axes{
        // Collapse all shapes up to and including axes to apply over
        let mut shp = shape.iter();
        let n_upper = shp.by_ref().take(*ax + 1).fold(1, |acc, e| acc * e);
        let n_lower = shp.fold(1, |acc, e| acc * e);
        
        if n_lower == 1{
            // n_upper calls
        }

    }

}

mod play{
    use crate::lwt::steps::{UpdateD, UpdateS, ScaleStep};

    struct Haar;

    pub trait LWTTest<U>{
        type StepListType;
        const STEPS: Self::StepListType;
    }

    macro_rules! make_haar_data {
        ( $( $t:ty ),* ) => {
            $(
            impl LWTTest<$t> for Haar{
                type StepListType = (UpdateD<$t, 1>, UpdateS<$t, 1>, ScaleStep<$t>);
                const STEPS: Self::StepListType = (
                    UpdateD{offset:0, coefs:[-1.0]},
                    UpdateS{offset:0, coefs:[0.5]},
                    ScaleStep{scale: 0.123}
                );
            }
            )*
        }
    }
    make_haar_data!(f32, f64);

}


#[cfg(test)]
mod tests{
    use super::*;

    use crate::lwt::steps::{UpdateD, UpdateS, ScaleStep};
    use wavelets_macros::LiftedTransform;
    use crate::test_approx_equal;

    use crate::boundarys::ZeroBoundary;
    

    const RTOL: f64 = 1E-6;
    const ATOL: f64 = 1E-14;

    #[derive(LiftedTransform)]
    struct TestWavelet<T: Copy + Num + Neg<Output=T>>{
        steps: (
            UpdateD<T, 2>,
            UpdateS<T, 2>,
            ScaleStep<T>
        ),
    }
    impl TestWavelet<f64>{
        pub fn new() -> Self{
            Self{steps:(
                UpdateD{offset: -1, coefs:[1.0, 2.0]},
                UpdateS{offset: 1, coefs:[-1.0, -2.0]},
                ScaleStep{scale: 0.5}
            )}
        }
    }


    #[test]
    fn test_multisteps_forward_inverse(){

        let wvlt = TestWavelet::new();

        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let mut s = [0.0; 8];
        let mut d = [0.0; 8];

        let (chunks, _rem) = input.as_chunks::<2>();
        chunks.iter().enumerate().for_each(|(i, xc)|{
            s[i] = xc[0];
            d[i] = xc[1];
        });

        let bc = ZeroBoundary{};

        wvlt.forward(&mut s, &mut d, &bc);

        wvlt.inverse(&mut s, &mut d, &bc);

        let mut output = [0.0; 8];
        let (chunks, _rem) = output.as_chunks_mut::<2>();
        chunks.iter_mut().enumerate().for_each(|(i, out)|{
            out[0] = s[i];
            out[1] = d[i];
        });

        test_approx_equal!(&input, &output, RTOL, ATOL);

    }
}