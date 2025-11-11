use crate::boundarys::BoundaryExtension;
use num_traits::{MulAdd, Num, NumAssignOps};
use std::ops::{Neg, MulAssign};

mod ops{
    use super::*;
    #[inline(always)]
    fn max_offset<const N: usize>(offset: isize) -> isize{(N as isize) - 1 - offset}
     #[inline(always)]
    fn max_offset_r<const N: usize>(offset: isize) -> isize{(N as isize) - 1 + offset_r::<N>(offset)}
     #[inline(always)]
    fn n_front<const N: usize>(offset: isize) -> usize{ if offset < 0 { (-offset) as usize} else {0}}
     #[inline(always)]
    fn n_back<const N: usize>(offset: isize) -> usize{ if max_offset::<N>(offset) < 0 {0} else {max_offset::<N>(offset) as usize}}
    #[inline(always)]
    fn offset_r<const N: usize>(offset: isize) -> isize{ - max_offset::<N>(offset)}
     #[inline(always)]
    fn n_front_r<const N: usize>(offset: isize) -> usize {if offset_r::<N>(offset) < 0 {(-offset_r::<N>(offset)) as usize} else {0}}
     #[inline(always)]
    fn n_back_r<const N: usize>(offset: isize) -> usize {if max_offset_r::<N>(offset) < 0 {0} else {(-max_offset_r::<N>(offset)) as usize}}

    
    pub fn update_step<T, const N: usize, U, BC>(offset: isize, coefs: &[T; N], x: &[U], y: &mut[U], _bc: &BC)
    where 
        T: Copy,
        U: Num + NumAssignOps + Copy + MulAdd<T, U, Output = U>,
        BC: BoundaryExtension,
    {
        let nf = n_front::<N>(offset);

        let mut y_iter = y.iter_mut().enumerate();
        // front boundary loop
        for (i, v) in y_iter.by_ref().take(nf){
            let i_offset = (i as isize) + offset;
            let mut c_iter = coefs.iter().enumerate();
            *v += c_iter.by_ref().take(nf - i)
                .fold(U::zero(), |acc, (idx, c)|
                    {
                        let xo = BC::extend_front(x, i_offset + (idx as isize));
                        xo.mul_add(
                            *c,
                            acc
                        )
                    }
                );

            *v += c_iter
                .zip(x.iter())
                .fold(U::zero(), |acc, ((_, c), xo)| xo.mul_add(*c, acc));
        }

        let nx_skip = if offset < 0 {0} else {offset as usize};
        // main loop
        y_iter.by_ref()
            .zip(x.windows(N).skip(nx_skip))
            .for_each(|((_, v), xs)|{
                *v += coefs.iter()
                    .zip(xs.iter())
                    .fold(
                        U::zero(), |acc, (c, xo)| xo.mul_add(*c, acc)
                    );
            });

        // back boundary loop
        for (i, v) in y_iter{
            let mut c_iter = coefs.iter().enumerate();
            
            let ix_start = ((i as isize) + offset) as usize;
            // iterate from ix_start until end of x
            *v += c_iter.by_ref()
                .zip(x.iter().skip(ix_start))
                .fold(U::zero(), |acc, ((_idx, c), xo)| {
                    xo.mul_add(*c,acc)
                });
            
            // iterate the rest with boundary extension
            *v += c_iter.fold(U::zero(), |acc, (idx, c)| {
                let xo = BC::extend_back(x, ix_start + idx);
                xo.mul_add(*c, acc)
            });
        }

    }

    pub fn scale_slice<T: Copy, U: MulAssign<T>>(s: T, x: &mut [U]){
        x.iter_mut().for_each(|v| *v *= s);
    }

    #[cfg(test)]
    mod tests{
        use super::*;
        use crate::boundarys::ZeroBoundary;

        #[test]
        fn test_offsets(){
            assert_eq!(max_offset::<5>(-2), 6);
            assert_eq!(n_front::<5>(-2), 2);
        }

        #[test]
        fn test_neg_offset_update_step(){

            let coefs = [1,1,1];
            let offset = -1;

            // x and y same length;
            let data = [1,2,3,4,5];
            let mut output = [0; 5];

            update_step(offset, &coefs, &data, &mut output, &ZeroBoundary{});

            assert_eq!(output, [3,6,9,12,9]);


            // x 1 smaller than y;
            let data = [1,2,3,4];
            let mut output = [0; 5];

            update_step(offset, &coefs, &data, &mut output, &ZeroBoundary{});

            assert_eq!(output, [3,6,9,7,4]);

            // x 1 larger than y;
            let data = [1, 2, 3, 4, 5];
            let mut output = [0; 4];

            update_step(offset, &coefs, &data, &mut output, &ZeroBoundary{});

            assert_eq!(output, [3,6,9, 12]);
        }


        #[test]
        fn test_pos_offset_update_step(){

            let coefs = [1,1,1];
            let offset = 1;

            // x and y same length;
            let data = [1,2,3,4,5];
            let mut output = [0; 5];

            update_step(offset, &coefs, &data, &mut output, &ZeroBoundary{});

            assert_eq!(output, [9,12,9,5,0]);


            // x 1 smaller than y;
            let data = [1,2,3,4];
            let mut output = [0; 5];

            update_step(offset, &coefs, &data, &mut output, &ZeroBoundary{});

            assert_eq!(output, [9,7,4,0,0]);

            // x 1 larger than y;
            let data = [1, 2, 3, 4, 5];
            let mut output = [0; 4];

            update_step(offset, &coefs, &data, &mut output, &ZeroBoundary{});

            assert_eq!(output, [9, 12, 9, 5]);
        }

    }
}

pub trait LiftedStep<SD>{
    fn forward<BC: BoundaryExtension>(&self, s: &mut[SD], d: &mut[SD], bc: &BC);
    fn inverse<BC: BoundaryExtension>(&self, s: &mut[SD], d: &mut[SD], bc: &BC);
}

pub struct UpdateD<T:Copy + Neg<Output=T>, const N: usize>{
    pub offset: isize,
    pub coefs: [T; N],
}

impl<T: Copy + Neg<Output=T>, U, const N: usize> LiftedStep<U> for UpdateD<T, N>
where
    U: Num + Copy + MulAdd<T, U, Output=U> + NumAssignOps,
{
    fn forward<BC: BoundaryExtension>(&self, s: &mut[U], d: &mut[U], bc: &BC){
        ops::update_step(self.offset, &self.coefs, s, d, bc);
    }
    fn inverse<BC: BoundaryExtension>(&self, s: &mut[U], d: &mut[U], bc: &BC){
        let inv_coefs: [T; N] = std::array::from_fn(|i| -self.coefs[i]);
        ops::update_step(self.offset, &inv_coefs, s, d, bc);
    }
}

pub struct UpdateS<T: Copy + Neg<Output=T>, const N: usize>{
    pub offset: isize,
    pub coefs: [T; N],
}

impl<T: Copy + Neg<Output=T>, U, const N: usize> LiftedStep<U> for UpdateS<T, N>
where
    U: Num + Copy + MulAdd<T, U, Output=U> + NumAssignOps
{
    fn forward<BC: BoundaryExtension>(&self, s: &mut[U], d: &mut[U], bc: &BC){
        ops::update_step(self.offset, &self.coefs, d, s, bc);
    }
    fn inverse<BC: BoundaryExtension>(&self, s: &mut[U], d: &mut[U], bc: &BC){
        let inv_coefs: [T; N] = std::array::from_fn(|i| -self.coefs[i]);
        ops::update_step(self.offset, &inv_coefs, d, s, bc);
    }
}

pub struct ScaleStep<T: Num + Copy>{
    pub scale: T
}
impl<T: Num + Copy, U: MulAssign<T>> LiftedStep<U> for ScaleStep<T>{
    fn forward<BC: BoundaryExtension>(&self, s: &mut[U], d: &mut[U], _bc: &BC){
        ops::scale_slice(self.scale, s);
        ops::scale_slice(T::one() / self.scale, d);
    }

    fn inverse<BC: BoundaryExtension>(&self, s: &mut[U], d: &mut[U], _bc: &BC){
        ops::scale_slice(T::one() / self.scale, s);
        ops::scale_slice(self.scale, d);
    }

}