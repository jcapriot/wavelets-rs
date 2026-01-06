use pulp::Simd;
use core::fmt::Debug;

use num_traits::{Num, NumRef, NumAssign, NumAssignRef};
use bytemuck::Pod;

pub trait SimdArch: Copy + Default + Send + Sync {
	fn dispatch<R>(self, f: impl pulp::WithSimd<Output = R>) -> R;
}
impl SimdArch for pulp::Arch {
	#[inline]
	fn dispatch<R>(self, f: impl pulp::WithSimd<Output = R>) -> R {
		self.dispatch(f)
	}
}
impl SimdArch for pulp::Scalar {
	#[inline]
	fn dispatch<R>(self, f: impl pulp::WithSimd<Output = R>) -> R {
		f.with_simd(self)
	}
}


pub trait SimdOps:
    Debug
    + Clone
    + Copy
    + Num
    + NumRef
    + NumAssign
    + NumAssignRef
{

    type SimdVec<S: Simd>: Pod + Debug;

	fn as_simd<S: Simd>(slice: &[Self]) -> (& [Self::SimdVec<S>], & [Self]);
	fn as_mut_simd<S: Simd>(slice: &mut [Self]) -> (& mut [Self::SimdVec<S>], & mut [Self]);
    fn simd_splat<S: Simd>(simd: &S, value: &Self) -> Self::SimdVec<S>;
    fn simd_reduce_sum<S: Simd>(simd: &S, a: Self::SimdVec<S>) -> Self;
    fn simd_neg<S: Simd>(simd: &S, a: Self::SimdVec<S>) -> Self::SimdVec<S>;
    fn simd_mul<S: Simd>(simd: &S, a: Self::SimdVec<S>, b: Self::SimdVec<S>) -> Self::SimdVec<S>;
    fn simd_add<S: Simd>(simd: &S, a: Self::SimdVec<S>, b: Self::SimdVec<S>) -> Self::SimdVec<S>;
    fn simd_sub<S: Simd>(simd: &S, a: Self::SimdVec<S>, b: Self::SimdVec<S>) -> Self::SimdVec<S>;
    fn simd_mul_add<S: Simd>(simd: &S, a: Self::SimdVec<S>, b: Self::SimdVec<S>, c: Self::SimdVec<S>) -> Self::SimdVec<S>;
}

impl SimdOps for f32{
	type SimdVec<S: Simd> = S::f32s;

	#[inline(always)]
	fn as_simd<S: Simd>(slice: &[Self]) -> (&[Self::SimdVec<S>], &[Self]){
		S::as_simd_f32s(slice)
	}

	#[inline(always)]
	fn as_mut_simd<S: Simd>(slice: & mut [Self]) -> (& mut [Self::SimdVec<S>], & mut [Self]){
		S::as_mut_simd_f32s(slice)
	}

	#[inline(always)]
    fn simd_splat<S: Simd>(simd: &S, value: &Self) -> Self::SimdVec<S>{
        simd.splat_f32s(*value)
    }
	#[inline(always)]
    fn simd_reduce_sum<S: Simd>(simd: &S, a: Self::SimdVec<S>) -> Self{
		simd.reduce_sum_f32s(a)
	}
	#[inline(always)]
    fn simd_neg<S: Simd>(simd: &S, a: Self::SimdVec<S>) -> Self::SimdVec<S>{
		simd.neg_f32s(a)
	}
	#[inline(always)]
    fn simd_mul<S: Simd>(simd: &S, a: Self::SimdVec<S>, b: Self::SimdVec<S>) -> Self::SimdVec<S>{
		simd.mul_f32s(a, b)
	}
	#[inline(always)]
    fn simd_add<S: Simd>(simd: &S, a: Self::SimdVec<S>, b: Self::SimdVec<S>) -> Self::SimdVec<S>{
		simd.add_f32s(a, b)
	}
	#[inline(always)]
    fn simd_sub<S: Simd>(simd: &S, a: Self::SimdVec<S>, b: Self::SimdVec<S>) -> Self::SimdVec<S>{
		simd.add_f32s(a, b)
	}
	#[inline(always)]
    fn simd_mul_add<S: Simd>(simd: &S, a: Self::SimdVec<S>, b: Self::SimdVec<S>, c: Self::SimdVec<S>) -> Self::SimdVec<S>{
		simd.mul_add_f32s(a, b, c)
	}
}