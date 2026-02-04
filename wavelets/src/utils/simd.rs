use itertools::Itertools;
use num_traits::{FromPrimitive, NumAssignOps, NumOps, Zero};
use std::ops::{Add, Div, Mul, Neg, Rem, Sub};
use std::ops::{AddAssign, DivAssign, MulAssign, RemAssign, SubAssign};

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Simd<T, const N: usize> {
    arr: [T; N],
}
impl<T: Clone, const N: usize> Simd<T, N> {
    #[inline(always)]
    pub fn clone(&mut self, arr: [T; N]) {
        self.arr.iter_mut().zip(arr).for_each(|(a, b)| *a = b);
    }
    #[inline(always)]
    pub fn load(&mut self, arr: &[T; N]) {
        self.arr
            .iter_mut()
            .zip(arr)
            .for_each(|(a, b)| *a = b.clone());
    }
    #[inline(always)]
    pub fn gather(&mut self, arr: [&T; N]) {
        self.arr
            .iter_mut()
            .zip(arr)
            .for_each(|(a, b)| *a = b.clone());
    }
    #[inline(always)]
    pub fn store(&self, arr: &mut [T; N]) {
        self.arr.iter().zip(arr).for_each(|(a, b)| *b = a.clone());
    }
    #[inline(always)]
    pub fn scatter(&self, arr: [&mut T; N]) {
        self.arr.iter().zip(arr).for_each(|(a, b)| *b = a.clone());
    }
}

impl<T: Zero, const N: usize> Zero for Simd<T, N> {
    #[inline(always)]
    fn zero() -> Self {
        Self {
            arr: std::array::from_fn(|_| T::zero()),
        }
    }
    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.arr.iter().all(|v| T::is_zero(v))
    }
    #[inline(always)]
    fn set_zero(&mut self) {
        self.arr.iter_mut().for_each(|v| T::set_zero(v));
    }
}

pub trait SimdSliceView<T, const N: usize> {
    fn as_simd_slice(&self) -> (&[T], &[Simd<T, N>], &[T]);
    fn as_simd_slice_mut(&mut self) -> (&mut [T], &mut [Simd<T, N>], &mut [T]);
}

impl<T, const N: usize> SimdSliceView<T, N> for [T] {
    #[inline(always)]
    fn as_simd_slice(&self) -> (&[T], &[Simd<T, N>], &[T]) {
        unsafe { self.align_to::<Simd<T, N>>() }
    }
    #[inline(always)]
    fn as_simd_slice_mut(&mut self) -> (&mut [T], &mut [Simd<T, N>], &mut [T]) {
        unsafe { self.align_to_mut::<Simd<T, N>>() }
    }
}

impl<T: Clone, const N: usize> From<&[T; N]> for Simd<T, N> {
    #[inline(always)]
    fn from(value: &[T; N]) -> Self {
        let arr = std::array::from_fn(|i| value[i].clone());
        Self { arr }
    }
}
impl<T: Clone, const N: usize> From<[&T; N]> for Simd<T, N> {
    #[inline(always)]
    fn from(value: [&T; N]) -> Self {
        let arr = std::array::from_fn(|i| value[i].clone());
        Self { arr }
    }
}

impl<T, const N: usize> From<[T; N]> for Simd<T, N> {
    #[inline(always)]
    fn from(arr: [T; N]) -> Self {
        Self { arr }
    }
}

impl<T: Neg<Output = T>, const N: usize> Neg for Simd<T, N> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self {
            arr: self.arr.map(|v| -v),
        }
    }
}

macro_rules! impl_bin_ops {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<T, const N: usize> $trait<Self> for Simd<T, N>
        where
            T: $trait<Output = T>
        {
            type Output = Self;
            #[inline(always)]
            fn $method(self, rhs: Simd<T, N>) -> Self::Output {
                let arr = self.arr.into_iter()
                    .zip(rhs.arr.into_iter())
                    .map(|(a, b)| a $op b)
                    .collect_array()
                    .expect("left and right are both const N length");
                Self { arr }
            }
        }

        impl<T, const N: usize> $trait<T> for Simd<T, N>
        where
            T: Clone + $trait<Output=T>,
        {
            type Output = Self;
            #[inline(always)]
            fn $method(self, rhs: T) -> Self::Output {
                let arr = self.arr.map(|v| v $op rhs.clone());
                Self { arr }
            }
        }

        impl<T, const N: usize> $trait<&T> for Simd<T, N>
        where
            T: Clone + $trait<Output=T>,
        {
            type Output = Self;
            #[inline(always)]
            fn $method(self, rhs: &T) -> Self::Output {
                let arr = self.arr.map(|v| v $op rhs.clone());
                Self { arr }
            }
        }
    };
}
impl_bin_ops!(Add, add, +);
impl_bin_ops!(Sub, sub, -);
impl_bin_ops!(Mul, mul, *);
impl_bin_ops!(Div, div, /);
impl_bin_ops!(Rem, rem, %);

macro_rules! impl_num_assign_ops {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<T, const N: usize> $trait<Simd<T, N>> for Simd<T, N>
        where
            T: $trait<T>,
        {
            #[inline(always)]
            fn $method(&mut self, rhs: Simd<T, N>){
                self.arr.iter_mut().zip(rhs.arr.into_iter()).for_each(|(a,b)| *a $op b);
            }
        }
        impl<T, const N: usize> $trait<T> for Simd<T, N>
        where
            T: Clone + $trait<T>,
        {
            #[inline(always)]
            fn $method(&mut self, rhs: T){
                self.arr.iter_mut().for_each(|a| *a $op rhs.clone());
            }
        }
        impl<T, const N: usize> $trait<&T> for Simd<T, N>
        where
            T: Clone + $trait<T>,
        {
            #[inline(always)]
            fn $method(&mut self, rhs: &T){
                self.arr.iter_mut().for_each(|a| *a $op rhs.clone());
            }
        }
    };
}
impl_num_assign_ops!(AddAssign, add_assign, +=);
impl_num_assign_ops!(SubAssign, sub_assign, -=);
impl_num_assign_ops!(MulAssign, mul_assign, *=);
impl_num_assign_ops!(DivAssign, div_assign, /=);
impl_num_assign_ops!(RemAssign, rem_assign, %=);

impl<T, const N: usize> crate::Transformable for Simd<T, N>
where
    T: Clone + NumOps + NumAssignOps + std::fmt::Debug + FromPrimitive + Neg<Output = T>,
{
    type ScalarType = T;
}
