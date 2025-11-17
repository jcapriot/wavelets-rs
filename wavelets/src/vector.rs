use std::ops::{Mul, Add, Sub, Div, Rem, Neg, AddAssign, SubAssign, MulAssign, DivAssign};
use num_traits::{MulAdd, Num, One, Zero};

use itertools::Itertools;

#[derive(PartialEq, Eq, Copy, Clone, Hash, Debug)]
pub struct Vector<T, const N: usize>{
    data: [T; N],
}

impl<T: Clone, const N: usize> Vector<T,N >{
    const WIDTH: usize = N;
    
    pub fn new(a: [T; N]) -> Self{
        Vector{data: a}
    }

    pub fn splat(a: T) -> Self{
        Self{data: std::array::from_fn(|_| a.clone())}
    }

    pub fn fill(&mut self, a: &[T; N]){
        self.data = a.clone();
    }
}

impl<T: Clone + PartialEq, const N: usize> PartialEq<T> for Vector<T, N>{
    fn eq(&self, other: &T) -> bool{
        self.data.iter().fold(true, |acc, v| acc && (v == other))
    }

}

impl<T: Neg<Output=T>, const N: usize> Neg for Vector<T, N>{
    type Output = Self;

    fn neg(self) -> Self::Output{
        Self{data: self.data.map(|v| v.neg())}
    }
}
impl<T: Clone + Neg<Output=T>, const N: usize> Neg for &Vector<T, N>{
    type Output = Vector<T, N>;

    fn neg(self) -> Self::Output{
        Self::Output{data: self.data.clone().map(|v| v.neg())}
    }
}

macro_rules! impl_vector_ops {
    ($($trait:ident, $method:ident);+ $(;)?) => {
        $(
            // Vector <op> Vector
            impl<T, const N: usize> $trait<Vector<T, N>> for Vector<T, N>
            where
                T: Clone + $trait<T, Output = T>,
            {
                type Output = Vector<T, N>;

                #[inline(always)]
                fn $method(self, rhs: Vector<T, N>) -> Self::Output {
                    let data = self.data
                        .into_iter()
                        .zip(rhs.data.into_iter())
                        .map(|(a, b)| a.$method(b))
                        .collect_array()
                        .unwrap();
                    Self { data }
                }
            }

            // Ref Vector <op> Ref Vector
            impl<T, const N: usize> $trait<&Vector<T, N>> for &Vector<T, N>
            where
                T: Clone + $trait<T, Output = T>,
            {
                type Output = Vector<T, N>;

                #[inline(always)]
                fn $method(self, rhs: &Vector<T, N>) -> Self::Output {
                    self.clone().$method(rhs.clone())
                }
            }

            // Vector <op> scalar T
            impl<T, const N: usize> $trait<T> for Vector<T, N>
            where
                T: Clone + $trait<T, Output = T>,
            {
                type Output = Vector<T, N>;

                #[inline(always)]
                fn $method(self, rhs: T) -> Self::Output {
                    Self { data: self.data.map(|v| v.$method(rhs.clone())) }
                }
            }

            // Ref Vector <op> Ref Scalar
            impl<T, const N: usize> $trait<&T> for &Vector<T, N>
            where
                T: Clone + $trait<T, Output = T>,
            {
                type Output = Vector<T, N>;

                #[inline(always)]
                fn $method(self, rhs: &T) -> Self::Output {
                    self.clone().$method(rhs.clone())
                }
            }
        )+
    };
}

impl_vector_ops!{
    Add, add;
    Sub, sub;
    Mul, mul;
    Div, div;
    Rem, rem;
}

impl<T: Clone + Num, const N: usize> Zero for Vector<T, N>{
    #[inline]
    fn zero() -> Self{
        Self{
            data: std::array::from_fn(|_| Zero::zero())
        }
    }

    #[inline]
    fn is_zero(&self) -> bool{
        self.data.iter().fold(true, |ac, ell| ac && ell.is_zero())
    }

    #[inline]
    fn set_zero(&mut self){
        self.data.iter_mut().for_each(|v| v.set_zero());
    }
}

impl<T: Clone + Num, const N: usize> One for Vector<T, N>{
    #[inline]
    fn one() -> Self{
        Self{
            data: std::array::from_fn(|_| One::one())
        }
    }

    #[inline]
    fn is_one(&self) -> bool{
        self.data.iter().fold(true, |ac, ell| ac && ell.is_one())
    }

    #[inline]
    fn set_one(&mut self){
        self.data.iter_mut().for_each(|v| v.set_one());
    }
}

macro_rules! impl_vector_op_assign {
    ($($trait:ident, $method:ident);+ $(;)?) => {
        $(
            // Vector <op> Vector
            impl<T, const N: usize> $trait<Vector<T, N>> for Vector<T, N>
            where
                T: $trait,
            {

                #[inline(always)]
                fn $method(&mut self, rhs: Vector<T, N>) {
                    self.data.iter_mut()
                        .zip(rhs.data.into_iter())
                        .for_each(|(a, b)| a.$method(b));
                }
            }

            // Ref Vector <op> Ref Vector
            impl<T, const N: usize> $trait<&Vector<T, N>> for Vector<T, N>
            where
                T: Clone + $trait,
            {
                #[inline(always)]
                fn $method(&mut self, rhs: &Vector<T, N>) {
                    self.$method(rhs.clone());
                }
            }

            // Vector <op> scalar T
            impl<T, const N: usize> $trait<T> for Vector<T, N>
            where
                T: Clone + $trait,
            {

                #[inline(always)]
                fn $method(&mut self, rhs: T){
                    self.data.iter_mut().for_each(|v| v.$method(rhs.clone()))
                }
            }

            // Ref Vector <op> Ref Scalar
            impl<T, const N: usize> $trait<&T> for Vector<T, N>
            where
                T: Clone + $trait,
            {

                #[inline(always)]
                fn $method(& mut self, rhs: &T){
                    self.$method(rhs.clone());
                }
            }
        )+
    };
}

impl_vector_op_assign!{
    AddAssign, add_assign;
    SubAssign, sub_assign;
    MulAssign, mul_assign;
    DivAssign, div_assign;
}

#[derive(Debug, PartialEq)]
pub struct ParseVectorError<E> {
    kind: VectorErrorKind<E>,
}

#[derive(Debug, PartialEq)]
enum VectorErrorKind<E> {
    ParseError(E),
    InconsistentLength
}

impl<E> ParseVectorError<E> {

    fn inconsistent_length() -> Self {
        ParseVectorError {
            kind: VectorErrorKind::InconsistentLength,
        }
    }

    fn from_error(error: E) -> Self {
        ParseVectorError {
            kind: VectorErrorKind::ParseError(error),
        }
    }
}

impl<T, const N: usize> Num for Vector<T, N>
where
    T: Num + Clone,
{
    type FromStrRadixErr = ParseVectorError<T::FromStrRadixErr>;

    fn from_str_radix(src: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        // This expects input like: "1,2,3" or "[1,2,3]"
        // You can adapt to your own format.
        let cleaned = src
            .trim()
            .trim_start_matches('[')
            .trim_end_matches(']');

        let mut out = Vec::with_capacity(N);

        for part in cleaned.split(',') {
            let val = T::from_str_radix(part.trim(), radix)
                .map_err(|err| Self::FromStrRadixErr::from_error(err))?;
            out.push(val);
        }

        // Too many or too few elements → panic or convert to error
        let arr: [T; N] = out.try_into()
            .map_err(|_| {
                Self::FromStrRadixErr::inconsistent_length()
            })?;

        Ok(Self { data: arr })
    }
}


#[cfg(test)]
mod tests{
    use super::*;

    const N: usize = 4;
    type P = f64;

    #[test]
    fn simple_implements(){
        let one = Vector::<P,N>::one();
        assert_eq!(one, 1.0);
        assert_eq!(one, Vector::splat(1.0));
        assert_ne!(one, 2.0);
        assert_ne!(one, Vector::splat(2.0));

        let zero = Vector::<P, N>::zero();
        assert_eq!(zero, 0.0);
    }

    #[test]
    fn ops(){
        let one = Vector::<P, N>::one();
        let two = Vector::<P, N>::splat(2.0);
        let half = Vector::splat(1.0 / 2.0);

        assert_eq!(one + 1.0, two);
        assert_eq!(one + one, two);

        assert_eq!(one * 2.0, two);
        assert_eq!(one * two, two);

        assert_eq!(one / 2.0, half);    
        assert_eq!(one / 2.0, half);
        
        assert_eq!(-one, -1.0);
        assert_eq!(-one, Vector::splat(-1.0));
    
        assert_eq!(one - 2.0, -one);
        assert_eq!(one - two, -one);
    
        assert_eq!((one + 2.0) % 2.0, one);
        assert_eq!((one + two) % two, one);
    }

    #[test]
    fn ref_ops(){
        let one = Vector::<P, N>::one();
        let two = Vector::<P, N>::splat(2.0);
        let half = Vector::splat(1.0 / 2.0);

        assert_eq!(&one + &1.0, two);
        assert_eq!(&one + &one, two);

        assert_eq!(&one * &2.0, two);
        assert_eq!(&one * &two, two);
        assert_eq!(&one / &2.0, half);
        assert_eq!(&one / &two, half);
        assert_eq!(&one - &2.0, -one);
        assert_eq!(&one - &two, -one);
    
        assert_eq!(&(one + 2.0) % &2.0, one);
        assert_eq!(&(one + two) % &two, one);
    }
    
    #[test]
    fn from_string(){
        let res: Result<Vector<P,N>, <Vector<P,N> as Num>::FromStrRadixErr> =
            Num::from_str_radix("1,1,1,1", 10);
        assert_eq!(res.unwrap(), Vector::<P, N>::splat(1.0));
        
        let res: Result<Vector<P,N>, <Vector<P,N> as Num>::FromStrRadixErr> =

            Num::from_str_radix("[1,1,1,1]", 10);
        assert_eq!(res.unwrap(), Vector::<P, N>::splat(1.0));
    }
}