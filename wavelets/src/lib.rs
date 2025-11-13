pub mod boundarys;
pub mod lwt;

pub enum TransformDirection{
    Forward,
    Inverse,
}

pub trait WaveletLength{
    const WIDTH: usize;
}


#[cfg(test)]
mod tests {
    #[macro_export]
    macro_rules! test_approx_equal {
        ($actual:expr, $desired:expr, $rtol:expr, $atol:expr) => {{
            let actual = $actual;
            let desired = $desired;
            let rtol = $rtol;
            let atol = $atol;

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
        }};
    }
}


mod hid{
    use num_traits::Num;
    use num_traits::MulAdd;
    
    #[derive(PartialEq, Eq, Copy, Clone, Hash, Debug, Default)]
    pub struct Complex<T>{
        pub re: T,
        pub im: T,
    }

    impl<T> Complex<T> {
    /// Create a new `Complex`
    #[inline]
    pub const fn new(re: T, im: T) -> Self {
        Complex { re, im }
    }
}


    // (a + i b) * (c + i d) + (e + i f) == ((a*c + e) - b*d) + i (a*d + (b*c + f))
    impl<T: Clone + Num + MulAdd<Output = T>> MulAdd<Complex<T>> for Complex<T> {
        type Output = Complex<T>;

        #[inline]
        fn mul_add(self, other: Complex<T>, add: Complex<T>) -> Complex<T> {
            let re = self.re.clone().mul_add(other.re.clone(), add.re)
                - (self.im.clone() * other.im.clone()); // FIXME: use mulsub when available in rust
            let im = self.re.mul_add(other.im, self.im.mul_add(other.re, add.im));
            Complex::new(re, im)
        }
    }
    impl<'a, 'b, T: Clone + Num + MulAdd<Output = T>> MulAdd<&'b Complex<T>> for &'a Complex<T> {
        type Output = Complex<T>;

        #[inline]
        fn mul_add(self, other: &Complex<T>, add: &Complex<T>) -> Complex<T> {
            self.clone().mul_add(other.clone(), add.clone())
        }
    }

    // (a + i b) * c + (e + i f) == (a*c + e) + i (b*c + f)
    impl<T: Clone + Num + MulAdd<Output = T>> MulAdd<T, Complex<T>> for Complex<T> {
        type Output = Complex<T>;

        #[inline]
        fn mul_add(self, other: T, add: Complex<T>) -> Complex<T> {
            let re = self.re.mul_add(other.clone(), add.re);
            let im = self.im.mul_add(other, add.im);
            Complex::new(re, im)
        }
    }
    impl<'a, 'b, T: Clone + Num + MulAdd<Output = T>> MulAdd<&'b T, &'b Complex<T>> for &'a Complex<T> {
        type Output = Complex<T>;

        #[inline]
        fn mul_add(self, other: &T, add: &Complex<T>) -> Complex<T> {
            self.clone().mul_add(other.clone(), add.clone())
        }
    }

    // (a + i b) * c + e == (a*c + e) + i (b*c)
    impl<T: Clone + Num + MulAdd<Output = T>> MulAdd<T, T> for Complex<T> {
        type Output = Complex<T>;

        #[inline]
        fn mul_add(self, other: T, add: T) -> Complex<T> {
            let re = self.re.mul_add(other.clone(), add);
            let im = self.im * other;
            Complex::new(re, im)
        }
    }
    impl<'a, 'b, T: Clone + Num + MulAdd<Output = T>> MulAdd<&'b T, &'b T> for &'a Complex<T> {
        type Output = Complex<T>;

        #[inline]
        fn mul_add(self, other: &T, add: &T) -> Complex<T> {
            self.clone().mul_add(other.clone(), add.clone())
        }
    }

    // (a + i b) * c + e == (a*c + e) + i (b*c)
    impl<T: Clone + Num + MulAdd<Output = T>> MulAdd<Complex<T>, T> for Complex<T> {
        type Output = Complex<T>;

        #[inline]
        fn mul_add(self, other: Complex<T>, add: T) -> Complex<T> {
            let re = self.re.clone().mul_add(other.re.clone(), add)
                - (self.im.clone() * other.im.clone()); // FIXME: use mulsub when available in rust
            let im = self.re.mul_add(other.im, self.im * other.re);
            Complex::new(re, im)
        }
    }
    impl<'a, 'b, T: Clone + Num + MulAdd<Output = T>> MulAdd<&'b Complex<T>, &'b T> for &'a Complex<T> {
        type Output = Complex<T>;

        #[inline]
        fn mul_add(self, other: &Complex<T>, add: &T) -> Complex<T> {
            self.clone().mul_add(other.clone(), add.clone())
        }
    }

}
