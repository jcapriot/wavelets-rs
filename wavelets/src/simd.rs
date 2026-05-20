use crate::Transformable;
use pulp::cast;
use std::marker::PhantomData;
use std::sync::LazyLock;

/// Types that know their SIMD lane width for the current CPU.
///
/// This is a marker/helper trait used by [`SimdTransformable`] to determine how many
/// elements fit in one SIMD register.  You generally do not need to implement or call
/// this trait directly.
pub trait Alignable {
    /// Number of `Self` elements that fit in one SIMD register under `simd`.
    fn simd_lanes<S: pulp::Simd>(_: S) -> usize;

    /// Number of `Self` elements per SIMD register for the best instruction set
    /// available at runtime (dispatched via [`ARCH`]).
    fn lanes() -> usize {
        struct Impl<T: ?Sized>(PhantomData<T>);
        impl<T> WithSimd for Impl<T>
        where
            T: Alignable + ?Sized,
        {
            type Output = usize;

            #[inline(always)]
            fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
                T::simd_lanes(simd)
            }
        }
        crate::simd::ARCH.dispatch_wvlt(Impl(PhantomData::<Self>))
    }
}

macro_rules! impl_alignable {
    ($t:ty, $n:tt) => {
        impl Alignable for $t {
            fn simd_lanes<S: pulp::Simd>(_: S) -> usize {
                S::$n
            }
        }
    };
}

impl_alignable!(i8, I8_LANES);
impl_alignable!(i16, I16_LANES);
impl_alignable!(i32, I32_LANES);
impl_alignable!(i64, I64_LANES);
impl_alignable!(f32, F32_LANES);
impl_alignable!(f64, F64_LANES);
impl_alignable!(num_complex::Complex32, C32_LANES);
impl_alignable!(num_complex::Complex64, C64_LANES);

/// Extension of [`pulp::Simd`] with mixed-type arithmetic operations needed by the
/// wavelet kernels.
///
/// The additional methods cover `f32 × c32` and `f64 × c64` fused multiply-add
/// (and its negated form), as well as plain multiply and divide variants.  Default
/// implementations are provided for all SIMD backends; the scalar backend overrides
/// the complex methods to avoid register-width casts that are invalid at scalar width.
pub trait Simd: pulp::Simd {
    /// Compute `(-a) * b + c` for `f32` SIMD vectors.
    #[inline(always)]
    fn neg_mul_add_f32s(self, a: Self::f32s, b: Self::f32s, c: Self::f32s) -> Self::f32s {
        let neg_a = self.neg_f32s(a);
        self.mul_add_f32s(neg_a, b, c)
    }
    /// Compute `(-a) * b + c` for `f64` SIMD vectors.
    #[inline(always)]
    fn neg_mul_add_f64s(self, a: Self::f64s, b: Self::f64s, c: Self::f64s) -> Self::f64s {
        let neg_a = self.neg_f64s(a);
        self.mul_add_f64s(neg_a, b, c)
    }
    /// Compute `(-a) * b + c` where `a`, `c` are `c32` vectors and `b` is an `f32` splat.
    #[inline(always)]
    fn neg_mul_add_c32s_f32s(self, a: Self::c32s, b: Self::f32s, c: Self::c32s) -> Self::c32s {
        cast(self.neg_mul_add_f32s(cast(a), b, cast(c)))
    }
    /// Compute `(-a) * b + c` where `a`, `c` are `c64` vectors and `b` is an `f64` splat.
    #[inline(always)]
    fn neg_mul_add_c64s_f64s(self, a: Self::c64s, b: Self::f64s, c: Self::c64s) -> Self::c64s {
        cast(self.neg_mul_add_f64s(cast(a), b, cast(c)))
    }
    /// Compute `a * b + c` where `a`, `c` are `c32` vectors and `b` is an `f32` splat.
    #[inline(always)]
    fn mul_add_c32s_f32s(self, a: Self::c32s, b: Self::f32s, c: Self::c32s) -> Self::c32s {
        cast(self.mul_add_f32s(cast(a), b, cast(c)))
    }
    /// Compute `a * b + c` where `a`, `c` are `c64` vectors and `b` is an `f64` splat.
    #[inline(always)]
    fn mul_add_c64s_f64s(self, a: Self::c64s, b: Self::f64s, c: Self::c64s) -> Self::c64s {
        cast(self.mul_add_f64s(cast(a), b, cast(c)))
    }
    /// Multiply a `c32` vector by an `f32` splat.
    #[inline(always)]
    fn mul_c32s_f32s(self, a: Self::c32s, b: Self::f32s) -> Self::c32s {
        cast(self.mul_f32s(cast(a), b))
    }
    /// Multiply a `c64` vector by an `f64` splat.
    #[inline(always)]
    fn mul_c64s_f64s(self, a: Self::c64s, b: Self::f64s) -> Self::c64s {
        cast(self.mul_f64s(cast(a), b))
    }
    /// Divide a `c32` vector by an `f32` splat.
    #[inline(always)]
    fn div_c32s_f32s(self, a: Self::c32s, b: Self::f32s) -> Self::c32s {
        cast(self.div_f32s(cast(a), b))
    }
    /// Divide a `c64` vector by an `f64` splat.
    #[inline(always)]
    fn div_c64s_f64s(self, a: Self::c64s, b: Self::f64s) -> Self::c64s {
        cast(self.div_f64s(cast(a), b))
    }

    /// Vectorize the wvlt function.
    fn vectorize_wvlt<Op: WithSimd>(self, op: Op) -> Op::Output;
}

/// Callback type passed to simd dispatch functions: receives a concrete SIMD
/// backend (one that implements both [`pulp::Simd`] and the local [`Simd`] extension)
/// and returns an `Output` value.
pub trait WithSimd {
    /// The return type of [`with_simd`](WithSimd::with_simd).
    type Output;
    /// Called with the concrete SIMD backend selected at runtime.
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output;
}

impl Simd for pulp::Scalar {
    #[inline(always)]
    fn mul_add_c32s_f32s(self, a: Self::c32s, b: Self::f32s, c: Self::c32s) -> Self::c32s {
        Self::c32s {
            re: f32::mul_add(a.re, b, c.re),
            im: f32::mul_add(a.im, b, c.im),
        }
    }
    #[inline(always)]
    fn mul_add_c64s_f64s(self, a: Self::c64s, b: Self::f64s, c: Self::c64s) -> Self::c64s {
        Self::c64s {
            re: f64::mul_add(a.re, b, c.re),
            im: f64::mul_add(a.im, b, c.im),
        }
    }

    #[inline(always)]
    fn neg_mul_add_c32s_f32s(self, a: Self::c32s, b: Self::f32s, c: Self::c32s) -> Self::c32s {
        Self::c32s {
            re: f32::mul_add(-a.re, b, c.re),
            im: f32::mul_add(-a.im, b, c.im),
        }
    }
    #[inline(always)]
    fn neg_mul_add_c64s_f64s(self, a: Self::c64s, b: Self::f64s, c: Self::c64s) -> Self::c64s {
        Self::c64s {
            re: f64::mul_add(-a.re, b, c.re),
            im: f64::mul_add(-a.im, b, c.im),
        }
    }

    #[inline(always)]
    fn mul_c32s_f32s(self, a: Self::c32s, b: Self::f32s) -> Self::c32s {
        a * b
    }
    #[inline(always)]
    fn mul_c64s_f64s(self, a: Self::c64s, b: Self::f64s) -> Self::c64s {
        a * b
    }
    #[inline(always)]
    fn div_c32s_f32s(self, a: Self::c32s, b: Self::f32s) -> Self::c32s {
        a / b
    }
    #[inline(always)]
    fn div_c64s_f64s(self, a: Self::c64s, b: Self::f64s) -> Self::c64s {
        a / b
    }

    #[inline]
    fn vectorize_wvlt<Op: WithSimd>(self, op: Op) -> Op::Output {
        op.with_simd(self)
    }
}

pub(crate) trait Dispatch {
    fn dispatch_wvlt<Op: WithSimd>(self, op: Op) -> Op::Output;
}

macro_rules! impl_vectorize_wvlt {
    ($t:ty) => {
        #[inline(always)]
        fn vectorize_wvlt<Op: WithSimd>(self, op: Op) -> Op::Output {
            struct Impl<Op> {
                this: $t,
                op: Op,
            }
            impl<Op: WithSimd> pulp::NullaryFnOnce for Impl<Op> {
                type Output = Op::Output;

                #[inline(always)]
                fn call(self) -> Self::Output {
                    self.op.with_simd(self.this)
                }
            }
            self.vectorize(Impl { this: self, op })
        }
    };
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub(crate) mod x86 {
    use super::*;
    #[cfg(feature = "x86-v4")]
    use pulp::x86::V4;
    use pulp::x86::{V2, V3};

    impl Simd for V2 {
        impl_vectorize_wvlt! {V2}
    }

    impl Simd for V3 {
        #[inline(always)]
        fn neg_mul_add_f32s(self, a: Self::f32s, b: Self::f32s, c: Self::f32s) -> Self::f32s {
            cast!(self.fma._mm256_fnmadd_ps(cast!(a), cast!(b), cast!(c)))
        }
        #[inline(always)]
        fn neg_mul_add_f64s(self, a: Self::f64s, b: Self::f64s, c: Self::f64s) -> Self::f64s {
            cast!(self.fma._mm256_fnmadd_pd(cast!(a), cast!(b), cast!(c)))
        }

        impl_vectorize_wvlt! {V3}
    }
    #[cfg(feature = "x86-v4")]
    impl Simd for V4 {
        #[inline(always)]
        fn neg_mul_add_f32s(self, a: Self::f32s, b: Self::f32s, c: Self::f32s) -> Self::f32s {
            cast!(self.avx512f._mm512_fnmadd_ps(cast!(a), cast!(b), cast!(c)))
        }
        #[inline(always)]
        fn neg_mul_add_f64s(self, a: Self::f64s, b: Self::f64s, c: Self::f64s) -> Self::f64s {
            cast!(self.avx512f._mm512_fnmadd_pd(cast!(a), cast!(b), cast!(c)))
        }
        impl_vectorize_wvlt! {V4}
    }

    impl Dispatch for pulp::x86::Arch {
        #[inline]
        fn dispatch_wvlt<Op: WithSimd>(self, op: Op) -> Op::Output {
            match self {
                #[cfg(feature = "x86-v4")]
                Self::V4(simd) => Simd::vectorize_wvlt(simd, op),
                #[cfg(feature = "x86-v3")]
                Self::V3(simd) => Simd::vectorize_wvlt(simd, op),

                _ => Simd::vectorize_wvlt(pulp::Scalar, op),
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub(crate) mod wasm {
    use super::*;
    use pulp::wasm::{RelaxedSimd, Simd128};
    impl Simd for Simd128 {
        impl_vectorize_wvlt! {Simd128}
    }
    impl Simd for RelaxedSimd {
        impl_vectorize_wvlt! {RelaxedSimd}
    }

    impl Dispatch for pulp::wasm::Arch {
        #[inline]
        fn dispatch_wvlt<Op: WithSimd>(self, op: Op) -> Op::Output {
            match self {
                Self::RelaxedSimd(simd) => Simd::vectorize_wvlt(simd, op),
                Self::Simd128(simd) => Simd::vectorize_wvlt(simd, op),

                _ => Simd::vectorize_wvlt(pulp::Scalar, op),
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub(crate) mod aarch64 {
    use super::*;
    use core::arch::aarch64::{vfmsq_f32, vfmsq_f64};
    use pulp::aarch64::Neon;

    impl Simd for Neon {
        #[inline(always)]
        fn neg_mul_add_f32s(self, a: Self::f32s, b: Self::f32s, c: Self::f32s) -> Self::f32s {
            //SAFETY: Required to be executed on aarch64 with Neon instruction set.
            unsafe { cast!(vfmsq_f32(cast!(c), cast!(a), cast!(b))) }
        }
        #[inline(always)]
        fn neg_mul_add_f64s(self, a: Self::f64s, b: Self::f64s, c: Self::f64s) -> Self::f64s {
            //SAFETY: Required to be on executed on aarch64 with Neon instruction set.
            unsafe { cast!(vfmsq_f64(cast!(c), cast!(a), cast!(b))) }
        }
        impl_vectorize_wvlt! {Neon}
    }

    impl Dispatch for pulp::aarch64::Arch {
        #[inline]
        fn dispatch_wvlt<Op: WithSimd>(self, op: Op) -> Op::Output {
            match self {
                Self::Neon(simd) => Simd::vectorize_wvlt(simd, op),

                _ => Simd::vectorize_wvlt(pulp::Scalar, op),
            }
        }
    }
}

/// A [`Transformable`] type that can be processed with SIMD instructions.
///
/// The trait abstracts over the platform-specific SIMD vector types exposed by
/// [`pulp`], allowing the lifting and DWT kernels to be written once and compiled to
/// SSE/AVX/NEON/SVE etc. transparently.
///
/// Implemented for `f32`, `f64`, `Complex32`, and `Complex64`.  Integer types do not
/// implement this trait because they lack SIMD mul-add support via `pulp`.
pub trait SimdTransformable: Sized + Transformable + Alignable {
    /// SIMD vector type holding `lanes()` elements of `Self`.
    type Vector<S: Simd>: Copy + std::fmt::Debug;
    /// SIMD scalar-splat vector (for broadcasting a [`Transformable::Scalar`]).
    type SplatVector<S: Simd>: Copy + std::fmt::Debug;

    /// Split `x` into a prefix of aligned SIMD vectors and a scalar remainder.
    fn as_simd<S: Simd>(simd: S, x: &[Self]) -> (&[Self::Vector<S>], &[Self]);

    /// Mutable version of [`as_simd`](SimdTransformable::as_simd).
    fn as_mut_simd<S: Simd>(simd: S, x: &mut [Self]) -> (&mut [Self::Vector<S>], &mut [Self]);

    /// Broadcast scalar `v` into a splat vector.
    fn simd_splat<S: Simd>(simd: S, v: Self::Scalar) -> Self::SplatVector<S>;

    /// Fused multiply-add: `a * b + c` on SIMD vectors.
    fn simd_mul_add<S: Simd>(
        simd: S,
        a: Self::Vector<S>,
        b: Self::SplatVector<S>,
        c: Self::Vector<S>,
    ) -> Self::Vector<S>;
    /// Fused negate-multiply-add: `(-a) * b + c` on SIMD vectors.
    fn simd_negate_mul_add<S: Simd>(
        simd: S,
        a: Self::Vector<S>,
        b: Self::SplatVector<S>,
        c: Self::Vector<S>,
    ) -> Self::Vector<S>;
    /// Element-wise addition of two SIMD vectors.
    fn simd_add<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Vector<S>;
    /// Element-wise subtraction of two SIMD vectors.
    fn simd_sub<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Vector<S>;
    /// Element-wise multiplication of a SIMD vector by a splat scalar.
    fn simd_mul<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::SplatVector<S>) -> Self::Vector<S>;
    /// Element-wise division of a SIMD vector by a splat scalar.
    fn simd_div<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::SplatVector<S>) -> Self::Vector<S>;
}

impl SimdTransformable for f32 {
    type Vector<S: Simd> = S::f32s;
    type SplatVector<S: Simd> = Self::Vector<S>;

    #[inline(always)]
    fn as_simd<S: Simd>(_: S, slice: &[Self]) -> (&[Self::Vector<S>], &[Self]) {
        S::as_simd_f32s(slice)
    }

    #[inline(always)]
    fn as_mut_simd<S: Simd>(_: S, slice: &mut [Self]) -> (&mut [Self::Vector<S>], &mut [Self]) {
        S::as_mut_simd_f32s(slice)
    }

    #[inline(always)]
    fn simd_splat<S: Simd>(simd: S, v: Self) -> Self::SplatVector<S> {
        simd.splat_f32s(v)
    }

    #[inline(always)]
    fn simd_mul_add<S: Simd>(
        simd: S,
        a: Self::Vector<S>,
        b: Self::Vector<S>,
        c: Self::Vector<S>,
    ) -> Self::Vector<S> {
        simd.mul_add_f32s(a, b, c)
    }

    #[inline(always)]
    fn simd_negate_mul_add<S: Simd>(
        simd: S,
        a: Self::Vector<S>,
        b: Self::Vector<S>,
        c: Self::Vector<S>,
    ) -> Self::Vector<S> {
        simd.neg_mul_add_f32s(a, b, c)
    }

    #[inline(always)]
    fn simd_add<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Vector<S> {
        simd.add_f32s(a, b)
    }

    #[inline(always)]
    fn simd_sub<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Vector<S> {
        simd.sub_f32s(a, b)
    }

    #[inline(always)]
    fn simd_mul<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Vector<S> {
        simd.mul_f32s(a, b)
    }

    #[inline(always)]
    fn simd_div<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Vector<S> {
        simd.div_f32s(a, b)
    }
}

impl SimdTransformable for f64 {
    type Vector<S: Simd> = S::f64s;
    type SplatVector<S: Simd> = Self::Vector<S>;

    #[inline(always)]
    fn as_simd<S: Simd>(_: S, slice: &[Self]) -> (&[Self::Vector<S>], &[Self]) {
        S::as_simd_f64s(slice)
    }

    #[inline(always)]
    fn as_mut_simd<S: Simd>(_: S, slice: &mut [Self]) -> (&mut [Self::Vector<S>], &mut [Self]) {
        S::as_mut_simd_f64s(slice)
    }

    #[inline(always)]
    fn simd_splat<S: Simd>(simd: S, v: Self) -> Self::Vector<S> {
        simd.splat_f64s(v)
    }

    #[inline(always)]
    fn simd_mul_add<S: Simd>(
        simd: S,
        a: Self::Vector<S>,
        b: Self::Vector<S>,
        c: Self::Vector<S>,
    ) -> Self::Vector<S> {
        simd.mul_add_f64s(a, b, c)
    }

    #[inline(always)]
    fn simd_negate_mul_add<S: Simd>(
        simd: S,
        a: Self::Vector<S>,
        b: Self::Vector<S>,
        c: Self::Vector<S>,
    ) -> Self::Vector<S> {
        simd.neg_mul_add_f64s(a, b, c)
    }

    #[inline(always)]
    fn simd_add<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Vector<S> {
        simd.add_f64s(a, b)
    }

    #[inline(always)]
    fn simd_sub<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Vector<S> {
        simd.sub_f64s(a, b)
    }

    #[inline(always)]
    fn simd_mul<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Vector<S> {
        simd.mul_f64s(a, b)
    }

    #[inline(always)]
    fn simd_div<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Vector<S> {
        simd.div_f64s(a, b)
    }
}

impl SimdTransformable for num_complex::Complex32 {
    type Vector<S: Simd> = S::c32s;
    type SplatVector<S: Simd> = S::f32s;

    #[inline(always)]
    fn as_simd<S: Simd>(_: S, slice: &[Self]) -> (&[Self::Vector<S>], &[Self]) {
        S::as_simd_c32s(slice)
    }

    #[inline(always)]
    fn as_mut_simd<S: Simd>(_: S, slice: &mut [Self]) -> (&mut [Self::Vector<S>], &mut [Self]) {
        S::as_mut_simd_c32s(slice)
    }

    #[inline(always)]
    fn simd_splat<S: Simd>(simd: S, v: Self::Scalar) -> Self::SplatVector<S> {
        simd.splat_f32s(v)
    }

    #[inline(always)]
    fn simd_mul_add<S: Simd>(
        simd: S,
        a: Self::Vector<S>,
        b: Self::SplatVector<S>,
        c: Self::Vector<S>,
    ) -> Self::Vector<S> {
        simd.mul_add_c32s_f32s(a, b, c)
    }

    #[inline(always)]
    fn simd_negate_mul_add<S: Simd>(
        simd: S,
        a: Self::Vector<S>,
        b: Self::SplatVector<S>,
        c: Self::Vector<S>,
    ) -> Self::Vector<S> {
        simd.neg_mul_add_c32s_f32s(a, b, c)
    }

    #[inline(always)]
    fn simd_add<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Vector<S> {
        simd.add_c32s(a, b)
    }

    #[inline(always)]
    fn simd_sub<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Vector<S> {
        simd.sub_c32s(a, b)
    }

    #[inline(always)]
    fn simd_mul<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::SplatVector<S>) -> Self::Vector<S> {
        simd.mul_c32s_f32s(a, b)
    }

    #[inline(always)]
    fn simd_div<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::SplatVector<S>) -> Self::Vector<S> {
        simd.div_c32s_f32s(a, b)
    }
}

impl SimdTransformable for num_complex::Complex64 {
    type Vector<S: Simd> = S::c64s;
    type SplatVector<S: Simd> = S::f64s;

    #[inline(always)]
    fn as_simd<S: Simd>(_: S, slice: &[Self]) -> (&[Self::Vector<S>], &[Self]) {
        S::as_simd_c64s(slice)
    }

    #[inline(always)]
    fn as_mut_simd<S: Simd>(_: S, slice: &mut [Self]) -> (&mut [Self::Vector<S>], &mut [Self]) {
        S::as_mut_simd_c64s(slice)
    }

    #[inline(always)]
    fn simd_splat<S: Simd>(simd: S, v: Self::Scalar) -> Self::SplatVector<S> {
        simd.splat_f64s(v)
    }

    #[inline(always)]
    fn simd_mul_add<S: Simd>(
        simd: S,
        a: Self::Vector<S>,
        b: Self::SplatVector<S>,
        c: Self::Vector<S>,
    ) -> Self::Vector<S> {
        simd.mul_add_c64s_f64s(a, b, c)
    }

    #[inline(always)]
    fn simd_negate_mul_add<S: Simd>(
        simd: S,
        a: Self::Vector<S>,
        b: Self::SplatVector<S>,
        c: Self::Vector<S>,
    ) -> Self::Vector<S> {
        simd.neg_mul_add_c64s_f64s(a, b, c)
    }

    #[inline(always)]
    fn simd_add<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Vector<S> {
        simd.add_c64s(a, b)
    }

    #[inline(always)]
    fn simd_sub<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Vector<S> {
        simd.sub_c64s(a, b)
    }

    #[inline(always)]
    fn simd_mul<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::SplatVector<S>) -> Self::Vector<S> {
        simd.mul_c64s_f64s(a, b)
    }

    #[inline(always)]
    fn simd_div<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::SplatVector<S>) -> Self::Vector<S> {
        simd.div_c64s_f64s(a, b)
    }
}

/// Runtime CPU feature detection singleton used to dispatch SIMD kernels.
///
/// Initialised once on first access via [`std::sync::LazyLock`].
pub static ARCH: LazyLock<pulp::Arch> = LazyLock::new(pulp::Arch::new);
