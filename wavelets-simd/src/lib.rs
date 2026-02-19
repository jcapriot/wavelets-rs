use itertools::izip;
use multiversion::multiversion;
use num_traits::{Float, NumAssignRef, NumRef};

use pulp::{Arch, Simd, WithSimd};

pub trait SimdImpl {
    type Vector<S: Simd>: Copy + std::fmt::Debug;
    type Scalar;

    fn simd_lanes<S: Simd>(simd: S) -> usize;

    fn as_simd<S: Simd>(simd: S, x: &[Self::Scalar]) -> (&[Self::Vector<S>], &[Self::Scalar]);

    fn as_mut_simd<S: Simd>(
        simd: S,
        x: &mut [Self::Scalar],
    ) -> (&mut [Self::Vector<S>], &mut [Self::Scalar]);

    fn simd_splat<S: Simd>(simd: S, v: Self::Scalar) -> Self::Vector<S>;

    fn simd_mul_add<S: Simd>(
        simd: S,
        a: Self::Vector<S>,
        b: Self::Vector<S>,
        c: Self::Vector<S>,
    ) -> Self::Vector<S>;
    fn simd_neg_mul_add<S: Simd>(
        simd: S,
        a: Self::Vector<S>,
        b: Self::Vector<S>,
        c: Self::Vector<S>,
    ) -> Self::Vector<S>;
    fn simd_add<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Vector<S>;
    fn simd_sub<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Vector<S>;
    fn simd_mul<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Vector<S>;
    fn simd_div<S: Simd>(simd: S, a: Self::Vector<S>, b: Self::Vector<S>) -> Self::Vector<S>;
}

impl SimdImpl for f32 {
    type Vector<S: Simd> = S::f32s;
    type Scalar = Self;

    #[inline(always)]
    fn simd_lanes<S: Simd>(_: S) -> usize {
        S::F32_LANES
    }

    #[inline(always)]
    fn as_simd<S: Simd>(_: S, slice: &[Self::Scalar]) -> (&[Self::Vector<S>], &[Self::Scalar]) {
        S::as_simd_f32s(slice)
    }

    #[inline(always)]
    fn as_mut_simd<S: Simd>(
        _: S,
        slice: &mut [Self::Scalar],
    ) -> (&mut [Self::Vector<S>], &mut [Self::Scalar]) {
        S::as_mut_simd_f32s(slice)
    }

    #[inline(always)]
    fn simd_splat<S: Simd>(simd: S, v: Self::Scalar) -> Self::Vector<S> {
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
    fn simd_neg_mul_add<S: Simd>(
        simd: S,
        a: Self::Vector<S>,
        b: Self::Vector<S>,
        c: Self::Vector<S>,
    ) -> Self::Vector<S> {
        let neg_a = simd.neg_f32s(a);
        simd.mul_add_f32s(neg_a, b, c)
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

impl SimdImpl for f64 {
    type Vector<S: Simd> = S::f64s;
    type Scalar = Self;

    #[inline(always)]
    fn simd_lanes<S: Simd>(_: S) -> usize {
        S::F64_LANES
    }

    #[inline(always)]
    fn as_simd<S: Simd>(_: S, slice: &[Self::Scalar]) -> (&[Self::Vector<S>], &[Self::Scalar]) {
        S::as_simd_f64s(slice)
    }

    #[inline(always)]
    fn as_mut_simd<S: Simd>(
        _: S,
        slice: &mut [Self::Scalar],
    ) -> (&mut [Self::Vector<S>], &mut [Self::Scalar]) {
        S::as_mut_simd_f64s(slice)
    }

    #[inline(always)]
    fn simd_splat<S: Simd>(simd: S, v: Self::Scalar) -> Self::Vector<S> {
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
    fn simd_neg_mul_add<S: Simd>(
        simd: S,
        a: Self::Vector<S>,
        b: Self::Vector<S>,
        c: Self::Vector<S>,
    ) -> Self::Vector<S> {
        let neg_a = simd.neg_f64s(a);
        simd.mul_add_f64s(neg_a, b, c)
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

struct Db2Forward<'a, 'b, T>(&'a mut [T], &'b mut [T]);

impl<'a, 'b, T: SimdImpl<Scalar = T> + Float + NumAssignRef + NumRef + std::fmt::Debug> WithSimd
    for Db2Forward<'a, 'b, T>
{
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let s = self.0;
        let d = self.1;

        let _n_lanes = T::simd_lanes(simd);

        let ns = s.len();
        let nd = d.len();

        assert!(ns == nd || ns == nd + 1);

        let c = T::from(-1.73205080756887729352744634150587236694280525381038062805581).unwrap();
        let c_vec = T::simd_splat(simd, c);

        let (d_h, d_t) = T::as_mut_simd(simd, d);
        let (s_h, s_t) = T::as_simd(simd, s);

        let (d_h4, d_h) = d_h.as_chunks_mut::<4>();
        let (s_h4, s_h) = s_h.as_chunks::<4>();
        d_h4.iter_mut()
            .zip(s_h4)
            .for_each(|([d0, d1, d2, d3], [s0, s1, s2, s3])| {
                *d0 = T::simd_mul_add(simd, *s0, c_vec, *d0);
                *d1 = T::simd_mul_add(simd, *s1, c_vec, *d1);
                *d2 = T::simd_mul_add(simd, *s2, c_vec, *d2);
                *d3 = T::simd_mul_add(simd, *s3, c_vec, *d3);
            });
        d_h.iter_mut().zip(s_h).for_each(|(d, s)| {
            *d = T::simd_mul_add(simd, *s, c_vec, *d);
        });

        d_t.iter_mut()
            .zip(s_t.iter().cloned())
            .for_each(|(d, s)| *d += c * s);

        let cs = (
            T::from(0.433012701892219323381861585376468091735701313452595157013952).unwrap(),
            T::from(-0.0669872981077806766181384146235319082642986865474048429860483).unwrap(),
        );

        let cs_vec = (T::simd_splat(simd, cs.0), T::simd_splat(simd, cs.1));

        if let Some(s) = s.get_mut(..nd - 1)
            && let Some(d0) = d.get(..nd - 1)
            && let Some(d1) = d.get(1..)
        {
            let (d0_h, d0_t) = T::as_simd(simd, d0);
            let (d1_h, d1_t) = T::as_simd(simd, d1);
            let (s_h, s_t) = T::as_mut_simd(simd, s);

            let (d0_h4, d0_h) = d0_h.as_chunks::<4>();
            let (d1_h4, d1_h) = d1_h.as_chunks::<4>();
            let (s_h4, s_h) = s_h.as_chunks_mut::<4>();

            s_h4.iter_mut().zip(izip!(d0_h4, d1_h4)).for_each(
                |([s0, s1, s2, s3], ([d0_0, d0_1, d0_2, d0_3], [d1_0, d1_1, d1_2, d1_3]))| {
                    *s0 = T::simd_mul_add(simd, *d0_0, cs_vec.0, *s0);
                    *s0 = T::simd_mul_add(simd, *d1_0, cs_vec.1, *s0);

                    *s1 = T::simd_mul_add(simd, *d0_1, cs_vec.0, *s1);
                    *s1 = T::simd_mul_add(simd, *d1_1, cs_vec.1, *s1);

                    *s2 = T::simd_mul_add(simd, *d0_2, cs_vec.0, *s2);
                    *s2 = T::simd_mul_add(simd, *d1_2, cs_vec.1, *s2);

                    *s3 = T::simd_mul_add(simd, *d0_3, cs_vec.0, *s3);
                    *s3 = T::simd_mul_add(simd, *d1_3, cs_vec.1, *s3);
                },
            );
            s_h.iter_mut().zip(izip!(d0_h, d1_h)).for_each(|(s, d)| {
                *s = T::simd_mul_add(simd, *d.0, cs_vec.0, *s);
                *s = T::simd_mul_add(simd, *d.1, cs_vec.1, *s);
            });

            s_t.iter_mut().zip(izip!(d0_t, d1_t)).for_each(|(s, d)| {
                *s += cs.0 * d.0 + cs.1 * d.1;
            });
        }

        if let Some(d) = d.last()
            && let Some(s) = s.get_mut(nd - 1)
        {
            *s = d.mul_add(cs.0, *s);
        }

        if let Some(d) = d.get_mut(1..)
            && let Some(s) = s.get(..nd - 1)
        {
            let (d_h, d_t) = T::as_mut_simd(simd, d);
            let (s_h, s_t) = T::as_simd(simd, s);

            let (d_h4, d_h) = d_h.as_chunks_mut::<4>();
            let (s_h4, s_h) = s_h.as_chunks::<4>();

            d_h4.iter_mut()
                .zip(s_h4)
                .for_each(|([d0, d1, d2, d3], [s0, s1, s2, s3])| {
                    *d0 = T::simd_add(simd, *s0, *d0);
                    *d1 = T::simd_add(simd, *s1, *d1);
                    *d2 = T::simd_add(simd, *s2, *d2);
                    *d3 = T::simd_add(simd, *s3, *d3);
                });

            d_h.iter_mut().zip(s_h.iter()).for_each(|(d, s)| {
                *d = T::simd_add(simd, *s, *d);
            });

            d_t.iter_mut().zip(s_t.iter()).for_each(|(d, s)| {
                *d += s;
            })
        }

        let c = T::from(1.93185165257813657349948639945779473526780967801680910080469).unwrap();
        let c_vec = T::simd_splat(simd, c);

        let (s_h, s_t) = T::as_mut_simd(simd, s);

        let (s_h4, s_h) = s_h.as_chunks_mut::<4>();
        s_h4.iter_mut().for_each(|[s0, s1, s2, s3]| {
            *s0 = T::simd_mul(simd, *s0, c_vec);
            *s1 = T::simd_mul(simd, *s1, c_vec);
            *s2 = T::simd_mul(simd, *s2, c_vec);
            *s3 = T::simd_mul(simd, *s3, c_vec);
        });
        s_h.iter_mut()
            .for_each(|s| *s = T::simd_mul(simd, *s, c_vec));
        s_t.iter_mut().for_each(|s| *s *= c);

        let c =
            T::from(1.0 / 1.93185165257813657349948639945779473526780967801680910080469).unwrap();
        let c_vec = T::simd_splat(simd, c);

        let (d_h, d_t) = T::as_mut_simd(simd, d);
        let (d_h4, d_h) = d_h.as_chunks_mut::<4>();
        d_h4.iter_mut().for_each(|[d0, d1, d2, d3]| {
            *d0 = T::simd_mul(simd, *d0, c_vec);
            *d1 = T::simd_mul(simd, *d1, c_vec);
            *d2 = T::simd_mul(simd, *d2, c_vec);
            *d3 = T::simd_mul(simd, *d3, c_vec);
        });
        d_h.iter_mut()
            .for_each(|d| *d = T::simd_mul(simd, *d, c_vec));
        d_t.iter_mut().for_each(|d| *d *= c);
    }
}

#[multiversion(targets = "simd")]
pub fn db2_forward<T: Float + NumAssignRef + NumRef + std::fmt::Debug>(s: &mut [T], d: &mut [T]) {
    let ns = s.len();
    let nd = d.len();

    assert!(ns == nd || ns == nd + 1);

    let c = T::from(-1.73205080756887729352744634150587236694280525381038062805581).unwrap();
    d.iter_mut().zip(s.iter()).for_each(|(d, s)| {
        #[cfg(any(target_feature = "neon", target_feature = "fma"))]
        {
            *d = T::mul_add(*s, c, *d);
        }
        #[cfg(not(any(target_feature = "neon", target_feature = "fma")))]
        {
            *d += c * s
        }
    });

    let cs = (
        T::from(0.433012701892219323381861585376468091735701313452595157013952).unwrap(),
        T::from(-0.0669872981077806766181384146235319082642986865474048429860483).unwrap(),
    );

    s.iter_mut().zip(d.windows(2)).for_each(|(s, d)| {
        #[cfg(any(target_feature = "neon", target_feature = "fma"))]
        {
            *s = T::mul_add(d[0], cs.0, *s);
            *s = T::mul_add(d[1], cs.1, *s);
        }
        #[cfg(not(any(target_feature = "neon", target_feature = "fma")))]
        {
            *s += cs.0 * d[0] + cs.1 * d[1];
        }
    });

    if ns == nd {
        if let Some(s) = s.last_mut()
            && let Some(d) = d.last()
        {
            #[cfg(any(target_feature = "neon", target_feature = "fma"))]
            {
                *s = T::mul_add(*d, cs.0, *s);
            }
            #[cfg(not(any(target_feature = "neon", target_feature = "fma")))]
            {
                *s += cs.0 * d;
            }
        }
    } else if ns > 1 {
        // ns = nd + 1
        // so main loop went up to the second to last element of s
        if let Some(s) = s.get_mut(ns - 2)
            && let Some(d) = d.last()
        {
            #[cfg(any(target_feature = "neon", target_feature = "fma"))]
            {
                *s = T::mul_add(*d, cs.0, *s);
            }
            #[cfg(not(any(target_feature = "neon", target_feature = "fma")))]
            {
                *s += cs.0 * d;
            }
        }
    }

    if let Some(d) = d.get_mut(1..) {
        d.iter_mut().zip(s.iter()).for_each(|(d, s)| {
            *d += s;
        })
    }

    let c = T::from(1.93185165257813657349948639945779473526780967801680910080469).unwrap();

    s.iter_mut().for_each(|s| *s *= c);
    d.iter_mut().for_each(|d| *d /= c);
}

pub fn db2_forward_simd<
    T: SimdImpl<Scalar = T> + Float + NumAssignRef + NumRef + std::fmt::Debug,
>(
    arch: Arch,
    s: &mut [T],
    d: &mut [T],
) {
    arch.dispatch(Db2Forward(s, d));
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn db2_forward_neon(s: &mut [f64], d: &mut [f64]) {
    use core::arch::aarch64::*;

    let ns = s.len();
    let nd = d.len();

    assert!(ns == nd || ns == nd + 1);

    let c = -1.73205080756887729352744634150587236694280525381038062805581;
    let cv = unsafe { vld1q_dup_f64(&c) };
    let (d_h, d_t) = d.as_chunks_mut::<2>();
    let (s_h, s_t) = s.as_chunks::<2>();

    let (d_h4, d_h) = d_h.as_chunks_mut::<4>();
    let (s_h4, s_h) = s_h.as_chunks::<4>();

    d_h4.iter_mut()
        .zip(s_h4)
        .for_each(|([d0, d1, d2, d3], [s0, s1, s2, s3])| unsafe {
            let dv = vld1q_f64(d0.as_ptr());
            let sv = vld1q_f64(s0.as_ptr());
            let dv = vfmaq_f64(dv, sv, cv);
            //let dv = vfmaq_n_f64(dv, sv, c);
            vst1q_f64(d0.as_mut_ptr(), dv);

            let dv = vld1q_f64(d1.as_ptr());
            let sv = vld1q_f64(s1.as_ptr());
            let dv = vfmaq_f64(dv, sv, cv);
            //let dv = vfmaq_n_f64(dv, sv, c);
            vst1q_f64(d1.as_mut_ptr(), dv);

            let dv = vld1q_f64(d2.as_ptr());
            let sv = vld1q_f64(s2.as_ptr());
            let dv = vfmaq_f64(dv, sv, cv);
            //let dv = vfmaq_n_f64(dv, sv, c);
            vst1q_f64(d2.as_mut_ptr(), dv);

            let dv = vld1q_f64(d3.as_ptr());
            let sv = vld1q_f64(s3.as_ptr());
            let dv = vfmaq_f64(dv, sv, cv);
            //let dv = vfmaq_n_f64(dv, sv, c);
            vst1q_f64(d3.as_mut_ptr(), dv);
        });

    d_h.iter_mut().zip(s_h.iter()).for_each(|(d, s)| unsafe {
        let dv = vld1q_f64(d.as_ptr());
        let sv = vld1q_f64(s.as_ptr());
        //let dv = vfmaq_n_f64(dv, sv, c);
        let dv = vfmaq_f64(dv, sv, cv);
        vst1q_f64(d.as_mut_ptr(), dv);
    });

    d_t.iter_mut().zip(s_t.iter()).for_each(|(d, s)| {
        *d = f64::mul_add(*s, c, *d);
    });

    let c = (
        0.433012701892219323381861585376468091735701313452595157013952,
        -0.0669872981077806766181384146235319082642986865474048429860483,
    );

    let cv = unsafe { (vld1q_dup_f64(&c.0), vld1q_dup_f64(&c.1)) };

    let (s_h, s_t) = s[..nd - 1].as_chunks_mut::<2>();
    let (d0_h, d0_t) = d[..nd - 1].as_chunks::<2>();
    let (d1_h, d1_t) = d[1..nd].as_chunks::<2>();

    let (d0_h4, d0_h) = d0_h.as_chunks::<4>();
    let (d1_h4, d1_h) = d1_h.as_chunks::<4>();
    let (s_h4, s_h) = s_h.as_chunks_mut::<4>();

    s_h4.iter_mut().zip(d0_h4.iter().zip(d1_h4)).for_each(
        |([s0, s1, s2, s3], ([d0_0, d0_1, d0_2, d0_3], [d1_0, d1_1, d1_2, d1_3]))| unsafe {
            let sv = vld1q_f64(s0.as_ptr());
            let dv = vld1q_f64(d0_0.as_ptr());
            //let sv = vfmaq_n_f64(sv, dv, c.0);
            let sv = vfmaq_f64(sv, dv, cv.0);
            let dv = vld1q_f64(d1_0.as_ptr());
            //let sv = vfmaq_n_f64(sv, dv, c.1);
            let sv = vfmaq_f64(sv, dv, cv.1);
            vst1q_f64(s0.as_mut_ptr(), sv);

            let sv = vld1q_f64(s1.as_ptr());
            let dv = vld1q_f64(d0_1.as_ptr());
            //let sv = vfmaq_n_f64(sv, dv, c.0);
            let sv = vfmaq_f64(sv, dv, cv.0);
            let dv = vld1q_f64(d1_1.as_ptr());
            //let sv = vfmaq_n_f64(sv, dv, c.1);
            let sv = vfmaq_f64(sv, dv, cv.1);
            vst1q_f64(s1.as_mut_ptr(), sv);

            let sv = vld1q_f64(s2.as_ptr());
            let dv = vld1q_f64(d0_2.as_ptr());
            //let sv = vfmaq_n_f64(sv, dv, c.0);
            let sv = vfmaq_f64(sv, dv, cv.0);
            let dv = vld1q_f64(d1_2.as_ptr());
            //let sv = vfmaq_n_f64(sv, dv, c.1);
            let sv = vfmaq_f64(sv, dv, cv.1);
            vst1q_f64(s2.as_mut_ptr(), sv);

            let sv = vld1q_f64(s3.as_ptr());
            let dv = vld1q_f64(d0_3.as_ptr());
            //let sv = vfmaq_n_f64(sv, dv, c.0);
            let sv = vfmaq_f64(sv, dv, cv.0);
            let dv = vld1q_f64(d1_3.as_ptr());
            //let sv = vfmaq_n_f64(sv, dv, c.1);
            let sv = vfmaq_f64(sv, dv, cv.1);
            vst1q_f64(s3.as_mut_ptr(), sv);
        },
    );

    s_h.iter_mut()
        .zip(d0_h.iter().zip(d1_h))
        .for_each(|(s, (d0, d1))| unsafe {
            let sv = vld1q_f64(s.as_ptr());
            let dv = vld1q_f64(d0.as_ptr());
            //let sv = vfmaq_n_f64(sv, dv, c.0);
            let sv = vfmaq_f64(sv, dv, cv.0);
            let dv = vld1q_f64(d1.as_ptr());
            //let sv = vfmaq_n_f64(sv, dv, c.1);
            let sv = vfmaq_f64(sv, dv, cv.1);
            vst1q_f64(s.as_mut_ptr(), sv);
        });

    s_t.iter_mut()
        .zip(d0_t.iter().zip(d1_t))
        .for_each(|(s, d)| {
            *s = d.0.mul_add(c.0, *s);
            *s = d.1.mul_add(c.1, *s);
        });

    if let Some(d) = d.last()
        && let Some(s) = s.get_mut(nd - 1)
    {
        *s = d.mul_add(c.0, *s);
    }

    if let Some(d) = d.get_mut(1..) {
        let (d_h, d_t) = d.as_chunks_mut::<2>();
        let (s_h, s_t) = s[..nd - 1].as_chunks::<2>();

        let (d_h4, d_h) = d_h.as_chunks_mut::<4>();
        let (s_h4, s_h) = s_h.as_chunks::<4>();

        d_h4.iter_mut()
            .zip(s_h4)
            .for_each(|([d0, d1, d2, d3], [s0, s1, s2, s3])| unsafe {
                let dv = vld1q_f64(d0.as_ptr());
                let sv = vld1q_f64(s0.as_ptr());
                let dv = vaddq_f64(dv, sv);
                vst1q_f64(d0.as_mut_ptr(), dv);

                let dv = vld1q_f64(d1.as_ptr());
                let sv = vld1q_f64(s1.as_ptr());
                let dv = vaddq_f64(dv, sv);
                vst1q_f64(d1.as_mut_ptr(), dv);

                let dv = vld1q_f64(d2.as_ptr());
                let sv = vld1q_f64(s2.as_ptr());
                let dv = vaddq_f64(dv, sv);
                vst1q_f64(d2.as_mut_ptr(), dv);

                let dv = vld1q_f64(d3.as_ptr());
                let sv = vld1q_f64(s3.as_ptr());
                let dv = vaddq_f64(dv, sv);
                vst1q_f64(d3.as_mut_ptr(), dv);
            });

        d_h.iter_mut().zip(s_h).for_each(|(d, s)| unsafe {
            let d0 = vld1q_f64(d.as_ptr());
            let s0 = vld1q_f64(s.as_ptr());
            let ds = vaddq_f64(d0, s0);
            vst1q_f64(d.as_mut_ptr(), ds);
        });
        d_t.iter_mut().zip(s_t.iter()).for_each(|(d, s)| {
            *d += s;
        })
    }

    let c = 1.93185165257813657349948639945779473526780967801680910080469;
    let cv = unsafe { vld1q_dup_f64(&c) };

    let (s_h, s_t) = s.as_chunks_mut::<2>();

    let (s_h4, s_h) = s_h.as_chunks_mut::<4>();

    s_h4.iter_mut().for_each(|[s0, s1, s2, s3]| unsafe {
        let sv = vld1q_f64(s0.as_ptr());
        let sv = vmulq_f64(sv, cv);
        //let sv = vmulq_n_f64(sv, c);
        vst1q_f64(s0.as_mut_ptr(), sv);

        let sv = vld1q_f64(s1.as_ptr());
        let sv = vmulq_f64(sv, cv);
        //let sv = vmulq_n_f64(sv, c);
        vst1q_f64(s1.as_mut_ptr(), sv);

        let sv = vld1q_f64(s2.as_ptr());
        let sv = vmulq_f64(sv, cv);
        //let sv = vmulq_n_f64(sv, c);
        vst1q_f64(s2.as_mut_ptr(), sv);

        let sv = vld1q_f64(s3.as_ptr());
        let sv = vmulq_f64(sv, cv);
        //let sv = vmulq_n_f64(sv, c);
        vst1q_f64(s3.as_mut_ptr(), sv);
    });

    s_h.iter_mut().for_each(|s| unsafe {
        let sv = vld1q_f64(s.as_ptr());
        let sv = vmulq_f64(sv, cv);
        //let sv = vmulq_n_f64(sv, c);
        vst1q_f64(s.as_mut_ptr(), sv);
    });

    s_t.iter_mut().for_each(|s| *s *= c);

    let cinv = 1.0 / c;
    let cinvv = unsafe { vld1q_dup_f64(&cinv) };

    let (d_h, d_t) = d.as_chunks_mut::<2>();
    let (d_h4, d_h) = d_h.as_chunks_mut::<4>();

    d_h4.iter_mut().for_each(|[d0, d1, d2, d3]| unsafe {
        let dv = vld1q_f64(d0.as_ptr());
        let dv = vmulq_f64(dv, cinvv);
        //let dv = vmulq_n_f64(dv, cinv);
        vst1q_f64(d0.as_mut_ptr(), dv);

        let dv = vld1q_f64(d1.as_ptr());
        let dv = vmulq_f64(dv, cinvv);
        //let dv = vmulq_n_f64(dv, cinv);
        vst1q_f64(d1.as_mut_ptr(), dv);

        let dv = vld1q_f64(d2.as_ptr());
        let dv = vmulq_f64(dv, cinvv);
        //let dv = vmulq_n_f64(dv, cinv);
        vst1q_f64(d2.as_mut_ptr(), dv);

        let dv = vld1q_f64(d3.as_ptr());
        let dv = vmulq_f64(dv, cinvv);
        //let dv = vmulq_n_f64(dv, cinv);
        vst1q_f64(d3.as_mut_ptr(), dv);
    });

    d_h.iter_mut().for_each(|d| unsafe {
        let dv = vld1q_f64(d.as_ptr());
        let dv = vmulq_f64(dv, cinvv);
        //let dv = vmulq_n_f64(dv, cinv);
        vst1q_f64(d.as_mut_ptr(), dv);
    });
    d_t.iter_mut().for_each(|d| *d *= cinv);
}

#[multiversion(targets = "simd")]
pub fn print_info() {
    #[cfg(any(target_feature = "fma", target_feature = "neon"))]
    {
        println!("would do a mul_add operation");
    }
    #[cfg(not(any(target_feature = "fma", target_feature = "neon")))]
    {
        println!("Simple multiply and accumulate");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn testable() {
        let n = 32;

        let mut s1 = (0..n).map(|v| v as f64).collect::<Vec<_>>();
        let mut d1 = (0..n).map(|v| -v as f64).collect::<Vec<_>>();

        db2_forward(&mut s1, &mut d1);

        let mut s2 = (0..n).map(|v| v as f64).collect::<Vec<_>>();
        let mut d2 = (0..n).map(|v| -v as f64).collect::<Vec<_>>();

        let arch = Arch::new();
        db2_forward_simd(arch, &mut s2, &mut d2);
        assert!(s1.iter().zip(s2.iter()).enumerate().all(|(i, (x, y))| {
            let a_diff = (x - y).abs();
            let ok = a_diff < 1E-14;
            if !ok {
                dbg!(i, x, y, a_diff);
            }
            ok
        }));

        assert!(d1.iter().zip(d2.iter()).enumerate().all(|(i, (x, y))| {
            let a_diff = (x - y).abs();
            let ok = a_diff < 1E-14;
            if !ok {
                dbg!(i, x, y, a_diff);
            }
            ok
        }));

        #[cfg(target_arch = "aarch64")]
        {
            let mut s3 = (0..n).map(|v| v as f64).collect::<Vec<_>>();
            let mut d3 = (0..n).map(|v| -v as f64).collect::<Vec<_>>();

            unsafe {
                db2_forward_neon(&mut s3, &mut d3);
            }

            assert!(s1.iter().zip(s3.iter()).enumerate().all(|(i, (x, y))| {
                let a_diff = (x - y).abs();
                let ok = a_diff < 1E-14;
                if !ok {
                    dbg!(i, x, y, a_diff);
                }
                ok
            }));

            assert!(d1.iter().zip(d3.iter()).enumerate().all(|(i, (x, y))| {
                let a_diff = (x - y).abs();
                let ok = a_diff < 1E-14;
                if !ok {
                    dbg!(i, x, y, a_diff);
                }
                ok
            }));
        }
    }
}
