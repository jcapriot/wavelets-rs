use std::sync::LazyLock;

use itertools::izip;
use multiversion::multiversion;
use num_traits::{Float, NumAssignRef, NumRef};

use pulp::{Simd, WithSimd};

pub static ARCH: LazyLock<pulp::Arch> = LazyLock::new(|| pulp::Arch::new());

pub trait FloatOps: Float + NumAssignRef + NumRef + std::fmt::Debug {}
impl<T: Float + NumAssignRef + NumRef + std::fmt::Debug> FloatOps for T {}

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

#[multiversion(targets = "simd")]
pub fn db2_forward<T: FloatOps>(s: &mut [T], d: &mut [T]) {
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

pub fn db2_forward_pulp<T: SimdImpl<Scalar = T> + FloatOps>(s: &mut [T], d: &mut [T]) {
    struct Impl<'a, 'b, T>(&'a mut [T], &'b mut [T]);

    impl<'a, 'b, T: SimdImpl<Scalar = T> + FloatOps> WithSimd for Impl<'a, 'b, T> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let s = self.0;
            let d = self.1;

            let _n_lanes = T::simd_lanes(simd);

            let ns = s.len();
            let nd = d.len();

            assert!(ns == nd || ns == nd + 1);

            let c =
                T::from(-1.73205080756887729352744634150587236694280525381038062805581).unwrap();
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

            let c = T::from(1.0 / 1.93185165257813657349948639945779473526780967801680910080469)
                .unwrap();
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

    ARCH.dispatch(Impl(s, d));
}

#[multiversion(targets = "simd")]
#[inline(always)]
pub fn forward_update_step<T: FloatOps, const N: usize, const OFF: isize>(
    c: [T; N],
    l: &mut [T],
    r: &[T],
) {
    let nl = l.len();
    let nr = r.len();
    let nd = std::cmp::min(nl, nr);

    let n_front = const { if OFF < 0 { (-OFF) as usize } else { 0 } };
    let max_offset = const { N as isize + OFF };

    let n1 = std::cmp::min(n_front, nl);
    if const { OFF < 0 } {
        (OFF..n1 as isize + OFF)
            .zip(&mut l[..n1])
            .for_each(|(i_o, l_i)| {
                c.iter().enumerate().for_each(|(j, c)| {
                    if let Some(r) = r.get(i_o as usize + j) {
                        *l_i = r.mul_add(*c, *l_i);
                    }
                })
            });
    }

    // `l` is slice from n1..nl;
    // need to find where it will completely overlap with `r`
    // end of the righthand side will be nl + offset, so...
    // r will start at max of 0 if (off < 0) or offset,
    let ir_0 = const { if OFF < 0 { 0 } else { OFF as usize } };
    // last position will be smaller of nd or nl + max_offset;
    let ir_end = std::cmp::min(nd as isize, nl as isize + max_offset) as usize;
    let nr = if let Some(nr) = (ir_end - ir_0).checked_sub(N) {
        let l = &mut l[n1..n1 + nr];
        let rs: [_; N] = std::array::from_fn(|i| {
            let ir = ir_0 + i;
            &r[ir..ir + nr]
        });

        debug_assert!(rs.iter().all(|r| r.len() == nr));
        l.iter_mut().enumerate().for_each(|(i, l_i)| {
            c.iter().zip(rs).for_each(|(c, r)| {
                *l_i = c.mul_add(r[i], *l_i);
            })
        });
        n1 + nr
    } else {
        n1
    };

    // if ir_end < nl do the back loop
    if let Some(l) = l.get_mut(nr..) {
        (nr as isize + OFF..nl as isize + OFF)
            .zip(l)
            .for_each(|(i_o, l_i)| {
                c.iter().enumerate().for_each(|(j, c)| {
                    if let Some(r) = r.get(i_o as usize + j) {
                        *l_i = r.mul_add(*c, *l_i);
                    }
                })
            });
    }
}

#[multiversion(targets = "simd")]
pub fn db2_forward_from_steps<T: FloatOps>(s: &mut [T], d: &mut [T]) {
    let cs = [T::from(-1.73205080756887729352744634150587236694280525381038062805581).unwrap()];
    forward_update_step::<_, _, 0>(cs, d, s);

    let cs = [
        T::from(0.433012701892219323381861585376468091735701313452595157013952).unwrap(),
        T::from(-0.0669872981077806766181384146235319082642986865474048429860483).unwrap(),
    ];

    forward_update_step::<_, _, 0>(cs, s, d);

    let cs = [T::from(1.0).unwrap()];

    forward_update_step::<_, _, -1>(cs, d, s);

    let c = T::from(1.93185165257813657349948639945779473526780967801680910080469).unwrap();

    s.iter_mut().for_each(|s| *s *= c);
    d.iter_mut().for_each(|d| *d /= c);
}

#[inline(always)]
pub fn forward_update_step_pulp<
    S: Simd,
    T: SimdImpl<Scalar = T> + FloatOps,
    const N: usize,
    const OFF: isize,
>(
    simd: S,
    c: [T; N],
    l: &mut [T],
    r: &[T],
) {
    let nl = l.len();
    let nr = r.len();
    let nd = std::cmp::min(nl, nr);

    let n_front = const { if OFF < 0 { (-OFF) as usize } else { 0 } };
    let max_offset = const { N as isize + OFF };

    let n1 = if const { OFF < 0 } {
        std::cmp::min(n_front, nl)
    } else {
        0
    };

    if const { OFF < 0 } {
        (OFF..n1 as isize + OFF)
            .zip(&mut l[..n1])
            .for_each(|(i_o, l_i)| {
                c.iter().enumerate().for_each(|(j, c)| {
                    if let Some(r) = r.get(i_o as usize + j) {
                        *l_i = r.mul_add(*c, *l_i);
                    }
                })
            });
    }

    // `l` is slice from n1..nl;
    // need to find where it will completely overlap with `r`
    // end of the righthand side will be nl + offset, so...
    // r will start at max of 0 if (off < 0) or offset,
    let ir_0 = const { if OFF < 0 { 0 } else { OFF as usize } };
    // last position will be smaller of nd or nl + max_offset;
    let ir_end = std::cmp::min(nd, nl.checked_add_signed(max_offset).unwrap_or(0));
    //let nr = (ir_end - ir_0).checked_sub(N).unwrap_or(0);

    let nr = if let Some(nr) = (ir_end - ir_0).checked_sub(N) {
        let cv = c.map(|c| T::simd_splat(simd, c));

        let (l_h, l) = T::as_mut_simd(simd, &mut l[n1..n1 + nr]);
        let (l_h4, l_h) = l_h.as_chunks_mut::<4>();

        match N {
            1 => {
                let (r_h, r) = T::as_simd(simd, &r[ir_0..ir_0 + nr]);
                let (r_h4, r_h) = r_h.as_chunks::<4>();

                l_h4.iter_mut()
                    .zip(r_h4)
                    .for_each(|([l0, l1, l2, l3], [r0, r1, r2, r3])| {
                        if c[0] == T::from(1.0).unwrap() {
                            *l0 = T::simd_add(simd, *r0, *l0);
                            *l1 = T::simd_add(simd, *r1, *l1);
                            *l2 = T::simd_add(simd, *r2, *l2);
                            *l3 = T::simd_add(simd, *r3, *l3);
                        } else if c[0] == T::from(-1.0).unwrap() {
                            *l0 = T::simd_sub(simd, *r0, *l0);
                            *l1 = T::simd_sub(simd, *r1, *l1);
                            *l2 = T::simd_sub(simd, *r2, *l2);
                            *l3 = T::simd_sub(simd, *r3, *l3);
                        } else {
                            *l0 = T::simd_mul_add(simd, *r0, cv[0], *l0);
                            *l1 = T::simd_mul_add(simd, *r1, cv[0], *l1);
                            *l2 = T::simd_mul_add(simd, *r2, cv[0], *l2);
                            *l3 = T::simd_mul_add(simd, *r3, cv[0], *l3);
                        }
                    });

                l_h.iter_mut().zip(r_h).for_each(|(l, r)| {
                    if c[0] == T::from(1.0).unwrap() {
                        *l = T::simd_add(simd, *r, *l);
                    } else if c[0] == T::from(-1.0).unwrap() {
                        *l = T::simd_sub(simd, *r, *l);
                    } else {
                        *l = T::simd_mul_add(simd, *r, cv[0], *l);
                    }
                });
                l.iter_mut().zip(r).for_each(|(l, r)| {
                    if c[0] == T::from(1.0).unwrap() {
                        *l += r;
                    } else if c[0] == T::from(-1.0).unwrap() {
                        *l -= r;
                    } else {
                        *l = r.mul_add(c[0], *l);
                    }
                });
            }
            2 => {
                let (r0_h, r0) = T::as_simd(simd, &r[ir_0..ir_0 + nr]);
                let (r0_h4, r0_h) = r0_h.as_chunks::<4>();

                let (r1_h, r1) = T::as_simd(simd, &r[ir_0 + 1..ir_0 + 1 + nr]);
                let (r1_h4, r1_h) = r1_h.as_chunks::<4>();

                l_h4.iter_mut().zip(izip!(r0_h4, r1_h4)).for_each(
                    |([l0, l1, l2, l3], (r0, r1))| {
                        if c[0] == T::from(1.0).unwrap() {
                            *l0 = T::simd_add(simd, r0[0], *l0);
                            *l1 = T::simd_add(simd, r0[1], *l1);
                            *l2 = T::simd_add(simd, r0[2], *l2);
                            *l3 = T::simd_add(simd, r0[3], *l3);
                        } else if c[0] == T::from(-1.0).unwrap() {
                            *l0 = T::simd_sub(simd, r0[0], *l0);
                            *l1 = T::simd_sub(simd, r0[1], *l1);
                            *l2 = T::simd_sub(simd, r0[2], *l2);
                            *l3 = T::simd_sub(simd, r0[3], *l3);
                        } else {
                            *l0 = T::simd_mul_add(simd, r0[0], cv[0], *l0);
                            *l1 = T::simd_mul_add(simd, r0[1], cv[0], *l1);
                            *l2 = T::simd_mul_add(simd, r0[2], cv[0], *l2);
                            *l3 = T::simd_mul_add(simd, r0[3], cv[0], *l3);
                        }

                        if c[1] == T::from(1.0).unwrap() {
                            *l0 = T::simd_add(simd, r1[0], *l0);
                            *l1 = T::simd_add(simd, r1[1], *l1);
                            *l2 = T::simd_add(simd, r1[2], *l2);
                            *l3 = T::simd_add(simd, r1[3], *l3);
                        } else if c[1] == T::from(-1.0).unwrap() {
                            *l0 = T::simd_sub(simd, r1[0], *l0);
                            *l1 = T::simd_sub(simd, r1[1], *l1);
                            *l2 = T::simd_sub(simd, r1[2], *l2);
                            *l3 = T::simd_sub(simd, r1[3], *l3);
                        } else {
                            *l0 = T::simd_mul_add(simd, r1[0], cv[1], *l0);
                            *l1 = T::simd_mul_add(simd, r1[1], cv[1], *l1);
                            *l2 = T::simd_mul_add(simd, r1[2], cv[1], *l2);
                            *l3 = T::simd_mul_add(simd, r1[3], cv[1], *l3);
                        }
                    },
                );

                l_h.iter_mut()
                    .zip(izip!(r0_h, r1_h))
                    .for_each(|(l, (r0, r1))| {
                        *l = T::simd_mul_add(simd, *r0, cv[0], *l);
                        *l = T::simd_mul_add(simd, *r1, cv[1], *l);
                    });
                l.iter_mut().zip(izip!(r0, r1)).for_each(|(l, (r0, r1))| {
                    *l = r0.mul_add(c[0], *l);
                    *l = r1.mul_add(c[1], *l);
                });
            }
            3 => {
                let (r0_h, r0) = T::as_simd(simd, &r[ir_0..ir_0 + nr]);
                let (r0_h4, r0_h) = r0_h.as_chunks::<4>();

                let (r1_h, r1) = T::as_simd(simd, &r[ir_0 + 1..ir_0 + 1 + nr]);
                let (r1_h4, r1_h) = r1_h.as_chunks::<4>();

                let (r2_h, r2) = T::as_simd(simd, &r[ir_0 + 2..ir_0 + 2 + nr]);
                let (r2_h4, r2_h) = r2_h.as_chunks::<4>();

                l_h4.iter_mut().zip(izip!(r0_h4, r1_h4, r2_h4)).for_each(
                    |([l0, l1, l2, l3], (r0, r1, r2))| {
                        *l0 = T::simd_mul_add(simd, r0[0], cv[0], *l0);
                        *l0 = T::simd_mul_add(simd, r1[0], cv[1], *l0);
                        *l0 = T::simd_mul_add(simd, r2[0], cv[2], *l0);

                        *l1 = T::simd_mul_add(simd, r0[1], cv[0], *l1);
                        *l1 = T::simd_mul_add(simd, r1[1], cv[1], *l1);
                        *l1 = T::simd_mul_add(simd, r2[1], cv[2], *l1);

                        *l2 = T::simd_mul_add(simd, r0[2], cv[0], *l2);
                        *l2 = T::simd_mul_add(simd, r1[2], cv[1], *l2);
                        *l2 = T::simd_mul_add(simd, r2[2], cv[2], *l2);

                        *l3 = T::simd_mul_add(simd, r0[3], cv[0], *l3);
                        *l3 = T::simd_mul_add(simd, r1[3], cv[1], *l3);
                        *l3 = T::simd_mul_add(simd, r2[3], cv[2], *l3);
                    },
                );

                l_h.iter_mut()
                    .zip(izip!(r0_h, r1_h, r2_h))
                    .for_each(|(l, (r0, r1, r2))| {
                        *l = T::simd_mul_add(simd, *r0, cv[0], *l);
                        *l = T::simd_mul_add(simd, *r1, cv[1], *l);
                        *l = T::simd_mul_add(simd, *r2, cv[2], *l);
                    });
                l.iter_mut()
                    .zip(izip!(r0, r1, r2))
                    .for_each(|(l, (r0, r1, r2))| {
                        *l = r0.mul_add(c[0], *l);
                        *l = r1.mul_add(c[1], *l);
                        *l = r2.mul_add(c[2], *l);
                    });
            }
            _ => {
                let mut rs_h4: Vec<_> = vec![];
                let mut rs_h: Vec<_> = vec![];
                let mut rs: Vec<_> = vec![];
                (0..N).for_each(|i| {
                    let (r_h, r) = T::as_simd(simd, &r[ir_0 + i..ir_0 + i + nr]);
                    let (r_h4, r_h) = r_h.as_chunks::<4>();
                    rs_h4.push(r_h4);
                    rs_h.push(r_h);
                    rs.push(r);
                });

                l_h4.iter_mut()
                    .enumerate()
                    .for_each(|(i, [l0, l1, l2, l3])| {
                        cv.iter().enumerate().for_each(|(j, cv)| {
                            *l0 = T::simd_mul_add(simd, *cv, rs_h4[j][i][0], *l0);
                        });

                        cv.iter().enumerate().for_each(|(j, cv)| {
                            *l1 = T::simd_mul_add(simd, *cv, rs_h4[j][i][1], *l1);
                        });

                        cv.iter().enumerate().for_each(|(j, cv)| {
                            *l2 = T::simd_mul_add(simd, *cv, rs_h4[j][i][2], *l2);
                        });

                        cv.iter().enumerate().for_each(|(j, cv)| {
                            *l3 = T::simd_mul_add(simd, *cv, rs_h4[j][i][3], *l3);
                        });
                    });

                l_h.iter_mut().enumerate().for_each(|(i, l)| {
                    cv.iter().enumerate().for_each(|(j, cv)| {
                        *l = T::simd_mul_add(simd, *cv, rs_h[j][i], *l);
                    });
                });
                l.iter_mut().enumerate().for_each(|(i, l)| {
                    c.iter().enumerate().for_each(|(j, c)| {
                        *l = c.mul_add(rs[j][i], *l);
                    });
                });
            }
        };
        n1 + nr
    } else {
        n1
    };

    // if ir_end < nl do the back loop
    if let Some(l) = l.get_mut(nr..) {
        (nr as isize + OFF..nl as isize + OFF)
            .zip(l)
            .for_each(|(i_o, l_i)| {
                c.iter().enumerate().for_each(|(j, c)| {
                    if let Some(r) = r.get(i_o as usize + j) {
                        *l_i = r.mul_add(*c, *l_i);
                    }
                })
            });
    }
}

pub fn db2_forward_pulp_from_steps<T: SimdImpl<Scalar = T> + FloatOps>(s: &mut [T], d: &mut [T]) {
    struct Impl<'a, 'b, T>(&'a mut [T], &'b mut [T]);

    impl<'a, 'b, T: SimdImpl<Scalar = T> + FloatOps> WithSimd for Impl<'a, 'b, T> {
        type Output = ();

        #[inline(always)]
        fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
            let s = self.0;
            let d = self.1;

            let cs = [
                T::from(-1.73205080756887729352744634150587236694280525381038062805581).unwrap(),
            ];
            forward_update_step_pulp::<_, _, _, 0>(simd, cs, d, s);

            let cs = [
                T::from(0.433012701892219323381861585376468091735701313452595157013952).unwrap(),
                T::from(-0.0669872981077806766181384146235319082642986865474048429860483).unwrap(),
            ];
            forward_update_step_pulp::<_, _, _, 0>(simd, cs, s, d);

            let cs = [T::from(1.0).unwrap()];
            forward_update_step_pulp::<_, _, _, -1>(simd, cs, d, s);

            let c = T::from(1.93185165257813657349948639945779473526780967801680910080469).unwrap();
            let cv = T::simd_splat(simd, c);

            let (s_h, s) = T::as_mut_simd(simd, s);
            let (s_h4, s_h) = s_h.as_chunks_mut::<4>();

            s_h4.iter_mut().for_each(|[s0, s1, s2, s3]| {
                *s0 = T::simd_mul(simd, *s0, cv);
                *s1 = T::simd_mul(simd, *s1, cv);
                *s2 = T::simd_mul(simd, *s2, cv);
                *s3 = T::simd_mul(simd, *s3, cv);
            });
            s_h.iter_mut().for_each(|s| *s = T::simd_mul(simd, *s, cv));

            let (d_h, d) = T::as_mut_simd(simd, d);
            let (d_h4, d_h) = d_h.as_chunks_mut::<4>();

            d_h4.iter_mut().for_each(|[d0, d1, d2, d3]| {
                *d0 = T::simd_div(simd, *d0, cv);
                *d1 = T::simd_div(simd, *d1, cv);
                *d2 = T::simd_div(simd, *d2, cv);
                *d3 = T::simd_div(simd, *d3, cv);
            });
            d_h.iter_mut().for_each(|d| *d = T::simd_div(simd, *d, cv));

            s.iter_mut().for_each(|s| *s *= c);
            d.iter_mut().for_each(|d| *d /= c);
        }
    }

    ARCH.dispatch(Impl(s, d));
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

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx", enable = "fma")]
pub unsafe fn db2_forward_avx_fma(s: &mut [f64], d: &mut [f64]) {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    const LANES: usize = 4;

    let ns = s.len();
    let nd = d.len();

    assert!(ns == nd || ns == nd + 1);

    let c = -1.73205080756887729352744634150587236694280525381038062805581;
    let cv = _mm256_broadcast_sd(&c);
    let (d_h, d_t) = d.as_chunks_mut::<LANES>();
    let (s_h, s_t) = s.as_chunks::<LANES>();

    let (d_h4, d_h) = d_h.as_chunks_mut::<4>();
    let (s_h4, s_h) = s_h.as_chunks::<4>();

    d_h4.iter_mut()
        .zip(s_h4)
        .for_each(|([d0, d1, d2, d3], [s0, s1, s2, s3])| unsafe {
            let dv = _mm256_loadu_pd(d0.as_ptr());
            let sv = _mm256_loadu_pd(s0.as_ptr());
            let dv = _mm256_fmadd_pd(sv, cv, dv);
            //let dv = vfmaq_n_f64(dv, sv, c);
            _mm256_storeu_pd(d0.as_mut_ptr(), dv);

            let dv = _mm256_loadu_pd(d1.as_ptr());
            let sv = _mm256_loadu_pd(s1.as_ptr());
            let dv = _mm256_fmadd_pd(sv, cv, dv);
            //let dv = vfmaq_n_f64(dv, sv, c);
            _mm256_storeu_pd(d1.as_mut_ptr(), dv);

            let dv = _mm256_loadu_pd(d2.as_ptr());
            let sv = _mm256_loadu_pd(s2.as_ptr());
            let dv = _mm256_fmadd_pd(sv, cv, dv);
            //let dv = vfmaq_n_f64(dv, sv, c);
            _mm256_storeu_pd(d2.as_mut_ptr(), dv);

            let dv = _mm256_loadu_pd(d3.as_ptr());
            let sv = _mm256_loadu_pd(s3.as_ptr());
            let dv = _mm256_fmadd_pd(sv, cv, dv);
            //let dv = vfmaq_n_f64(dv, sv, c);
            _mm256_storeu_pd(d3.as_mut_ptr(), dv);
        });

    d_h.iter_mut().zip(s_h.iter()).for_each(|(d, s)| unsafe {
        let dv = _mm256_loadu_pd(d.as_ptr());
        let sv = _mm256_loadu_pd(s.as_ptr());
        //let dv = vfmaq_n_f64(dv, sv, c);
        let dv = _mm256_fmadd_pd(sv, cv, dv);
        _mm256_storeu_pd(d.as_mut_ptr(), dv);
    });

    d_t.iter_mut().zip(s_t.iter()).for_each(|(d, s)| {
        *d = f64::mul_add(*s, c, *d);
    });

    let c = (
        0.433012701892219323381861585376468091735701313452595157013952,
        -0.0669872981077806766181384146235319082642986865474048429860483,
    );

    let cv = (_mm256_broadcast_sd(&c.0), _mm256_broadcast_sd(&c.1));

    let (s_h, s_t) = s[..nd - 1].as_chunks_mut::<LANES>();
    let (d0_h, d0_t) = d[..nd - 1].as_chunks::<LANES>();
    let (d1_h, d1_t) = d[1..nd].as_chunks::<LANES>();

    let (d0_h4, d0_h) = d0_h.as_chunks::<4>();
    let (d1_h4, d1_h) = d1_h.as_chunks::<4>();
    let (s_h4, s_h) = s_h.as_chunks_mut::<4>();

    s_h4.iter_mut().zip(d0_h4.iter().zip(d1_h4)).for_each(
        |([s0, s1, s2, s3], ([d0_0, d0_1, d0_2, d0_3], [d1_0, d1_1, d1_2, d1_3]))| unsafe {
            let sv = _mm256_loadu_pd(s0.as_ptr());
            let dv = _mm256_loadu_pd(d0_0.as_ptr());
            //let sv = vfmaq_n_f64(sv, dv, c.0);
            let sv = _mm256_fmadd_pd(dv, cv.0, sv);
            let dv = _mm256_loadu_pd(d1_0.as_ptr());
            //let sv = vfmaq_n_f64(sv, dv, c.1);
            let sv = _mm256_fmadd_pd(dv, cv.1, sv);
            _mm256_storeu_pd(s0.as_mut_ptr(), sv);

            let sv = _mm256_loadu_pd(s1.as_ptr());
            let dv = _mm256_loadu_pd(d0_1.as_ptr());
            //let sv = vfmaq_n_f64(sv, dv, c.0);
            let sv = _mm256_fmadd_pd(dv, cv.0, sv);
            let dv = _mm256_loadu_pd(d1_1.as_ptr());
            //let sv = vfmaq_n_f64(sv, dv, c.1);
            let sv = _mm256_fmadd_pd(dv, cv.1, sv);
            _mm256_storeu_pd(s1.as_mut_ptr(), sv);

            let sv = _mm256_loadu_pd(s2.as_ptr());
            let dv = _mm256_loadu_pd(d0_2.as_ptr());
            //let sv = vfmaq_n_f64(sv, dv, c.0);
            let sv = _mm256_fmadd_pd(dv, cv.0, sv);
            let dv = _mm256_loadu_pd(d1_2.as_ptr());
            //let sv = vfmaq_n_f64(sv, dv, c.1);
            let sv = _mm256_fmadd_pd(dv, cv.1, sv);
            _mm256_storeu_pd(s2.as_mut_ptr(), sv);

            let sv = _mm256_loadu_pd(s3.as_ptr());
            let dv = _mm256_loadu_pd(d0_3.as_ptr());
            //let sv = vfmaq_n_f64(sv, dv, c.0);
            let sv = _mm256_fmadd_pd(dv, cv.0, sv);
            let dv = _mm256_loadu_pd(d1_3.as_ptr());
            //let sv = vfmaq_n_f64(sv, dv, c.1);
            let sv = _mm256_fmadd_pd(dv, cv.1, sv);
            _mm256_storeu_pd(s3.as_mut_ptr(), sv);
        },
    );

    s_h.iter_mut()
        .zip(d0_h.iter().zip(d1_h))
        .for_each(|(s, (d0, d1))| unsafe {
            let sv = _mm256_loadu_pd(s.as_ptr());
            let dv = _mm256_loadu_pd(d0.as_ptr());
            //let sv = vfmaq_n_f64(sv, dv, c.0);
            let sv = _mm256_fmadd_pd(dv, cv.0, sv);
            let dv = _mm256_loadu_pd(d1.as_ptr());
            //let sv = vfmaq_n_f64(sv, dv, c.1);
            let sv = _mm256_fmadd_pd(dv, cv.1, sv);
            _mm256_storeu_pd(s.as_mut_ptr(), sv);
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
        let (d_h, d_t) = d.as_chunks_mut::<LANES>();
        let (s_h, s_t) = s[..nd - 1].as_chunks::<LANES>();

        let (d_h4, d_h) = d_h.as_chunks_mut::<4>();
        let (s_h4, s_h) = s_h.as_chunks::<4>();

        d_h4.iter_mut()
            .zip(s_h4)
            .for_each(|([d0, d1, d2, d3], [s0, s1, s2, s3])| unsafe {
                let dv = _mm256_loadu_pd(d0.as_ptr());
                let sv = _mm256_loadu_pd(s0.as_ptr());
                let dv = _mm256_add_pd(dv, sv);
                _mm256_storeu_pd(d0.as_mut_ptr(), dv);

                let dv = _mm256_loadu_pd(d1.as_ptr());
                let sv = _mm256_loadu_pd(s1.as_ptr());
                let dv = _mm256_add_pd(dv, sv);
                _mm256_storeu_pd(d1.as_mut_ptr(), dv);

                let dv = _mm256_loadu_pd(d2.as_ptr());
                let sv = _mm256_loadu_pd(s2.as_ptr());
                let dv = _mm256_add_pd(dv, sv);
                _mm256_storeu_pd(d2.as_mut_ptr(), dv);

                let dv = _mm256_loadu_pd(d3.as_ptr());
                let sv = _mm256_loadu_pd(s3.as_ptr());
                let dv = _mm256_add_pd(dv, sv);
                _mm256_storeu_pd(d3.as_mut_ptr(), dv);
            });

        d_h.iter_mut().zip(s_h).for_each(|(d, s)| unsafe {
            let dv = _mm256_loadu_pd(d.as_ptr());
            let sv = _mm256_loadu_pd(s.as_ptr());
            let ds = _mm256_add_pd(dv, sv);
            _mm256_storeu_pd(d.as_mut_ptr(), ds);
        });
        d_t.iter_mut().zip(s_t.iter()).for_each(|(d, s)| {
            *d += s;
        })
    }

    let c = 1.93185165257813657349948639945779473526780967801680910080469;
    let cv = _mm256_broadcast_sd(&c);

    let (s_h, s_t) = s.as_chunks_mut::<LANES>();

    let (s_h4, s_h) = s_h.as_chunks_mut::<4>();

    s_h4.iter_mut().for_each(|[s0, s1, s2, s3]| unsafe {
        let sv = _mm256_loadu_pd(s0.as_ptr());
        let sv = _mm256_mul_pd(sv, cv);
        //let sv = vmulq_n_f64(sv, c);
        _mm256_storeu_pd(s0.as_mut_ptr(), sv);

        let sv = _mm256_loadu_pd(s1.as_ptr());
        let sv = _mm256_mul_pd(sv, cv);
        //let sv = vmulq_n_f64(sv, c);
        _mm256_storeu_pd(s1.as_mut_ptr(), sv);

        let sv = _mm256_loadu_pd(s2.as_ptr());
        let sv = _mm256_mul_pd(sv, cv);
        //let sv = vmulq_n_f64(sv, c);
        _mm256_storeu_pd(s2.as_mut_ptr(), sv);

        let sv = _mm256_loadu_pd(s3.as_ptr());
        let sv = _mm256_mul_pd(sv, cv);
        //let sv = vmulq_n_f64(sv, c);
        _mm256_storeu_pd(s3.as_mut_ptr(), sv);
    });

    s_h.iter_mut().for_each(|s| unsafe {
        let sv = _mm256_loadu_pd(s.as_ptr());
        let sv = _mm256_mul_pd(sv, cv);
        //let sv = vmulq_n_f64(sv, c);
        _mm256_storeu_pd(s.as_mut_ptr(), sv);
    });

    s_t.iter_mut().for_each(|s| *s *= c);

    let cinv = 1.0 / c;
    let cinvv = _mm256_broadcast_sd(&cinv);

    let (d_h, d_t) = d.as_chunks_mut::<LANES>();
    let (d_h4, d_h) = d_h.as_chunks_mut::<4>();

    d_h4.iter_mut().for_each(|[d0, d1, d2, d3]| unsafe {
        let dv = _mm256_loadu_pd(d0.as_ptr());
        let dv = _mm256_mul_pd(dv, cinvv);
        //let dv = vmulq_n_f64(dv, cinv);
        _mm256_storeu_pd(d0.as_mut_ptr(), dv);

        let dv = _mm256_loadu_pd(d1.as_ptr());
        let dv = _mm256_mul_pd(dv, cinvv);
        //let dv = vmulq_n_f64(dv, cinv);
        _mm256_storeu_pd(d1.as_mut_ptr(), dv);

        let dv = _mm256_loadu_pd(d2.as_ptr());
        let dv = _mm256_mul_pd(dv, cinvv);
        //let dv = vmulq_n_f64(dv, cinv);
        _mm256_storeu_pd(d2.as_mut_ptr(), dv);

        let dv = _mm256_loadu_pd(d3.as_ptr());
        let dv = _mm256_mul_pd(dv, cinvv);
        //let dv = vmulq_n_f64(dv, cinv);
        _mm256_storeu_pd(d3.as_mut_ptr(), dv);
    });

    d_h.iter_mut().for_each(|d| unsafe {
        let dv = _mm256_loadu_pd(d.as_ptr());
        let dv = _mm256_mul_pd(dv, cinvv);
        //let dv = vmulq_n_f64(dv, cinv);
        _mm256_storeu_pd(d.as_mut_ptr(), dv);
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

pub mod check {
    use super::*;
    // use wavelets_macros::generate_forward_func_simd;

    // struct DB2 {}

    // generate_forward_func_simd! {
    //     DB2,
    //     UpdateD(
    //         0,
    //         [-1.73205080756887729352744634150587236694280525381038062805581]
    //     ),
    //     UpdateS(
    //         0,
    //         [
    //             0.433012701892219323381861585376468091735701313452595157013952,
    //             -0.0669872981077806766181384146235319082642986865474048429860483
    //         ]
    //     ),
    //     UpdateD(-1, [1.0]),
    //     Scale(1.93185165257813657349948639945779473526780967801680910080469)
    // }

    pub fn forward_simd<T, BC>(s: &mut [T], d: &mut [T], bc: &BC)
    where
        T: crate::SimdImpl<Scalar = T> + crate::FloatOps + wavelets::Transformable,
        BC: wavelets::boundarys::BoundaryExtension,
    {
        use ::itertools::izip;
        struct Impl<'a, 'b, 'c, T, BC>(&'a mut [T], &'b mut [T], &'c BC);
        impl<'a, 'b, 'c, T, BC> WithSimd for Impl<'a, 'b, 'c, T, BC>
        where
            T: crate::SimdImpl<Scalar = T> + crate::FloatOps + wavelets::Transformable,
            BC: wavelets::boundarys::BoundaryExtension,
        {
            type Output = ();
            #[inline(always)]
            fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
                let s = self.0;
                let d = self.1;
                let bc = self.2;
                let _ns = s.len();
                let nd = d.len();
                assert!(
                    d.len() == s.len() || d.len() + 1 == s.len(),
                    "detail and scaling coefficient arrays must have compatible lengths"
                );
                let c = (T::from(-1.7320508075688772f64).unwrap(),);
                let ir_end = d.len().checked_add_signed(1isize).unwrap_or(0);
                let nr = (ir_end - 0usize).checked_sub(1usize).unwrap_or(0);
                if nr > 0 {
                    let cv = (T::simd_splat(simd, c.0),);
                    let (l_h, l) = T::as_mut_simd(simd, &mut d[0usize..nr + 0usize]);
                    let (l_h4, l_h) = l_h.as_chunks_mut::<4>();
                    let (r0_h, r0) = T::as_simd(simd, &s[0..nr + 0]);
                    let (r0_h4, r0_h) = r0_h.as_chunks::<4>();
                    l_h4.iter_mut()
                        .zip(izip!(r0_h4))
                        .for_each(|([l0, l1, l2, l3], r0)| {
                            *l0 = T::simd_mul_add(simd, r0[0], cv.0, *l0);
                            *l1 = T::simd_mul_add(simd, r0[1], cv.0, *l1);
                            *l2 = T::simd_mul_add(simd, r0[2], cv.0, *l2);
                            *l3 = T::simd_mul_add(simd, r0[3], cv.0, *l3);
                        });
                    l_h.iter_mut().zip(izip!(r0_h)).for_each(|(l, r0)| {
                        *l = T::simd_mul_add(simd, *r0, cv.0, *l);
                    });
                    l.iter_mut().zip(izip!(r0)).for_each(|(l, r0)| {
                        *l += c.0 * r0;
                    });
                }
                let c = (
                    T::from(0.4330127018922193f64).unwrap(),
                    T::from(-0.06698729810778067f64).unwrap(),
                );
                let ir_end = std::cmp::min(nd, s.len().checked_add_signed(2isize).unwrap_or(0));
                let nr = (ir_end - 0usize).checked_sub(2usize).unwrap_or(0);
                if nr > 0 {
                    let cv = (T::simd_splat(simd, c.0), T::simd_splat(simd, c.1));
                    let (l_h, l) = T::as_mut_simd(simd, &mut s[0usize..nr + 0usize]);
                    let (l_h4, l_h) = l_h.as_chunks_mut::<4>();
                    let (r0_h, r0) = T::as_simd(simd, &d[0..nr + 0]);
                    let (r0_h4, r0_h) = r0_h.as_chunks::<4>();
                    let (r1_h, r1) = T::as_simd(simd, &d[1..nr + 1]);
                    let (r1_h4, r1_h) = r1_h.as_chunks::<4>();
                    l_h4.iter_mut().zip(izip!(r0_h4, r1_h4)).for_each(
                        |([l0, l1, l2, l3], (r0, r1))| {
                            *l0 = T::simd_mul_add(simd, r0[0], cv.0, *l0);
                            *l1 = T::simd_mul_add(simd, r0[1], cv.0, *l1);
                            *l2 = T::simd_mul_add(simd, r0[2], cv.0, *l2);
                            *l3 = T::simd_mul_add(simd, r0[3], cv.0, *l3);
                            *l0 = T::simd_mul_add(simd, r1[0], cv.1, *l0);
                            *l1 = T::simd_mul_add(simd, r1[1], cv.1, *l1);
                            *l2 = T::simd_mul_add(simd, r1[2], cv.1, *l2);
                            *l3 = T::simd_mul_add(simd, r1[3], cv.1, *l3);
                        },
                    );
                    l_h.iter_mut()
                        .zip(izip!(r0_h, r1_h))
                        .for_each(|(l, (r0, r1))| {
                            *l = T::simd_mul_add(simd, *r0, cv.0, *l);
                            *l = T::simd_mul_add(simd, *r1, cv.1, *l);
                        });
                    l.iter_mut().zip(izip!(r0, r1)).for_each(|(l, (r0, r1))| {
                        *l += c.0 * r0;
                        *l += c.1 * r1;
                    });
                }
                let n2 = std::cmp::min(0usize + nr, s.len());
                (n2 as isize..s.len() as isize)
                    .zip(&mut s[n2..])
                    .for_each(|(i, s_i)| {
                        if let Some(r_i) = bc.get_bc(d, i + 0isize) {
                            *s_i += r_i * c.0;
                        }
                        if let Some(r_i) = bc.get_bc(d, i + 1isize) {
                            *s_i += r_i * c.1;
                        }
                    });
                let n1 = std::cmp::min(1usize, d.len());
                (0..1usize as isize).zip(&mut d[..n1]).for_each(|(i, d_i)| {
                    if let Some(r_i) = bc.get_bc(s, i + -1isize) {
                        *d_i += r_i;
                    }
                });
                let ir_end = d.len().checked_add_signed(0isize).unwrap_or(0);
                let nr = (ir_end - 0usize).checked_sub(1usize).unwrap_or(0);
                if nr > 0 {
                    let (l_h, l) = T::as_mut_simd(simd, &mut d[1usize..nr + 1usize]);
                    let (l_h4, l_h) = l_h.as_chunks_mut::<4>();
                    let (r0_h, r0) = T::as_simd(simd, &s[0..nr + 0]);
                    let (r0_h4, r0_h) = r0_h.as_chunks::<4>();
                    l_h4.iter_mut()
                        .zip(izip!(r0_h4))
                        .for_each(|([l0, l1, l2, l3], r0)| {
                            *l0 = T::simd_add(simd, r0[0], *l0);
                            *l1 = T::simd_add(simd, r0[1], *l1);
                            *l2 = T::simd_add(simd, r0[2], *l2);
                            *l3 = T::simd_add(simd, r0[3], *l3);
                        });
                    l_h.iter_mut().zip(izip!(r0_h)).for_each(|(l, r0)| {
                        *l = T::simd_add(simd, *r0, *l);
                    });
                    l.iter_mut().zip(izip!(r0)).for_each(|(l, r0)| {
                        *l += r0;
                    });
                }
                let scaling =
                    T::from(1.93185165257813657349948639945779473526780967801680910080469).unwrap();
                let scaling_vec = T::simd_splat(simd, scaling);
                let (s_h, s_t) = T::as_mut_simd(simd, s);
                let (s_h4, s_h) = s_h.as_chunks_mut::<4>();
                s_h4.iter_mut().for_each(|[s0, s1, s2, s3]| {
                    *s0 = T::simd_mul(simd, *s0, scaling_vec);
                    *s1 = T::simd_mul(simd, *s1, scaling_vec);
                    *s2 = T::simd_mul(simd, *s2, scaling_vec);
                    *s3 = T::simd_mul(simd, *s3, scaling_vec);
                });
                s_h.iter_mut()
                    .for_each(|s| *s = T::simd_mul(simd, *s, scaling_vec));
                s_t.iter_mut().for_each(|s| *s *= scaling);
                let scaling =
                    T::from(1.0 / 1.93185165257813657349948639945779473526780967801680910080469)
                        .unwrap();
                let scaling_vec = T::simd_splat(simd, scaling);
                let (d_h, d_t) = T::as_mut_simd(simd, d);
                let (d_h4, d_h) = d_h.as_chunks_mut::<4>();
                d_h4.iter_mut().for_each(|[d0, d1, d2, d3]| {
                    *d0 = T::simd_mul(simd, *d0, scaling_vec);
                    *d1 = T::simd_mul(simd, *d1, scaling_vec);
                    *d2 = T::simd_mul(simd, *d2, scaling_vec);
                    *d3 = T::simd_mul(simd, *d3, scaling_vec);
                });
                d_h.iter_mut()
                    .for_each(|d| *d = T::simd_mul(simd, *d, scaling_vec));
                d_t.iter_mut().for_each(|d| *d *= scaling);
            }
        }
        crate::ARCH.dispatch(Impl(s, d, bc));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn testable() {
        let n = 32;

        let mut s_ref = (0..n).map(|v| v as f64).collect::<Vec<_>>();
        let mut d_ref = (0..n).map(|v| -v as f64).collect::<Vec<_>>();

        db2_forward(&mut s_ref, &mut d_ref);

        {
            let mut s_test = (0..n).map(|v| v as f64).collect::<Vec<_>>();
            let mut d_test = (0..n).map(|v| -v as f64).collect::<Vec<_>>();

            db2_forward_from_steps(&mut s_test, &mut d_test);
            assert!(
                s_ref
                    .iter()
                    .zip(s_test.iter())
                    .enumerate()
                    .all(|(i, (x, y))| {
                        let a_diff = (x - y).abs();
                        let ok = a_diff < 1E-14;
                        if !ok {
                            dbg!(i, x, y, a_diff);
                        }
                        ok
                    })
            );

            assert!(
                d_ref
                    .iter()
                    .zip(d_test.iter())
                    .enumerate()
                    .all(|(i, (x, y))| {
                        let a_diff = (x - y).abs();
                        let ok = a_diff < 1E-14;
                        if !ok {
                            dbg!(i, x, y, a_diff);
                        }
                        ok
                    })
            );
        }

        {
            let mut s_test = (0..n).map(|v| v as f64).collect::<Vec<_>>();
            let mut d_test = (0..n).map(|v| -v as f64).collect::<Vec<_>>();
            db2_forward_pulp(&mut s_test, &mut d_test);
            assert!(
                s_ref
                    .iter()
                    .zip(s_test.iter())
                    .enumerate()
                    .all(|(i, (x, y))| {
                        let a_diff = (x - y).abs();
                        let ok = a_diff < 1E-14;
                        if !ok {
                            dbg!(i, x, y, a_diff);
                        }
                        ok
                    })
            );

            assert!(
                d_ref
                    .iter()
                    .zip(d_test.iter())
                    .enumerate()
                    .all(|(i, (x, y))| {
                        let a_diff = (x - y).abs();
                        let ok = a_diff < 1E-14;
                        if !ok {
                            dbg!(i, x, y, a_diff);
                        }
                        ok
                    })
            );
        }

        {
            let mut s_test = (0..n).map(|v| v as f64).collect::<Vec<_>>();
            let mut d_test = (0..n).map(|v| -v as f64).collect::<Vec<_>>();
            db2_forward_pulp_from_steps(&mut s_test, &mut d_test);
            assert!(
                s_ref
                    .iter()
                    .zip(s_test.iter())
                    .enumerate()
                    .all(|(i, (x, y))| {
                        let a_diff = (x - y).abs();
                        let ok = a_diff < 1E-14;
                        if !ok {
                            dbg!(i, x, y, a_diff);
                        }
                        ok
                    })
            );

            assert!(
                d_ref
                    .iter()
                    .zip(d_test.iter())
                    .enumerate()
                    .all(|(i, (x, y))| {
                        let a_diff = (x - y).abs();
                        let ok = a_diff < 1E-14;
                        if !ok {
                            dbg!(i, x, y, a_diff);
                        }
                        ok
                    })
            );
        }

        {
            let mut s_test = (0..n).map(|v| v as f64).collect::<Vec<_>>();
            let mut d_test = (0..n).map(|v| -v as f64).collect::<Vec<_>>();
            check::forward_simd(
                &mut s_test,
                &mut d_test,
                &wavelets::boundarys::ZeroBoundary {},
            );
            assert!(
                s_ref
                    .iter()
                    .zip(s_test.iter())
                    .enumerate()
                    .all(|(i, (x, y))| {
                        let a_diff = (x - y).abs();
                        let ok = a_diff < 1E-13;
                        if !ok {
                            dbg!(i, x, y, a_diff);
                        }
                        ok
                    })
            );

            assert!(
                d_ref
                    .iter()
                    .zip(d_test.iter())
                    .enumerate()
                    .all(|(i, (x, y))| {
                        let a_diff = (x - y).abs();
                        let ok = a_diff < 1E-13;
                        if !ok {
                            dbg!(i, x, y, a_diff);
                        }
                        ok
                    })
            );
        }

        #[cfg(target_arch = "aarch64")]
        {
            let mut s_test = (0..n).map(|v| v as f64).collect::<Vec<_>>();
            let mut d_test = (0..n).map(|v| -v as f64).collect::<Vec<_>>();

            unsafe {
                db2_forward_neon(&mut s_test, &mut d_test);
            }
            assert!(
                s_ref
                    .iter()
                    .zip(s_test.iter())
                    .enumerate()
                    .all(|(i, (x, y))| {
                        let a_diff = (x - y).abs();
                        let ok = a_diff < 1E-14;
                        if !ok {
                            dbg!(i, x, y, a_diff);
                        }
                        ok
                    })
            );

            assert!(
                d_ref
                    .iter()
                    .zip(d_test.iter())
                    .enumerate()
                    .all(|(i, (x, y))| {
                        let a_diff = (x - y).abs();
                        let ok = a_diff < 1E-14;
                        if !ok {
                            dbg!(i, x, y, a_diff);
                        }
                        ok
                    })
            );
        }

        #[cfg(all(any(target_arch = "x86_64", target_arch = "x86"),))]
        {
            if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
                let mut s_test = (0..n).map(|v| v as f64).collect::<Vec<_>>();
                let mut d_test = (0..n).map(|v| -v as f64).collect::<Vec<_>>();

                unsafe {
                    db2_forward_avx_fma(&mut s_test, &mut d_test);
                }
                assert!(
                    s_ref
                        .iter()
                        .zip(s_test.iter())
                        .enumerate()
                        .all(|(i, (x, y))| {
                            let a_diff = (x - y).abs();
                            let ok = a_diff < 1E-13;
                            if !ok {
                                dbg!(i, x, y, a_diff);
                            }
                            ok
                        })
                );

                assert!(
                    d_ref
                        .iter()
                        .zip(d_test.iter())
                        .enumerate()
                        .all(|(i, (x, y))| {
                            let a_diff = (x - y).abs();
                            let ok = a_diff < 1E-13;
                            if !ok {
                                dbg!(i, x, y, a_diff);
                            }
                            ok
                        })
                );
            }
        }
    }
}
