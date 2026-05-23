use aligned_vec::{AVec, avec};
use criterion::{Criterion, criterion_group, criterion_main};
use itertools::Itertools;

mod extras {
    use wavelets::simd::*;
    use wavelets::{boundarys::BoundaryExtension, simd::SimdTransformable};

    /// Placeholder
    pub(super) fn db2_forward_arr<T, BC, const N: usize>(
        s: &mut [[T; N]],
        d: &mut [[T; N]],
        bc: &BC,
    ) where
        T: SimdTransformable,
        BC: BoundaryExtension,
    {
        use wavelets::simd::Dispatch;
        let n_lanes = T::lanes();

        debug_assert_eq!(N, n_lanes);

        let ns = s.len();
        let nd = d.len();
        assert!(
            ns == nd || nd + 1 == ns,
            "detail and smooth coefficient arrays must have compatible lengths, got {nd} d-chunks and {ns} s-chunks."
        );

        struct Impl<'a, 'b, 'c, T, BC>(&'a mut [T], &'b mut [T], &'c BC);

        impl<'a, 'b, 'c, T: SimdTransformable, BC: BoundaryExtension> WithSimd for Impl<'a, 'b, 'c, T, BC>
        where
            T: SimdTransformable,
            BC: BoundaryExtension,
        {
            type Output = ();
            #[inline(always)]
            fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
                let s = T::as_mut_simd(simd, self.0).0;
                let d = T::as_mut_simd(simd, self.1).0;
                let ns = s.len();
                let nd = d.len();
                let bc = self.2;

                let c = T::simd_splat(
                    simd,
                    T::scalar_type_from_f64(
                        -1.73205080756887729352744634150587236694280525381038062805581,
                    ),
                );

                d.iter_mut().zip(s.iter()).for_each(|(l, r)| {
                    *l = T::simd_mul_add(simd, *r, c, *l);
                });

                let c = [
                    T::simd_splat(
                        simd,
                        T::scalar_type_from_f64(
                            0.433012701892219323381861585376468091735701313452595157013952,
                        ),
                    ),
                    T::simd_splat(
                        simd,
                        T::scalar_type_from_f64(
                            -0.0669872981077806766181384146235319082642986865474048429860483,
                        ),
                    ),
                ];

                let (sf, sb) = s.split_at_mut(nd - 1);

                sf.iter_mut()
                    .zip(d.array_windows())
                    .for_each(|(l, [r0, r1])| {
                        *l = T::simd_mul_add(simd, *r0, c[0], *l);
                        *l = T::simd_mul_add(simd, *r1, c[1], *l);
                    });

                (nd as isize - 1..ns as isize).zip(sb).for_each(|(io, l)| {
                    c.iter().enumerate().for_each(|(i, c)| {
                        let bc_parts = bc.get_parts::<T>(nd, io + i as isize);
                        for (coef, i_bc) in bc_parts {
                            let rv = match coef {
                                Some(coef) => {
                                    let c = T::simd_splat(simd, coef);
                                    T::simd_mul(simd, d[i_bc], c)
                                }
                                None => d[i_bc],
                            };
                            *l = T::simd_mul_add(simd, rv, *c, *l);
                        }
                    });
                });

                let (df, dv) = d.split_at_mut(1);

                (-1..0).zip(df).for_each(|(io, l)| {
                    let bc_parts = bc.get_parts::<T>(nd, io);
                    for (coef, i_bc) in bc_parts {
                        match coef {
                            Some(coef) => {
                                let c = T::simd_splat(simd, coef);
                                *l = T::simd_mul_add(simd, s[i_bc], c, *l);
                            }
                            None => {
                                *l = T::simd_add(simd, s[i_bc], *l);
                            }
                        };
                    }
                });

                dv.iter_mut().zip(s.iter()).for_each(|(l, r)| {
                    *l = T::simd_add(simd, *r, *l);
                });

                let scale = T::simd_splat(
                    simd,
                    T::scalar_type_from_f64(
                        1.93185165257813657349948639945779473526780967801680910080469,
                    ),
                );
                let inv_scale = T::simd_splat(
                    simd,
                    T::scalar_type_from_f64(
                        1.0 / 1.93185165257813657349948639945779473526780967801680910080469,
                    ),
                );

                s.iter_mut().for_each(|s| *s = T::simd_mul(simd, *s, scale));
                d.iter_mut()
                    .for_each(|d| *d = T::simd_mul(simd, *d, inv_scale));
            }
        }

        wavelets::simd::ARCH.dispatch_wvlt(Impl(s.as_flattened_mut(), d.as_flattened_mut(), bc));
    }
}

fn db2_benchmark(c: &mut Criterion) {
    use wavelets::boundarys::BoundaryCondition;
    use wavelets::daubechies;
    type WVLT = daubechies::Daubechies2;
    let n = 1000;
    let ns = (n + 1) / 2;
    let nd = n / 2;

    let x = AVec::<_>::from_iter(128, (0..n).map(|i| i as f64));

    let mut s = AVec::<_>::from_iter(128, (0..ns).map(|i| i as f64));
    let mut d = AVec::<_>::from_iter(128, (0..nd).map(|i| (i + ns) as f64));

    let bc = BoundaryCondition::Zero;

    let mut group = c.benchmark_group("Daubechies 2");

    group.bench_function("lifted inplace", |b| {
        b.iter(|| {
            use wavelets::lwt::LiftingTransform;
            WVLT::forward(&mut s, &mut d, &bc);
        })
    });

    group.bench_function("lifted out of place", |b| {
        b.iter(|| {
            // this is closest in operation to the filtered version
            use wavelets::lwt::LiftingTransform;
            wavelets::utils::deinterleave(&x, &mut s, &mut d);
            WVLT::forward(&mut s, &mut d, &bc);
        })
    });

    let x2 = (0..n).map(|i| i as f64).collect_vec();
    let nsd = wavelets::dwt::get_outlen(WVLT::WIDTH, n);
    let mut s2 = avec![0.0; nsd];
    let mut d2 = avec![0.0; nsd];

    group.bench_function("filtered", |b| {
        b.iter(|| {
            use wavelets::dwt::DiscreteTransform;
            WVLT::forward(&x2, &mut s2, &mut d2, &bc);
        })
    });

    group.finish();
}

fn db6_benchmark(c: &mut Criterion) {
    use wavelets::boundarys::BoundaryCondition;
    use wavelets::daubechies;
    type WVLT = daubechies::Daubechies6;
    let n = 1000;
    let ns = (n + 1) / 2;
    let nd = n / 2;

    let x = AVec::<_>::from_iter(128, (0..n).map(|i| i as f64));

    let mut s = AVec::<_>::from_iter(128, (0..ns).map(|i| i as f64));
    let mut d = AVec::<_>::from_iter(128, (0..nd).map(|i| (i + ns) as f64));

    let bc = BoundaryCondition::Zero;

    let mut group = c.benchmark_group("Daubechies 6");

    group.bench_function("lifted inplace", |b| {
        b.iter(|| {
            use wavelets::lwt::LiftingTransform;
            WVLT::forward(&mut s, &mut d, &bc);
        })
    });

    group.bench_function("lifted out of place", |b| {
        b.iter(|| {
            // this is closest in operation to the filtered version
            use wavelets::lwt::LiftingTransform;
            wavelets::utils::deinterleave(&x, &mut s, &mut d);
            WVLT::forward(&mut s, &mut d, &bc);
        })
    });

    let x2 = (0..n).map(|i| i as f64).collect_vec();
    let nsd = wavelets::dwt::get_outlen(WVLT::WIDTH, n);
    let mut s2 = vec![0.0; nsd];
    let mut d2 = vec![0.0; nsd];

    group.bench_function("filtered", |b| {
        b.iter(|| {
            use wavelets::dwt::DiscreteTransform;
            WVLT::forward(&x2, &mut s2, &mut d2, &bc);
        })
    });

    group.finish();
}

fn interleave_slice_benchmark(c: &mut Criterion) {
    use wavelets::utils::{interleave, interleave_inplace};

    let n = 1042;
    let evens = (0..n).step_by(2).collect_vec();
    let odds = (1..n).step_by(2).collect_vec();
    let mut x1 = vec![0; n];
    let mut x2 = evens.iter().chain(odds.iter()).collect_vec();

    let mut group = c.benchmark_group("slice");

    group.bench_function("out of place", |b| {
        b.iter(|| {
            interleave(&evens, &odds, &mut x1);
        })
    });

    group.bench_function("in place", |b| {
        b.iter(|| {
            interleave_inplace(&mut x2);
        })
    });

    group.finish();
}

fn interleave_strided_benchmark(c: &mut Criterion) {
    use wavelets::iter::LanesIterator;

    let n: usize = 1000;

    let mut group = c.benchmark_group("interleave");

    let shape = [n, n];
    let n_total: usize = shape.iter().product();
    let x1 = (0..n_total as i32).collect_vec();
    let mut x2 = (0..n_total as i32).collect_vec();

    let nf = (n + 1) / 2;
    let ns = n / 2;

    let mut work_f = vec![0; nf];
    let mut work_s = vec![0; ns];

    group.bench_function("lanes/across", |b| {
        let ax = 0;
        b.iter(|| {
            for (lane_in, mut lane_out) in
                x1.iter_lanes(&shape, ax).zip(x2.iter_lanes_mut(&shape, ax))
            {
                lane_in.split(&mut work_f, &mut work_s);
                lane_out.interleave(&work_f, &work_s);
            }
        })
    });

    group.bench_function("lanes/along", |b| {
        let ax = 1;
        b.iter(|| {
            for (lane_in, mut lane_out) in
                x1.iter_lanes(&shape, ax).zip(x2.iter_lanes_mut(&shape, ax))
            {
                lane_in.split(&mut work_f, &mut work_s);
                lane_out.interleave(&work_f, &work_s);
            }
        })
    });

    const N: usize = 8;

    let mut work_f: [_; N] = core::array::from_fn(|_| avec![0; nf]);
    let mut work_s: [_; N] = core::array::from_fn(|_| avec![0; nf]);

    let mut work_f2 = avec![0; nf];
    let mut work_s2 = avec![0; ns];

    group.bench_function("chunks/across", |b| {
        let ax = 0;
        b.iter(|| {
            let in_chunk = x1.iter_lane_chunks::<N>(&shape, ax);
            let in_rem = in_chunk.remainder();
            let out_chunk = x2.iter_lane_chunks_mut::<N>(&shape, ax);
            let out_rem = out_chunk.remainder();

            for (in_chunk, mut out_chunk) in in_chunk.zip(out_chunk) {
                in_chunk.split(&mut work_f, &mut work_s);
                out_chunk.interleave(&&work_f, &work_s);
            }
            for (in_rem, mut out_rem) in in_rem.zip(out_rem) {
                in_rem.split(&mut work_f2, &mut work_s2);
                out_rem.interleave(&work_f2, &work_s2);
            }
        })
    });

    group.bench_function("chunks/along", |b| {
        let ax = 1;
        b.iter(|| {
            let in_chunk = x1.iter_lane_chunks::<N>(&shape, ax);
            let in_rem = in_chunk.remainder();
            let out_chunk = x2.iter_lane_chunks_mut::<N>(&shape, ax);
            let out_rem = out_chunk.remainder();

            for (in_chunk, mut out_chunk) in in_chunk.zip(out_chunk) {
                in_chunk.split(&mut work_f, &mut work_s);
                out_chunk.interleave(&work_f, &work_s);
            }
            for (in_rem, mut out_rem) in in_rem.zip(out_rem) {
                in_rem.split(&mut work_f2, &mut work_s2);
                out_rem.interleave(&work_f2, &work_s2);
            }
        })
    });

    group.bench_function("chunked", |b| {
        b.iter(|| {
            // construct vector of slices of the input array
            //let mut arrs = x1.chunks_exact(n).collect_vec();

            //interleave_inplace(&mut arrs);
            let n_total = x1.len();
            let n0 = x1.len() / n;
            let n_first = (n0 + 1) / 2;

            let (first, second) = x1.split_at(n_first * n);

            let mut first_chunks = first.chunks_exact(n);

            first_chunks
                .by_ref()
                .zip(second.chunks_exact(n))
                .zip(x2.chunks_exact_mut(2 * n))
                .for_each(|((f, s), out)| {
                    let (evens, odds) = out.split_at_mut(n);
                    evens
                        .iter_mut()
                        .zip(f.iter().cloned())
                        .for_each(|(e, f)| *e = f);
                    odds.iter_mut()
                        .zip(s.iter().cloned())
                        .for_each(|(o, s)| *o = s);
                });

            if let Some(last_chunk) = first_chunks.next() {
                x2[n_total - n..n_total]
                    .iter_mut()
                    .zip(last_chunk.iter().cloned())
                    .for_each(|(e, f)| *e = f);
            }
        })
    });

    group.finish();
}

fn deinterleave_benchmark(c: &mut Criterion) {
    use wavelets::iter::LanesIterator;
    use wavelets::utils::deinterleave_nd;

    let n = 1000;

    let mut group = c.benchmark_group("deinterleave");

    let shape: Vec<usize> = vec![n, n];
    let n_total: usize = shape.iter().product();
    let x1 = (0..n_total as i32).collect_vec();
    let mut x2 = (0..n_total as i32).collect_vec();

    group.bench_function("lanes/across", |b| {
        let ax = 0;
        let n = shape[ax];
        let n_e = (n + 1) / 2;
        let n_o = n / 2;
        let mut work_e = vec![0; n_e];
        let mut work_o = vec![0; n_o];
        b.iter(|| {
            for (lane_in, mut lane_out) in
                x1.iter_lanes(&shape, ax).zip(x2.iter_lanes_mut(&shape, ax))
            {
                lane_in.deinterleave(&mut work_e, &mut work_o);
                lane_out.stack(&work_e, &work_o);
            }
        })
    });

    group.bench_function("lanes/along", |b| {
        let ax = 1;
        let n = shape[ax];
        let n_e = (n + 1) / 2;
        let n_o = n / 2;
        let mut work_e = vec![0; n_e];
        let mut work_o = vec![0; n_o];
        b.iter(|| {
            for (lane_in, mut lane_out) in
                x1.iter_lanes(&shape, ax).zip(x2.iter_lanes_mut(&shape, ax))
            {
                lane_in.deinterleave(&mut work_e, &mut work_o);
                lane_out.stack(&work_e, &work_o);
            }
        })
    });

    const N: usize = 8;

    group.bench_function("chunks/across", |b| {
        let ax = 0;
        let n = shape[ax];
        let n_e = (n + 1) / 2;
        let n_o = n / 2;

        let mut work_e: [_; N] = core::array::from_fn(|_| avec![0; n_e]);
        let mut work_o: [_; N] = core::array::from_fn(|_| avec![0; n_o]);

        let mut work_e2 = avec![0; n_e];
        let mut work_o2 = avec![0; n_o];

        b.iter(|| {
            let in_chunk = x1.iter_lane_chunks::<N>(&shape, ax);
            let in_rem = in_chunk.remainder();
            let out_chunk = x2.iter_lane_chunks_mut::<N>(&shape, ax);
            let out_rem = out_chunk.remainder();

            for (in_chunk, mut out_chunk) in in_chunk.zip(out_chunk) {
                in_chunk.deinterleave(&mut work_e, &mut work_o);
                out_chunk.stack(&work_e, &work_o);
            }
            for (in_rem, mut out_rem) in in_rem.zip(out_rem) {
                in_rem.deinterleave(&mut work_e2, &mut work_o2);
                out_rem.stack(&work_e2, &work_o2);
            }
        })
    });

    group.bench_function("chunks/along", |b| {
        let ax = 1;
        let n = shape[ax];
        let n_e = (n + 1) / 2;
        let n_o = n / 2;

        let mut work_e: [_; N] = core::array::from_fn(|_| avec![0; n_e]);
        let mut work_o: [_; N] = core::array::from_fn(|_| avec![0; n_o]);

        let mut work_e2 = avec![0; n_e];
        let mut work_o2 = avec![0; n_o];

        b.iter(|| {
            let in_chunk = x1.iter_lane_chunks::<N>(&shape, ax);
            let in_rem = in_chunk.remainder();
            let out_chunk = x2.iter_lane_chunks_mut::<N>(&shape, ax);
            let out_rem = out_chunk.remainder();

            for (in_chunk, mut out_chunk) in in_chunk.zip(out_chunk) {
                in_chunk.deinterleave(&mut work_e, &mut work_o);
                out_chunk.stack(&work_e, &work_o);
            }
            for (in_rem, mut out_rem) in in_rem.zip(out_rem) {
                in_rem.deinterleave(&mut work_e2, &mut work_o2);
                out_rem.stack(&work_e2, &work_o2);
            }
        })
    });

    group.bench_function("array_chunks/across", |b| {
        let ax = 0;
        let n = shape[ax];
        let n_e = (n + 1) / 2;
        let n_o = n / 2;

        let mut work_e = avec![[0; N]; n_e];
        let mut work_o = avec![[0; N]; n_o];

        let mut work_e2 = avec![0; n_e];
        let mut work_o2 = avec![0; n_o];

        b.iter(|| {
            let in_chunk = x1.iter_lane_chunks::<N>(&shape, ax);
            let in_rem = in_chunk.remainder();
            let out_chunk = x2.iter_lane_chunks_mut::<N>(&shape, ax);
            let out_rem = out_chunk.remainder();

            for (in_chunk, mut out_chunk) in in_chunk.zip(out_chunk) {
                in_chunk.deinterleave_arrays(&mut work_e, &mut work_o);
                out_chunk.stack_arrays(&work_e, &work_o);
            }
            for (in_rem, mut out_rem) in in_rem.zip(out_rem) {
                in_rem.deinterleave(&mut work_e2, &mut work_o2);
                out_rem.stack(&work_e2, &work_o2);
            }
        })
    });

    group.bench_function("array_chunks/along", |b| {
        let ax = 1;
        let n = shape[ax];
        let n_e = (n + 1) / 2;
        let n_o = n / 2;

        let mut work_e = avec![[0; N]; n_e];
        let mut work_o = avec![[0; N]; n_o];

        let mut work_e2 = avec![0; n_e];
        let mut work_o2 = avec![0; n_o];

        b.iter(|| {
            let in_chunk = x1.iter_lane_chunks::<N>(&shape, ax);
            let in_rem = in_chunk.remainder();
            let out_chunk = x2.iter_lane_chunks_mut::<N>(&shape, ax);
            let out_rem = out_chunk.remainder();

            for (in_chunk, mut out_chunk) in in_chunk.zip(out_chunk) {
                in_chunk.deinterleave_arrays(&mut work_e, &mut work_o);
                out_chunk.stack_arrays(&work_e, &work_o);
            }
            for (in_rem, mut out_rem) in in_rem.zip(out_rem) {
                in_rem.deinterleave(&mut work_e2, &mut work_o2);
                out_rem.stack(&work_e2, &work_o2);
            }
        })
    });

    group.bench_function("recursive", |b| {
        b.iter(|| {
            deinterleave_nd(&x1, &mut x2, &shape);
        })
    });

    group.finish();
}

fn driver_vs_array_db2(c: &mut Criterion) {
    use extras::db2_forward_arr;
    use wavelets::Wavelets;
    use wavelets::boundarys::ZeroBoundary;
    use wavelets::daubechies::Daubechies2;
    use wavelets::iter::LanesIterator;
    use wavelets::lwt::LiftingTransform;
    use wavelets::lwt::driver::WaveletTransform;
    use wavelets::simd::Alignable;

    let n = 1000;
    let wvlt = Wavelets::Daubechies2;
    let bc = ZeroBoundary;

    let mut group = c.benchmark_group("driver_vs_array");

    let shape = [n, n];
    let n_total: usize = shape.iter().product();
    let x1 = (0..n_total).map(|v| v as f64).collect_vec();
    let mut out = vec![0.0; n_total];

    let trans = WaveletTransform::new(wvlt, bc);

    group.bench_function("driver/along", |b| {
        let ax = 1;
        b.iter(|| {
            trans.forward_nd(&x1, &mut out, &shape, &[ax]);
        });
    });

    group.bench_function("driver/across", |b| {
        let ax = 1;
        b.iter(|| {
            trans.forward_nd(&x1, &mut out, &shape, &[ax]);
        });
    });

    macro_rules! impl_arm {
        ($N:tt, $ax:ident, $ns:ident, $nd:ident) => {
            const N: usize = $N;
            let in_chunks = x1.iter_lane_chunks::<N>(&shape, $ax);
            let in_rem = in_chunks.remainder();
            let out_chunks = out.iter_lane_chunks_mut::<N>(&shape, $ax);
            let out_rem = out_chunks.remainder();


            let mut s = avec![[0.0;N]; $ns];
            let mut d = avec![[0.0;N]; $nd];

            for (inc, mut outc) in in_chunks.zip(out_chunks){
                inc.deinterleave_arrays(&mut s, &mut d);
                db2_forward_arr(&mut s, &mut d, &bc);
                outc.stack_arrays(&s, &d);
            };

            let mut s = avec![0.0; $ns];
            let mut d = avec![0.0; $nd];
            in_rem.zip(out_rem).for_each(|(ins, mut outs)| {
                ins.deinterleave(&mut s, &mut d);
                Daubechies2::forward(&mut s, &mut d, &bc);
                outs.stack(&s, &d);
            });
        };
    }

    group.bench_function("arrays/along", |b| {
        b.iter(|| {
            let ax = 1;
            let lanes = f64::lanes();
            let n_ax = shape[ax];
            let ns = n_ax.div_ceil(2);
            let nd = n_ax / 2;
            match lanes {
                2 => {
                    impl_arm! {2, ax, ns, nd};
                }
                4 => {
                    impl_arm! {4, ax, ns, nd};
                }
                8 => {
                    impl_arm! {8, ax, ns, nd}
                }
                16 => {
                    impl_arm! {16, ax, ns, nd}
                }
                _ => {
                    unimplemented!()
                }
            }
        });
    });

    group.bench_function("arrays/across", |b| {
        b.iter(|| {
            let ax = 0;
            let lanes = f64::lanes();
            let n_ax = shape[ax];
            let ns = n_ax.div_ceil(2);
            let nd = n_ax / 2;
            match lanes {
                2 => {
                    impl_arm! {2, ax, ns, nd};
                }
                4 => {
                    impl_arm! {4, ax, ns, nd};
                }
                8 => {
                    impl_arm! {8, ax, ns, nd}
                }
                16 => {
                    impl_arm! {16, ax, ns, nd}
                }
                _ => {
                    unimplemented!()
                }
            }
        });
    });
}

fn broadcasted_vs_strided_db2(c: &mut Criterion) {
    use wavelets::Wavelets;
    use wavelets::boundarys::ZeroBoundary;
    use wavelets::lwt::LiftingTransform;
    use wavelets::lwt::driver::WaveletTransform;
    use wavelets::utils::deinterleave_nd;

    let n = 500;

    let shape = [n, n];
    let n_total = shape.iter().product();
    let x = (0..n_total).map(|i| i as f64).collect_vec();

    let wvlt = Wavelets::Daubechies2;
    let trans = WaveletTransform::new(wvlt, ZeroBoundary {});

    let mut sd = vec![0.0; n_total];
    let axes = [1, 0];

    let mut group = c.benchmark_group("broadcasted_vs_lanes/db2");

    group.bench_function("using driver", |b| {
        b.iter(|| {
            trans.par_forward_nd(&x, &mut sd, &shape, &axes);
        })
    });

    let ne = (shape[0] + 1) / 2;
    let mut x2 = vec![0.0; n_total];

    group.bench_function("using recursive", |b| {
        b.iter(|| {
            let bc = ZeroBoundary;
            deinterleave_nd(&x, &mut x2, &shape);
            let (s, d) = x2.split_at_mut(ne * shape[1]);
            wavelets::lwt::daubechies::Daubechies2::forward_chunk(s, d, shape[1], &bc);
            x2.chunks_exact_mut(shape[1]).for_each(|slc| {
                let (s, d) = slc.split_at_mut(ne);
                wavelets::lwt::daubechies::Daubechies2::forward(s, d, &bc);
            });
        })
    });

    group.finish();
}

fn broadcasted_vs_strided_db4(c: &mut Criterion) {
    use wavelets::Wavelets;
    use wavelets::boundarys::ZeroBoundary;
    use wavelets::lwt::LiftingTransform;
    use wavelets::lwt::driver::WaveletTransform;
    use wavelets::utils::deinterleave_nd;

    let n = 500;

    let shape = [n, n];
    let n_total = shape.iter().product();
    let x = (0..n_total).map(|i| i as f64).collect_vec();

    let wvlt = Wavelets::Daubechies4;
    let trans = WaveletTransform::new(wvlt, ZeroBoundary {});

    let mut sd = vec![0.0; n_total];
    let axes = [1, 0];

    let mut group = c.benchmark_group("broadcasted_vs_lanes/db4");

    group.bench_function("using driver", |b| {
        b.iter(|| {
            trans.par_forward_nd(&x, &mut sd, &shape, &axes);
        })
    });

    let ne = (shape[0] + 1) / 2;
    let mut x2 = vec![0.0; n_total];

    group.bench_function("using recursive", |b| {
        b.iter(|| {
            let bc = ZeroBoundary;
            deinterleave_nd(&x, &mut x2, &shape);
            let (s, d) = x2.split_at_mut(ne * shape[1]);
            wavelets::lwt::daubechies::Daubechies4::forward_chunk(s, d, shape[1], &bc);

            x2.chunks_exact_mut(shape[1]).for_each(|slc| {
                let (s, d) = slc.split_at_mut(ne);
                wavelets::lwt::daubechies::Daubechies4::forward(s, d, &bc);
            });
        })
    });

    group.finish();
}

fn broadcasted_vs_strided_db6(c: &mut Criterion) {
    use wavelets::Wavelets;
    use wavelets::boundarys::ZeroBoundary;
    use wavelets::lwt::LiftingTransform;
    use wavelets::lwt::driver::WaveletTransform;
    use wavelets::utils::deinterleave_nd;

    let n = 500;

    let shape = [n, n];
    let n_total = shape.iter().product();
    let x = (0..n_total).map(|i| i as f64).collect_vec();

    let wvlt = Wavelets::Daubechies6;
    let trans = WaveletTransform::new(wvlt, ZeroBoundary {});

    let mut sd = vec![0.0; n_total];
    let axes = [1, 0];

    let mut group = c.benchmark_group("broadcasted_vs_lanes/db6");

    group.bench_function("using driver", |b| {
        b.iter(|| {
            trans.par_forward_nd(&x, &mut sd, &shape, &axes);
        })
    });

    let ne = (shape[0] + 1) / 2;
    let mut x2 = vec![0.0; n_total];

    group.bench_function("using recursive", |b| {
        b.iter(|| {
            let bc = ZeroBoundary;
            deinterleave_nd(&x, &mut x2, &shape);
            let (s, d) = x2.split_at_mut(ne * shape[1]);
            wavelets::lwt::daubechies::Daubechies6::forward_chunk(s, d, shape[1], &bc);

            x2.chunks_exact_mut(shape[1]).for_each(|slc| {
                let (s, d) = slc.split_at_mut(ne);
                wavelets::lwt::daubechies::Daubechies6::forward(s, d, &bc);
            });
        })
    });

    group.finish();
}

criterion_group!(
    interleave_deinterleave,
    interleave_slice_benchmark,
    interleave_strided_benchmark,
    deinterleave_benchmark,
);
criterion_group!(lifted_vs_filtered, db2_benchmark, db6_benchmark);
criterion_group!(
    broadcasted_vs_lanes,
    broadcasted_vs_strided_db2,
    broadcasted_vs_strided_db4,
    broadcasted_vs_strided_db6,
    driver_vs_array_db2,
);
criterion_main!(
    interleave_deinterleave,
    lifted_vs_filtered,
    broadcasted_vs_lanes
);
