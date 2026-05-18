use aligned_vec::{AVec, avec};
use criterion::{Criterion, criterion_group, criterion_main};
use itertools::Itertools;
use wavelets::Alignable;
use wavelets::boundarys::ZeroBoundary;
use wavelets::lwt::LiftingTransform;

use wavelets::daubechies::Daubechies2;

pub mod simd {
    use pulp::{Simd, WithSimd};
    use wavelets::SimdTransformable;
    use wavelets::boundarys::BoundaryExtension;

    pub fn db2_simd_along<T: SimdTransformable, BC: BoundaryExtension>(
        s: &mut [T],
        d: &mut [T],
        bc: &BC,
    ) {
        struct Impl<'a, 'b, 'c, T, BC>(&'a mut [T], &'b mut [T], &'c BC);

        impl<'a, 'b, 'c, T: SimdTransformable, BC: BoundaryExtension> WithSimd for Impl<'a, 'b, 'c, T, BC> {
            type Output = ();

            #[inline(always)]
            fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
                let s = self.0;
                let d = self.1;
                let bc = self.2;

                let n_lanes = T::simd_lanes(simd);
                let ns = s.len() / n_lanes;
                let nd = d.len() / n_lanes;

                assert!(ns == nd || ns == nd + 1);

                let c = T::simd_splat(
                    simd,
                    T::scalar_type_from_f64(
                        -1.73205080756887729352744634150587236694280525381038062805581,
                    ),
                );

                let (s_v, _) = T::as_mut_simd(simd, s);
                let (d_v, _) = T::as_mut_simd(simd, d);

                d_v.iter_mut().zip(s_v.iter()).for_each(|(u, v)| {
                    *u = T::simd_mul_add(simd, *v, c, *u);
                });

                if let Some((s_vf, s_vb)) = s_v.split_at_mut_checked(nd - 1) {
                    let (c0, c1) = (
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
                    );
                    s_vf.iter_mut()
                        .zip(d_v.array_windows::<2>())
                        .for_each(|(u, [v0, v1])| {
                            *u = T::simd_mul_add(simd, *v0, c0, *u);
                            *u = T::simd_mul_add(simd, *v1, c1, *u);
                        });
                    if nd > 0 {
                        (nd - 1..).zip(s_vb).for_each(|(i, u)| {
                            let io = i as isize - 1;
                            let ps = bc.get_parts::<T>(nd, io);
                            for (b, j) in ps {
                                let v = if let Some(b) = b {
                                    let b = T::simd_splat(simd, b);
                                    T::simd_mul(simd, d_v[j], b)
                                } else {
                                    d_v[j]
                                };
                                *u = T::simd_mul_add(simd, v, c0, *u);
                            }
                            let ps = bc.get_parts::<T>(nd, io + 1);
                            for (b, j) in ps {
                                let v = if let Some(b) = b {
                                    let b = T::simd_splat(simd, b);
                                    T::simd_mul(simd, d_v[j], b)
                                } else {
                                    d_v[j]
                                };
                                *u = T::simd_mul_add(simd, v, c1, *u);
                            }
                        });
                    }
                }

                if let Some((d_vf, d_vb)) = d_v.split_at_mut_checked(1) {
                    d_vf.iter_mut().enumerate().for_each(|(i, u)| {
                        let io = i as isize - 1;
                        let ps = bc.get_parts::<T>(ns, io);
                        for (b, j) in ps {
                            let v = if let Some(b) = b {
                                let b = T::simd_splat(simd, b);
                                T::simd_mul(simd, s_v[j], b)
                            } else {
                                s_v[j]
                            };
                            *u = T::simd_add(simd, v, *u);
                        }
                    });
                    d_vb.iter_mut().zip(s_v.iter()).for_each(|(u, v)| {
                        *u = T::simd_add(simd, *v, *u);
                    });
                }

                let c = T::simd_splat(
                    simd,
                    T::scalar_type_from_f64(
                        1.93185165257813657349948639945779473526780967801680910080469,
                    ),
                );
                let c_inv = T::simd_splat(
                    simd,
                    T::scalar_type_from_f64(
                        1.0 / 1.93185165257813657349948639945779473526780967801680910080469,
                    ),
                );
                s_v.iter_mut().for_each(|u| {
                    *u = T::simd_mul(simd, *u, c);
                });
                d_v.iter_mut().for_each(|u| {
                    *u = T::simd_mul(simd, *u, c_inv);
                });
            }
        }

        wavelets::ARCH.dispatch(Impl(s, d, bc));
    }
}

fn db2_benchmark(c: &mut Criterion) {
    let n_lanes = f64::lanes();
    let nv = 1012;
    let ns = (nv + 1) / 2;
    let nd = nv / 2;

    let bc = ZeroBoundary;

    let mut s_chunks: AVec<_, aligned_vec::ConstAlign<256>> =
        AVec::from_iter(n_lanes * 64, (0..ns * n_lanes).map(|i| i as f64));
    let mut d_chunks: AVec<_, aligned_vec::RuntimeAlign> =
        AVec::from_iter(n_lanes * 64, (0..nd * n_lanes).map(|i| i as f64));

    let mut s_lanes = (0..n_lanes)
        .map(|i| {
            AVec::<_, aligned_vec::RuntimeAlign>::from_iter(
                n_lanes * 64,
                (i..ns * n_lanes).step_by(n_lanes).map(|i| i as f64),
            )
        })
        .collect::<Vec<_>>();
    let mut d_lanes = (0..n_lanes)
        .map(|i| {
            AVec::<_, aligned_vec::RuntimeAlign>::from_iter(
                n_lanes * 64,
                (i..ns * n_lanes).step_by(n_lanes).map(|i| i as f64),
            )
        })
        .collect::<Vec<_>>();

    assert_eq!(s_chunks.len(), s_lanes[0].len() * n_lanes);
    assert_eq!(d_chunks.len(), d_lanes[0].len() * n_lanes);

    let mut group = c.benchmark_group("simd_direction");

    group.bench_function("along_ax", |b| {
        b.iter(|| {
            s_lanes
                .iter_mut()
                .zip(d_lanes.iter_mut())
                .for_each(|(s, d)| {
                    Daubechies2::forward(s, d, &bc);
                });
        })
    });

    group.bench_function("across_ax", |b| {
        b.iter(|| {
            simd::db2_simd_along(&mut s_chunks, &mut d_chunks, &bc);
        })
    });
}

fn driver_across(c: &mut Criterion) {
    let n_lanes = f64::lanes();
    println!("Number of f64 lanes: {n_lanes}");

    let shape = [1024, 1024];
    let n_total = shape.iter().product();
    let axes = [0];

    let ns = (shape[1] + 1) / 2;
    let nd = shape[1] / 2;

    let wvlt = wavelets::Wavelets::Daubechies2;
    let bc = ZeroBoundary;

    let trans = wavelets::lwt::driver::WaveletTransform::new(wvlt, bc);

    let x = (0..n_total).map(|i| i as f64).collect::<Vec<_>>();
    let mut xw1 = vec![0.0; n_total];

    let mut group = c.benchmark_group("driver_direction");

    group.bench_function("across_driver", |b| {
        b.iter(|| {
            trans.forward_nd(&x, &mut xw1, &shape, &axes);
        })
    });

    group.bench_function("across_simd", |b| {
        b.iter(|| {
            use wavelets::iter::LanesIterator;
            let mut swork = avec![0.0; ns * n_lanes];
            let mut dwork = avec![0.0; nd * n_lanes];
            match n_lanes {
                2 => {
                    const N: usize = 2;
                    let in_chunks = x.iter_lane_chunks::<N>(&shape, 0);
                    let out_chunks = xw1.iter_lane_chunks_mut::<N>(&shape, 0);

                    in_chunks.zip(out_chunks).for_each(|(ic, mut oc)| {
                        // copy and deinterleave chunk to work arrays.
                        if let Some(ic) = ic.as_chunks() {
                            ic.iter()
                                .zip(
                                    swork
                                        .chunks_exact_mut(N)
                                        .interleave(dwork.chunks_exact_mut(N)),
                                )
                                .for_each(|(a, b)| {
                                    a.into_iter().zip(b).for_each(|(a, b)| *b = *a);
                                });
                        } else {
                            ic.iter()
                                .zip(
                                    swork
                                        .chunks_exact_mut(N)
                                        .interleave(dwork.chunks_exact_mut(N)),
                                )
                                .for_each(|(a, b)| {
                                    a.into_iter().zip(b).for_each(|(a, b)| *b = *a);
                                });
                        }

                        // transform
                        simd::db2_simd_along(&mut swork, &mut dwork, &bc);

                        // copy to output.
                        if let Some(oc) = oc.as_chunks_mut() {
                            oc.iter_mut()
                                .zip(swork.chunks_exact(N).chain(dwork.chunks_exact(N)))
                                .for_each(|(a, b)| {
                                    a.into_iter().zip(b).for_each(|(a, b)| *a = *b);
                                });
                        } else {
                            oc.iter_mut()
                                .zip(swork.chunks_exact(N).chain(dwork.chunks_exact(N)))
                                .for_each(|(a, b)| {
                                    a.into_iter().zip(b).for_each(|(a, b)| *a = *b);
                                });
                        }
                    });
                }
                4 => {
                    const N: usize = 4;
                    let in_chunks = x.iter_lane_chunks::<N>(&shape, 0);
                    let out_chunks = xw1.iter_lane_chunks_mut::<N>(&shape, 0);

                    in_chunks.zip(out_chunks).for_each(|(ic, mut oc)| {
                        // copy and deinterleave chunk to work arrays.
                        if let Some(ic) = ic.as_chunks() {
                            ic.iter()
                                .zip(
                                    swork
                                        .chunks_exact_mut(N)
                                        .interleave(dwork.chunks_exact_mut(N)),
                                )
                                .for_each(|(a, b)| {
                                    a.into_iter().zip(b).for_each(|(a, b)| *b = *a);
                                });
                        } else {
                            ic.iter()
                                .zip(
                                    swork
                                        .chunks_exact_mut(N)
                                        .interleave(dwork.chunks_exact_mut(N)),
                                )
                                .for_each(|(a, b)| {
                                    a.into_iter().zip(b).for_each(|(a, b)| *b = *a);
                                });
                        }

                        // transform
                        simd::db2_simd_along(&mut swork, &mut dwork, &bc);

                        // copy to output.
                        if let Some(oc) = oc.as_chunks_mut() {
                            oc.iter_mut()
                                .zip(swork.chunks_exact(N).chain(dwork.chunks_exact(N)))
                                .for_each(|(a, b)| {
                                    a.into_iter().zip(b).for_each(|(a, b)| *a = *b);
                                });
                        } else {
                            oc.iter_mut()
                                .zip(swork.chunks_exact(N).chain(dwork.chunks_exact(N)))
                                .for_each(|(a, b)| {
                                    a.into_iter().zip(b).for_each(|(a, b)| *a = *b);
                                });
                        }
                    });
                }
                8 => {
                    const N: usize = 8;
                    let in_chunks = x.iter_lane_chunks::<N>(&shape, 0);
                    let out_chunks = xw1.iter_lane_chunks_mut::<N>(&shape, 0);

                    in_chunks.zip(out_chunks).for_each(|(ic, mut oc)| {
                        // copy and deinterleave chunk to work arrays.
                        if let Some(ic) = ic.as_chunks() {
                            ic.iter()
                                .zip(
                                    swork
                                        .chunks_exact_mut(N)
                                        .interleave(dwork.chunks_exact_mut(N)),
                                )
                                .for_each(|(a, b)| {
                                    a.into_iter().zip(b).for_each(|(a, b)| *b = *a);
                                });
                        } else {
                            ic.iter()
                                .zip(
                                    swork
                                        .chunks_exact_mut(N)
                                        .interleave(dwork.chunks_exact_mut(N)),
                                )
                                .for_each(|(a, b)| {
                                    a.into_iter().zip(b).for_each(|(a, b)| *b = *a);
                                });
                        }

                        // transform
                        simd::db2_simd_along(&mut swork, &mut dwork, &bc);

                        // copy to output.
                        if let Some(oc) = oc.as_chunks_mut() {
                            oc.iter_mut()
                                .zip(swork.chunks_exact(N).chain(dwork.chunks_exact(N)))
                                .for_each(|(a, b)| {
                                    a.into_iter().zip(b).for_each(|(a, b)| *a = *b);
                                });
                        } else {
                            oc.iter_mut()
                                .zip(swork.chunks_exact(N).chain(dwork.chunks_exact(N)))
                                .for_each(|(a, b)| {
                                    a.into_iter().zip(b).for_each(|(a, b)| *a = *b);
                                });
                        }
                    });
                }
                _ => {
                    panic!("Unmatchable number of lanes {n_lanes}.");
                }
            }
        })
    });
}

criterion_group!(simd_direction, db2_benchmark, driver_across);
criterion_main!(simd_direction);
