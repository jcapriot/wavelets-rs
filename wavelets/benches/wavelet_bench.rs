use criterion::{Criterion, criterion_group, criterion_main};
use itertools::Itertools;
use wavelets::utils::{
    interleave_strided, interleave_strided_chunk, interleave_strided_simd, simd::Simd,
    split_strided, split_strided_chunk, split_strided_simd, stack_to_strided,
};

use num_traits::Zero;

fn db2_benchmark(c: &mut Criterion) {
    use wavelets::boundarys::BoundaryCondition;
    use wavelets::daubechies;
    type WVLT = daubechies::Daubechies2;
    let n = 1000;
    let x = (0..n).map(|i| i as f64).collect_vec();
    //let (mut s, mut d) = x.split_at_mut((n + 1) / 2);
    let mut s = vec![0.0; (n + 1) / 2];
    let mut d = vec![0.0; n / 2];

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
    let nsd = wavelets::dwt::get_outlen::<{ WVLT::WIDTH }>(n);
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

fn db6_benchmark(c: &mut Criterion) {
    use wavelets::boundarys::BoundaryCondition;
    use wavelets::daubechies;
    type WVLT = daubechies::Daubechies6;
    let n = 1000;
    let x = (0..n).map(|i| i as f64).collect_vec();
    //let (mut s, mut d) = x.split_at_mut((n + 1) / 2);
    let mut s = vec![0.0; (n + 1) / 2];
    let mut d = vec![0.0; n / 2];

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
    let nsd = wavelets::dwt::get_outlen::<{ WVLT::WIDTH }>(n);
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
    use wavelets::iter::slice::LanesIterator;
    use wavelets::utils::interleave;

    let n = 100;

    let mut group = c.benchmark_group("interleave_strided");

    let shape = [n, n];
    let n_total = shape.iter().product();
    let x1 = (0..n_total).collect_vec();
    let mut x2 = (0..n_total).collect_vec();

    let mut work1 = vec![0; n];
    let mut work2 = vec![0; n];

    let nf = (n + 1) / 2;
    let ns = n / 2;

    group.bench_function("lanes - out of place", |b| {
        b.iter(|| {
            for (lane_in, lane_out) in x1.iter_lanes(&shape, 0).zip(x2.iter_lanes_mut(&shape, 0)) {
                lane_in
                    .iter()
                    .cloned()
                    .zip(work1.iter_mut())
                    .for_each(|(a, b)| *b = a);
                let (first, second) = work1.split_at((n + 1) / 2);
                interleave(&first, &second, &mut work2);

                lane_out
                    .iter_mut()
                    .zip(work2.iter().cloned())
                    .for_each(|(a, b)| *a = b);
            }
        })
    });
    const N: usize = 8;

    let mut work_f = vec![0; N * nf];
    let mut work_s = vec![0; N * ns];

    let mut work_f2 = vec![0; nf];
    let mut work_s2 = vec![0; ns];

    group.bench_function("lane chunks - out of place", |b| {
        b.iter(|| {
            let (in_chunk, in_rem) = x1.iter_lane_chunks::<N>(&shape, 0);
            let (out_chunk, out_rem) = x2.iter_lane_chunks_mut::<N>(&shape, 0);

            for (in_chunk, out_chunk) in in_chunk.zip(out_chunk) {
                split_strided_chunk(in_chunk, &mut work_f, &mut work_s);
                interleave_strided_chunk(&work_f, &work_s, out_chunk);
            }
            for (in_rem, out_rem) in in_rem.zip(out_rem) {
                split_strided(in_rem, &mut work_f2, &mut work_s2);
                interleave_strided(&work_f2, &work_s2, out_rem);
            }
        })
    });

    group.bench_function("chunked - out of place", |b| {
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
    let mut work_f = vec![Simd::<usize, N>::zero(); nf];
    let mut work_s = vec![Simd::<usize, N>::zero(); ns];

    let mut work_f2 = vec![0; nf];
    let mut work_s2 = vec![0; ns];
    group.bench_function("simd(chunk) - out of place", |b| {
        b.iter(|| {
            let (in_chunk, in_rem) = x1.iter_lane_chunks::<N>(&shape, 0);
            let (out_chunk, out_rem) = x2.iter_lane_chunks_mut::<N>(&shape, 0);

            for (in_chunk, out_chunk) in in_chunk.zip(out_chunk) {
                split_strided_simd(in_chunk, &mut work_f, &mut work_s);
                interleave_strided_simd(&work_f, &work_s, out_chunk);
            }
            for (in_rem, out_rem) in in_rem.zip(out_rem) {
                split_strided(in_rem, &mut work_f2, &mut work_s2);
                interleave_strided(&work_f2, &work_s2, out_rem);
            }
        })
    });

    group.finish();
}

fn deinterleave_benchmark(c: &mut Criterion) {
    use wavelets::iter::slice::LanesIterator;
    use wavelets::utils::{deinterleave_nd, deinterleave_strided};

    const D: usize = 2;
    let n = 100;

    let mut group = c.benchmark_group("deinterleave-strided");

    let shape = vec![n; 2];
    let n_total = shape.iter().product();
    let x1 = (0..n_total).collect_vec();
    let mut x2 = (0..n_total).collect_vec();

    group.bench_function("using lanes", |b| {
        b.iter(|| {
            for ax in 0..D {
                let n = shape[ax];
                let n_e = (n + 1) / 2;
                let n_o = n / 2;
                let mut work_e = vec![0; n_e];
                let mut work_o = vec![0; n_o];
                for (lane_in, lane_out) in
                    x1.iter_lanes(&shape, ax).zip(x2.iter_lanes_mut(&shape, ax))
                {
                    deinterleave_strided(lane_in, &mut work_e, &mut work_o);
                    stack_to_strided(&work_e, &work_o, lane_out);
                }
            }
        })
    });

    group.bench_function("using recursive", |b| {
        b.iter(|| {
            deinterleave_nd(&x1, &mut x2, &shape);
        })
    });

    group.finish();
}

fn broadcasted_vs_strided_db2(c: &mut Criterion) {
    use wavelets::Wavelets;
    use wavelets::boundarys::ZeroBoundary;
    use wavelets::driver::Wavelet;
    use wavelets::lwt::LiftingTransform;
    use wavelets::utils::deinterleave_nd;

    let n = 500;

    let shape = [n, n];
    let n_total = shape.iter().product();
    let x = (0..n_total).map(|i| i as f64).collect_vec();

    let wvlt = Wavelets::Daubechies2;
    let trans = Wavelet::new(wvlt, ZeroBoundary {});

    let mut sd = vec![0.0; n_total];
    let axes = [1, 0];

    let mut group = c.benchmark_group("broadcasted_vs_lanes/db2");

    group.bench_function("using driver", |b| {
        b.iter(|| {
            trans.forward_nd(&x, &mut sd, &shape, &axes);
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
            for slc in x2.chunks_exact_mut(shape[1]) {
                let (s, d) = slc.split_at_mut(ne);
                wavelets::lwt::daubechies::Daubechies2::forward(s, d, &bc);
            }
        })
    });

    group.finish();
}

fn broadcasted_vs_strided_db4(c: &mut Criterion) {
    use wavelets::Wavelets;
    use wavelets::boundarys::ZeroBoundary;
    use wavelets::driver::Wavelet;
    use wavelets::lwt::LiftingTransform;
    use wavelets::utils::deinterleave_nd;

    let n = 500;

    let shape = [n, n];
    let n_total = shape.iter().product();
    let x = (0..n_total).map(|i| i as f64).collect_vec();

    let wvlt = Wavelets::Daubechies4;
    let trans = Wavelet::new(wvlt, ZeroBoundary {});

    let mut sd = vec![0.0; n_total];
    let axes = [1, 0];

    let mut group = c.benchmark_group("broadcasted_vs_lanes/db4");

    group.bench_function("using driver", |b| {
        b.iter(|| {
            trans.forward_nd(&x, &mut sd, &shape, &axes);
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

            for slc in x2.chunks_exact_mut(shape[1]) {
                let (s, d) = slc.split_at_mut(ne);
                wavelets::lwt::daubechies::Daubechies4::forward(s, d, &bc);
            }
        })
    });

    group.finish();
}

fn broadcasted_vs_strided_db6(c: &mut Criterion) {
    use wavelets::Wavelets;
    use wavelets::boundarys::ZeroBoundary;
    use wavelets::driver::Wavelet;
    use wavelets::lwt::LiftingTransform;
    use wavelets::utils::deinterleave_nd;

    let n = 500;

    let shape = [n, n];
    let n_total = shape.iter().product();
    let x = (0..n_total).map(|i| i as f64).collect_vec();

    let wvlt = Wavelets::Daubechies6;
    let trans = Wavelet::new(wvlt, ZeroBoundary {});

    let mut sd = vec![0.0; n_total];
    let axes = [1, 0];

    let mut group = c.benchmark_group("broadcasted_vs_lanes/db6");

    group.bench_function("using driver", |b| {
        b.iter(|| {
            trans.forward_nd(&x, &mut sd, &shape, &axes);
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

            for slc in x2.chunks_exact_mut(shape[1]) {
                let (s, d) = slc.split_at_mut(ne);
                wavelets::lwt::daubechies::Daubechies6::forward(s, d, &bc);
            }
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
);
criterion_main!(
    interleave_deinterleave,
    lifted_vs_filtered,
    broadcasted_vs_lanes
);
