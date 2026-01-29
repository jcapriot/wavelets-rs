use criterion::{Criterion, criterion_group, criterion_main};
use itertools::Itertools;
use wavelets::utils::stack_to_strided;

fn db2_benchmark(c: &mut Criterion) {
    use wavelets::boundarys::BoundaryCondition;
    use wavelets::wavelets::daubechies;
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

fn interleave_benchmark(c: &mut Criterion) {
    use wavelets::utils::{interleave, interleave_inplace};

    let n = 1042;
    let evens = (0..n).step_by(2).collect_vec();
    let odds = (1..n).step_by(2).collect_vec();
    let mut x1 = vec![0; n];
    let mut x2 = evens.iter().chain(odds.iter()).collect_vec();

    let mut group = c.benchmark_group("interleave");

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
    use wavelets::utils::{interleave, interleave_inplace};

    let n = 100;

    let mut group = c.benchmark_group("interleave_strided");

    let shape = [n, n];
    let n_total = shape.iter().product();
    let x1 = (0..n_total).collect_vec();
    let mut x2 = (0..n_total).collect_vec();

    let mut work1 = vec![0; n];
    let mut work2 = vec![0; n];

    group.bench_function("strided - out of place", |b| {
        b.iter(|| {
            for (lane_in, mut lane_out) in
                x1.iter_lanes(&shape, 0).zip(x2.iter_lanes_mut(&shape, 0))
            {
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

    group.bench_function("out of place", |b| {
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
    use wavelets::iter::slice::LanesIterator;
    use wavelets::utils::{deinterleave_nd, deinterleave_strided, deinterleave_strided_chunk};

    const D: usize = 2;
    let n = 100;

    let mut group = c.benchmark_group("interleave_strided");

    let shape = vec![n; 2];
    let n_total = shape.iter().product();
    let x1 = (0..n_total).collect_vec();
    let mut x2 = (0..n_total).collect_vec();

    let mut work_e = vec![0; n];
    let mut work_o = vec![0; n];

    group.bench_function("using lanes", |b| {
        b.iter(|| {
            for ax in 0..D {
                let n = shape[ax];
                let n_e = (n + 1) / 2;
                let n_o = n / 2;
                let mut work_e = vec![0; n_e];
                let mut work_o = vec![0; n_o];
                for (lane_in, mut lane_out) in
                    x1.iter_lanes(&shape, ax).zip(x2.iter_lanes_mut(&shape, ax))
                {
                    deinterleave_strided(&lane_in, &mut work_e, &mut work_o);
                    stack_to_strided(&work_e, &work_o, &mut lane_out);
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

    const D: usize = 2;
    let n = 500;

    let shape = [n, n];
    let n_total = shape.iter().product();
    let x = (0..n_total).map(|i| i as f64).collect_vec();

    let wvlt = Wavelets::Daubechies2;
    let trans = Wavelet::new(wvlt, ZeroBoundary {});

    let mut sd = vec![0.0; n_total];
    let axes = [1, 0];

    let mut group = c.benchmark_group("db2_driver_vs_broadcasted");

    group.bench_function("using driver", |b| {
        b.iter(|| {
            trans.forward_nd(&x, &mut sd, &shape, &axes);
        })
    });

    let ne = (shape[0] + 1) / 2;
    let mut x2 = vec![0.0; n_total];

    group.bench_function("using recursive", |b| {
        b.iter(|| {
            deinterleave_nd(&x, &mut x2, &shape);
            let (s, d) = x2.split_at_mut(ne * shape[1]);
            wavelets::lwt::broadcasted_db2(s, d, shape[1]);

            let bc = ZeroBoundary;
            for slc in x2.chunks_exact_mut(shape[1]) {
                let (s, d) = slc.split_at_mut(ne);
                wavelets::lwt::daubechies::Daubechies2::forward(s, d, &bc);
            }
        })
    });

    group.finish();
}
criterion_group!(
    benches,
    db2_benchmark,
    interleave_benchmark,
    interleave_strided_benchmark,
    deinterleave_benchmark,
    broadcasted_vs_strided_db2
);
criterion_main!(benches);
