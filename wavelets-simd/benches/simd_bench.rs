use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use wavelets_simd::{
    db2_forward, db2_forward_from_steps, db2_forward_pulp, db2_forward_pulp_from_steps,
};

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
use wavelets_simd::db2_forward_avx_fma;
#[cfg(target_arch = "aarch64")]
use wavelets_simd::db2_forward_neon;

fn db2_benchmark(c: &mut Criterion) {
    let n = 1000;
    let mut x = (0..n).map(|i| i as f64).collect::<Vec<_>>();
    let (mut s, mut d) = x.split_at_mut((n + 1) / 2);

    let mut group = c.benchmark_group("Daubechies 2");

    group.bench_function("auto-vectorized", |b| {
        b.iter(|| {
            db2_forward(black_box(&mut s), black_box(&mut d));
        })
    });

    group.bench_function("auto-vectorized-from-steps", |b| {
        b.iter(|| {
            db2_forward_from_steps(black_box(&mut s), black_box(&mut d));
        })
    });

    group.bench_function("pulp", |b| {
        b.iter(|| {
            db2_forward_pulp(black_box(&mut s), black_box(&mut d));
        })
    });

    group.bench_function("pulp-from-steps", |b| {
        b.iter(|| {
            db2_forward_pulp_from_steps(black_box(&mut s), black_box(&mut d));
        })
    });

    group.bench_function("pulp-from-macro", |b| {
        b.iter(|| {
            wavelets_simd::check::forward_simd(
                black_box(&mut s),
                black_box(&mut d),
                &wavelets::boundarys::ZeroBoundary {},
            );
        })
    });

    #[cfg(target_arch = "aarch64")]
    {
        group.bench_function("hand-vectorized", |b| {
            b.iter(|| unsafe {
                db2_forward_neon(black_box(&mut s), black_box(&mut d));
            })
        });
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("avx") && is_x86_feature_detected!("fma") {
            group.bench_function("hand-vectorized", |b| {
                b.iter(|| unsafe {
                    db2_forward_avx_fma(black_box(&mut s), black_box(&mut d));
                })
            });
        }
    }

    group.finish();
}

criterion_group!(db2, db2_benchmark);
criterion_main!(db2,);
