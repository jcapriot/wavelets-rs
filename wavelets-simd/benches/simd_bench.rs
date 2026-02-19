use criterion::{Criterion, criterion_group, criterion_main};
use pulp::Arch;
use wavelets_simd::{db2_forward, db2_forward_neon, db2_forward_simd};

fn db2_benchmark(c: &mut Criterion) {
    let n = 1000;
    let mut x = (0..n).map(|i| i as f64).collect::<Vec<_>>();
    let (mut s, mut d) = x.split_at_mut((n + 1) / 2);
    // let mut s = vec![0.0; (n + 1) / 2];
    // let mut d = vec![0.0; n / 2];

    let mut group = c.benchmark_group("Daubechies 2");

    group.bench_function("auto-vectorized", |b| {
        b.iter(|| {
            db2_forward(&mut s, &mut d);
        })
    });

    let arch = Arch::new();

    group.bench_function("pulp", |b| {
        b.iter(|| {
            db2_forward_simd(arch, &mut s, &mut d);
        })
    });

    #[cfg(target_arch = "aarch64")]
    {
        group.bench_function("hand-vectorized", |b| {
            b.iter(|| unsafe {
                db2_forward_neon(&mut s, &mut d);
            })
        });
    }

    group.finish();
}

criterion_group!(db2, db2_benchmark);
criterion_main!(db2,);
