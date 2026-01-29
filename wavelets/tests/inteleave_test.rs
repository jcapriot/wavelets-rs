use itertools::Itertools;
use wavelets::iter::slice::LanesIterator;
use wavelets::utils::{deinterleave_nd, deinterleave_strided, stack_to_strided};

fn ref_deinterleave_nd<T: Clone + num_traits::Zero>(x: &[T], shape: &[usize]) -> Vec<T> {
    let mut out = x.to_vec();

    for ax in 0..shape.len() {
        let n = shape[ax];
        let ne = (n + 1) / 2;
        let no = n / 2;

        let mut work_e = vec![T::zero(); ne];
        let mut work_o = vec![T::zero(); no];

        for mut lane in out.iter_lanes_mut(shape, ax) {
            deinterleave_strided(&lane, &mut work_e, &mut work_o);
            stack_to_strided(&work_e, &work_o, &mut lane);
        }
    }
    out
}

#[test]
fn test_deinterleave_2d() {
    let ns = [10, 11];

    for dim in [1, 2, 3, 4, 5] {
        for n in ns {
            let shape = vec![n; dim];
            let n_total = shape.iter().product();
            let x = (0..n_total).collect_vec();

            let mut out = vec![0; n_total];
            deinterleave_nd(&x, &mut out, &shape);

            let desired = ref_deinterleave_nd(&x, &shape);

            assert_eq!(out, desired);
        }
    }
}
