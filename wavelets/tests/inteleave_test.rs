use aligned_vec::{AVec, avec};
use itertools::Itertools;
use wavelets::iter::LanesIterator;
use wavelets::utils::deinterleave_nd;

fn ref_deinterleave_nd<T: Clone + num_traits::Zero>(x: &[T], shape: &[usize]) -> Vec<T> {
    let mut out = x.to_vec();

    for ax in 0..shape.len() {
        let n = shape[ax];
        let ne = (n + 1) / 2;
        let no = n / 2;

        let mut work_e = vec![T::zero(); ne];
        let mut work_o = vec![T::zero(); no];

        for mut lane in out.iter_lanes_mut(shape, ax) {
            lane.deinterleave(&mut work_e, &mut work_o);
            lane.stack(&work_e, &work_o);
        }
    }
    out
}

fn ref_deinterleave_chunk_nd<T: Clone + num_traits::Zero>(x: &[T], shape: &[usize]) -> Vec<T> {
    let mut out = x.to_vec();

    const N: usize = 4;

    for ax in 0..shape.len() {
        let n = shape[ax];
        let ne = (n + 1) / 2;
        let no = n / 2;

        let chunks = out.iter_lane_chunks_mut::<4>(shape, ax);
        let rem = chunks.remainder();

        let mut work_e: [AVec<T>; N] = core::array::from_fn(|_| avec![T::zero(); ne]);
        let mut work_o: [AVec<T>; N] = core::array::from_fn(|_| avec![T::zero(); no]);

        for mut chunk in chunks {
            chunk.deinterleave(&mut work_e, &mut work_o);
            chunk.stack(&work_e, &work_o);
        }

        let mut work_e = vec![T::zero(); ne];
        let mut work_o = vec![T::zero(); no];

        for mut lane in rem {
            lane.deinterleave(&mut work_e, &mut work_o);
            lane.stack(&work_e, &work_o);
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

#[test]
fn test_deinterleave_chunked_2d() {
    let ns = [10, 11];

    for dim in [1, 2, 3, 4, 5] {
        for n in ns {
            let shape = vec![n; dim];
            let n_total = shape.iter().product();
            let x = (0..n_total).collect_vec();

            let mut out = vec![0; n_total];
            deinterleave_nd(&x, &mut out, &shape);

            let desired = ref_deinterleave_chunk_nd(&x, &shape);

            assert_eq!(out, desired);
        }
    }
}
