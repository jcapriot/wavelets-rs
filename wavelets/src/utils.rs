use crate::iter::slice::{ChunkStridedSliceRef, StridedSliceRef};
use aligned_vec::AVec;
use itertools::{Itertools, izip};

#[inline]
pub fn stride_from_shape(shape: &[usize]) -> Vec<usize> {
    let mut stride = vec![1; shape.len()];
    for i in (1..shape.len()).rev() {
        stride[i - 1] = stride[i] * shape[i];
    }
    stride
}

#[inline]
pub fn deinterleave<T: Clone>(x: &[T], evens: &mut [T], odds: &mut [T]) {
    let nx = x.len();
    let n_e = evens.len();
    let n_o = odds.len();

    assert_eq!(
        nx / 2,
        n_o,
        "incorrect odd length, {n_o}, for slice deinterleave"
    );
    assert_eq!(
        (nx + 1) / 2,
        n_e,
        "incorrect even length, {n_e}, for slice deinterleave"
    );

    let (chunks, rem) = x.as_chunks::<2>();
    chunks
        .iter()
        .zip(evens.iter_mut().zip(odds.iter_mut()))
        .for_each(|(x, (e, o))| {
            *e = x[0].clone();
            *o = x[1].clone();
        });
    if let Some(x) = rem.last()
        && let Some(e) = evens.last_mut()
    {
        *e = x.clone();
    }
}

#[inline]
pub fn deinterleave_2d<T: Clone>(input: &[T], output: &mut [T], shape: &[usize; 2]) {
    let n_total: usize = shape.iter().product();
    assert_eq!(input.len(), n_total);
    assert_eq!(output.len(), n_total);

    let n_first = (shape[0] + 1) / 2;
    let n_sub: usize = shape[1..].iter().product();

    let (first, second) = output.split_at_mut(n_first * n_sub);

    let mut in_chunks = input.chunks_exact(2 * n_sub);

    let mut first_chunks = first.chunks_exact_mut(n_sub);

    let n_first = (shape[1] + 1) / 2;
    first_chunks
        .by_ref()
        .zip(second.chunks_exact_mut(n_sub))
        .zip(in_chunks.by_ref())
        .for_each(|((f, s), inp)| {
            let (f_f, f_s) = f.split_at_mut(n_first);
            deinterleave(&inp[0..n_sub], f_f, f_s);
            let (s_f, s_s) = s.split_at_mut(n_first);
            deinterleave(&inp[n_sub..2 * n_sub], s_f, s_s);
        });
    first_chunks.for_each(|f| {
        let (evens, odds) = f.split_at_mut(n_first);
        deinterleave(in_chunks.remainder(), evens, odds);
    });
}

#[inline]
pub fn deinterleave_nd<T: Clone>(input: &[T], output: &mut [T], shape: &[usize]) {
    match shape.len() {
        0 => {}
        1 => {
            let (f, s) = output.split_at_mut((shape[0] + 1) / 2);
            deinterleave(input, f, s);
        }
        2 => deinterleave_2d(
            input,
            output,
            shape
                .try_into()
                .expect("shape length was already checked to be 2"),
        ),
        _ => {
            let n_total: usize = shape.iter().product();
            assert_eq!(input.len(), n_total);
            assert_eq!(input.len(), n_total);

            deinterleave_nd_unchecked(input, output, shape);
        }
    }
}
#[inline]
fn deinterleave_nd_unchecked<T: Clone>(input: &[T], output: &mut [T], shape: &[usize]) {
    match shape.len() {
        0 => {}
        1 => {
            let (f, s) = output.split_at_mut((shape[0] + 1) / 2);
            deinterleave(input, f, s);
        }
        2 => deinterleave_2d(
            input,
            output,
            shape
                .try_into()
                .expect("shape length was already checked to be 2"),
        ),
        _ => {
            let n_first = (shape[0] + 1) / 2;
            let n_sub: usize = shape[1..].iter().product();

            let (first, second) = output.split_at_mut(n_first * n_sub);

            let mut first_chunks = first.chunks_exact_mut(n_sub);
            let mut in_chunks = input.chunks_exact(2 * n_sub);

            first_chunks
                .by_ref()
                .zip(second.chunks_exact_mut(n_sub))
                .zip(in_chunks.by_ref())
                .for_each(|((f, s), inp)| {
                    let (in_even, in_odd) = inp.split_at(n_sub);
                    deinterleave_nd_unchecked(in_even, f, &shape[1..]);
                    deinterleave_nd_unchecked(in_odd, s, &shape[1..]);
                });
            first_chunks.for_each(|f| {
                deinterleave_nd_unchecked(in_chunks.remainder(), f, &shape[1..]);
            });
        }
    }
}

#[inline]
pub fn deinterleave_strided<T: Clone>(x: &StridedSliceRef<T>, evens: &mut [T], odds: &mut [T]) {
    if let Some(x) = x.as_slice() {
        deinterleave(x, evens, odds);
    } else {
        let nx = x.len();
        let n_e = evens.len();
        let n_o = odds.len();

        assert_eq!(
            nx / 2,
            n_o,
            "incorrect odd length, {n_o}, for strided deinterleave"
        );
        assert_eq!(
            (nx + 1) / 2,
            n_e,
            "incorrect even length, {n_e}, for strided deinterleave"
        );
        x.iter()
            .zip(evens.iter_mut().interleave(odds.iter_mut()))
            .for_each(|(v, ou)| {
                *ou = v.clone();
            });
    }
}

#[inline]
pub fn deinterleave_strided_chunk<T: Clone, const N: usize>(
    x: &ChunkStridedSliceRef<T, N>,
    evens: &mut [AVec<T>; N],
    odds: &mut [AVec<T>; N],
) {
    assert_ne!(N, 0);

    let mut e_chunks = evens.each_mut().map(|v| v.as_mut_slice());
    let mut o_chunks = odds.each_mut().map(|v| v.as_mut_slice());

    let mut do_even = true;
    let mut ind_io = 0;

    if let Some(x_iter) = x.chunks() {
        x_iter.for_each(|x| {
            match do_even {
                true => x
                    .iter()
                    .cloned()
                    .zip(e_chunks.iter_mut())
                    .for_each(|(x, v)| {
                        unsafe { *v.get_unchecked_mut(ind_io) = x };
                    }),
                false => {
                    x.iter()
                        .cloned()
                        .zip(o_chunks.iter_mut())
                        .for_each(|(x, v)| {
                            unsafe { *v.get_unchecked_mut(ind_io) = x };
                        });
                    ind_io += 1;
                }
            }
            do_even = !do_even;
        })
    } else {
        x.iter().for_each(|x| {
            match do_even {
                true => x
                    .into_iter()
                    .cloned()
                    .zip(e_chunks.iter_mut())
                    .for_each(|(x, v)| {
                        unsafe { *v.get_unchecked_mut(ind_io) = x };
                    }),
                false => {
                    x.into_iter()
                        .cloned()
                        .zip(o_chunks.iter_mut())
                        .for_each(|(x, v)| {
                            unsafe { *v.get_unchecked_mut(ind_io) = x };
                        });
                    ind_io += 1;
                }
            }
            do_even = !do_even;
        });
    }
}

#[inline]
pub fn stack<T: Clone>(first: &[T], second: &[T], out: &mut [T]) {
    assert_eq!(
        first.len() + second.len(),
        out.len(),
        "invalid lengths for slice stack, first: {}, second: {}, third: {}",
        first.len(),
        second.len(),
        out.len()
    );
    out.iter_mut()
        .zip(first.iter().chain(second).cloned())
        .for_each(|(a, b)| *a = b);
}

#[inline]
pub fn stack_to_strided<T: Clone>(first: &[T], second: &[T], out: &mut StridedSliceRef<T>) {
    if let Some(out) = out.as_slice_mut() {
        stack(first, second, out);
    } else {
        assert_eq!(
            first.len() + second.len(),
            out.len(),
            "invalid lengths for strided stack, first: {}, second: {}, third: {}",
            first.len(),
            second.len(),
            out.len()
        );
        out.iter_mut()
            .zip(first.iter().chain(second).cloned())
            .for_each(|(a, b)| *a = b);
    }
}

#[inline]
pub fn stack_to_strided_chunk<T: Clone, const N: usize>(
    first: &[AVec<T>; N],
    second: &[AVec<T>; N],
    out: &mut ChunkStridedSliceRef<T, N>,
) {
    assert_ne!(N, 0);

    let nx = out.len();

    let n_first = first[0].len();
    let n_second = second[0].len();
    assert_eq!(
        n_first + n_second,
        nx,
        "invalid lengths for strided chunk stack, first: {n_first}, second: {n_second}, third: {nx}"
    );

    let f_chunks = first.each_ref().map(|v| v.as_slice());
    let s_chunks = second.each_ref().map(|v| v.as_slice());

    if let Some(mut out_iter) = out.chunks_mut() {
        out_iter
            .by_ref()
            .take(n_first)
            .enumerate()
            .for_each(|(i, out)| {
                out.iter_mut().zip(f_chunks.iter()).for_each(|(out, v)| {
                    *out = unsafe { v.get_unchecked(i) }.clone();
                })
            });
        out_iter.enumerate().for_each(|(i, out)| {
            out.iter_mut().zip(s_chunks.iter()).for_each(|(out, v)| {
                *out = unsafe { v.get_unchecked(i) }.clone();
            })
        });
    } else {
        let mut out_iter = out.iter_mut();
        out_iter
            .by_ref()
            .take(n_first)
            .enumerate()
            .for_each(|(i, out)| {
                out.into_iter().zip(f_chunks.iter()).for_each(|(out, v)| {
                    *out = unsafe { v.get_unchecked(i) }.clone();
                })
            });
        out_iter.enumerate().for_each(|(i, out)| {
            out.into_iter().zip(s_chunks.iter()).for_each(|(out, v)| {
                *out = unsafe { v.get_unchecked(i) }.clone();
            })
        });
    }
}

// #[inline]
// pub(crate) fn stack_to_strided_aligned_chunk<T: Clone + Alignable, const N: usize>(
//     first: &[T],
//     second: &[T],
//     out: &mut ChunkStridedSliceRef<T, N>,
// ) {
//     assert_ne!(N, 0);

//     let nx = out.len();
//     let n_s = (nx + 1) / 2;
//     let n_d = nx / 2;

//     let f_chunks: [_; N] = first.aligned_chunks_exact(n_s).collect_array().unwrap();
//     let s_chunks: [_; N] = second.aligned_chunks_exact(n_d).collect_array().unwrap();

//     if let Some(mut out_iter) = out.chunks_mut() {
//         out_iter
//             .by_ref()
//             .take(n_s)
//             .enumerate()
//             .for_each(|(i, out)| {
//                 out.iter_mut().zip(f_chunks.iter()).for_each(|(out, v)| {
//                     *out = unsafe { v.get_unchecked(i) }.clone();
//                 })
//             });
//         out_iter.enumerate().for_each(|(i, out)| {
//             out.iter_mut().zip(s_chunks.iter()).for_each(|(out, v)| {
//                 *out = unsafe { v.get_unchecked(i) }.clone();
//             })
//         });
//     } else {
//         let mut out_iter = out.iter_mut();
//         out_iter
//             .by_ref()
//             .take(n_s)
//             .enumerate()
//             .for_each(|(i, out)| {
//                 out.into_iter().zip(f_chunks.iter()).for_each(|(out, v)| {
//                     *out = unsafe { v.get_unchecked(i) }.clone();
//                 })
//             });
//         out_iter.enumerate().for_each(|(i, out)| {
//             out.into_iter().zip(s_chunks.iter()).for_each(|(out, v)| {
//                 *out = unsafe { v.get_unchecked(i) }.clone();
//             })
//         });
//     }
// }

#[inline]
pub fn interleave<T: Clone>(evens: &[T], odds: &[T], x: &mut [T]) {
    let nx = x.len();
    let n_e = evens.len();
    let n_o = odds.len();

    assert_eq!(nx / 2, n_o);
    assert_eq!((nx + 1) / 2, n_e);

    let (chunks, rem) = x.as_chunks_mut::<2>();
    let mut ev_iter = evens.iter();
    izip!(chunks.iter_mut(), ev_iter.by_ref(), odds.iter()).for_each(|(xc, even, odd)| {
        xc[0] = even.clone();
        xc[1] = odd.clone();
    });
    if let Some(x) = rem.last_mut()
        && let Some(e) = evens.last()
    {
        *x = e.clone();
    }
}

#[inline]
pub fn interleave_strided<T: Clone>(evens: &[T], odds: &[T], x: &mut StridedSliceRef<T>) {
    if let Some(x) = x.as_slice_mut() {
        interleave(evens, odds, x);
    } else {
        let nx = x.len();
        let n_e = evens.len();
        let n_o = odds.len();

        assert_eq!(nx / 2, n_o);
        assert_eq!((nx + 1) / 2, n_e);

        x.iter_mut()
            .zip(evens.iter().interleave(odds.iter()).cloned())
            .for_each(|(l, r)| *l = r);
    }
}

#[inline]
pub fn interleave_strided_chunk<T: Clone, const N: usize>(
    evens: &[AVec<T>; N],
    odds: &[AVec<T>; N],
    x: &mut ChunkStridedSliceRef<T, N>,
) {
    assert_ne!(N, 0);

    let e_chunks = evens.each_ref().map(|v| v.as_slice());
    let o_chunks = odds.each_ref().map(|v| v.as_slice());

    let mut do_even = true;
    let mut ind_io = 0;

    if let Some(x_iter) = x.chunks_mut() {
        x_iter.for_each(|x| {
            match do_even {
                true => x.into_iter().zip(e_chunks.iter()).for_each(|(x, v)| {
                    *x = unsafe { v.get_unchecked(ind_io) }.clone();
                }),
                false => {
                    x.into_iter().zip(o_chunks.iter()).for_each(|(x, v)| {
                        *x = unsafe { v.get_unchecked(ind_io) }.clone();
                    });
                    ind_io += 1;
                }
            }
            do_even = !do_even;
        });
    } else {
        x.iter_mut().for_each(|x| {
            match do_even {
                true => x.into_iter().zip(e_chunks.iter()).for_each(|(x, v)| {
                    *x = unsafe { v.get_unchecked(ind_io) }.clone();
                }),
                false => {
                    x.into_iter().zip(o_chunks.iter()).for_each(|(x, v)| {
                        *x = unsafe { v.get_unchecked(ind_io) }.clone();
                    });
                    ind_io += 1;
                }
            }
            do_even = !do_even;
        });
    }
}

#[inline]
pub fn split<T: Clone>(x: &[T], first: &mut [T], second: &mut [T]) {
    assert_eq!(x.len(), first.len() + second.len());

    x.iter()
        .cloned()
        .zip(first.iter_mut().chain(second))
        .for_each(|(b, a)| *a = b);
}

#[inline]
pub fn split_strided<T: Clone>(x: &StridedSliceRef<T>, first: &mut [T], second: &mut [T]) {
    if let Some(x) = x.as_slice() {
        split(x, first, second);
    } else {
        assert_eq!(x.len(), first.len() + second.len());
        x.iter()
            .cloned()
            .zip(first.iter_mut().chain(second))
            .for_each(|(b, a)| *a = b);
    }
}

#[inline]
pub fn split_strided_chunk<T: Clone, const N: usize>(
    x: &ChunkStridedSliceRef<T, N>,
    first: &mut [AVec<T>; N],
    second: &mut [AVec<T>; N],
) {
    assert_ne!(N, 0);

    let nx = x.len();

    let n_first = first[0].len();
    let n_second = second[0].len();
    assert_eq!(nx, n_first + n_second);

    let mut f_chunks = first.each_mut().map(|v| v.as_mut_slice());
    let mut s_chunks = second.each_mut().map(|v| v.as_mut_slice());

    if let Some(mut x_iter) = x.chunks() {
        x_iter
            .by_ref()
            .take(n_first)
            .enumerate()
            .for_each(|(i, out)| {
                out.into_iter()
                    .cloned()
                    .zip(f_chunks.iter_mut())
                    .for_each(|(out, v)| {
                        *unsafe { v.get_unchecked_mut(i) } = out;
                    })
            });
        x_iter.enumerate().for_each(|(i, out)| {
            out.into_iter()
                .cloned()
                .zip(s_chunks.iter_mut())
                .for_each(|(out, v)| {
                    *unsafe { v.get_unchecked_mut(i) } = out;
                })
        });
    } else {
        let mut x_iter = x.iter();

        x_iter
            .by_ref()
            .take(n_first)
            .enumerate()
            .for_each(|(i, out)| {
                out.into_iter()
                    .cloned()
                    .zip(f_chunks.iter_mut())
                    .for_each(|(out, v)| {
                        *unsafe { v.get_unchecked_mut(i) } = out;
                    })
            });
        x_iter.enumerate().for_each(|(i, out)| {
            out.into_iter()
                .cloned()
                .zip(s_chunks.iter_mut())
                .for_each(|(out, v)| {
                    *unsafe { v.get_unchecked_mut(i) } = out;
                })
        });
    }
}

#[inline]
pub fn interleave_inplace<T: Clone>(x: &mut [T]) {
    let n = x.len();
    if n < 2 {
        return;
    } else if n == 3 {
        x.swap(1, 2);
        return;
    }
    let do_sub = n % 2 == 1;
    let x = match do_sub {
        true => &mut x[1..],
        false => x,
    };
    let n = x.len();
    let mut m = 0;
    while m < n {
        let i = lookup(n - m);
        let slice_start = m + (i - 1) / 2;
        let slice_len = (n - m) / 2;
        shift_n(&mut x[slice_start..slice_start + slice_len], (i - 1) / 2);
        perfect_shuffle(&mut x[m..m + i - 1]);
        m += i - 1;
    }
    if !do_sub {
        x.chunks_exact_mut(2).for_each(|x| x.reverse());
    }
}

#[inline(always)]
fn cycle<T: Clone>(x: &mut [T], start: usize) {
    let n = x.len();
    let mut i_c = (start * 2).rem_euclid(n + 1);

    let mut t1 = x[start - 1].clone();
    std::mem::swap(&mut x[i_c - 1], &mut t1);
    while i_c != start {
        let i = (i_c * 2).rem_euclid(n + 1);
        std::mem::swap(&mut x[i - 1], &mut t1);
        i_c = i;
    }
}

#[inline(always)]
fn shift_n<T>(x: &mut [T], n: usize) {
    assert!(n <= x.len());
    let (left, right) = x.split_at_mut(x.len() - n);
    left.reverse();
    right.reverse();
    x.reverse();
}

#[inline(always)]
fn lookup(n: usize) -> usize {
    let mut i = 3;
    while i <= n + 1 {
        i *= 3
    }
    if i > 3 {
        i /= 3
    };
    i
}

#[inline(always)]
fn perfect_shuffle<T: Clone>(x: &mut [T]) {
    let n = x.len();
    match n {
        2 => x.swap(0, 1),
        _ => {
            let mut i = 1;
            while i < n {
                cycle(x, i);
                i *= 3;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iter::LanesIterator;
    use aligned_vec::avec;

    use rstest::rstest;

    #[rstest]
    fn test_interleave_inplace(#[values(20, 21, 22, 32, 423, 553)] n: usize) {
        let evens = (0..n).step_by(2).collect_vec();
        let odds = (1..n).step_by(2).collect_vec();

        let mut x = vec![0; n];
        stack(&evens, &odds, &mut x);

        interleave_inplace(&mut x);
        assert_eq!(x, (0..n).collect_vec());
    }

    #[rstest]
    fn test_interleave(#[values(20, 21, 22, 32, 423, 553)] n: usize) {
        let ns = (n + 1) / 2;
        let nd = n / 2;

        let s = (0..ns).collect_vec();
        let d = (ns..ns + nd).collect_vec();

        let mut out = vec![0; n];

        interleave(&s, &d, &mut out);

        let expected = (0..ns).interleave(ns..ns + nd).collect_vec();
        assert_eq!(out, expected);
    }

    #[rstest]
    fn test_deinterleave(#[values(20, 21, 22, 32, 423, 553)] n: usize) {
        let ns = (n + 1) / 2;
        let nd = n / 2;

        let mut s = vec![0; ns];
        let mut d = vec![0; nd];

        let inp = (0..ns).interleave(ns..ns + nd).collect_vec();

        deinterleave(&inp, &mut s, &mut d);

        assert_eq!(s, (0..ns).collect_vec());
        assert_eq!(d, (ns..ns + nd).collect_vec());
    }

    #[rstest]
    fn test_stack(#[values(20, 21, 22, 32, 423, 553)] n: usize) {
        let ns = (n + 1) / 2;
        let nd = n / 2;

        let first = (0..ns).collect_vec();
        let second = (ns..ns + nd).collect_vec();

        let mut out = vec![0; n];
        stack(&first, &second, &mut out);

        assert_eq!(out, (0..n).collect_vec());
    }

    #[rstest]
    fn test_split(#[values(20, 21, 22, 32, 423, 553)] n: usize) {
        let ns = (n + 1) / 2;
        let nd = n / 2;

        let inp = (0..n).collect_vec();

        let mut first = vec![0; ns];
        let mut second = vec![0; nd];

        split(&inp, &mut first, &mut second);

        assert_eq!(first, (0..ns).collect_vec());
        assert_eq!(second, (ns..ns + nd).collect_vec());
    }

    #[rstest]
    fn test_interleave_strided(
        #[values(10, 11)] n0: usize,
        #[values(10, 11)] n1: usize,
        #[values(0, 1)] ax: usize,
    ) {
        let shape = [n0, n1];
        let n_total: usize = shape.iter().product();
        let n = shape[ax];
        let ns = (n + 1) / 2;

        let mut out = vec![0; n_total];
        for (i, mut slc) in out.iter_lanes_mut(&shape, ax).enumerate() {
            let s = (i..ns + i).collect_vec();
            let d = (i + ns..n + i).collect_vec();
            interleave_strided(&s, &d, &mut slc);
        }

        for (i, slc) in out.iter_lanes(&shape, ax).enumerate() {
            let expected = (i..ns + i).interleave(i + ns..n + i).collect_vec();
            assert_eq!(slc.iter().cloned().collect_vec(), expected);
        }
    }

    #[rstest]
    fn test_deinterleave_strided(
        #[values(10, 11)] n0: usize,
        #[values(10, 11)] n1: usize,
        #[values(0, 1)] ax: usize,
    ) {
        let shape = [n0, n1];
        let n_total: usize = shape.iter().product();
        let n = shape[ax];
        let ns = (n + 1) / 2;
        let nd = n / 2;

        let mut out = vec![0; n_total];
        for (i, mut slc) in out.iter_lanes_mut(&shape, ax).enumerate() {
            slc.iter_mut()
                .zip((i..ns + i).interleave(ns + i..n + i))
                .for_each(|(v1, v2)| *v1 = v2);
        }

        for (i, slc) in out.iter_lanes(&shape, ax).enumerate() {
            let mut s = vec![0; ns];
            let mut d = vec![0; nd];
            deinterleave_strided(&slc, &mut s, &mut d);

            assert_eq!(s, (i..ns + i).collect_vec());
            assert_eq!(d, (ns + i..n + i).collect_vec());
        }
    }

    #[rstest]
    fn test_stack_strided(
        #[values(10, 11)] n0: usize,
        #[values(10, 11)] n1: usize,
        #[values(0, 1)] ax: usize,
    ) {
        let shape = [n0, n1];
        let n_total: usize = shape.iter().product();
        let n = shape[ax];
        let ns = (n + 1) / 2;

        let mut out = vec![0; n_total];
        for (i, mut slc) in out.iter_lanes_mut(&shape, ax).enumerate() {
            let first = (i..ns + i).collect_vec();
            let second = (i + ns..n + i).collect_vec();
            stack_to_strided(&first, &second, &mut slc);
        }

        for (i, slc) in out.iter_lanes(&shape, ax).enumerate() {
            let expected = (i..n + i).collect_vec();
            assert_eq!(slc.iter().cloned().collect_vec(), expected);
        }
    }

    #[rstest]
    fn test_split_strided(
        #[values(10, 11)] n0: usize,
        #[values(10, 11)] n1: usize,
        #[values(0, 1)] ax: usize,
    ) {
        let shape = [n0, n1];
        let n_total: usize = shape.iter().product();
        let n = shape[ax];
        let ns = (n + 1) / 2;
        let nd = n / 2;

        let mut out = vec![0; n_total];
        for (i, mut slc) in out.iter_lanes_mut(&shape, ax).enumerate() {
            slc.iter_mut().zip(i..n + i).for_each(|(v1, v2)| *v1 = v2);
        }

        for (i, slc) in out.iter_lanes(&shape, ax).enumerate() {
            let mut first = vec![0; ns];
            let mut second = vec![0; nd];
            split_strided(&slc, &mut first, &mut second);

            assert_eq!(first, (i..ns + i).collect_vec());
            assert_eq!(second, (ns + i..n + i).collect_vec());
        }
    }

    #[rstest]
    fn test_split_and_interleave_strided_chunk(
        #[values(10, 11)] n0: usize,
        #[values(10, 11)] n1: usize,
        #[values(0, 1)] ax: usize,
    ) {
        const N: usize = 4;

        let shape = [n0, n1];
        let n_total: usize = shape.iter().product();
        let n = shape[ax];
        let ns = (n + 1) / 2;
        let nd = n / 2;

        let mut out = (0..n_total).collect_vec();
        let mut out2 = (0..n_total).collect_vec();

        let chunks = out.iter_lane_chunks_mut::<N>(&shape, ax);
        let lanes = chunks.remainder();

        let mut work_e: [AVec<usize>; N] = core::array::from_fn(|_| avec![0; ns]);
        let mut work_o: [AVec<usize>; N] = core::array::from_fn(|_| avec![0; nd]);

        chunks.for_each(|mut chunk| {
            split_strided_chunk(&chunk, &mut work_e, &mut work_o);
            interleave_strided_chunk(&work_e, &work_o, &mut chunk);
        });
        let mut work_e = vec![0; ns];
        let mut work_o = vec![0; nd];

        lanes.for_each(|mut slc| {
            split_strided(&slc, &mut work_e, &mut work_o);
            interleave_strided(&work_e, &work_o, &mut slc);
        });

        // iterate over all of them using single lanes
        out2.iter_lanes_mut(&shape, ax).for_each(|mut slc| {
            split_strided(&slc, &mut work_e, &mut work_o);
            interleave_strided(&work_e, &work_o, &mut slc);
        });

        assert_eq!(out, out2);
    }

    #[rstest]
    fn test_deinterleave_and_stack_strided_chunk(
        #[values(10, 11)] n0: usize,
        #[values(10, 11)] n1: usize,
        #[values(0, 1)] ax: usize,
    ) {
        const N: usize = 4;

        let shape = [n0, n1];
        let n_total: usize = shape.iter().product();
        let n = shape[ax];
        let ns = (n + 1) / 2;
        let nd = n / 2;

        let mut out = (0..n_total).collect_vec();
        let mut out2 = (0..n_total).collect_vec();

        let chunks = out.iter_lane_chunks_mut::<N>(&shape, ax);
        let lanes = chunks.remainder();

        let mut work_e: [AVec<usize>; N] = core::array::from_fn(|_| avec![0; ns]);
        let mut work_o: [AVec<usize>; N] = core::array::from_fn(|_| avec![0; nd]);

        chunks.for_each(|mut chunk| {
            deinterleave_strided_chunk(&chunk, &mut work_e, &mut work_o);
            stack_to_strided_chunk(&work_e, &work_o, &mut chunk);
        });
        let mut work_e = vec![0; ns];
        let mut work_o = vec![0; nd];

        lanes.for_each(|mut slc| {
            deinterleave_strided(&slc, &mut work_e, &mut work_o);
            stack_to_strided(&work_e, &work_o, &mut slc);
        });

        // iterate over all of them using single lanes
        out2.iter_lanes_mut(&shape, ax).for_each(|mut slc| {
            deinterleave_strided(&slc, &mut work_e, &mut work_o);
            stack_to_strided(&work_e, &work_o, &mut slc);
        });

        assert_eq!(out, out2);
    }

    #[test]
    fn test_aligned_vec() {
        let n = 12;
        let mut v = avec![0.0; n];

        v.iter_mut().enumerate().for_each(|(i, v)| {
            *v = i as f64;
        });

        assert_eq!(v[n - 1], (n - 1) as f64);
        assert_eq!(v.len(), n);
    }
}
