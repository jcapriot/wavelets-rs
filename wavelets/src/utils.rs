use crate::iter::slice::{ChunkStridedSlice, MutChunkStridedSlice, MutStridedSlice, StridedSlice};
use itertools::{Itertools, izip};

#[inline]
pub fn stride_from_shape(shape: &[usize]) -> Vec<usize> {
    let mut stride = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        stride[i] = stride[i + 1] * shape[i + 1];
    }
    stride
}

#[inline]
pub fn deinterleave<T: Clone>(x: &[T], evens: &mut [T], odds: &mut [T]) {
    let nx = x.len();
    let n_e = evens.len();
    let n_o = odds.len();

    assert_eq!(nx / 2, n_o);
    assert_eq!((nx + 1) / 2, n_e);

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
    let n_total = shape.iter().product();
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
pub fn deinterleave_strided<T: Clone>(x: StridedSlice<T>, evens: &mut [T], odds: &mut [T]) {
    let nx = x.len();
    let n_e = evens.len();
    let n_o = odds.len();

    assert_eq!(nx / 2, n_o);
    assert_eq!((nx + 1) / 2, n_e);

    x.iter()
        .zip(evens.iter_mut().interleave(odds.iter_mut()))
        .for_each(|(v, ou)| {
            *ou = v.clone();
        });
}

#[inline]
pub fn deinterleave_strided_chunk<T: Clone, const N: usize>(
    x: ChunkStridedSlice<T, N>,
    evens: &mut [T],
    odds: &mut [T],
) {
    assert_ne!(N, 0);

    let nx = x.len();
    let n_o = nx / 2;
    let n_e = nx - n_o;

    assert_eq!(evens.len(), N * n_e);
    assert_eq!(odds.len(), N * n_o);

    let mut e_chunks: [_; N] = evens.chunks_exact_mut(n_e).collect_array().unwrap();
    let mut o_chunks: [_; N] = odds.chunks_exact_mut(n_o).collect_array().unwrap();

    let mut do_even = true;
    let mut ind_io = 0;

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

#[inline]
pub fn stack<T: Clone>(first: &[T], second: &[T], out: &mut [T]) {
    assert_eq!(first.len() + second.len(), out.len());
    first
        .iter()
        .chain(second.iter())
        .cloned()
        .zip(out.iter_mut())
        .for_each(|(v, o)| *o = v);
}

#[inline]
pub fn stack_to_strided<T: Clone>(first: &[T], second: &[T], out: MutStridedSlice<T>) {
    assert_eq!(first.len() + second.len(), out.len());
    first
        .iter()
        .chain(second.iter())
        .zip(out.iter_mut())
        .for_each(|(v_in, v_out)| *v_out = v_in.clone());
}

#[inline]
pub fn stack_to_strided_chunk<T: Clone, const N: usize>(
    first: &[T],
    second: &[T],
    out: MutChunkStridedSlice<T, N>,
) {
    assert_ne!(N, 0);

    let nx = out.len();

    let n_first = first.len() / N;
    let n_second = second.len() / N;
    assert_eq!(nx, n_first + n_second);

    let f_chunks: [_; N] = first.chunks_exact(n_first).collect_array().unwrap();
    let s_chunks: [_; N] = second.chunks_exact(n_second).collect_array().unwrap();

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
pub fn interleave_strided<T: Clone>(evens: &[T], odds: &[T], x: MutStridedSlice<T>) {
    let nx = x.len();
    let n_e = evens.len();
    let n_o = odds.len();

    assert_eq!(nx / 2, n_o);
    assert_eq!((nx + 1) / 2, n_e);

    x.iter_mut()
        .zip(evens.iter().interleave(odds.iter()).cloned())
        .for_each(|(l, r)| *l = r);
}

#[inline]
pub fn interleave_strided_chunk<T: Clone, const N: usize>(
    evens: &[T],
    odds: &[T],
    x: MutChunkStridedSlice<T, N>,
) {
    assert_ne!(N, 0);

    let nx = x.len();
    let n_o = nx / 2;
    let n_e = nx - n_o;

    assert_eq!(evens.len(), N * n_e);
    assert_eq!(odds.len(), N * n_o);

    let e_chunks: [_; N] = evens.chunks_exact(n_e).collect_array().unwrap();
    let o_chunks: [_; N] = odds.chunks_exact(n_o).collect_array().unwrap();

    let mut do_even = true;
    let mut ind_io = 0;
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

#[inline]
pub fn split<T: Clone>(x: &[T], first: &mut [T], second: &mut [T]) {
    assert_eq!(x.len(), first.len() + second.len());

    x.iter()
        .cloned()
        .zip(first.iter_mut().chain(second.iter_mut()))
        .for_each(|(x, v)| *v = x);
}

#[inline]
pub fn split_strided<T: Clone>(x: StridedSlice<T>, first: &mut [T], second: &mut [T]) {
    assert_eq!(x.len(), first.len() + second.len());

    x.iter()
        .cloned()
        .zip(first.iter_mut().chain(second.iter_mut()))
        .for_each(|(x, v)| *v = x);
}

#[inline]
pub fn split_strided_chunk<T: Clone, const N: usize>(
    x: ChunkStridedSlice<T, N>,
    first: &mut [T],
    second: &mut [T],
) {
    assert_ne!(N, 0);

    let nx = x.len();

    let n_first = first.len() / N;
    let n_second = second.len() / N;
    assert_eq!(nx, n_first + n_second);

    let mut f_chunks: [_; N] = first.chunks_exact_mut(n_first).collect_array().unwrap();
    let mut s_chunks: [_; N] = second.chunks_exact_mut(n_second).collect_array().unwrap();

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
    use crate::iter::slice::LanesIterator;

    #[test]
    fn test_interleave_inplace() {
        for n in [20, 21, 22, 32, 432, 553] {
            let evens = (0..n).step_by(2).collect_vec();
            let odds = (1..n).step_by(2).collect_vec();

            let mut x = vec![0; n];
            stack(&evens, &odds, &mut x);

            interleave_inplace(&mut x);
            assert_eq!(x, (0..n).collect_vec());
        }
    }

    #[test]
    fn test_interleave() {
        for n in [10, 11] {
            let ns = (n + 1) / 2;
            let nd = n / 2;

            let s = (0..ns).collect_vec();
            let d = (ns..ns + nd).collect_vec();

            let mut out = vec![0; n];

            interleave(&s, &d, &mut out);

            let expected = (0..ns).interleave(ns..ns + nd).collect_vec();
            assert_eq!(out, expected);
        }
    }
    #[test]
    fn test_deinterleave() {
        for n in [10, 11] {
            let ns = (n + 1) / 2;
            let nd = n / 2;

            let mut s = vec![0; ns];
            let mut d = vec![0; nd];

            let inp = (0..ns).interleave(ns..ns + nd).collect_vec();

            deinterleave(&inp, &mut s, &mut d);

            assert_eq!(s, (0..ns).collect_vec());
            assert_eq!(d, (ns..ns + nd).collect_vec());
            // deinterleave_strided(&inp, &mut s, &mut d);
            // assert_eq!(s, (0..ns).collect_vec());
            // assert_eq!(d, (ns..ns + nd).collect_vec());
        }
    }
    #[test]
    fn test_stack() {
        for n in [10, 11] {
            let ns = (n + 1) / 2;
            let nd = n / 2;

            let first = (0..ns).collect_vec();
            let second = (ns..ns + nd).collect_vec();

            let mut out = vec![0; n];
            stack(&first, &second, &mut out);

            assert_eq!(out, (0..n).collect_vec());
        }
    }
    #[test]
    fn test_split() {
        for n in [10, 11] {
            let ns = (n + 1) / 2;
            let nd = n / 2;

            let inp = (0..n).collect_vec();

            let mut first = vec![0; ns];
            let mut second = vec![0; nd];

            split(&inp, &mut first, &mut second);

            assert_eq!(first, (0..ns).collect_vec());
            assert_eq!(second, (ns..ns + nd).collect_vec());
        }
    }

    #[test]
    fn test_interleave_strided() {
        for n1 in [10, 11] {
            for n2 in [10, 11] {
                for ax in [0, 1] {
                    let n = match ax {
                        0 => n1,
                        1 => n2,
                        _ => unreachable!(),
                    };
                    let ns = (n + 1) / 2;

                    let mut out = vec![0; n1 * n2];
                    let shape = [n1, n2];
                    for (i, slc) in out.iter_lanes_mut(&shape, ax).enumerate() {
                        let s = (i..ns + i).collect_vec();
                        let d = (i + ns..n + i).collect_vec();
                        interleave_strided(&s, &d, slc);
                    }

                    for (i, slc) in out.iter_lanes(&shape, ax).enumerate() {
                        let expected = (i..ns + i).interleave(i + ns..n + i).collect_vec();
                        assert_eq!(slc.iter().cloned().collect_vec(), expected);
                    }
                }
            }
        }
    }

    #[test]
    fn test_deinterleave_strided() {
        for n1 in [10, 11] {
            for n2 in [10, 11] {
                for ax in [0, 1] {
                    let n = match ax {
                        0 => n1,
                        1 => n2,
                        _ => unreachable!(),
                    };
                    let ns = (n + 1) / 2;
                    let nd = n / 2;

                    let mut out = vec![0; n1 * n2];

                    let shape = [n1, n2];
                    for (i, slc) in out.iter_lanes_mut(&shape, ax).enumerate() {
                        slc.iter_mut()
                            .zip((i..ns + i).interleave(ns + i..n + i))
                            .for_each(|(v1, v2)| *v1 = v2);
                    }

                    for (i, slc) in out.iter_lanes(&shape, ax).enumerate() {
                        let mut s = vec![0; ns];
                        let mut d = vec![0; nd];
                        deinterleave_strided(slc, &mut s, &mut d);

                        assert_eq!(s, (i..ns + i).collect_vec());
                        assert_eq!(d, (ns + i..n + i).collect_vec());
                    }
                }
            }
        }
    }

    #[test]
    fn test_stack_strided() {
        for n1 in [10, 11] {
            for n2 in [10, 11] {
                for ax in [0, 1] {
                    let n = match ax {
                        0 => n1,
                        1 => n2,
                        _ => unreachable!(),
                    };
                    let ns = (n + 1) / 2;

                    let mut out = vec![0; n1 * n2];
                    let shape = [n1, n2];
                    for (i, slc) in out.iter_lanes_mut(&shape, ax).enumerate() {
                        let first = (i..ns + i).collect_vec();
                        let second = (i + ns..n + i).collect_vec();
                        stack_to_strided(&first, &second, slc);
                    }

                    for (i, slc) in out.iter_lanes(&shape, ax).enumerate() {
                        let expected = (i..n + i).collect_vec();
                        assert_eq!(slc.iter().cloned().collect_vec(), expected);
                    }
                }
            }
        }
    }

    #[test]
    fn test_split_strided() {
        for n1 in [10, 11] {
            for n2 in [10, 11] {
                for ax in [0, 1] {
                    let n = match ax {
                        0 => n1,
                        1 => n2,
                        _ => unreachable!(),
                    };
                    let ns = (n + 1) / 2;
                    let nd = n / 2;

                    let mut out = vec![0; n1 * n2];

                    let shape = [n1, n2];
                    for (i, slc) in out.iter_lanes_mut(&shape, ax).enumerate() {
                        slc.iter_mut().zip(i..n + i).for_each(|(v1, v2)| *v1 = v2);
                    }

                    for (i, slc) in out.iter_lanes(&shape, ax).enumerate() {
                        let mut first = vec![0; ns];
                        let mut second = vec![0; nd];
                        split_strided(slc, &mut first, &mut second);

                        assert_eq!(first, (i..ns + i).collect_vec());
                        assert_eq!(second, (ns + i..n + i).collect_vec());
                    }
                }
            }
        }
    }

    #[test]
    fn test_interleave_strided_chunk() {
        const N: usize = 4;

        for n1 in [10, 11] {
            for n2 in [10, 11] {
                for ax in [0, 1] {
                    let n = match ax {
                        0 => n1,
                        1 => n2,
                        _ => unreachable!(),
                    };
                    let ns = (n + 1) / 2;

                    let mut out = vec![0; n1 * n2];
                    let shape = [n1, n2];
                    let (chunks, lanes) = out.iter_lane_chunks_mut::<N>(&shape, ax);
                    let n_chunks = chunks.len();

                    for (i_c, slc) in chunks.enumerate() {
                        let s = (0..N)
                            .map(|j| {
                                let i = N * i_c + j;
                                i..ns + i
                            })
                            .flatten()
                            .collect_vec();
                        let d = (0..N)
                            .map(|j| {
                                let i = N * i_c + j;
                                ns + i..n + i
                            })
                            .flatten()
                            .collect_vec();
                        interleave_strided_chunk(&s, &d, slc);
                    }

                    for (i, slc) in lanes.enumerate() {
                        let i = i + n_chunks * N;

                        let s = (i..ns + i).collect_vec();
                        let d = (i + ns..n + i).collect_vec();
                        interleave_strided(&s, &d, slc);
                    }

                    for (i, slc) in out.iter_lanes(&shape, ax).enumerate() {
                        let expected = (i..ns + i).interleave(i + ns..n + i).collect_vec();
                        assert_eq!(slc.iter().cloned().collect_vec(), expected);
                    }
                }
            }
        }
    }

    #[test]
    fn test_deinterleave_strided_chunk() {
        const N: usize = 4;

        for n1 in [10, 11] {
            for n2 in [10, 11] {
                for ax in [0, 1] {
                    let n = match ax {
                        0 => n1,
                        1 => n2,
                        _ => unreachable!(),
                    };
                    let ns = (n + 1) / 2;
                    let nd = n / 2;

                    let mut out = vec![0; n1 * n2];

                    let shape = [n1, n2];

                    for (i, slc) in out.iter_lanes_mut(&shape, ax).enumerate() {
                        slc.iter_mut()
                            .zip((0..ns).interleave(ns..ns + nd))
                            .for_each(|(v1, v2)| *v1 = v2 + i);
                    }

                    let (chunks, lanes) = out.iter_lane_chunks::<N>(&shape, ax);
                    let n_chunks = chunks.len();

                    for (i_c, slc) in chunks.enumerate() {
                        let mut s = vec![0; ns * N];
                        let mut d = vec![0; nd * N];
                        deinterleave_strided_chunk(slc, &mut s, &mut d);

                        let s_ref = (0..N)
                            .map(|j| {
                                let i = N * i_c + j;
                                i..ns + i
                            })
                            .flatten()
                            .collect_vec();
                        let d_ref = (0..N)
                            .map(|j| {
                                let i = N * i_c + j;
                                ns + i..n + i
                            })
                            .flatten()
                            .collect_vec();
                        assert_eq!(s, s_ref);
                        assert_eq!(d, d_ref);
                    }

                    for (i, slc) in lanes.enumerate() {
                        let mut s = vec![0; ns];
                        let mut d = vec![0; nd];
                        deinterleave_strided(slc, &mut s, &mut d);

                        let i = N * n_chunks + i;
                        assert_eq!(s, (i..ns + i).collect_vec());
                        assert_eq!(d, (ns + i..ns + nd + i).collect_vec());
                    }
                }
            }
        }
    }

    #[test]
    fn test_stack_strided_chunk() {
        const N: usize = 4;

        for n1 in [10, 11] {
            for n2 in [10, 11] {
                for ax in [0, 1] {
                    let n = match ax {
                        0 => n1,
                        1 => n2,
                        _ => unreachable!(),
                    };
                    let ns = (n + 1) / 2;

                    let mut out = vec![0; n1 * n2];
                    let shape = [n1, n2];
                    let (chunks, lanes) = out.iter_lane_chunks_mut::<N>(&shape, ax);
                    let n_chunks = chunks.len();

                    for (i_c, slc) in chunks.enumerate() {
                        let first = (0..N)
                            .map(|j| {
                                let i = N * i_c + j;
                                i..ns + i
                            })
                            .flatten()
                            .collect_vec();
                        let second = (0..N)
                            .map(|j| {
                                let i = N * i_c + j;
                                ns + i..n + i
                            })
                            .flatten()
                            .collect_vec();
                        stack_to_strided_chunk(&first, &second, slc);
                    }

                    for (i, slc) in lanes.enumerate() {
                        let i = i + n_chunks * N;
                        let first = (i..ns + i).collect_vec();
                        let second = (i + ns..n + i).collect_vec();
                        stack_to_strided(&first, &second, slc);
                    }

                    for (i, slc) in out.iter_lanes(&shape, ax).enumerate() {
                        let expected = (i..n + i).collect_vec();
                        assert_eq!(slc.iter().cloned().collect_vec(), expected);
                    }
                }
            }
        }
    }

    #[test]
    fn test_split_strided_chunk() {
        const N: usize = 4;

        for n1 in [10, 11] {
            for n2 in [10, 11] {
                for ax in [0, 1] {
                    let n = match ax {
                        0 => n1,
                        1 => n2,
                        _ => unreachable!(),
                    };
                    let ns = (n + 1) / 2;
                    let nd = n / 2;

                    let mut out = vec![0; n1 * n2];

                    let shape = [n1, n2];

                    for (i, slc) in out.iter_lanes_mut(&shape, ax).enumerate() {
                        slc.iter_mut().zip(i..n + i).for_each(|(v1, v2)| *v1 = v2);
                    }

                    let (chunks, lanes) = out.iter_lane_chunks::<N>(&shape, ax);
                    let n_chunks = chunks.len();

                    for (i_c, slc) in chunks.enumerate() {
                        let mut s = vec![0; ns * N];
                        let mut d = vec![0; nd * N];
                        split_strided_chunk(slc, &mut s, &mut d);

                        let s_ref = (0..N)
                            .map(|j| {
                                let i = N * i_c + j;
                                i..ns + i
                            })
                            .flatten()
                            .collect_vec();
                        let d_ref = (0..N)
                            .map(|j| {
                                let i = N * i_c + j;
                                ns + i..n + i
                            })
                            .flatten()
                            .collect_vec();
                        assert_eq!(s, s_ref);
                        assert_eq!(d, d_ref);
                    }

                    for (i, slc) in lanes.enumerate() {
                        let mut s = vec![0; ns];
                        let mut d = vec![0; nd];
                        split_strided(slc, &mut s, &mut d);

                        let i = N * n_chunks + i;
                        assert_eq!(s, (i..ns + i).collect_vec());
                        assert_eq!(d, (ns + i..ns + nd + i).collect_vec());
                    }
                }
            }
        }
    }
}
