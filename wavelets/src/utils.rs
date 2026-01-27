use crate::iter::slice::{ChunkStridedSlice, MutChunkStridedSlice, MutStridedSlice, StridedSlice};
use itertools::{Itertools, izip};

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
pub fn deinterleave_strided<T: Clone>(x: &StridedSlice<T>, evens: &mut [T], odds: &mut [T]) {
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
    x: &ChunkStridedSlice<T, N>,
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
            true => x.cloned().zip(e_chunks.iter_mut()).for_each(|(x, v)| {
                unsafe { *v.get_unchecked_mut(ind_io) = x };
            }),
            false => {
                x.cloned().zip(o_chunks.iter_mut()).for_each(|(x, v)| {
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
pub fn stack_to_strided<'a, T: Clone>(first: &[T], second: &[T], out: &mut MutStridedSlice<T>) {
    assert_eq!(first.len() + second.len(), out.len());
    first
        .iter()
        .chain(second.iter())
        .zip(out.iter_mut())
        .for_each(|(v_in, v_out)| *v_out = v_in.clone());
}

#[inline]
pub fn stack_to_strided_chunk<'a, T: Clone, const N: usize>(
    first: &[T],
    second: &[T],
    out: &'a mut MutChunkStridedSlice<'a, T, N>,
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
            out.zip(f_chunks.iter()).for_each(|(out, v)| {
                *out = unsafe { v.get_unchecked(i) }.clone();
            })
        });
    out_iter.enumerate().for_each(|(i, out)| {
        out.zip(s_chunks.iter()).for_each(|(out, v)| {
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
pub fn interleave_strided<T: Clone>(evens: &[T], odds: &[T], x: &mut MutStridedSlice<T>) {
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
pub fn interleave_strided_chunk<'a, T: Clone, const N: usize>(
    evens: &[T],
    odds: &[T],
    x: &'a mut MutChunkStridedSlice<'a, T, N>,
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
            true => x.zip(e_chunks.iter()).for_each(|(x, v)| {
                *x = unsafe { v.get_unchecked(ind_io) }.clone();
            }),
            false => {
                x.zip(o_chunks.iter()).for_each(|(x, v)| {
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
pub fn split_strided<T: Clone>(x: &StridedSlice<T>, first: &mut [T], second: &mut [T]) {
    assert_eq!(x.len(), first.len() + second.len());

    x.iter()
        .cloned()
        .zip(first.iter_mut().chain(second.iter_mut()))
        .for_each(|(x, v)| *v = x);
}

#[inline]
pub fn split_strided_chunk<T: Clone, const N: usize>(
    x: &ChunkStridedSlice<T, N>,
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
            out.cloned().zip(f_chunks.iter_mut()).for_each(|(out, v)| {
                *unsafe { v.get_unchecked_mut(i) } = out;
            })
        });
    x_iter.enumerate().for_each(|(i, out)| {
        out.cloned().zip(s_chunks.iter_mut()).for_each(|(out, v)| {
            *unsafe { v.get_unchecked_mut(i) } = out;
        })
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iter::slice::LanesIterator;

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
                    let nd = n / 2;

                    let mut out = vec![0; n1 * n2];
                    let shape = [n1, n2];
                    for (i, mut slc) in out.iter_lanes_mut(&shape, ax).enumerate() {
                        let s = (i..ns + i).collect_vec();
                        let d = (i + ns..ns + nd + i).collect_vec();
                        interleave_strided(&s, &d, &mut slc);
                    }

                    for (i, slc) in out.iter_lanes(&shape, ax).enumerate() {
                        let expected = (i..ns + i).interleave(i + ns..ns + nd + i).collect_vec();
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
                    for (i, mut slc) in out.iter_lanes_mut(&shape, ax).enumerate() {
                        slc.iter_mut()
                            .zip((0..ns).interleave(ns..ns + nd))
                            .for_each(|(v1, v2)| *v1 = v2 + i);
                    }

                    for (i, slc) in out.iter_lanes(&shape, ax).enumerate() {
                        let mut s = vec![0; ns];
                        let mut d = vec![0; nd];
                        deinterleave_strided(&slc, &mut s, &mut d);

                        assert_eq!(s, (0 + i..ns + i).collect_vec());
                        assert_eq!(d, (ns + i..ns + nd + i).collect_vec());
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
                    let nd = n / 2;

                    let mut out = vec![0; n1 * n2];
                    let shape = [n1, n2];
                    for (i, mut slc) in out.iter_lanes_mut(&shape, ax).enumerate() {
                        let first = (i..ns + i).collect_vec();
                        let second = (i + ns..ns + nd + i).collect_vec();
                        stack_to_strided(&first, &second, &mut slc);
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
                    for (i, mut slc) in out.iter_lanes_mut(&shape, ax).enumerate() {
                        slc.iter_mut().zip(i..n + i).for_each(|(v1, v2)| *v1 = v2);
                    }

                    for (i, slc) in out.iter_lanes(&shape, ax).enumerate() {
                        let mut first = vec![0; ns];
                        let mut second = vec![0; nd];
                        split_strided(&slc, &mut first, &mut second);

                        assert_eq!(first, (i..ns + i).collect_vec());
                        assert_eq!(second, (ns + i..ns + nd + i).collect_vec());
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

                    for (i_c, mut slc) in chunks.enumerate() {
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
                        interleave_strided_chunk(&s, &d, &mut slc);
                    }

                    for (i, mut slc) in lanes.enumerate() {
                        let i = i + n_chunks * N;
                        let expected = (i..ns + i).interleave(i + ns..n + i).collect_vec();

                        let s = (i..ns + i).collect_vec();
                        let d = (i + ns..n + i).collect_vec();
                        interleave_strided(&s, &d, &mut slc);
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

                    for (i, mut slc) in out.iter_lanes_mut(&shape, ax).enumerate() {
                        slc.iter_mut()
                            .zip((0..ns).interleave(ns..ns + nd))
                            .for_each(|(v1, v2)| *v1 = v2 + i);
                    }

                    let (chunks, lanes) = out.iter_lane_chunks::<N>(&shape, ax);
                    let n_chunks = chunks.len();

                    for (i_c, slc) in chunks.enumerate() {
                        let mut s = vec![0; ns * N];
                        let mut d = vec![0; nd * N];
                        deinterleave_strided_chunk(&slc, &mut s, &mut d);

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
                        deinterleave_strided(&slc, &mut s, &mut d);

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

                    for (i_c, mut slc) in chunks.enumerate() {
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
                        stack_to_strided_chunk(&first, &second, &mut slc);
                    }

                    for (i, mut slc) in lanes.enumerate() {
                        let i = i + n_chunks * N;
                        let first = (i..ns + i).collect_vec();
                        let second = (i + ns..n + i).collect_vec();
                        stack_to_strided(&first, &second, &mut slc);
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

                    for (i, mut slc) in out.iter_lanes_mut(&shape, ax).enumerate() {
                        slc.iter_mut().zip(i..n + i).for_each(|(v1, v2)| *v1 = v2);
                    }

                    let (chunks, lanes) = out.iter_lane_chunks::<N>(&shape, ax);
                    let n_chunks = chunks.len();

                    for (i_c, slc) in chunks.enumerate() {
                        let mut s = vec![0; ns * N];
                        let mut d = vec![0; nd * N];
                        split_strided_chunk(&slc, &mut s, &mut d);

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
                        split_strided(&slc, &mut s, &mut d);

                        let i = N * n_chunks + i;
                        assert_eq!(s, (i..ns + i).collect_vec());
                        assert_eq!(d, (ns + i..ns + nd + i).collect_vec());
                    }
                }
            }
        }
    }
}
