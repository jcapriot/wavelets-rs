//! Array layout utilities for wavelet sub-band manipulation.
//!
//! This module provides helpers for the even/odd index splitting (deinterleave) and
//! merging (interleave/stack) that the lifting transform requires, as well as strided
//! variants used for N-D axis traversal.

use crate::iter::{ChunkStridedSliceRef, StridedSliceRef};
use aligned_vec::AVec;
use itertools::{Itertools, izip};
use num_traits::Zero;

/// Compute the C-order (row-major) strides for a given shape.
///
/// `stride[i]` is the number of elements to skip to advance by one step along axis `i`.
#[inline]
pub fn stride_from_shape(shape: &[usize]) -> Vec<usize> {
    let mut stride = vec![1; shape.len()];
    for i in (1..shape.len()).rev() {
        stride[i - 1] = stride[i] * shape[i];
    }
    stride
}

/// Compile-time assertion that `N` is > 0.
///
/// Evaluate `CheckPositive::<N>::VALID` in a const context to trigger the assert.
struct CheckPositive<const N: usize>();
impl<const N: usize> CheckPositive<N> {
    /// Asserts at compile time that `N >= 2` and `N % 2 == 0`.
    const VALID: () = {
        assert!(N > 0, "N must be positive.");
    };
}

/// Assert at compile time that a chunk width `N` is positive.
///
/// Emits a compile error if `N` is 0.  Call this inside `const` blocks that
/// accept a coefficient-length type parameter to get a clearer error message.
macro_rules! static_assert_positive {
    ($N: ty) => {
        let _ = $crate::utils::CheckPositive::<$N>::VALID;
    };
}

/// Split a slice into its even-indexed and odd-indexed elements.
///
/// `evens[i] = x[2*i]` and `odds[i] = x[2*i + 1]`.  For odd-length `x` the extra
/// element goes into `evens`, so `evens.len() == (x.len() + 1) / 2` and
/// `odds.len() == x.len() / 2`.
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
        nx.div_ceil(2),
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

/// Deinterleave a 2-D row-major array: separate even- and odd-indexed elements along the
/// first axis, then recursively along the second axis.
///
/// `shape` is `[rows, cols]`.  `output` receives the same total number of elements rearranged
/// so that approximation coefficients precede detail coefficients along each axis.
#[inline]
pub fn deinterleave_2d<T: Clone>(input: &[T], output: &mut [T], shape: &[usize; 2]) {
    let n_total: usize = shape.iter().product();
    assert_eq!(input.len(), n_total);
    assert_eq!(output.len(), n_total);

    let n_first = shape[0].div_ceil(2);
    let n_sub: usize = shape[1..].iter().product();

    let (first, second) = output.split_at_mut(n_first * n_sub);

    let mut in_chunks = input.chunks_exact(2 * n_sub);

    let mut first_chunks = first.chunks_exact_mut(n_sub);

    let n_first = shape[1].div_ceil(2);
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

/// Deinterleave an N-D row-major array along the first axis at every level of the shape.
///
/// Dispatches to [`deinterleave`], [`deinterleave_2d`], or a recursive N-D implementation
/// based on `shape.len()`.
#[inline]
pub fn deinterleave_nd<T: Clone>(input: &[T], output: &mut [T], shape: &[usize]) {
    match shape.len() {
        0 => {}
        1 => {
            let (f, s) = output.split_at_mut(shape[0].div_ceil(2));
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
            let (f, s) = output.split_at_mut(shape[0].div_ceil(2));
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
            let n_first = shape[0].div_ceil(2);
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

/// Deinterleave a strided lane into even- and odd-indexed flat buffers.
///
/// Equivalent to [`deinterleave`] but reads from a [`StridedSliceRef`] instead of a plain
/// slice; takes a fast path when the view happens to be contiguous.
#[inline]
pub fn deinterleave_strided<T: Clone>(x: &StridedSliceRef<T>, evens: &mut [T], odds: &mut [T]) {
    if let Ok(x) = x.try_into() {
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
            nx.div_ceil(2),
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

/// Deinterleave `N` parallel strided lanes into `N` aligned even-buffer and odd-buffer arrays.
///
/// `x` presents `N` interleaved lanes; `evens[j]` and `odds[j]` receive the even- and
/// odd-indexed elements of lane `j`, respectively.  Takes a fast path when the chunk is
/// contiguous.
#[inline]
pub fn deinterleave_strided_chunk<T: Clone, const N: usize, A: aligned_vec::Alignment>(
    x: &ChunkStridedSliceRef<T, N>,
    evens: &mut [AVec<T, A>; N],
    odds: &mut [AVec<T, A>; N],
) {
    static_assert_positive!(N);
    let ne = x.len().div_ceil(2);
    let no = x.len() / 2;
    debug_assert_eq!(x.len(), ne + no);
    assert!(evens.iter().all(|v| v.len() == ne));
    assert!(odds.iter().all(|v| v.len() == no));

    if let Ok(mut x_iter) = x.try_array_chunks() {
        let mut i = 0;
        while let Some([xe, xo]) = x_iter.next_chunk::<2>() {
            xe.iter().cloned().zip(evens.iter_mut()).for_each(|(x, v)| {
                // SAFETY: Every evens' length is equal to ne = x.len()/2 + 1 and i < x.len()/2, so i < v.len().
                *unsafe { v.get_unchecked_mut(i) } = x;
            });
            xo.iter().cloned().zip(odds.iter_mut()).for_each(|(x, v)| {
                // SAFETY: Every odds' length is equal to no = x.len()/2 and i < x.len()/2, so i < v.len().
                *unsafe { v.get_unchecked_mut(i) } = x;
            });
            i += 1;
        }
        if let Some(x) = x_iter.next() {
            x.iter().cloned().zip(evens.iter_mut()).for_each(|(x, v)| {
                *unsafe { v.get_unchecked_mut(no) } = x;
            });
        }
    } else {
        let mut x_iter = x.iter();

        let mut i = 0;
        while let Some([xe, xo]) = x_iter.next_chunk::<2>() {
            xe.into_iter()
                .cloned()
                .zip(evens.iter_mut())
                .for_each(|(x, v)| {
                    // SAFETY: Every evens' length is equal to ne = x.len()/2 + 1 and i < x.len()/2, so i < v.len().
                    *unsafe { v.get_unchecked_mut(i) } = x;
                });
            xo.into_iter()
                .cloned()
                .zip(odds.iter_mut())
                .for_each(|(x, v)| {
                    // SAFETY: Every odds' length is equal to no = x.len()/2 and i < x.len()/2, so i < v.len().
                    *unsafe { v.get_unchecked_mut(i) } = x;
                });
            i += 1;
        }
        if let Some(x) = x_iter.next() {
            x.into_iter()
                .cloned()
                .zip(evens.iter_mut())
                .for_each(|(x, v)| {
                    // SAFETY: no < v.len() since v has length ne = no + 1 if there are any leftover slice chunks.
                    *unsafe { v.get_unchecked_mut(no) } = x;
                });
        }
    }
}

/// Write `first` at the start of `out` and `second` at the very end, zero-filling the gap.
///
/// Unlike a simple concatenation, the second half is placed at the tail of `out` rather than
/// immediately after `first`.  This matches the layout expected by the inverse LWT.
#[inline]
pub fn stack<T: Clone + Zero>(first: &[T], second: &[T], out: &mut [T]) {
    // stacks first and second into out, but with the second half at the very end of out, instead of immediately after the first half.
    assert!(
        first.len() + second.len() <= out.len(),
        "invalid lengths for slice stack, first: {}, second: {}, third: {}",
        first.len(),
        second.len(),
        out.len()
    );

    let (of, ol) = out.split_at_mut(first.len());
    let (om, os) = ol.split_at_mut(ol.len() - second.len());
    debug_assert_eq!(of.len(), first.len());
    debug_assert_eq!(os.len(), second.len());

    of.iter_mut()
        .zip(first.iter().cloned())
        .for_each(|(a, b)| *a = b);
    om.iter_mut().for_each(|v| *v = T::zero());
    os.iter_mut()
        .zip(second.iter().cloned())
        .for_each(|(a, b)| *a = b);
}

/// Strided variant of [`stack`]: write `first` and `second` into a [`StridedSliceRef`] lane.
#[inline]
pub fn stack_to_strided<T: Clone + Zero>(first: &[T], second: &[T], out: &mut StridedSliceRef<T>) {
    if let Ok(out) = out.try_into() {
        stack(first, second, out);
    } else {
        let no = out.len();
        let nf = first.len();
        let ns = second.len();
        assert!(
            nf + ns <= no,
            "invalid lengths for slice stack, first: {nf}, second: {ns}, third: {no}",
        );
        let n_mid = no - (nf + ns);
        let mut out_iter = out.iter_mut();
        out_iter
            .by_ref()
            .take(nf)
            .zip(first.iter().cloned())
            .for_each(|(a, b)| *a = b);
        out_iter.by_ref().take(n_mid).for_each(|v| *v = T::zero());
        out_iter
            .zip(second.iter().cloned())
            .for_each(|(a, b)| *a = b);
    }
}

/// Chunk-strided variant of [`stack`]: write `N` aligned lanes into a [`ChunkStridedSliceRef`].
#[inline]
pub fn stack_to_strided_chunk<T: Clone + Zero, const N: usize, A: aligned_vec::Alignment>(
    first: &[AVec<T, A>; N],
    second: &[AVec<T, A>; N],
    out: &mut ChunkStridedSliceRef<T, N>,
) {
    static_assert_positive!(N);
    let nf = first[0].len();
    let ns = second[0].len();
    let no = out.len();
    let n_mid = no - (nf + ns);
    assert!(first.iter().all(|v| v.len() == nf));
    assert!(second.iter().all(|v| v.len() == ns));
    assert!(
        nf + ns <= no,
        "invalid lengths for slice stack, first: {nf}, second: {ns}, third: {no}",
    );

    if let Ok(mut out_iter) = out.try_array_chunks_mut() {
        out_iter.by_ref().take(nf).enumerate().for_each(|(i, o)| {
            o.iter_mut().zip(first.iter()).for_each(|(o, v)| {
                // SAFETY: i comes from enumerate().take(nf), so i < nf
                *o = unsafe { v.get_unchecked(i) }.clone();
            })
        });
        out_iter
            .by_ref()
            .take(n_mid)
            .for_each(|o| o.iter_mut().for_each(|v| *v = T::zero()));
        out_iter.enumerate().for_each(|(i, o)| {
            o.iter_mut().zip(second.iter()).for_each(|(o, v)| {
                // SAFETY: i comes from enumerate() over the remaining ns positions, so i < ns
                *o = unsafe { v.get_unchecked(i) }.clone();
            })
        });
    } else {
        let mut out_iter = out.iter_mut();
        out_iter.by_ref().take(nf).enumerate().for_each(|(i, o)| {
            o.into_iter().zip(first.iter()).for_each(|(o, v)| {
                // SAFETY: i comes from enumerate().take(nf), so i < nf
                *o = unsafe { v.get_unchecked(i) }.clone();
            })
        });
        out_iter
            .by_ref()
            .take(n_mid)
            .for_each(|o| o.into_iter().for_each(|v| *v = T::zero()));
        out_iter.enumerate().for_each(|(i, o)| {
            o.into_iter().zip(second.iter()).for_each(|(o, v)| {
                // SAFETY: i comes from enumerate() over the remaining ns positions, so i < ns
                *o = unsafe { v.get_unchecked(i) }.clone();
            })
        });
    }
}

/// Interleave even- and odd-indexed elements back into a single flat slice.
///
/// Inverse of [`deinterleave`]: `x[2*i] = evens[i]`, `x[2*i+1] = odds[i]`.
#[inline]
pub fn interleave<T: Clone>(evens: &[T], odds: &[T], x: &mut [T]) {
    let nx = x.len();
    let n_e = evens.len();
    let n_o = odds.len();

    assert_eq!(nx / 2, n_o);
    assert_eq!(nx.div_ceil(2), n_e);

    let (chunks, rem) = x.as_chunks_mut::<2>();
    let mut ev_iter = evens.iter();
    izip!(chunks.iter_mut(), ev_iter.by_ref(), odds.iter()).for_each(|([xe, xo], even, odd)| {
        *xe = even.clone();
        *xo = odd.clone();
    });
    if let Some(x) = rem.last_mut()
        && let Some(e) = evens.last()
    {
        *x = e.clone();
    }
}

/// Strided variant of [`interleave`]: write interleaved values into a [`StridedSliceRef`] lane.
#[inline]
pub fn interleave_strided<T: Clone>(evens: &[T], odds: &[T], x: &mut StridedSliceRef<T>) {
    if let Ok(x) = x.try_into() {
        interleave(evens, odds, x);
    } else {
        let nx = x.len();
        let n_e = evens.len();
        let n_o = odds.len();

        assert_eq!(nx / 2, n_o);
        assert_eq!(nx.div_ceil(2), n_e);

        x.iter_mut()
            .zip(evens.iter().interleave(odds.iter()).cloned())
            .for_each(|(l, r)| *l = r);
    }
}

/// Chunk-strided variant of [`interleave`]: write `N` aligned lanes interleaved into a
/// [`ChunkStridedSliceRef`].
#[inline]
pub fn interleave_strided_chunk<T: Clone, const N: usize, A: aligned_vec::Alignment>(
    evens: &[AVec<T, A>; N],
    odds: &[AVec<T, A>; N],
    x: &mut ChunkStridedSliceRef<T, N>,
) {
    static_assert_positive!(N);
    let ne = x.len().div_ceil(2);
    let no = x.len() / 2;
    debug_assert_eq!(x.len(), ne + no);
    assert!(evens.iter().all(|v| v.len() == ne));
    assert!(odds.iter().all(|v| v.len() == no));

    // Note: Uses unsafe indexing to avoid bounds checks that are difficult to elide.
    // But are gauranteed by the assertions above and the loop conditions.
    if let Ok(mut x_iter) = x.try_array_chunks_mut() {
        let mut i = 0;
        while let Some([xe, xo]) = x_iter.next_chunk::<2>() {
            xe.iter_mut().zip(evens.iter()).for_each(|(x, v)| {
                // SAFETY: Every evens' length is equal to ne = x.len()/2 + 1 and i < x.len()/2, so i < v.len().
                *x = unsafe { v.get_unchecked(i) }.clone();
            });
            xo.iter_mut().zip(odds.iter()).for_each(|(x, v)| {
                // SAFETY: Every odds' length is equal to no = x.len()/2 and i < x.len()/2, so i < v.len().
                *x = unsafe { v.get_unchecked(i) }.clone();
            });
            i += 1;
        }
        if let Some(x) = x_iter.next() {
            x.iter_mut().zip(evens.iter()).for_each(|(x, v)| {
                // SAFETY: no < v.len() since v has length ne = no + 1 if there are any leftover slice chunks.
                *x = unsafe { v.get_unchecked(no) }.clone();
            });
        }
    } else {
        let mut x_iter = x.iter_mut();
        let mut i = 0;
        while let Some([xe, xo]) = x_iter.next_chunk::<2>() {
            xe.into_iter().zip(evens.iter()).for_each(|(x, v)| {
                // SAFETY: Same as above (chunks branch, evens).
                *x = unsafe { v.get_unchecked(i) }.clone();
            });
            xo.into_iter().zip(odds.iter()).for_each(|(x, v)| {
                // SAFETY: Same as above (chunks branch, evens).
                *x = unsafe { v.get_unchecked(i) }.clone();
            });
            i += 1;
        }
        if let Some(x) = x_iter.next() {
            x.into_iter().zip(evens.iter()).for_each(|(x, v)| {
                // SAFETY: Same as above (chunks branch remainder, evens).
                *x = unsafe { v.get_unchecked(no) }.clone();
            });
        }
    }
}

/// Split `x` into a leading `first` segment and a trailing `second` segment, skipping the gap.
///
/// `second` is taken from the tail of `x`, not from immediately after `first`.  This is the
/// inverse of [`stack`].
#[inline]
pub fn split<T: Clone>(x: &[T], first: &mut [T], second: &mut [T]) {
    // splits x into first and second, but with the second at the very end of x, instead of immediately after the first.
    let nf = first.len();
    let ns = second.len();
    let nx = x.len();
    assert!(
        nf + ns <= nx,
        "invalid lengths for slice stack, first: {nf}, second: {ns}, third: {nx}"
    );

    let (xf, xs) = x.split_at(nx - ns);
    let (xf, _) = xf.split_at(nf);
    debug_assert_eq!(xf.len(), nf);
    debug_assert_eq!(xs.len(), ns);

    xf.iter().cloned().zip(first).for_each(|(a, b)| *b = a);
    xs.iter().cloned().zip(second).for_each(|(a, b)| *b = a);
}

/// Strided variant of [`split`]: read from a [`StridedSliceRef`] lane.
#[inline]
pub fn split_strided<T: Clone>(x: &StridedSliceRef<T>, first: &mut [T], second: &mut [T]) {
    if let Ok(x) = x.try_into() {
        split(x, first, second);
    } else {
        let nf = first.len();
        let ns = second.len();
        let nx = x.len();
        assert!(
            nf + ns <= nx,
            "invalid lengths for slice stack, first: {nf}, second: {ns}, third: {nx}"
        );
        let n_mid = nx - (nf + ns);
        let mut x_iter = x.iter().cloned();
        x_iter
            .by_ref()
            .take(nf)
            .zip(first)
            .for_each(|(b, a)| *a = b);
        x_iter.skip(n_mid).zip(second).for_each(|(b, a)| *a = b);
    }
}

/// Chunk-strided variant of [`split`]: read `N` interleaved lanes from a [`ChunkStridedSliceRef`].
#[inline]
pub fn split_strided_chunk<T: Clone, const N: usize, A: aligned_vec::Alignment>(
    x: &ChunkStridedSliceRef<T, N>,
    first: &mut [AVec<T, A>; N],
    second: &mut [AVec<T, A>; N],
) {
    static_assert_positive!(N);
    let nf = first[0].len();
    let ns = second[0].len();
    let nx = x.len();
    assert!(
        nf + ns <= nx,
        "invalid lengths for slice stack, first: {nf}, second: {ns}, third: {nx}",
    );
    assert!(first.iter().all(|v| v.len() == nf));
    assert!(second.iter().all(|v| v.len() == ns));

    let n_mid = nx - (nf + ns);

    if let Ok(mut x_iter) = x.try_array_chunks() {
        x_iter.by_ref().take(nf).enumerate().for_each(|(i, out)| {
            out.iter()
                .cloned()
                .zip(first.iter_mut())
                .for_each(|(out, v)| {
                    // SAFETY: i comes from enumerate().take(nf), so i < nf.
                    // first[lane] has length nf,
                    // so i < v.len().
                    *unsafe { v.get_unchecked_mut(i) } = out;
                })
        });
        x_iter.skip(n_mid).enumerate().for_each(|(i, out)| {
            out.iter()
                .cloned()
                .zip(second.iter_mut())
                .for_each(|(out, v)| {
                    // SAFETY: i comes from enumerate() over the remaining ns positions,
                    // so i < ns = second[lane].len().
                    *unsafe { v.get_unchecked_mut(i) } = out;
                })
        });
    } else {
        let mut x_iter = x.iter();

        x_iter.by_ref().take(nf).enumerate().for_each(|(i, out)| {
            out.into_iter()
                .cloned()
                .zip(first.iter_mut())
                .for_each(|(out, v)| {
                    // SAFETY: see above (chunks branch, first half).
                    *unsafe { v.get_unchecked_mut(i) } = out;
                })
        });
        x_iter.skip(n_mid).enumerate().for_each(|(i, out)| {
            out.into_iter()
                .cloned()
                .zip(second.iter_mut())
                .for_each(|(out, v)| {
                    // SAFETY: see above (chunks branch, second half).
                    *unsafe { v.get_unchecked_mut(i) } = out;
                })
        });
    }
}

/// In-place interleave of a slice: rearranges so that the even-half and odd-half are merged.
///
/// The first half of `x` is treated as even elements and the second half as odd elements;
/// after the call `x[2*i] == old_x[i]` and `x[2*i+1] == old_x[n/2 + i]`.
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

/// Copy elements of `x` into `out`, stopping at whichever slice is shorter.
#[inline]
pub fn clone_slice<T: Clone>(x: &[T], out: &mut [T]) {
    // clones x into out, based on the shorter of the two slices' lengths.
    x.iter().cloned().zip(out).for_each(|(a, b)| *b = a);
}

/// Copy elements of a [`StridedSliceRef`] lane into a flat slice.
#[inline]
pub fn clone_strided_to_slice<T: Clone>(x: &StridedSliceRef<T>, out: &mut [T]) {
    // clones x into out, based on the shorter of the two slices' lengths.
    if let Ok(x) = x.try_into() {
        clone_slice(x, out);
    } else {
        x.iter().cloned().zip(out).for_each(|(a, b)| *b = a);
    }
}

/// Copy the first `out[0].len()` positions from a [`ChunkStridedSliceRef`] into `N` aligned vecs.
#[inline]
pub fn clone_strided_chunk_to_avecs<T: Clone, const N: usize, A: aligned_vec::Alignment>(
    x: &ChunkStridedSliceRef<T, N>,
    out: &mut [AVec<T, A>; N],
) {
    static_assert_positive!(N);
    let n = out[0].len();
    assert!(
        out.iter().all(|v| v.len() == n),
        "all output AVecs must have the same length"
    );
    assert!(
        n <= x.len(),
        "invalid lengths for strided chunk clone, input: {}, output: {}",
        x.len(),
        n
    );
    if let Ok(x) = x.try_array_chunks() {
        x.take(n).enumerate().for_each(|(i, x)| {
            x.iter().cloned().zip(out.iter_mut()).for_each(|(x, v)| {
                // SAFETY: Every out lane has length n and i < n since i comes from iterating over x's slices.
                *unsafe { v.get_unchecked_mut(i) } = x;
            })
        });
    } else {
        x.iter().take(n).enumerate().for_each(|(i, x)| {
            x.into_iter()
                .cloned()
                .zip(out.iter_mut())
                .for_each(|(x, v)| {
                    // SAFETY: Same as above (slices branch).
                    *unsafe { v.get_unchecked_mut(i) } = x;
                });
        });
    }
}

/// Copy elements of a flat slice into a [`StridedSliceRef`] lane.
#[inline]
pub fn clone_slice_to_strided<T: Clone>(x: &[T], out: &mut StridedSliceRef<T>) {
    if let Ok(out) = out.try_into() {
        clone_slice(x, out);
    } else {
        x.iter()
            .cloned()
            .zip(out.iter_mut())
            .for_each(|(a, b)| *b = a);
    }
}

/// Copy `N` aligned vecs into a [`ChunkStridedSliceRef`] lane.
///
/// `x[j][i]` is written to position `(i, j)` of `out`.  Writes exactly `x[0].len()` positions.
#[inline]
pub fn clone_avecs_to_strided_chunk<T: Clone, const N: usize, A: aligned_vec::Alignment>(
    x: &[AVec<T, A>; N],
    out: &mut ChunkStridedSliceRef<T, N>,
) {
    static_assert_positive!(N);
    let n = x[0].len();
    assert!(
        x.iter().all(|v| v.len() == n),
        "all input AVecs must have the same length"
    );
    assert!(
        n <= out.len(),
        "invalid lengths for strided chunk clone, input: {}, output: {}",
        n,
        out.len()
    );
    if let Ok(out) = out.try_array_chunks_mut() {
        out.enumerate().take(n).for_each(|(i, out)| {
            out.iter_mut().zip(x.iter()).for_each(|(x, v)| {
                // SAFETY: Every out lane has length n and i < n since i comes from iterating over x's slices.
                *x = unsafe { v.get_unchecked(i) }.clone();
            })
        });
    } else {
        out.iter_mut().enumerate().take(n).for_each(|(i, out)| {
            out.into_iter().zip(x.iter()).for_each(|(x, v)| {
                // SAFETY: Same as above (slices branch).
                *x = unsafe { v.get_unchecked(i) }.clone();
            });
        });
    }
}

/// creates an array of slices from an array of aligned vectors ['AVec']
#[inline(always)]
pub fn avecs_to_slices<T, const N: usize, A: aligned_vec::Alignment>(
    x: &[AVec<T, A>; N],
) -> [&[T]; N] {
    x.iter().map(|v| v.as_slice()).collect_array().unwrap()
}

/// creates an array of slices from an array of aligned vectors ['AVec']
#[inline(always)]
pub fn avecs_to_mut_slices<T, const N: usize, A: aligned_vec::Alignment>(
    x: &mut [AVec<T, A>; N],
) -> [&mut [T]; N] {
    x.iter_mut()
        .map(|v| v.as_mut_slice())
        .collect_array()
        .unwrap()
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
    use crate::iter::strided_slice::{StridedSlice, StridedSliceMut};
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

    #[rstest]
    fn test_clone_slice(#[values(10, 12)] n0: usize, #[values(10, 12)] n1: usize) {
        let inp = (1..n0 + 1).collect_vec();
        let mut out = vec![0; n1];

        let n_min = n0.min(n1);
        clone_slice(&inp, &mut out);
        assert_eq!(out[..n_min], inp[..n_min]);
    }

    #[rstest]
    fn test_clone_slice_to_strided(
        #[values(10, 11, 12)] n0: usize,
        #[values(10, 11, 12)] n1: usize,
        #[values(1, 2, 3)] stride: usize,
    ) {
        let inp = (1..n0 + 1).collect_vec();
        let mut out = vec![0; n1 * stride];

        let n_min = n0.min(n1);
        let mut out_strided = StridedSliceMut::from_mut_slice(&mut out, stride);
        clone_slice_to_strided(&inp, &mut out_strided);

        let output = out.iter().step_by(stride).cloned().collect_vec();

        assert_eq!(output[..n_min], inp[..n_min]);
    }

    #[rstest]
    fn test_clone_slice_to_strided_chunk(
        #[values(10, 11)] n0: usize,
        #[values(10, 11)] n1: usize,
        #[values(10, 11)] m0: usize,
        #[values(0, 1)] ax: usize,
    ) {
        const N: usize = 4;

        let n_shape = [n0, n1];

        let m_shape = if ax == 0 { [m0, n1] } else { [n1, m0] };
        let m_total: usize = m_shape.iter().product();

        let n_ax = n_shape[ax].min(m_shape[ax]);

        let mut out = vec![0; m_total];

        let mut chunks = out.iter_lane_chunks_mut::<N>(&m_shape, ax);
        let n_c = chunks.len() * N;
        chunks.by_ref().enumerate().for_each(|(i, mut chunk)| {
            let vecs = core::array::from_fn(|j| {
                let start = i * N + j + 1;
                let end = start + n_ax;
                AVec::<_>::from_iter(128, start..end)
            });
            clone_avecs_to_strided_chunk(&vecs, &mut chunk);
        });
        chunks.remainder().enumerate().for_each(|(i, mut slice)| {
            let start = n_c + i + 1;
            let end = start + n_ax;
            let vec = (start..end).collect_vec();
            clone_slice_to_strided(&vec, &mut slice);
        });

        out.iter_lanes(&m_shape, ax)
            .enumerate()
            .for_each(|(i, slice)| {
                let start = i + 1;
                let end = start + n_ax;
                let expected = (start..end).collect_vec();
                let actual = slice.iter().take(n_ax).cloned().collect_vec();
                assert_eq!(actual, expected);
            })
    }

    #[rstest]
    fn test_clone_strided_to_slice(
        #[values(10, 11, 12)] n0: usize,
        #[values(10, 11, 12)] n1: usize,
        #[values(1, 2, 3)] stride: usize,
    ) {
        let inp = (1..n0 * stride + 1).collect_vec();
        let inp_strided = StridedSlice::from_slice(&inp, stride);
        let mut out = vec![0; n1];

        clone_strided_to_slice(&inp_strided, &mut out);

        let n_min = n0.min(n1);

        let out_ref = inp.iter().step_by(stride).cloned().collect_vec();

        assert_eq!(out[..n_min], out_ref[..n_min]);
    }

    #[rstest]
    fn test_clone_strided_chunk_to_slice(
        #[values(10, 11)] n0: usize,
        #[values(10, 11)] n1: usize,
        #[values(10, 11)] m0: usize,
        #[values(0, 1)] ax: usize,
    ) {
        const N: usize = 4;

        let n_shape = [n0, n1];

        let m_shape = if ax == 0 { [m0, n1] } else { [n1, m0] };
        let m_total: usize = m_shape.iter().product();

        let mut inp = vec![0; m_total];

        inp.iter_lanes_mut(&m_shape, ax)
            .enumerate()
            .for_each(|(i, mut slice)| {
                let start = i + 1;
                let end = start + m_shape[ax];
                slice.iter_mut().zip(start..end).for_each(|(v, i)| {
                    *v = i;
                });
            });
        let n_lanes = if ax == 0 { m_shape[1] } else { m_shape[0] };

        let n_ax = n_shape[ax].min(m_shape[ax]);

        let chunks = inp.iter_lane_chunks::<N>(&m_shape, ax);
        let rem = chunks.remainder();
        let n_c = chunks.len();

        chunks.enumerate().for_each(|(i, chunk)| {
            let mut vecs = core::array::from_fn(|_| avec![0;n_ax]);
            clone_strided_chunk_to_avecs(&chunk, &mut vecs);

            vecs.into_iter().enumerate().for_each(|(j, vec)| {
                let start = i * N + j + 1;
                let end = start + n_ax;
                assert_eq!(vec.as_slice(), &(start..end).collect_vec());
            })
        });

        (n_c * N..n_lanes).zip(rem).for_each(|(i, slice)| {
            let mut vec = vec![0; n_ax];
            clone_strided_to_slice(&slice, &mut vec);

            let start = i + 1;
            let end = start + n_ax;
            assert_eq!(vec, (start..end).collect_vec());
        });
    }

    #[rstest]
    fn test_stack_slices_to_outer(#[values(10, 11)] n0: usize, #[values(0, 2, 5)] n_pad: usize) {
        let inp = (1..n0 + 1).collect_vec();
        let n1 = n0 + n_pad;
        let mut out = vec![0; n1];

        let nf = (n0 + 1) / 2;
        let ns = n0 / 2;

        let (in_1, in_2) = inp.split_at(nf);

        assert_eq!(in_1.len(), nf);
        assert_eq!(in_2.len(), ns);

        stack(&in_1, &in_2, &mut out);

        assert_eq!(&out[..nf], in_1);
        assert_eq!(&out[n1 - ns..], in_2);
    }

    #[rstest]
    fn test_stack_slices_to_outer_strided(
        #[values(10, 11)] n0: usize,
        #[values(0, 2, 5)] n_pad: usize,
        #[values(1, 2, 3)] stride: usize,
    ) {
        let inp = (1..n0 + 1).collect_vec();
        let n1 = n0 + n_pad;
        let mut out = vec![0; n1 * stride];

        let nf = (n0 + 1) / 2;
        let ns = n0 / 2;

        let (in_1, in_2) = inp.split_at(nf);

        assert_eq!(in_1.len(), nf);
        assert_eq!(in_2.len(), ns);

        let mut out_strided = StridedSliceMut::from_mut_slice(&mut out, stride);
        stack_to_strided(&in_1, &in_2, &mut out_strided);

        let out_f = out.iter().step_by(stride).take(nf).cloned().collect_vec();
        let out_s = out
            .iter()
            .step_by(stride)
            .skip(n1 - ns)
            .cloned()
            .collect_vec();
        assert_eq!(&out_f, in_1);
        assert_eq!(&out_s, in_2);
    }

    #[rstest]
    fn test_stack_slice_to_outer_strided_chunk(
        #[values(10, 11)] n0: usize,
        #[values(10, 11)] n1: usize,
        #[values(0, 2, 5)] n_pad: usize,
        #[values(0, 1)] ax: usize,
    ) {
        const N: usize = 4;

        let n_shape = [n0, n1];

        let m_shape = if ax == 0 {
            [n0 + n_pad, n1]
        } else {
            [n0, n1 + n_pad]
        };
        let m_total: usize = m_shape.iter().product();

        let n_ax = n_shape[ax];
        let nf = (n_ax + 1) / 2;
        let ns = n_ax / 2;

        let mut out = vec![0; m_total];

        let mut chunks = out.iter_lane_chunks_mut::<N>(&m_shape, ax);
        let n_c = chunks.len() * N;
        chunks.by_ref().enumerate().for_each(|(i, mut chunk)| {
            let vecs_f = core::array::from_fn(|j| {
                let start = i * N + j + 1;
                let end = start + nf;
                AVec::<_>::from_iter(128, start..end)
            });
            let vecs_s = core::array::from_fn(|j| {
                let start = i * N + j + 1 + nf;
                let end = start + ns;
                AVec::<_>::from_iter(128, start..end)
            });
            stack_to_strided_chunk(&vecs_f, &vecs_s, &mut chunk);
        });
        chunks.remainder().enumerate().for_each(|(i, mut slice)| {
            let start = n_c + i + 1;
            let end = start + n_ax;
            let vec = (start..end).collect_vec();
            let (vf, vs) = vec.split_at(nf);
            stack_to_strided(vf, vs, &mut slice);
        });

        out.iter_lanes(&m_shape, ax)
            .enumerate()
            .for_each(|(i, slice)| {
                let start = i + 1;
                let end = start + n_ax;
                let expected = (start..end).collect_vec();

                let a_f = slice.iter().take(nf).cloned();
                let a_s = slice.iter().skip(nf + n_pad).cloned();
                assert_eq!(a_f.len(), nf);
                assert_eq!(a_s.len(), ns);

                let actual = a_f.chain(a_s).collect_vec();
                assert_eq!(actual, expected);
            })
    }

    #[rstest]
    fn test_split_outer_to_slices(#[values(10, 11)] n0: usize, #[values(0, 2, 5)] n_pad: usize) {
        let n1 = n0 + n_pad;
        let inp = (1..n1 + 1).collect_vec();
        let mut out = vec![0; n0];

        let nf = (n0 + 1) / 2;
        let ns = n0 / 2;

        let (out_f, out_s) = out.split_at_mut(nf);

        assert_eq!(out_f.len(), nf);
        assert_eq!(out_s.len(), ns);

        split(&inp, out_f, out_s);

        assert_eq!(out_f, &inp[..nf]);
        assert_eq!(out_s, &inp[n1 - ns..]);
    }

    #[rstest]
    fn test_split_strided_outer_to_slices(
        #[values(10, 11)] n0: usize,
        #[values(0, 2, 5)] n_pad: usize,
        #[values(1, 2, 3)] stride: usize,
    ) {
        let n1 = n0 + n_pad;
        let inp = (1..n1 * stride + 1).collect_vec();
        let mut out = vec![0; n0];

        let nf = (n0 + 1) / 2;
        let ns = n0 / 2;

        let (out_f, out_s) = out.split_at_mut(nf);

        assert_eq!(out_f.len(), nf);
        assert_eq!(out_s.len(), ns);

        let inp_strided = StridedSlice::from_slice(&inp, stride);
        split_strided(&inp_strided, out_f, out_s);

        let inp_f = inp.iter().step_by(stride).take(nf).cloned().collect_vec();
        let inp_s = inp
            .iter()
            .step_by(stride)
            .skip(n1 - ns)
            .cloned()
            .collect_vec();
        assert_eq!(out_f, &inp_f);
        assert_eq!(out_s, &inp_s);
    }

    #[rstest]
    fn test_split_strided_chunk_outer_to_slices(
        #[values(10, 11)] n0: usize,
        #[values(10, 11)] n1: usize,
        #[values(0, 2, 5)] n_pad: usize,
        #[values(0, 1)] ax: usize,
    ) {
        const N: usize = 4;

        let n_shape = [n0, n1];

        let m_shape = if ax == 0 {
            [n0 + n_pad, n1]
        } else {
            [n0, n1 + n_pad]
        };
        let m_total: usize = m_shape.iter().product();

        let mut inp = vec![0; m_total];

        let m_ax = m_shape[ax];
        inp.iter_lanes_mut(&m_shape, ax)
            .enumerate()
            .for_each(|(i, mut slice)| {
                let start = i + 1;
                let end = start + m_ax;
                assert_eq!(slice.len(), m_ax);
                slice.iter_mut().zip(start..end).for_each(|(v, i)| {
                    *v = i;
                });
            });

        let chunks = inp.iter_lane_chunks::<N>(&m_shape, ax);
        let rem = chunks.remainder();

        let n_lanes = if ax == 0 { m_shape[1] } else { m_shape[0] };
        let n_c = chunks.len();

        let n_ax = n_shape[ax];
        let nf = (n_ax + 1) / 2;
        let ns = n_ax / 2;

        chunks.enumerate().for_each(|(i, chunk)| {
            let mut vecs_f = core::array::from_fn(|_| avec![0; nf]);
            let mut vecs_s = core::array::from_fn(|_| avec![0; ns]);

            split_strided_chunk(&chunk, &mut vecs_f, &mut vecs_s);

            vecs_f
                .into_iter()
                .zip(vecs_s.into_iter())
                .enumerate()
                .for_each(|(j, (vf, vs))| {
                    let start = i * N + j + 1;
                    let end = start + nf;
                    assert_eq!(vf.as_slice(), &(start..end).collect_vec());
                    let start = end + n_pad;
                    let end = start + ns;
                    assert_eq!(vs.as_slice(), &(start..end).collect_vec());
                });
        });

        (n_c * N..n_lanes).zip(rem).for_each(|(i, slice)| {
            let mut vec_f = vec![0; nf];
            let mut vec_s = vec![0; ns];
            split_strided(&slice, &mut vec_f, &mut vec_s);

            let start = i + 1;
            let end = start + nf;
            assert_eq!(vec_f, (start..end).collect_vec());

            let start = end + n_pad;
            let end = start + ns;
            assert_eq!(vec_s, (start..end).collect_vec());
        });
    }
}
