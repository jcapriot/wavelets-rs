//! Array layout utilities for wavelet sub-band manipulation.
//!
//! This module provides helpers for the even/odd index splitting (deinterleave) and
//! merging (interleave/stack) that the lifting transform requires, as well as strided
//! variants used for N-D axis traversal.

use itertools::izip;

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

/// Split a slice into its even-indexed and odd-indexed elements.
///
/// `evens[i] = x[2*i]` and `odds[i] = x[2*i + 1]`.  For odd-length `x` the extra
/// element goes into `evens`, so `evens.len() == (x.len() + 1) / 2` and
/// `odds.len() == x.len() / 2`.
///
/// # Panics
///
/// Panics if `odds.len() != x.len() / 2` or `evens.len() != (x.len() + 1) / 2`.
#[inline]
#[track_caller]
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
///
/// # Panics
///
/// Panics if `input.len()` or `output.len()` is not equal to `shape[0] * shape[1]`.
#[inline]
#[track_caller]
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
///
/// # Panics
///
/// Panics if `input.len()` is not equal to `shape.iter().product()` (for N ≥ 3),
/// or with the same constraints as [`deinterleave`] for 1-D or [`deinterleave_2d`] for 2-D.
#[inline]
#[track_caller]
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

/// Interleave even- and odd-indexed elements back into a single flat slice.
///
/// Inverse of [`deinterleave`]: `x[2*i] = evens[i]`, `x[2*i+1] = odds[i]`.
///
/// # Panics
///
/// Panics if `odds.len() != x.len() / 2` or `evens.len() != (x.len() + 1) / 2`.
#[inline]
#[track_caller]
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

/// In-place interleave of a split slice: rearranges elements without extra allocation.
///
/// The first `ceil(n/2)` elements are treated as even-indexed and the remainder as
/// odd-indexed.  After the call `x[2*i] == old_x[i]` and `x[2*i+1] == old_x[ceil(n/2) + i]`.
/// Inverse of [`deinterleave`] applied in-place.
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
    use itertools::Itertools;

    use rstest::rstest;

    #[rstest]
    fn test_interleave_inplace(#[values(20, 21, 22, 32, 423, 553)] n: usize) {
        let evens = (0..n).step_by(2).collect_vec();
        let odds = (1..n).step_by(2).collect_vec();

        let mut x = evens.iter().chain(odds.iter()).cloned().collect::<Vec<_>>();

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
}
