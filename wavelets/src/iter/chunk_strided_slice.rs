//! Chunked Strided slice views and lane iterators over flat multi-dimensional arrays.
//!
//! This module provides [`ChunkStridedSliceRef`] — a lightweight non-owning view into a
//! fixed number of strided regions of memory — and the concrete chunked iterator types returned by
//! [`super::LanesIterator`] and [`super::parallel::LanesParallelIterator`].

use num_traits::Zero;

use super::strided_slice::{IterLanes, IterLanesMut};
use super::*;
use crate::utils::stride_from_shape;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

/// Raw backing storage for a chunk-strided slice of `N` parallel lanes.
struct ChunkStrideParts<T, const N: usize> {
    base: NonNull<T>,
    offsets: [isize; N],
    length: usize,
    stride: isize,
}

/// `N`-lane parallel strided slice view parameterised by a lifetime marker `L`.
///
/// This is used to traverse `N` lanes simultaneously, enabling SIMD gather/scatter
/// operations over non-contiguous memory layouts.
pub struct ChunkStridedSliceBase<L, const N: usize, T = <L as Data>::Elem>
where
    L: Data<Elem = T>,
{
    parts: ChunkStrideParts<T, N>,
    _member: SliceLifetime<L>,
}

/// Read-only `N`-lane parallel strided view with an explicit lifetime.
pub type ChunkStridedSlice<'a, T, const N: usize> =
    ChunkStridedSliceBase<SliceLifetime<&'a T>, N, T>;
/// Mutable `N`-lane parallel strided view with an explicit lifetime.
pub type ChunkStridedSliceMut<'a, T, const N: usize> =
    ChunkStridedSliceBase<SliceLifetime<&'a mut T>, N, T>;

impl<'a, T, const N: usize> ChunkStridedSlice<'a, T, N> {
    /// Create an `N`-lane strided view for the `ind`-th group of lanes along axis `ax`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is empty, `shape.iter().product() != slice.len()`, `ax >= shape.len()`,
    /// or `ind + N > shape.iter().product::<usize>() / shape[ax]` (insufficient lanes remaining).
    #[track_caller]
    pub fn from_slice(slice: &'a [T], shape: &[usize], ax: usize, ind: usize) -> Self {
        assert_ne!(slice.len(), 0);
        assert_eq!(shape.iter().product::<usize>(), slice.len());
        assert!(ax < shape.len());

        let mut strides = stride_from_shape(shape);
        let stride = strides.remove(ax);
        let mut shape = shape.to_owned();
        let length = shape.remove(ax);

        // gauranteed that maximum linear index will not go past the leftover shapes.
        assert!(ind + N <= shape.iter().product());

        let offsets = core::array::from_fn(|i| {
            let dim_inds = unravel(ind + i, &shape);
            dim_inds
                .into_iter()
                .zip(&strides)
                .fold(0, |acc, vs| acc + vs.0 * vs.1) as isize
        });

        // SAFETY: slice length > 0 so ptr is non-null.
        Self {
            parts: ChunkStrideParts {
                base: unsafe { NonNull::new_unchecked(slice.as_ptr() as *mut T) },
                offsets,
                length,
                stride: stride as isize,
            },
            _member: SliceLifetime {
                _member: PhantomData,
            },
        }
    }
}

impl<'a, T, const N: usize> ChunkStridedSliceMut<'a, T, N> {
    /// Create a mutable `N`-lane strided view for the `ind`-th group of lanes along axis `ax`.
    ///
    /// # Panics
    ///
    /// Panics if `slice` is empty, `shape.iter().product() != slice.len()`, `ax >= shape.len()`,
    /// or `ind + N > shape.iter().product::<usize>() / shape[ax]` (insufficient lanes remaining).
    #[track_caller]
    pub fn from_mut_slice(slice: &'a mut [T], shape: &[usize], ax: usize, ind: usize) -> Self {
        assert_ne!(slice.len(), 0);
        assert_eq!(shape.iter().cloned().product::<usize>(), slice.len());
        assert!(ax < shape.len());

        let mut strides = stride_from_shape(shape);
        let stride = strides.remove(ax);
        let mut shape = shape.to_owned();
        let length = shape.remove(ax);

        // gauranteed that maximum linear index will not go past the leftover shapes.
        assert!(ind + N <= shape.iter().cloned().product());

        let offsets = core::array::from_fn(|i| {
            let dim_inds = unravel(ind + i, &shape);
            dim_inds
                .into_iter()
                .zip(&strides)
                .fold(0, |acc, vs| acc + vs.0 * vs.1) as isize
        });

        // SAFETY: slice length > 0 so ptr is non-null.
        Self {
            parts: ChunkStrideParts {
                base: unsafe { NonNull::new_unchecked(slice.as_ptr() as *mut T) },
                offsets,
                length,
                stride: stride as isize,
            },
            _member: SliceLifetime {
                _member: PhantomData,
            },
        }
    }
}

/// Non-owning strided view over `N` interleaved lanes in a flat buffer.
///
/// `ChunkStridedSliceRef<T, N>` presents `N` parallel lanes of `T` values that step
/// through memory with a common `stride`.  It is the `N`-lane analogue of
/// [`StridedSliceRef`]: element `(i, j)` lives at `base + i * stride + offsets[j]`.
///
/// This type is `#[repr(transparent)]` over its internal `ChunkStrideParts`, so it
/// can be safely constructed via pointer casts inside [`ChunkStridedSliceBase`].
#[repr(transparent)]
pub struct ChunkStridedSliceRef<T, const N: usize>(ChunkStrideParts<T, N>);

impl<T, const N: usize> ChunkStridedSliceRef<T, N> {
    /// Return a const pointer to the base element of this view.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.0.base.as_ptr()
    }

    /// Return a mut pointer to the base element of this view.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.0.base.as_ptr()
    }

    /// Number of positions (rows) in the view.  Each position holds `N` elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.0.length
    }

    /// Whether or not this chunk of strided slices has any elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.length == 0
    }

    /// Return a reference to element `(i0, i1)`, or `None` if either index is out of bounds.
    #[inline]
    pub fn get(&self, (i0, i1): (usize, usize)) -> Option<&T> {
        if (i0 >= self.0.length) || (i1 >= N) {
            None
        } else {
            Some(unsafe { self.get_unchecked((i0, i1)) })
        }
    }

    /// Return a reference to element `(i0, i1)` without bounds checking.
    ///
    /// # Safety
    /// `i0 < self.len()` and `i1 < N` must both hold.
    #[inline]
    pub unsafe fn get_unchecked(&self, (i0, i1): (usize, usize)) -> &T {
        // SAFETY: Caller gaurantees that i0 is less than the slice length, and i1 is less than the chunk size N

        unsafe {
            &*self
                .as_ptr()
                .offset(i0 as isize * self.0.stride + self.0.offsets[i1])
        }
    }

    /// Return a mutable reference to element `(i0, i1)`, or `None` if either index is out of bounds.
    #[inline]
    pub fn get_mut(&mut self, (i0, i1): (usize, usize)) -> Option<&mut T> {
        if (i0 >= self.0.length) || (i1 >= N) {
            None
        } else {
            Some(unsafe { self.get_unchecked_mut((i0, i1)) })
        }
    }

    /// Return a mutable reference to element `(i0, i1)` without bounds checking.
    ///
    /// # Safety
    /// `i0 < self.len()` and `i1 < N` must both hold.
    #[inline]
    pub unsafe fn get_unchecked_mut(&mut self, (i0, i1): (usize, usize)) -> &mut T {
        unsafe {
            &mut *self
                .as_mut_ptr()
                .offset(i0 as isize * self.0.stride + self.0.offsets[i1])
        }
    }

    /// Return `true` if the `N` elements at each position are stored at consecutive addresses.
    #[inline(always)]
    pub fn is_chunk_contiguous(&self) -> bool {
        self.0.offsets.windows(2).all(|v| v[0] + 1 == v[1])
    }

    /// Return `true` if both the inter-position stride equals `N` and the chunk is contiguous,
    /// meaning the entire view is a flat `&[[T; N]]`.
    #[inline(always)]
    pub fn is_contiguous(&self) -> bool {
        (self.0.stride == N as isize) && self.is_chunk_contiguous()
    }

    /// Return an iterator that yields `N`-element arrays at each position.
    #[inline]
    pub fn iter(&self) -> Iter<'_, T, N> {
        let start = self.0.base;
        let end = unsafe {
            start
                .as_ptr()
                .offset(self.0.stride * self.0.length as isize)
        };
        Iter {
            ptr: start,
            end,
            offsets: self.0.offsets,
            stride: self.0.stride,
            _member: PhantomData,
        }
    }

    /// Return an iterator over `&[T; N]` slices if the `N` elements at each position are contiguous,
    /// or an error if they are not.
    #[inline]
    pub fn try_array_chunks(&self) -> Result<ArrayChunks<'_, T, N>, &'static str> {
        if self.is_chunk_contiguous() {
            let start = unsafe { self.0.base.offset(self.0.offsets[0]) };
            let end = unsafe {
                start
                    .as_ptr()
                    .offset(self.0.stride * self.0.length as isize)
            };
            Ok(ArrayChunks {
                ptr: start,
                end,
                stride: self.0.stride,
                _member: PhantomData,
            })
        } else {
            Err("Cannot create ArrayChunks iterator because the chunk stride is not contiguous")
        }
    }

    /// Return a mutable iterator that yields `N`-element arrays at each position.
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, T, N> {
        let start = self.0.base;
        let end = unsafe {
            start
                .as_ptr()
                .offset(self.0.stride * self.0.length as isize)
        };
        IterMut {
            ptr: start,
            end,
            offsets: self.0.offsets,
            stride: self.0.stride,
            _member: PhantomData,
        }
    }

    /// Return a mutable iterator over `&mut [T; N]` slices if the `N` elements at each position are
    /// contiguous, or an error if they are not.
    #[inline]
    pub fn try_array_chunks_mut(&mut self) -> Result<ArrayChunksMut<'_, T, N>, &'static str> {
        if self.is_chunk_contiguous() {
            let start = unsafe { self.0.base.offset(self.0.offsets[0]) };
            let end = unsafe {
                start
                    .as_ptr()
                    .offset(self.0.stride * self.0.length as isize)
            };
            Ok(ArrayChunksMut {
                ptr: start,
                end,
                stride: self.0.stride,
                _member: PhantomData,
            })
        } else {
            Err("Cannot create ArrayChunks iterator because the chunk stride is not contiguous")
        }
    }
}

impl<T: Clone, const N: usize> ChunkStridedSliceRef<T, N> {
    /// Deinterleave `N` strided slices into `N` slices
    ///
    /// `self` presents `N` interleaved lanes; `evens[j]` and `odds[j]` receive the even- and
    /// odd-indexed elements of lane `j`, respectively.
    ///
    /// # Panics
    ///
    /// Panics if any `evens[j].len() != (self.len() + 1) / 2` or `odds[j].len() != self.len() / 2`.
    #[inline(always)]
    #[track_caller]
    pub fn deinterleave<V>(&self, evens: &mut [V; N], odds: &mut [V; N])
    where
        V: DerefMut<Target = [T]>,
    {
        if const { N == 0 } {
            return;
        }
        let ne = self.len().div_ceil(2);
        let no = self.len() / 2;
        assert!(evens.iter().all(|v| v.len() == ne));
        assert!(odds.iter().all(|v| v.len() == no));

        let mut i = 0;
        if let Ok(mut x_iter) = self.try_array_chunks() {
            while let Some([xes, xos]) = x_iter.next_chunk::<2>() {
                (0..N)
                    .zip(xes)
                    // SAFETY: Lengths checked above to be valid.
                    .for_each(|(j, xe)| unsafe { *evens[j].get_unchecked_mut(i) = xe.clone() });
                (0..N)
                    .zip(xos)
                    // SAFETY: Lengths checked above to be valid.
                    .for_each(|(j, xo)| unsafe { *odds[j].get_unchecked_mut(i) = xo.clone() });
                i += 1;
            }
            if let Some(xes) = x_iter.next() {
                (0..N)
                    .zip(xes)
                    // SAFETY: Lengths checked above to be valid.
                    .for_each(|(j, xe)| unsafe { *evens[j].get_unchecked_mut(i) = xe.clone() });
            }
        } else {
            let mut x_iter = self.iter();

            while let Some([xes, xos]) = x_iter.next_chunk::<2>() {
                (0..N)
                    .zip(xes)
                    .for_each(|(j, xe)| unsafe { *evens[j].get_unchecked_mut(i) = xe.clone() });
                (0..N)
                    .zip(xos)
                    .for_each(|(j, xo)| unsafe { *odds[j].get_unchecked_mut(i) = xo.clone() });
                i += 1;
            }
            if let Some(xes) = x_iter.next() {
                (0..N)
                    .zip(xes)
                    // SAFETY: Lengths checked above to be valid.
                    .for_each(|(j, xe)| unsafe { *evens[j].get_unchecked_mut(i) = xe.clone() });
            }
        }
    }

    /// Deinterleave into a slice of arrays.
    ///
    /// # Panics
    ///
    /// Panics if `evens.len() != (self.len() + 1) / 2` or `odds.len() != self.len() / 2`.
    #[inline(always)]
    #[track_caller]
    pub fn deinterleave_arrays(&self, evens: &mut [[T; N]], odds: &mut [[T; N]]) {
        if const { N == 0 } {
            return;
        }
        let ne = self.len().div_ceil(2);
        let no = self.len() / 2;
        assert!(evens.len() == ne);
        assert!(odds.len() == no);

        let mut eo_iter = evens.iter_mut().zip(odds);
        if let Ok(mut x_iter) = self.try_array_chunks() {
            while let (Some([xes, xos]), Some((es, os))) =
                (x_iter.next_chunk::<2>(), eo_iter.next())
            {
                *es = xes.clone();
                *os = xos.clone();
            }
            if let Some(xes) = x_iter.next() {
                *evens.last_mut().unwrap() = xes.clone();
            }
        } else {
            let mut x_iter = self.iter();

            while let (Some([xes, xos]), Some((es, os))) =
                (x_iter.next_chunk::<2>(), eo_iter.next())
            {
                es.iter_mut().zip(xes).for_each(|(e, x)| *e = x.clone());
                os.iter_mut().zip(xos).for_each(|(o, x)| *o = x.clone());
            }
            if let Some(xes) = x_iter.next() {
                evens
                    .last_mut()
                    .unwrap()
                    .iter_mut()
                    .zip(xes)
                    .for_each(|(e, x)| *e = x.clone());
            }
        }
    }

    /// `N` slices are interleaved into `N` strided slices.
    ///
    /// `self` presents `N` lanes; `evens[j]` and `odds[j]` are the source of the even- and
    /// odd-indexed elements of lane `j`, respectively.
    ///
    /// # Panics
    ///
    /// Panics if any `evens[j].len() != (self.len() + 1) / 2` or `odds[j].len() != self.len() / 2`.
    #[inline(always)]
    #[track_caller]
    pub fn interleave<V>(&mut self, evens: &[V; N], odds: &[V; N])
    where
        V: Deref<Target = [T]>,
    {
        if const { N == 0 } {
            return;
        }
        let n = self.len();
        let ne = n.div_ceil(2);
        let no = n / 2;
        debug_assert_eq!(n, ne + no);
        assert!(evens.iter().all(|v| v.len() == ne));
        assert!(odds.iter().all(|v| v.len() == no));

        let mut i = 0;
        if let Ok(mut x_iter) = self.try_array_chunks_mut() {
            while let Some([xes, xos]) = x_iter.next_chunk::<2>() {
                (0..N)
                    .zip(xes)
                    // SAFETY: Lengths checked above to be valid.
                    .for_each(|(j, xe)| unsafe { *xe = evens[j].get_unchecked(i).clone() });
                (0..N)
                    .zip(xos)
                    // SAFETY: Lengths checked above to be valid.
                    .for_each(|(j, xo)| unsafe { *xo = odds[j].get_unchecked(i).clone() });
                i += 1;
            }
            if let Some(xes) = x_iter.next() {
                (0..N)
                    .zip(xes)
                    // SAFETY: Lengths checked above to be valid.
                    .for_each(|(j, xe)| unsafe { *xe = evens[j].get_unchecked(i).clone() });
            }
        } else {
            let mut x_iter = self.iter_mut();

            while let Some([xes, xos]) = x_iter.next_chunk::<2>() {
                (0..N)
                    .zip(xes)
                    // SAFETY: Lengths checked above to be valid.
                    .for_each(|(j, xe)| unsafe { *xe = evens[j].get_unchecked(i).clone() });
                (0..N)
                    .zip(xos)
                    // SAFETY: Lengths checked above to be valid.
                    .for_each(|(j, xo)| unsafe { *xo = odds[j].get_unchecked(i).clone() });
                i += 1;
            }
            if let Some(xes) = x_iter.next() {
                (0..N)
                    .zip(xes)
                    // SAFETY: Lengths checked above to be valid.
                    .for_each(|(j, xe)| unsafe { *xe = evens[j].get_unchecked(i).clone() });
            }
        }
    }

    /// `N` slices are interleaved into `N` strided slices.
    ///
    /// # Panics
    ///
    /// Panics if `evens.len() != (self.len() + 1) / 2` or `odds.len() != self.len() / 2`.
    #[inline(always)]
    #[track_caller]
    pub fn interleave_arrays(&mut self, evens: &[[T; N]], odds: &[[T; N]]) {
        if const { N == 0 } {
            return;
        }
        let n = self.len();
        let ne = n.div_ceil(2);
        let no = n / 2;
        debug_assert_eq!(n, ne + no);
        assert_eq!(evens.len(), ne);
        assert_eq!(odds.len(), no);

        let mut eo_iter = evens.iter().zip(odds);
        if let Ok(mut x_iter) = self.try_array_chunks_mut() {
            while let (Some([xes, xos]), Some((es, os))) =
                (x_iter.next_chunk::<2>(), eo_iter.next())
            {
                *xes = es.clone();
                *xos = os.clone();
            }
            if let Some(xes) = x_iter.next() {
                *xes = evens.last().unwrap().clone();
            }
        } else {
            let mut x_iter = self.iter_mut();

            while let (Some([xes, xos]), Some((es, os))) =
                (x_iter.next_chunk::<2>(), eo_iter.next())
            {
                xes.into_iter().zip(es).for_each(|(x, e)| *x = e.clone());
                xos.into_iter().zip(os).for_each(|(x, o)| *x = o.clone());
            }
            if let Some(xes) = x_iter.next() {
                xes.into_iter()
                    .zip(evens.last().unwrap())
                    .for_each(|(x, e)| *x = e.clone());
            }
        }
    }

    /// Chunk-strided variant of [`StridedSliceRef::split`]: read `N` lanes from a [`ChunkStridedSliceRef`].
    ///
    /// Fills `first` and `second` with the head and tail ends, respectively, of the `N` strided lanes.
    ///
    /// # Panics
    ///
    /// Panics if `first[0].len() + second[0].len() > self.len()`, or if not all slices in `first`
    /// have the same length, or if not all slices in `second` have the same length.
    #[inline(always)]
    #[track_caller]
    pub fn split<V>(&self, first: &mut [V; N], second: &mut [V; N])
    where
        V: DerefMut<Target = [T]>,
    {
        if const { N == 0 } {
            return;
        }
        let nf = first[0].len();
        let ns = second[0].len();
        let nx = self.len();
        assert!(
            nf + ns <= nx,
            "invalid lengths for slice stack, first: {nf}, second: {ns}, third: {nx}",
        );
        assert!(first.iter().all(|v| v.len() == nf));
        assert!(second.iter().all(|v| v.len() == ns));

        let n_mid = nx - (nf + ns);

        if let Ok(mut x_iter) = self.try_array_chunks() {
            x_iter.by_ref().take(nf).enumerate().for_each(|(i, vs)| {
                (0..N)
                    .zip(vs)
                    // SAFETY: Lengths checked above to be valid.
                    .for_each(|(j, v)| unsafe { *first[j].get_unchecked_mut(i) = v.clone() });
            });
            x_iter.skip(n_mid).enumerate().for_each(|(i, vs)| {
                (0..N)
                    .zip(vs)
                    // SAFETY: Lengths checked above to be valid.
                    .for_each(|(j, v)| unsafe { *second[j].get_unchecked_mut(i) = v.clone() });
            });
        } else {
            let mut x_iter = self.iter();

            x_iter.by_ref().take(nf).enumerate().for_each(|(i, vs)| {
                (0..N)
                    .zip(vs)
                    // SAFETY: Lengths checked above to be valid.
                    .for_each(|(j, v)| unsafe { *first[j].get_unchecked_mut(i) = v.clone() });
            });
            x_iter.skip(n_mid).enumerate().for_each(|(i, vs)| {
                (0..N)
                    .zip(vs)
                    // SAFETY: Lengths checked above to be valid.
                    .for_each(|(j, v)| unsafe { *second[j].get_unchecked_mut(i) = v.clone() });
            });
        }
    }

    /// Chunk-strided variant of [`StridedSliceRef::split`]: read `N` interleaved lanes from a [`ChunkStridedSliceRef`].
    ///
    /// # Panics
    ///
    /// Panics if `first.len() + second.len() > self.len()`.
    #[inline(always)]
    #[track_caller]
    pub fn split_arrays(&self, first: &mut [[T; N]], second: &mut [[T; N]]) {
        if const { N == 0 } {
            return;
        }
        let nf = first.len();
        let ns = second.len();
        let nx = self.len();
        assert!(
            nf + ns <= nx,
            "invalid lengths for slice stack, first: {nf}, second: {ns}, third: {nx}",
        );

        let n_mid = nx - (nf + ns);

        if let Ok(mut x_iter) = self.try_array_chunks() {
            x_iter
                .by_ref()
                .take(nf)
                .zip(first)
                .for_each(|(xs, fs)| *fs = xs.clone());
            x_iter
                .skip(n_mid)
                .zip(second)
                .for_each(|(xs, ss)| *ss = xs.clone());
        } else {
            let mut x_iter = self.iter();

            x_iter
                .by_ref()
                .take(nf)
                .zip(first)
                .for_each(|(xs, fs)| xs.into_iter().zip(fs).for_each(|(x, f)| *f = x.clone()));
            x_iter
                .skip(n_mid)
                .zip(second)
                .for_each(|(xs, ss)| xs.into_iter().zip(ss).for_each(|(x, s)| *s = x.clone()));
        }
    }

    /// Fill the `N` slices `sink` with cloned elements of `self`.
    ///
    /// # Panics
    ///
    /// Panics if `sink[0].len() > self.len()` or if not all slices in `sink` have the same length.
    #[inline(always)]
    #[track_caller]
    pub fn pour_into<V>(&self, sink: &mut [V; N])
    where
        V: DerefMut<Target = [T]>,
    {
        if const { N == 0 } {
            return;
        }
        let n = self.len();
        let no = sink[0].len();
        assert!(
            no <= n,
            "Output slice with length {no} too long for strided slice with length {n}."
        );
        assert!(sink.iter().all(|v| v.len() == no));

        if let Ok(source) = self.try_array_chunks() {
            source.enumerate().take(no).for_each(|(i, vs)| {
                (0..N)
                    .zip(vs)
                    // SAFETY: Lengths checked above to be valid.
                    .for_each(|(j, v)| unsafe { *sink[j].get_unchecked_mut(i) = v.clone() })
            });
        } else {
            let source = self.iter();

            source.enumerate().take(no).for_each(|(i, vs)| {
                (0..N)
                    .zip(vs)
                    // SAFETY: Lengths checked above to be valid.
                    .for_each(|(j, v)| unsafe { *sink[j].get_unchecked_mut(i) = v.clone() })
            });
        }
    }
}

impl<T: Clone + Zero, const N: usize> ChunkStridedSliceRef<T, N> {
    /// Stack `N` slices into `N` strided slices
    ///
    /// This is the chunked version of [`StridedSliceRef::stack`]
    ///
    /// # Panics
    ///
    /// Panics if `first[0].len() + second[0].len() > self.len()`, or if not all slices in `first`
    /// have the same length, or if not all slices in `second` have the same length.
    #[inline(always)]
    #[track_caller]
    pub fn stack<V>(&mut self, first: &[V; N], second: &[V; N])
    where
        V: Deref<Target = [T]>,
    {
        if const { N == 0 } {
            return;
        }
        let nf = first[0].len();
        let ns = second[0].len();
        let no = self.len();

        assert!(first.iter().all(|v| v.len() == nf));
        assert!(second.iter().all(|v| v.len() == ns));
        assert!(
            nf + ns <= no,
            "invalid lengths for slice stack, first: {nf}, second: {ns}, third: {no}",
        );
        let n_mid = no - (nf + ns);

        if let Ok(mut x_iter) = self.try_array_chunks_mut() {
            x_iter.by_ref().take(nf).enumerate().for_each(|(i, vs)| {
                (0..N)
                    .zip(vs)
                    // SAFETY: Lengths checked above to be valid.
                    .for_each(|(j, v)| *v = unsafe { first[j].get_unchecked(i).clone() });
            });
            x_iter
                .by_ref()
                .take(n_mid)
                .for_each(|vs| vs.fill(T::zero()));
            x_iter.enumerate().for_each(|(i, vs)| {
                (0..N)
                    .zip(vs)
                    // SAFETY: Lengths checked above to be valid.
                    .for_each(|(j, v)| *v = unsafe { second[j].get_unchecked(i).clone() });
            });
        } else {
            let mut x_iter = self.iter_mut();

            x_iter.by_ref().take(nf).enumerate().for_each(|(i, vs)| {
                (0..N)
                    .zip(vs)
                    // SAFETY: Lengths checked above to be valid.
                    .for_each(|(j, v)| *v = unsafe { first[j].get_unchecked(i).clone() });
            });
            x_iter
                .by_ref()
                .take(n_mid)
                .for_each(|vs| vs.into_iter().for_each(|v| *v = T::zero()));
            x_iter.enumerate().for_each(|(i, vs)| {
                (0..N)
                    .zip(vs)
                    // SAFETY: Lengths checked above to be valid.s
                    .for_each(|(j, v)| *v = unsafe { second[j].get_unchecked(i).clone() });
            });
        }
    }

    /// Stack arrays of N elements into the strided slice.
    ///
    /// # Panics
    ///
    /// Panics if `first.len() + second.len() > self.len()`.
    #[inline(always)]
    #[track_caller]
    pub fn stack_arrays(&mut self, first: &[[T; N]], second: &[[T; N]]) {
        if const { N == 0 } {
            return;
        }
        let nf = first.len();
        let ns = second.len();
        let no = self.len();
        assert!(
            nf + ns <= no,
            "invalid lengths for slice stack, first: {nf}, second: {ns}, third: {no}",
        );
        let n_mid = no - (nf + ns);

        if let Ok(mut x_iter) = self.try_array_chunks_mut() {
            x_iter.by_ref().take(nf).zip(first).for_each(|(xs, fs)| {
                *xs = fs.clone();
            });
            x_iter
                .by_ref()
                .take(n_mid)
                .for_each(|vs| vs.fill(T::zero()));
            x_iter.zip(second).for_each(|(xs, ss)| {
                *xs = ss.clone();
            });
        } else {
            let mut x_iter = self.iter_mut();
            x_iter.by_ref().take(nf).zip(first).for_each(|(xs, fs)| {
                std::iter::zip(xs, fs).for_each(|(x, f)| *x = f.clone());
            });
            x_iter
                .by_ref()
                .take(n_mid)
                .for_each(|xs| xs.into_iter().for_each(|x| *x = T::zero()));
            x_iter.zip(second).for_each(|(xs, ss)| {
                std::iter::zip(xs, ss).for_each(|(x, s)| *x = s.clone());
            });
        }
    }

    /// Fill the `N` lanes of `self` with cloned elements from the `N` `source` slices, filling the leftover with zero values.
    ///
    /// # Panics
    ///
    /// Panics if `source[0].len() > self.len()` or if not all slices in `source` have the same length.
    #[inline(always)]
    #[track_caller]
    pub fn fill_from<V>(&mut self, source: &[V; N])
    where
        V: Deref<Target = [T]>,
    {
        if const { N == 0 } {
            return;
        }
        let n = self.len();
        let no = source[0].len();
        assert!(
            no <= n,
            "Output slice with length {no} too long for strided slice with length {n}."
        );
        assert!(source.iter().all(|v| v.len() == no));

        if let Ok(mut sink) = self.try_array_chunks_mut() {
            sink.by_ref().take(no).enumerate().for_each(|(i, vs)| {
                (0..N)
                    .zip(vs)
                    // SAFETY: Lengths checked above to be valid.
                    .for_each(|(j, v)| unsafe { *v = source[j].get_unchecked(i).clone() })
            });
            sink.for_each(|v| v.fill(T::zero()))
        } else {
            let mut sink = self.iter_mut();
            sink.by_ref().take(no).enumerate().for_each(|(i, vs)| {
                (0..N)
                    .zip(vs)
                    // SAFETY: Lengths checked above to be valid.s
                    .for_each(|(j, v)| unsafe { *v = source[j].get_unchecked(i).clone() })
            });
            sink.for_each(|v| v.into_iter().for_each(|v| *v = T::zero()));
        }
    }
}

impl<T, const N: usize> TryFrom<&ChunkStridedSliceRef<T, N>> for &[T] {
    type Error = &'static str;

    #[inline]
    fn try_from(value: &ChunkStridedSliceRef<T, N>) -> Result<Self, Self::Error> {
        if value.is_contiguous() {
            // SAFETY: I'm gauranteed to point to a contiguous set of values.
            let slc = unsafe { std::slice::from_raw_parts(value.as_ptr(), value.0.length * N) };
            // SAFETY: slc is gauranteed to be divisible by N, so there will be no remainder slice.
            Ok(slc)
        } else {
            Err("Cannot convert to &[T] because the strided chunk is not contiguous")
        }
    }
}

impl<T, const N: usize> TryFrom<&mut ChunkStridedSliceRef<T, N>> for &mut [T] {
    type Error = &'static str;

    #[inline]
    fn try_from(value: &mut ChunkStridedSliceRef<T, N>) -> Result<Self, Self::Error> {
        if value.is_contiguous() {
            // SAFETY: I'm gauranteed to point to a contiguous set of values.
            let slc =
                unsafe { std::slice::from_raw_parts_mut(value.as_mut_ptr(), value.0.length * N) };

            // SAFETY: slc is gauranteed to be divisible by N, so there will be no remainder slice.
            Ok(slc)
        } else {
            Err("Cannot convert to &mut [T] because the strided chunk is not contiguous")
        }
    }
}

impl<T, const N: usize> TryFrom<&ChunkStridedSliceRef<T, N>> for &[[T; N]] {
    type Error = &'static str;

    #[inline]
    fn try_from(value: &ChunkStridedSliceRef<T, N>) -> Result<Self, Self::Error> {
        let slc: &[T] = value.try_into()?;
        Ok(slc.as_chunks::<N>().0)
    }
}

impl<T, const N: usize> TryFrom<&mut ChunkStridedSliceRef<T, N>> for &mut [[T; N]] {
    type Error = &'static str;

    #[inline]
    fn try_from(value: &mut ChunkStridedSliceRef<T, N>) -> Result<Self, Self::Error> {
        let slc: &mut [T] = value.try_into()?;
        Ok(slc.as_chunks_mut::<N>().0)
    }
}

macro_rules! implement_chunk_strided_iter {
    ($name:ident -> $ptr:ty, $elem:ty, $into_ref:ident) => {
        /// Iterator over `N`-element chunks within a strided lane, yielding non-contiguous
        /// element references.
        pub struct $name<'a, T, const N: usize> {
            ptr: NonNull<T>,
            end: $ptr,
            offsets: [isize; N],
            stride: isize,
            _member: PhantomData<$elem>,
        }

        impl<'a, T, const N: usize> $name<'a, T, N> {
            #[inline(always)]
            unsafe fn get_items(&self, ptr: NonNull<T>) -> $elem {
                core::array::from_fn(|i| unsafe { ptr.offset(self.offsets[i]).$into_ref() })
            }

            #[inline]
            unsafe fn next_unchecked(&mut self) -> $elem {
                // SAFETY: The caller promised there's at least one more chunk of items.
                unsafe {
                    let ptr = self.post_inc_start(1);
                    self.get_items(ptr)
                }
            }

            #[inline]
            unsafe fn next_back_unchecked(&mut self) -> $elem {
                // SAFETY: the caller promised it's not empty, so
                // the offsetting is in-bounds and there's an element to return.
                unsafe {
                    let ptr = self.pre_dec_end(1);
                    self.get_items(ptr)
                }
            }

            #[inline(always)]
            unsafe fn post_inc_start(&mut self, offset: usize) -> NonNull<T> {
                let address = self.ptr;

                // SAFETY: the caller guarantees that `offset` doesn't exceed `self.len()`,
                // so this new pointer is inside `self` and thus guaranteed to be non-null.
                unsafe {
                    self.ptr = self.ptr.offset(self.stride * offset as isize);
                }
                address
            }

            #[inline(always)]
            unsafe fn pre_dec_end(&mut self, offset: usize) -> NonNull<T> {
                // SAFETY: the caller guarantees that `offset` doesn't exceed `self.len()`,
                // which is guaranteed to not overflow an `isize`. Also, the resulting pointer
                // is in bounds of `slice`, which fulfills the other requirements for `offset`.
                let end = unsafe { &mut *(&raw mut self.end).cast::<NonNull<T>>() };
                *end = unsafe { end.offset(-self.stride * offset as isize) };
                *end
            }

            /// Return `true` if the iterator has been exhausted.
            #[inline(always)]
            pub fn is_iter_empty(&self) -> bool {
                unsafe { self.ptr == std::mem::transmute::<$ptr, NonNull<T>>(self.end) }
            }

            /// Advance by `M` items and return them as an array, or return `None` if fewer
            /// than `M` items remain.
            #[inline(always)]
            pub fn next_chunk<const M: usize>(&mut self) -> Option<[$elem; M]> {
                if M > self.len() {
                    None
                } else {
                    // SAFETY: We just checked that there are at least M chunks remaining, so this is in bounds.
                    Some(core::array::from_fn(|_| unsafe { self.next_unchecked() }))
                }
            }
        }

        impl<T, const N: usize> ExactSizeIterator for $name<'_, T, N> {
            #[inline(always)]
            fn len(&self) -> usize {
                let end = unsafe { std::mem::transmute::<*const T, NonNull<T>>(self.end) };
                let offset = unsafe { end.offset_from(self.ptr) };
                (offset / self.stride) as usize
            }
        }

        impl<'a, T, const N: usize> Iterator for $name<'a, T, N> {
            type Item = $elem;

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                unsafe {
                    if self.is_iter_empty() {
                        return None;
                    }
                    Some(self.next_unchecked())
                }
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                let len = self.len();
                (len, Some(len))
            }

            #[inline]
            fn count(self) -> usize {
                self.len()
            }

            #[inline]
            fn nth(&mut self, n: usize) -> Option<$elem> {
                unsafe {
                    if n >= self.len() {
                        // This iterator is now empty.
                        self.ptr = std::mem::transmute::<$ptr, NonNull<T>>(self.end);
                        return None;
                    }
                    // SAFETY: We are in bounds.
                    self.post_inc_start(n);
                    Some(self.next_unchecked())
                }
            }

            #[inline]
            fn last(mut self) -> Option<$elem> {
                self.next_back()
            }

            // We override the default implementation, which uses `try_fold`,
            // because this simple implementation generates less LLVM IR and is
            // faster to compile.
            #[inline]
            fn for_each<F>(mut self, mut f: F)
            where
                Self: Sized,
                F: FnMut(Self::Item),
            {
                while let Some(x) = self.next() {
                    f(x);
                }
            }
        }

        impl<'a, T, const N: usize> DoubleEndedIterator for $name<'a, T, N> {
            #[inline]
            fn next_back(&mut self) -> Option<Self::Item> {
                unsafe {
                    if self.is_iter_empty() {
                        return None;
                    }
                    Some(self.next_back_unchecked())
                }
            }

            #[inline]
            fn nth_back(&mut self, n: usize) -> Option<$elem> {
                if n >= self.len() {
                    // This iterator is now empty.
                    unsafe {
                        self.ptr = std::mem::transmute::<$ptr, NonNull<T>>(self.end);
                    }
                    return None;
                }
                // SAFETY: We are in bounds. `pre_dec_end` does the right thing even for ZSTs.
                unsafe {
                    self.pre_dec_end(n);
                    Some(self.next_back_unchecked())
                }
            }
        }

        impl<T, const N: usize> std::iter::FusedIterator for $name<'_, T, N> {}
    };
}

implement_chunk_strided_iter!(Iter -> *const T, [&'a T; N], as_ref);
implement_chunk_strided_iter!(IterMut -> *mut T, [&'a mut T; N], as_mut);

macro_rules! implement_continuous_chunk_strided_iter {
    ($name:ident -> $ptr:ty, $elem:ty, $into_ref:ident) => {
        /// Iterator over `N`-element contiguous chunks within a strided lane, yielding
        /// `&[T; N]` (or `&mut [T; N]`) slices.
        pub struct $name<'a, T, const N: usize> {
            ptr: NonNull<T>,
            end: $ptr,
            stride: isize,
            _member: PhantomData<$elem>,
        }

        impl<'a, T, const N: usize> $name<'a, T, N> {
            #[inline]
            unsafe fn next_unchecked(&mut self) -> $elem {
                // SAFETY: The caller promised there's at least one more chunk of items.
                unsafe { self.post_inc_start(1).cast::<[T; N]>().$into_ref() }
            }

            #[inline]
            unsafe fn next_back_unchecked(&mut self) -> $elem {
                // SAFETY: the caller promised it's not empty, so
                // the offsetting is in-bounds and there's an element to return.
                unsafe { self.pre_dec_end(1).cast::<[T; N]>().$into_ref() }
            }

            #[inline(always)]
            unsafe fn post_inc_start(&mut self, offset: usize) -> NonNull<T> {
                let address = self.ptr;

                // SAFETY: the caller guarantees that `offset` doesn't exceed `self.len()`,
                // so this new pointer is inside `self` and thus guaranteed to be non-null.
                unsafe {
                    self.ptr = self.ptr.offset(self.stride * offset as isize);
                }
                address
            }

            #[inline(always)]
            unsafe fn pre_dec_end(&mut self, offset: usize) -> NonNull<T> {
                // SAFETY: the caller guarantees that `offset` doesn't exceed `self.len()`,
                // which is guaranteed to not overflow an `isize`. Also, the resulting pointer
                // is in bounds of `slice`, which fulfills the other requirements for `offset`.
                let end = unsafe { &mut *(&raw mut self.end).cast::<NonNull<T>>() };
                *end = unsafe { end.offset(-self.stride * offset as isize) };
                *end
            }

            /// Return `true` if the iterator has been exhausted.
            #[inline(always)]
            pub fn is_iter_empty(&self) -> bool {
                unsafe { self.ptr == std::mem::transmute::<$ptr, NonNull<T>>(self.end) }
            }

            /// Advance by `M` contiguous chunks and return them as an array, or return `None`
            /// if fewer than `M` chunks remain.
            #[inline(always)]
            pub fn next_chunk<const M: usize>(&mut self) -> Option<[$elem; M]> {
                if M > self.len() {
                    None
                } else {
                    // SAFETY: We just checked that there are at least M chunks remaining, so this is in bounds.
                    Some(core::array::from_fn(|_| unsafe { self.next_unchecked() }))
                }
            }
        }

        impl<T, const N: usize> ExactSizeIterator for $name<'_, T, N> {
            #[inline(always)]
            fn len(&self) -> usize {
                let end = unsafe { std::mem::transmute::<*const T, NonNull<T>>(self.end) };
                let offset = unsafe { end.offset_from(self.ptr) };
                (offset / self.stride) as usize
            }
        }

        impl<'a, T, const N: usize> Iterator for $name<'a, T, N> {
            type Item = $elem;

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                unsafe {
                    if self.is_iter_empty() {
                        return None;
                    }
                    Some(self.next_unchecked())
                }
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                let len = self.len();
                (len, Some(len))
            }

            #[inline]
            fn count(self) -> usize {
                self.len()
            }

            #[inline]
            fn nth(&mut self, n: usize) -> Option<$elem> {
                unsafe {
                    if n >= self.len() {
                        // This iterator is now empty.
                        self.ptr = std::mem::transmute::<$ptr, NonNull<T>>(self.end);
                        return None;
                    }
                    // SAFETY: We are in bounds.
                    self.post_inc_start(n);
                    Some(self.next_unchecked())
                }
            }

            #[inline]
            fn last(mut self) -> Option<$elem> {
                self.next_back()
            }

            // We override the default implementation, which uses `try_fold`,
            // because this simple implementation generates less LLVM IR and is
            // faster to compile.
            #[inline]
            fn for_each<F>(mut self, mut f: F)
            where
                Self: Sized,
                F: FnMut(Self::Item),
            {
                while let Some(x) = self.next() {
                    f(x);
                }
            }
        }

        impl<'a, T, const N: usize> DoubleEndedIterator for $name<'a, T, N> {
            #[inline]
            fn next_back(&mut self) -> Option<Self::Item> {
                unsafe {
                    if self.is_iter_empty() {
                        return None;
                    }
                    Some(self.next_back_unchecked())
                }
            }

            #[inline]
            fn nth_back(&mut self, n: usize) -> Option<$elem> {
                if n >= self.len() {
                    // This iterator is now empty.
                    unsafe {
                        self.ptr = std::mem::transmute::<$ptr, NonNull<T>>(self.end);
                    }
                    return None;
                }
                // SAFETY: We are in bounds. `pre_dec_end` does the right thing even for ZSTs.
                unsafe {
                    self.pre_dec_end(n);
                    Some(self.next_back_unchecked())
                }
            }
        }

        impl<T, const N: usize> std::iter::FusedIterator for $name<'_, T, N> {}
    };
}

implement_continuous_chunk_strided_iter!(ArrayChunks -> *const T, &'a [T; N], as_ref);
implement_continuous_chunk_strided_iter!(ArrayChunksMut -> *mut T, &'a mut [T; N], as_mut);

impl<L, const N: usize, T> Deref for ChunkStridedSliceBase<L, N, T>
where
    L: Data<Elem = T>,
{
    type Target = ChunkStridedSliceRef<T, N>;

    fn deref(&self) -> &Self::Target {
        // SAFETY:
        // - The pointer is aligned because neither type uses repr(align)
        // - It is "dereferencable" because it comes from a reference
        // - For the same reason, it is initialized
        // - The cast is valid because StridedSliceRef uses #[repr(transparent)]
        let Self { parts, _member } = self;
        let ptr = (parts as *const ChunkStrideParts<T, N>) as *const ChunkStridedSliceRef<T, N>;
        unsafe { &*ptr }
    }
}

impl<L, const N: usize, T> DerefMut for ChunkStridedSliceBase<L, N, T>
where
    L: DataMut<Elem = T>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY:
        // - The pointer is aligned because neither type uses repr(align)
        // - It is "dereferencable" because it comes from a reference
        // - For the same reason, it is initialized
        // - The cast is valid because StridedSliceRef uses #[repr(transparent)]
        let Self { parts, _member } = self;
        let ptr = (parts as *mut ChunkStrideParts<T, N>) as *mut ChunkStridedSliceRef<T, N>;
        unsafe { &mut *ptr }
    }
}

macro_rules! implement_lane_chunk_iter {
    ($name:ident -> $ptr:ty, $memb:ty, $elem:ty, { $rem:tt }, {$( $mut_:tt )?}, $into_ref:ident) => {
        /// Iterator over groups of `N` consecutive lanes within an N-dimensional array.
        ///
        /// Each item is a [`ChunkStridedSlice`] (or mutable variant) representing `N` parallel
        /// lanes along the chosen axis.  Use [`remainder`](Self::remainder) to obtain the
        /// leftover lanes when the total is not divisible by `N`.
        pub struct $name<'a, T, const N: usize> {
            base: NonNull<T>,
            arr_info: ArrayInfo,
            front_pos: Vec<usize>,
            front_offset: isize,
            rear_offset: isize,
            rear_pos: Vec<usize>,
            remaining: usize,
            _member: PhantomData<$memb>,
        }

        unsafe impl<T: Send, const N: usize> Send for $name<'_, T, N> {}
        unsafe impl<T: Sync, const N: usize> Sync for $name<'_, T, N> {}

        impl<'a, T, const N: usize> $name<'a, T, N> {
            /// Construct from a flat slice with the given `shape`, iterating chunks along `axis`.
            ///
            /// # Panics
            ///
            /// Panics if `arr` is empty, `axis >= shape.len()`, or `arr.len()` does not equal
            /// `shape.iter().product()`.
            #[track_caller]
            pub fn from_slice(arr: &'a $( $mut_ )? [T], shape: &[usize], axis: usize) -> Self {
                let (ptr, arr_info) = lane_parts_from_slice(arr,  shape, axis);
                Self::new(ptr, arr_info)
            }

            /// Construct from a sub-region of a flat slice: the outer layout is `shape`, but
            /// only the first `sub_shape[axis]` elements along `axis` are iterated.
            ///
            /// # Panics
            ///
            /// Panics if `arr` is empty, `axis >= shape.len()`, `shape.len() != sub_shape.len()`,
            /// any `sub_shape[i] > shape[i]`, or `arr.len()` does not equal `shape.iter().product()`.
            #[track_caller]
            pub fn from_sub_slice(
                arr: &'a $( $mut_ )? [T],
                shape: &[usize],
                sub_shape: &[usize],
                axis: usize,
            ) ->  Self {
                let (ptr, arr_info) = lane_parts_from_sub_slice(arr, shape, sub_shape, axis);
                Self::new(ptr, arr_info)
            }

            /// Construct from an ndarray, iterating lane chunks along `axis`.
            ///
            /// # Panics
            ///
            /// Panics if `axis >= shape.len()` or `arr.len()` does not equal `shape.iter().product()`.
            #[cfg(feature="ndarray")]
            #[track_caller]
            pub fn from_ndarray<D: Dimension>(
                arr: &'a $( $mut_ )? ArrayRef<T, D>,
                shape: &[usize],
                axis: usize,
            ) -> Self
            {
                let (ptr, arr_info) = lane_parts_from_ndarray(arr, shape, axis);
                Self::new(ptr, arr_info)
            }

            fn new(base: NonNull<T>, arr_info: ArrayInfo) -> Self {
                let n_lanes = arr_info.n_lanes();
                let n_remainder = n_lanes % N;
                let n_chunkable = n_lanes - n_remainder;

                let front_offset = 0;
                let front_pos = arr_info.get_position_at(0);

                let rear_pos = arr_info.get_position_at(n_chunkable);
                let rear_offset = arr_info.get_offset_at(&rear_pos);

                Self {
                    base,
                    arr_info:arr_info.clone(),
                    front_offset,
                    front_pos,
                    rear_offset,
                    rear_pos,
                    remaining:n_chunkable,
                    _member: PhantomData,
                }
            }

            #[inline(always)]
            unsafe fn post_inc_start(&mut self, i: usize) -> NonNull<T>{

                // SAFETY: caller guarantees i <= remaining;

                let ptr = unsafe{self.base.offset(self.front_offset)};

                for _ in 0..i{
                    self.arr_info.advance_position_and_offset(&mut self.front_pos, &mut self.front_offset)
                }
                self.remaining -= i;
                ptr
            }

            #[inline(always)]
            unsafe fn pre_dec_end(&mut self, i: usize) -> NonNull<T>{

                // SAFETY: caller guarantees i <= remaining;

                for _ in 0..i{
                    self.arr_info.retreat_position_and_offset(&mut self.rear_pos, &mut self.rear_offset)
                }
                self.remaining -= i;

                unsafe{self.base.offset(self.rear_offset)}
            }

            /// Return an iterator over the leftover lanes that do not fill a complete chunk of `N`.
            pub fn remainder(&self) -> $rem<'a, T>{
                let n_lanes = self.arr_info.n_lanes();
                let n_remainder = n_lanes % N;
                let front_pos = self.arr_info.get_position_at(n_lanes - n_remainder);
                let front_offset = self.arr_info.get_offset_at(&front_pos);

                let rear_pos = self.arr_info.get_position_at(n_lanes);
                let rear_offset = self.arr_info.get_offset_at(&rear_pos);

                $rem
                {
                    base: self.base.into(),
                    arr_info: self.arr_info.clone(),
                    front_offset,
                    front_pos,
                    rear_offset,
                    rear_pos,
                    remaining: n_remainder,
                    _member: PhantomData,
                }
            }
        }

        impl<'a, T, const N: usize> ExactSizeIterator for $name<'a, T, N> {
            #[inline(always)]
            fn len(&self) -> usize{
                self.remaining / N
            }
        }

        impl<'a ,T, const N: usize> Iterator for $name<'a ,T, N>{
            type Item = $elem;

            #[inline]
            fn next(&mut self) -> Option<Self::Item>{
                if self.remaining < N{
                    return None
                }
                // already checked to ensure there are at least N remaining items.
                let offsets = core::array::from_fn(|_|{
                    let off = self.front_offset;
                    let _ptr = unsafe{self.post_inc_start(1)};
                    off
                });

                Some(
                    Self::Item{
                        parts: ChunkStrideParts {
                            base: self.base,
                            offsets,
                            length: self.arr_info.lane_length,
                            stride: self.arr_info.lane_stride,
                        },
                        _member: SliceLifetime {
                            _member: PhantomData,
                        },
                    }
                )
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                let len = self.len();
                (len, Some(len))
            }

            #[inline]
            fn count(self) -> usize {
                self.len()
            }

            #[inline]
            fn nth(&mut self, n: usize) -> Option<$elem> {
                if n >= self.len() {
                    self.remaining = 0;
                    return None;
                }
                unsafe {
                    // SAFETY: We are in bounds.
                    self.post_inc_start(n);
                }
                self.next()
            }

            #[inline]
            fn last(mut self) -> Option<$elem> {
                self.next_back()
            }

            // We override the default implementation, which uses `try_fold`,
            // because this simple implementation generates less LLVM IR and is
            // faster to compile.
            #[inline]
            fn for_each<F>(mut self, mut f: F)
            where
                Self: Sized,
                F: FnMut(Self::Item),
            {
                while let Some(x) = self.next() {
                    f(x);
                }
            }
        }

        impl<'a, T, const N: usize> DoubleEndedIterator for $name<'a, T, N> {
            #[inline]
            fn next_back(&mut self) -> Option<Self::Item> {
                if self.remaining < N {
                    return None;
                }
                // already checked to ensure there are at least N remaining items.
                let offsets = core::array::from_fn(|_|{
                    let off = self.rear_offset;
                    let _ptr = unsafe{self.pre_dec_end(1)};
                    off
                });
                Some(
                    Self::Item{
                        parts: ChunkStrideParts {
                            base: self.base,
                            offsets,
                            length: self.arr_info.lane_length,
                            stride: self.arr_info.lane_stride,
                        },
                        _member: SliceLifetime {
                            _member: PhantomData,
                        },
                    }
                )
            }

            #[inline]
            fn nth_back(&mut self, n: usize) -> Option<$elem> {
                if n >= self.len() {
                    self.remaining = 0;
                    return None;
                }
                unsafe {
                    // SAFETY: We are in bounds.
                    self.pre_dec_end(n);
                }
                self.next_back()
            }
        }

        impl<T, const N: usize> std::iter::FusedIterator for $name<'_, T, N> {}
    };
}

implement_lane_chunk_iter!(IterLaneChunks -> *const T, &'a T, ChunkStridedSlice<'a, T, N>, {IterLanes}, {}, as_ref);
implement_lane_chunk_iter!(IterLaneChunksMut -> *mut T, &'a mut T, ChunkStridedSliceMut<'a, T, N>, {IterLanesMut},  {mut}, as_mut);

unsafe impl<L: Sync + Data, const N: usize> Send for ChunkStridedSliceBase<L, N> {}
unsafe impl<T: Sync, const N: usize> Sync for ChunkStridedSliceRef<T, N> {}
unsafe impl<T: Send, const N: usize> Send for ChunkStridedSliceRef<T, N> {}

#[cfg(feature = "rayon")]
/// Rayon-parallel lane iterators for flat slices and ndarray arrays.
pub mod parallel {
    use super::super::strided_slice::parallel::{ParIterLanes, ParIterLanesMut};
    use super::*;

    use rayon::iter::plumbing::{Consumer, Producer, ProducerCallback, UnindexedConsumer, bridge};
    pub use rayon::iter::{IndexedParallelIterator, ParallelIterator};

    macro_rules! implement_lane_chunk_par_iter {
        ($par_name:ident, $prod_name:ident, $memb:ty, $item:ident, $into_iter:ident, {$rem_iter:tt}, {$( $mut_:tt )?}) => {
            /// Rayon parallel iterator over groups of `N` lanes in an N-dimensional array.
            pub struct $par_name<'a, T, const N: usize> {
                base: NonNull<T>,
                arr_info: ArrayInfo,
                _member: PhantomData<$memb>,
            }
            unsafe impl<T: Send, const N: usize> Send for $par_name<'_, T, N> {}
            unsafe impl<T: Sync, const N: usize> Sync for $par_name<'_, T, N> {}

            impl<'a, T, const N: usize> $par_name<'a, T, N> {
                /// Construct from a flat slice with the given `shape`, chunking lanes along `axis`.
                ///
                /// # Panics
                ///
                /// Panics if `arr` is empty, `axis >= shape.len()`, or `arr.len()` does not equal
                /// `shape.iter().product()`.
                #[track_caller]
                pub fn from_slice(arr: &'a $( $mut_ )? [T], shape: &[usize], axis: usize) -> Self {
                    let (ptr, arr_info) = lane_parts_from_slice(arr, shape, axis);
                    Self::new(ptr, arr_info)
                }

                /// Construct from a sub-region of a flat slice.
                ///
                /// # Panics
                ///
                /// Panics if `arr` is empty, `axis >= shape.len()`, `shape.len() != sub_shape.len()`,
                /// any `sub_shape[i] > shape[i]`, or `arr.len()` does not equal `shape.iter().product()`.
                #[track_caller]
                pub fn from_sub_slice(
                    arr: &'a $( $mut_ )? [T],
                    shape: &[usize],
                    sub_shape: &[usize], // this excepts only usize for use safety (i.e. it's difficult to get negative strides correct.)
                    axis: usize,
                ) -> Self {
                    let (ptr, arr_info) = lane_parts_from_sub_slice(arr, shape, sub_shape, axis);
                    Self::new(ptr, arr_info)
                }

                /// Construct from an ndarray.
                ///
                /// # Panics
                ///
                /// Panics if `axis >= shape.len()` or `arr.len()` does not equal `shape.iter().product()`.
                #[cfg(feature="ndarray")]
                #[track_caller]
                pub fn from_ndarray<D: Dimension>(
                    arr: &'a $( $mut_ )? ArrayRef<T, D>,
                    shape: &[usize],
                    axis: usize,
                ) -> Self
                {
                    let (ptr, arr_info) = lane_parts_from_ndarray(arr, shape, axis);
                    Self::new(ptr, arr_info)
                }

                pub(super) fn new(base: NonNull<T>, arr_info: ArrayInfo) -> Self {
                    Self {
                        base,
                        arr_info,
                        _member: PhantomData,
                    }
                }

                /// Return a parallel iterator over the leftover lanes that do not fill a chunk of `N`.
                pub fn remainder(&self) -> $rem_iter<'a, T>{
                    let n_lanes = self.arr_info.n_lanes();
                    let n_remainder = n_lanes % N;

                    $rem_iter{
                        base: self.base,
                        arr_info: self.arr_info.clone(),
                        start: n_lanes - n_remainder,
                        end: n_lanes,
                        _member: PhantomData
                    }
                }
            }

            impl<'a, T: Sync + Send, const N: usize> ParallelIterator for $par_name<'a, T, N> {
                type Item = $item<'a, T, N>;
                fn drive_unindexed<C>(self, consumer: C) -> C::Result
                where
                    C: UnindexedConsumer<Self::Item>,
                {
                    bridge(self, consumer)
                }
            }

            impl<'a, T: Sync + Send, const N: usize> IndexedParallelIterator for $par_name<'a, T, N> {
                fn drive<C>(self, consumer: C) -> C::Result
                where
                    C: Consumer<Self::Item>,
                {
                    bridge(self, consumer)
                }

                #[inline(always)]
                fn len(&self) -> usize {
                    self.arr_info.n_lanes() / N
                }

                fn with_producer<CB: ProducerCallback<Self::Item>>(
                    self,
                    callback: CB,
                ) -> CB::Output {
                    callback.callback($prod_name {
                        base: self.base,
                        arr_info: &self.arr_info,
                        start: 0,
                        end: self.len() * N,
                        _member: PhantomData,
                    })
                }
            }

            struct $prod_name<'a, 'b, T, const N: usize> {
                base: NonNull<T>,
                arr_info: &'b ArrayInfo,
                start: usize,
                end: usize,
                _member: PhantomData<$memb>,
            }

            unsafe impl<'a, 'b, T: Send, const N: usize> Send for $prod_name<'a, 'b, T, N> {}

            impl<'a, 'b, T: Send + Sync, const N: usize> Producer for $prod_name<'a, 'b, T, N> {
                type Item = $item<'a, T, N>;
                type IntoIter = $into_iter<'a, T,  N>;

                fn into_iter(self) -> Self::IntoIter {
                    let front_pos = self.arr_info.get_position_at(self.start);
                    let front_offset = self.arr_info.get_offset_at(&front_pos);
                    let rear_pos = self.arr_info.get_position_at(self.end);
                    let rear_offset = self.arr_info.get_offset_at(&rear_pos);

                    Self::IntoIter {
                        base: self.base,
                        arr_info: self.arr_info.clone(),
                        front_pos,
                        front_offset,
                        rear_pos,
                        rear_offset,
                        remaining: self.end - self.start,
                        _member: PhantomData,
                    }
                }

                fn split_at(self, index: usize) -> (Self, Self) {
                    let index = self.start + index * N;
                    let elem_index = Ord::min(index, self.end);
                    (
                        Self {
                            end: elem_index,
                            ..self
                        },
                        Self {
                            start: elem_index,
                            ..self
                        },
                    )
                }
            }
        };
    }
    implement_lane_chunk_par_iter!(
        ParIterLaneChunks,
        IterLaneChunksProducer,
        &'a T,
        ChunkStridedSlice,
        IterLaneChunks,
        { ParIterLanes },
        {}
    );

    implement_lane_chunk_par_iter!(
        ParIterLaneChunksMut,
        IterLaneChunksMutProducer,
        &'a mut T,
        ChunkStridedSliceMut,
        IterLaneChunksMut,
        { ParIterLanesMut },
        {}
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    use itertools::Itertools;
    use rstest::rstest;

    #[inline]
    fn dot_product(v1: &[usize], v2: &[usize]) -> usize {
        v1.iter()
            .zip(v2)
            .fold(0, |acc, (v1, v2)| acc + v1.clone() * v2.clone())
    }

    #[rstest]
    fn test_strided_chunk_iter(
        #[values(51, 52, 62, 63, 64)] n: usize,
        #[values(0, 5, 8, 20, 25)] ind: usize,
        #[values(2, 3, 4)] chunk_size: usize,
        #[values(0, 1)] ax: usize,
    ) {
        let shape = [n, n];
        let strides = stride_from_shape(&shape)
            .into_iter()
            .map(|i| i as usize)
            .collect::<Vec<_>>();
        let n_total = n * n;
        let data = (0..n_total).collect::<Vec<_>>();
        let other_ax = match ax {
            0 => 1,
            1 => 0,
            _ => unimplemented!(), // only called for ax = 0 and 1
        };

        let mut data_mut = data.clone();

        macro_rules! test_for_N {
            () => {
                // test immutable
                let slice = ChunkStridedSlice::<_, N>::from_slice(&data, &shape, ax, ind);

                assert_eq!(slice.len(), shape[ax]);
                // test getting
                for i in 0..shape[ax] {
                    let ind_0 = strides[ax] * i;
                    for j in 0..N {
                        let ind_1 = strides[other_ax] * (ind + j) + ind_0;
                        assert_eq!(slice.get((i, j)), data.get(ind_1));
                    }
                }

                // test iterating
                slice.iter().enumerate().for_each(|(i, chunk)| {
                    let ind_0 = strides[ax] * i;
                    chunk.into_iter().enumerate().for_each(|(j, v)| {
                        let ind_1 = strides[other_ax] * (ind + j) + ind_0;

                        assert_eq!(*v, data[ind_1]);
                    })
                });

                // test mutable
                let mut slice =
                    ChunkStridedSliceMut::<_, N>::from_mut_slice(&mut data_mut, &shape, ax, ind);

                // test getting mut
                for i in 0..shape[ax] {
                    for j in 0..N {
                        slice.get_mut((i, j)).and_then(|v| Some(*v *= 2));
                    }
                }
                // test that it modified only the correct elements of the slice.
                for i in 0..shape[ax] {
                    let ind_0 = strides[ax] * i;
                    for j in 0..ind {
                        let ind_1 = strides[other_ax] * j + ind_0;
                        assert_eq!(data_mut.get(ind_1), data.get(ind_1));
                    }
                    for j in ind..N + ind {
                        let ind_1 = strides[other_ax] * j + ind_0;
                        assert_eq!(
                            data_mut.get(ind_1).cloned(),
                            data.get(ind_1).cloned().and_then(|v| Some(v * 2))
                        );
                    }
                    for j in N + ind..shape[other_ax] {
                        let ind_1 = strides[other_ax] * j + ind_0;
                        assert_eq!(data_mut.get(ind_1), data.get(ind_1));
                    }
                }

                let mut slice =
                    ChunkStridedSliceMut::<_, N>::from_mut_slice(&mut data_mut, &shape, ax, ind);

                // test iterating
                slice.iter_mut().for_each(|chunk| {
                    chunk.into_iter().for_each(|v| {
                        *v *= 2;
                    })
                });
                // test that it modified only the correct elements of the slice.
                for i in 0..shape[ax] {
                    let ind_0 = strides[ax] * i;
                    for j in 0..ind {
                        let ind_1 = strides[other_ax] * j + ind_0;
                        assert_eq!(data_mut.get(ind_1), data.get(ind_1));
                    }
                    for j in ind..N + ind {
                        let ind_1 = strides[other_ax] * j + ind_0;
                        assert_eq!(
                            data_mut.get(ind_1).cloned(),
                            data.get(ind_1).cloned().and_then(|v| Some(v * 4))
                        );
                    }
                    for j in N + ind..shape[other_ax] {
                        let ind_1 = strides[other_ax] * j + ind_0;
                        assert_eq!(data_mut.get(ind_1), data.get(ind_1));
                    }
                }
            };
        }

        match chunk_size {
            2 => {
                const N: usize = 2;
                test_for_N! {}
            }
            3 => {
                const N: usize = 3;
                test_for_N! {}
            }
            4 => {
                const N: usize = 4;
                test_for_N! {}
            }
            _ => {
                unimplemented!() // function is only called for n = 2, 3, or 4
            }
        }
    }

    #[rstest]
    fn test_strided_chunk_deref(
        #[values(51, 52, 62, 63, 64)] n: usize,
        #[values(0, 5, 8, 20, 25)] ind: usize,
        #[values(2, 3, 4)] chunk_size: usize,
        #[values(0, 1)] ax: usize,
    ) {
        let shape = [n, n];
        let n_total: usize = shape.iter().product();
        let mut data = (0..n_total).collect::<Vec<_>>();

        fn test_sum<const N: usize>(chunk: &ChunkStridedSliceRef<usize, N>) -> usize {
            chunk.iter().map(|row| row.into_iter().sum::<usize>()).sum()
        }

        macro_rules! test_for_N {
            () => {
                // test immutable
                let slice = ChunkStridedSlice::<_, N>::from_slice(&data, &shape, ax, ind);

                let expected: usize = slice.iter().map(|row| row.into_iter().sum::<usize>()).sum();

                assert_eq!(test_sum(&slice), expected);

                let slice =
                    ChunkStridedSliceMut::<_, N>::from_mut_slice(&mut data, &shape, ax, ind);
                assert_eq!(test_sum(&slice), expected);
            };
        }

        match chunk_size {
            2 => {
                const N: usize = 2;
                test_for_N! {}
            }
            3 => {
                const N: usize = 3;
                test_for_N! {}
            }
            4 => {
                const N: usize = 4;
                test_for_N! {}
            }
            _ => {
                unimplemented!() // function is only called for n = 2, 3, or 4
            }
        }
    }

    #[rstest]
    fn test_strided_chunk_deref_mut(
        #[values(51, 52, 62, 63, 64)] n: usize,
        #[values(0, 5, 8, 20, 25)] ind: usize,
        #[values(2, 3, 4)] chunk_size: usize,
        #[values(0, 1)] ax: usize,
    ) {
        let other_ax = match ax {
            0 => 1,
            1 => 0,
            _ => unimplemented!(), // only called for ax = 0 and 1
        };
        let shape = [n, n];
        let strides = stride_from_shape(&shape)
            .into_iter()
            .map(|i| i as usize)
            .collect::<Vec<_>>();
        let n_total: usize = shape.iter().product();
        let data = (0..n_total).collect::<Vec<_>>();
        let mut data_mut = (0..n_total).collect::<Vec<_>>();

        fn test_scale<const N: usize>(chunk: &mut ChunkStridedSliceRef<usize, N>) {
            chunk
                .iter_mut()
                .for_each(|row| row.into_iter().for_each(|v| *v *= 2));
        }

        macro_rules! test_for_N {
            () => {
                // test immutable
                let mut slice =
                    ChunkStridedSliceMut::<_, N>::from_mut_slice(&mut data_mut, &shape, ax, ind);

                test_scale(&mut slice);

                // test that it modified only the correct elements of the slice.
                for i in 0..shape[ax] {
                    let ind_0 = strides[ax] * i;
                    for j in 0..ind {
                        let ind_1 = strides[other_ax] * j + ind_0;
                        assert_eq!(data_mut.get(ind_1), data.get(ind_1));
                    }
                    for j in ind..N + ind {
                        let ind_1 = strides[other_ax] * j + ind_0;
                        assert_eq!(
                            data_mut.get(ind_1).cloned(),
                            data.get(ind_1).cloned().and_then(|v| Some(v * 2))
                        );
                    }
                    for j in N + ind..shape[other_ax] {
                        let ind_1 = strides[other_ax] * j + ind_0;
                        assert_eq!(data_mut.get(ind_1), data.get(ind_1));
                    }
                }
            };
        }

        match chunk_size {
            2 => {
                const N: usize = 2;
                test_for_N! {}
            }
            3 => {
                const N: usize = 3;
                test_for_N! {}
            }
            4 => {
                const N: usize = 4;
                test_for_N! {}
            }
            _ => {
                unimplemented!() // function is only called for n = 2, 3, or 4
            }
        }
    }

    #[test]
    fn test_strided_chunk_as_chunks() {
        const N: usize = 4;
        let n = 50;
        let shape = [n, N];
        let n_total: usize = shape.iter().product();

        let data = (0..n_total).collect::<Vec<_>>();

        let slice = ChunkStridedSlice::<_, N>::from_slice(&data, &shape, 0, 0);

        assert_eq!(slice.len(), n);

        fn as_slice(chunks: &ChunkStridedSliceRef<usize, N>) -> &[usize] {
            chunks.try_into().unwrap()
        }
        // The chunk points to a slice of the same size
        let chunks = as_slice(&slice).as_chunks::<N>().0;

        let expected = data.as_chunks::<N>().0;

        assert_eq!(chunks, expected);

        fn as_chunks(chunks: &ChunkStridedSliceRef<usize, N>) -> &[[usize; N]] {
            chunks.try_into().unwrap()
        }
        // The chunk points to a slice of the same size
        let chunks = as_chunks(&slice);

        let expected = data.as_chunks::<N>().0;

        assert_eq!(chunks, expected);
    }

    #[test]
    fn test_strided_chunk_as_chunks_mut() {
        const N: usize = 4;
        let n = 50;
        let shape = [n, N];
        let n_total: usize = shape.iter().product();

        let data = (0..n_total).map(|i| i * 2).collect::<Vec<_>>();

        let mut data_mut = (0..n_total).collect::<Vec<_>>();

        let mut slice = ChunkStridedSliceMut::<_, N>::from_mut_slice(&mut data_mut, &shape, 0, 0);

        assert_eq!(slice.len(), n);

        fn as_slice_mut(chunks: &mut ChunkStridedSliceRef<usize, N>) -> &mut [usize] {
            chunks.try_into().unwrap()
        }

        let chunks = as_slice_mut(&mut slice).as_chunks_mut::<N>().0; // The chunk points to a slice of the same size

        chunks
            .iter_mut()
            .map(|v| v.iter_mut())
            .flatten()
            .for_each(|v| *v *= 2);

        assert_eq!(data_mut, data);

        let mut data_mut = (0..n_total).collect::<Vec<_>>();
        let mut slice = ChunkStridedSliceMut::<_, N>::from_mut_slice(&mut data_mut, &shape, 0, 0);

        fn as_chunks_mut(chunks: &mut ChunkStridedSliceRef<usize, N>) -> &mut [[usize; N]] {
            chunks.try_into().unwrap()
        }

        let chunks = as_chunks_mut(&mut slice); // The chunk points to a slice of the same size

        chunks
            .iter_mut()
            .map(|v| v.iter_mut())
            .flatten()
            .for_each(|v| *v *= 2);

        assert_eq!(data_mut, data);
    }

    #[rstest]
    fn test_strided_chunk_chunks(#[values(0, 1)] ax: usize, #[values(0, 10, 20)] ind: usize) {
        const N: usize = 4;
        let n = 50;
        let shape = [n, n];
        let n_total: usize = shape.iter().product();

        let data = (0..n_total).collect::<Vec<_>>();

        let slice = ChunkStridedSlice::<_, N>::from_slice(&data, &shape, ax, ind);

        assert_eq!(slice.len(), n);

        if let Ok(chunks) = slice.try_array_chunks() {
            let actual = chunks
                .map(|row| row.iter())
                .flatten()
                .cloned()
                .collect::<Vec<_>>();
            let expected = slice
                .iter()
                .map(|row| row.into_iter())
                .flatten()
                .cloned()
                .collect::<Vec<_>>();
            assert_eq!(actual, expected);
        } else {
            assert_eq!(ax, 1);
        }
    }

    #[rstest]
    fn test_strided_chunk_chunks_mut(#[values(0, 1)] ax: usize, #[values(0, 10, 20)] ind: usize) {
        let other_ax = match ax {
            0 => 1,
            1 => 0,
            _ => unimplemented!(), // only called for ax = 0 and 1
        };
        const N: usize = 4;
        let n = 50;
        let shape = [n, n];
        let strides = stride_from_shape(&shape)
            .into_iter()
            .map(|i| i as usize)
            .collect::<Vec<_>>();
        let n_total: usize = shape.iter().product();

        let data = (0..n_total).collect::<Vec<_>>();

        let mut data_mut = (0..n_total).collect::<Vec<_>>();

        let mut slice =
            ChunkStridedSliceMut::<_, N>::from_mut_slice(&mut data_mut, &shape, ax, ind);

        assert_eq!(slice.len(), n);

        if let Ok(chunks) = slice.try_array_chunks_mut() {
            chunks
                .map(|row| row.iter_mut())
                .flatten()
                .for_each(|v| *v *= 2);

            // test that it modified only the correct elements of the slice.
            for i in 0..shape[ax] {
                let ind_0 = strides[ax] * i;
                for j in 0..ind {
                    let ind_1 = strides[other_ax] * j + ind_0;
                    assert_eq!(data_mut.get(ind_1), data.get(ind_1));
                }
                for j in ind..N + ind {
                    let ind_1 = strides[other_ax] * j + ind_0;
                    assert_eq!(
                        data_mut.get(ind_1).cloned(),
                        data.get(ind_1).cloned().and_then(|v| Some(v * 2))
                    );
                }
                for j in N + ind..shape[other_ax] {
                    let ind_1 = strides[other_ax] * j + ind_0;
                    assert_eq!(data_mut.get(ind_1), data.get(ind_1));
                }
            }
        } else {
            assert_eq!(ax, 1);
        }
    }

    #[rstest]
    #[case::one_d_axis_0(1, 0)]
    #[case::two_d_axis_0(2, 0)]
    #[case::two_d_axis_1(2, 1)]
    #[case::three_d_axis_0(3, 0)]
    #[case::three_d_axis_1(3, 1)]
    #[case::three_d_axis_2(3, 2)]
    #[case::four_d_axis_0(4, 0)]
    #[case::four_d_axis_1(4, 1)]
    #[case::four_d_axis_2(4, 2)]
    #[case::four_d_axis_2(4, 3)]
    fn test_strided_lane_chunks_iter(
        #[case] dim: usize,
        #[case] axis: usize,
        #[values(4, 5, 6)] n: usize,
    ) {
        const N: usize = 4;
        let shape = (0..dim).map(|i| n + i).collect_vec();
        let n_t = shape.iter().product();
        let arr = (0..n_t).collect::<Vec<_>>();

        let strides = stride_from_shape(&shape);
        let mut shape_sub = shape.clone();
        let _ = shape_sub.remove(axis);
        let mut stride_sub = strides.clone();
        let _ = stride_sub.remove(axis);

        let n_lanes_expected: usize = shape_sub.iter().product();
        let n_chunks_expected = n_lanes_expected / N;
        let n_rem_expected = n_lanes_expected % N;

        let chunks = IterLaneChunks::<_, N>::from_slice(&arr, &shape, axis);
        let rem = chunks.remainder();
        assert_eq!(chunks.len(), n_chunks_expected);
        assert_eq!(rem.len(), n_rem_expected);

        let mut actual = chunks
            .map(|chunk| {
                chunk
                    .iter()
                    .map(|v| v.into_iter().cloned().collect_vec())
                    .concat()
            })
            .concat();
        actual.extend(rem.map(|slc| slc.iter().map(|v| *v).collect_vec()).concat());

        let mut expected = (0..n_chunks_expected)
            .map(|i_chunk| {
                (0..shape[axis])
                    .map(|j| {
                        (i_chunk * N..(i_chunk + 1) * N)
                            .map(|i| {
                                let inds_sub = unravel(i, &shape_sub);
                                let offset = dot_product(&inds_sub, &stride_sub);
                                let io = offset + j * strides[axis];
                                arr[io]
                            })
                            .collect_vec()
                    })
                    .concat()
            })
            .concat();
        expected.extend(
            (n_chunks_expected * N..n_lanes_expected)
                .map(|i| {
                    let inds_sub = unravel(i, &shape_sub);
                    let offset = dot_product(&inds_sub, &stride_sub);
                    (0..shape[axis])
                        .map(|j| {
                            let io = offset + j * strides[axis];
                            arr[io]
                        })
                        .collect_vec()
                })
                .concat(),
        );
        assert_eq!(actual, expected);
    }

    #[rstest]
    #[case::one_d_axis_0(1, 0)]
    #[case::two_d_axis_0(2, 0)]
    #[case::two_d_axis_1(2, 1)]
    #[case::three_d_axis_0(3, 0)]
    #[case::three_d_axis_1(3, 1)]
    #[case::three_d_axis_2(3, 2)]
    #[case::four_d_axis_0(4, 0)]
    #[case::four_d_axis_1(4, 1)]
    #[case::four_d_axis_2(4, 2)]
    #[case::four_d_axis_2(4, 3)]
    fn test_strided_lane_chunks_iter_mut(
        #[case] dim: usize,
        #[case] axis: usize,
        #[values(4, 5, 6)] n: usize,
    ) {
        let shape = (0..dim).map(|i| n + i).collect_vec();
        let n_t = shape.iter().product();
        let mut arr = (0..n_t).collect::<Vec<_>>();

        let strides = stride_from_shape(&shape);
        let mut shape_sub = shape.clone();
        let _ = shape_sub.remove(axis);
        let mut stride_sub = strides.clone();
        let _ = stride_sub.remove(axis);

        let n_lanes_expected: usize = shape_sub.iter().product();
        let n_chunks_expected = n_lanes_expected / N;
        let n_rem_expected = n_lanes_expected % N;

        const N: usize = 4;

        let chunks = IterLaneChunksMut::<_, N>::from_slice(&mut arr, &shape, axis);
        let rem = chunks.remainder();
        assert_eq!(chunks.len(), n_chunks_expected);
        assert_eq!(rem.len(), n_rem_expected);

        chunks.enumerate().for_each(|(i_chunk, mut chunk)| {
            chunk.iter_mut().for_each(|v| {
                v.into_iter().enumerate().for_each(|(i, v)| {
                    let i_lane = i_chunk * N + i;
                    *v *= i_lane;
                })
            });
        });
        rem.enumerate().for_each(|(i_lane, mut slc)| {
            slc.iter_mut().for_each(|v| {
                let i_lane = i_lane + N * n_chunks_expected;
                *v *= i_lane;
            })
        });

        let stride_sub = stride_from_shape(&shape_sub);
        // use a new stride_sub so as to correctly calulate the i_lane value.
        let expected = (0..n_t)
            .map(|i_flat| {
                let mut pos = unravel(i_flat, &shape);
                let _ = pos.remove(axis);
                let i_lane = dot_product(&pos, &stride_sub);
                i_flat * i_lane
            })
            .collect_vec();

        assert_eq!(arr, expected);
    }

    #[cfg(feature = "rayon")]
    mod parallel {
        use super::super::parallel::*;
        use super::*;

        #[rstest]
        #[case::one_d_axis_0(1, 0)]
        #[case::two_d_axis_0(2, 0)]
        #[case::two_d_axis_1(2, 1)]
        #[case::three_d_axis_0(3, 0)]
        #[case::three_d_axis_1(3, 1)]
        #[case::three_d_axis_2(3, 2)]
        #[case::four_d_axis_0(4, 0)]
        #[case::four_d_axis_1(4, 1)]
        #[case::four_d_axis_2(4, 2)]
        #[case::four_d_axis_2(4, 3)]
        fn test_strided_lane_chunks_par_iter(
            #[case] dim: usize,
            #[case] axis: usize,
            #[values(4, 5, 6)] n: usize,
        ) {
            const N: usize = 4;
            let shape = (0..dim).map(|i| n + i).collect_vec();
            let n_t = shape.iter().product();
            let arr = (0..n_t).collect::<Vec<_>>();

            let strides = stride_from_shape(&shape);
            let mut shape_sub = shape.clone();
            let _ = shape_sub.remove(axis);
            let mut stride_sub = strides.clone();
            let _ = stride_sub.remove(axis);

            let n_lanes_expected: usize = shape_sub.iter().product();
            let n_chunks_expected = n_lanes_expected / N;
            let n_rem_expected = n_lanes_expected % N;

            let chunks = ParIterLaneChunks::<_, N>::from_slice(&arr, &shape, axis);
            let rem = chunks.remainder();
            assert_eq!(chunks.len(), n_chunks_expected);
            assert_eq!(rem.len(), n_rem_expected);

            let mut actual = chunks
                .map(|chunk| {
                    chunk
                        .iter()
                        .map(|v| v.into_iter().cloned().collect_vec())
                        .concat()
                })
                .collect::<Vec<_>>();
            actual.extend(
                rem.map(|slc| slc.iter().map(|v| *v).collect_vec())
                    .collect::<Vec<_>>(),
            );
            let actual = actual.concat();

            let mut expected = (0..n_chunks_expected)
                .map(|i_chunk| {
                    (0..shape[axis])
                        .map(|j| {
                            (i_chunk * N..(i_chunk + 1) * N)
                                .map(|i| {
                                    let inds_sub = unravel(i, &shape_sub);
                                    let offset = dot_product(&inds_sub, &stride_sub);
                                    let io = offset + j * strides[axis];
                                    arr[io]
                                })
                                .collect_vec()
                        })
                        .concat()
                })
                .concat();
            expected.extend(
                (n_chunks_expected * N..n_lanes_expected)
                    .map(|i| {
                        let inds_sub = unravel(i, &shape_sub);
                        let offset = dot_product(&inds_sub, &stride_sub);
                        (0..shape[axis])
                            .map(|j| {
                                let io = offset + j * strides[axis];
                                arr[io]
                            })
                            .collect_vec()
                    })
                    .concat(),
            );
            assert_eq!(actual, expected);
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

        let mut work_e = core::array::from_fn(|_| vec![0; ns]);
        let mut work_o = core::array::from_fn(|_| vec![0; nd]);

        chunks.for_each(|mut chunk| {
            chunk.split(&mut work_e, &mut work_o);
            chunk.interleave(&work_e, &work_o);
        });
        let mut work_e = vec![0; ns];
        let mut work_o = vec![0; nd];

        lanes.for_each(|mut slc| {
            slc.split(&mut work_e, &mut work_o);
            slc.interleave(&work_e, &work_o);
        });

        // iterate over all of them using single lanes
        out2.iter_lanes_mut(&shape, ax).for_each(|mut slc| {
            slc.split(&mut work_e, &mut work_o);
            slc.interleave(&work_e, &work_o);
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

        let mut work_e = core::array::from_fn(|_| vec![0; ns]);
        let mut work_o = core::array::from_fn(|_| vec![0; nd]);

        chunks.for_each(|mut chunk| {
            chunk.deinterleave(&mut work_e, &mut work_o);
            chunk.stack(&work_e, &work_o);
        });
        let mut work_e = vec![0; ns];
        let mut work_o = vec![0; nd];

        lanes.for_each(|mut slc| {
            slc.deinterleave(&mut work_e, &mut work_o);
            slc.stack(&work_e, &work_o);
        });

        // iterate over all of them using single lanes
        out2.iter_lanes_mut(&shape, ax).for_each(|mut slc| {
            slc.deinterleave(&mut work_e, &mut work_o);
            slc.stack(&work_e, &work_o);
        });

        assert_eq!(out, out2);
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
                (start..end).collect_vec()
            });
            chunk.fill_from(&vecs);
        });
        chunks.remainder().enumerate().for_each(|(i, mut slice)| {
            let start = n_c + i + 1;
            let end = start + n_ax;
            let vec = (start..end).collect_vec();
            slice.fill_from(&vec);
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
            let mut vecs = core::array::from_fn(|_| vec![0; n_ax]);
            chunk.pour_into(&mut vecs);

            vecs.into_iter().enumerate().for_each(|(j, vec)| {
                let start = i * N + j + 1;
                let end = start + n_ax;
                assert_eq!(vec.as_slice(), &(start..end).collect_vec());
            })
        });

        (n_c * N..n_lanes).zip(rem).for_each(|(i, slice)| {
            let mut vec = vec![0; n_ax];
            slice.pour_into(&mut vec);

            let start = i + 1;
            let end = start + n_ax;
            assert_eq!(vec, (start..end).collect_vec());
        });
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
            let mut vecs_f = core::array::from_fn(|_| vec![0; nf]);
            let mut vecs_s = core::array::from_fn(|_| vec![0; ns]);

            chunk.split(&mut vecs_f, &mut vecs_s);
            //split_strided_chunk(&chunk, &mut vecs_f, &mut vecs_s);

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
            slice.split(&mut vec_f, &mut vec_s);

            let start = i + 1;
            let end = start + nf;
            assert_eq!(vec_f, (start..end).collect_vec());

            let start = end + n_pad;
            let end = start + ns;
            assert_eq!(vec_s, (start..end).collect_vec());
        });
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
                (start..end).collect_vec()
            });
            let vecs_s = core::array::from_fn(|j| {
                let start = i * N + j + 1 + nf;
                let end = start + ns;
                (start..end).collect_vec()
            });
            chunk.stack(&vecs_f, &vecs_s);
            //stack_to_strided_chunk(&vecs_f, &vecs_s, &mut chunk);
        });
        chunks.remainder().enumerate().for_each(|(i, mut slice)| {
            let start = n_c + i + 1;
            let end = start + n_ax;
            let vec = (start..end).collect_vec();
            let (vf, vs) = vec.split_at(nf);
            slice.stack(vf, vs);
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
}
