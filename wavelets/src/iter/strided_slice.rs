//! Strided slice views and lane iterators over flat multi-dimensional arrays.
//!
//! This module provides [`StridedSliceRef`] — a lightweight non-owning view into a
//! strided region of memory — and the concrete iterator types returned by
//! [`super::LanesIterator`] and [`super::parallel::LanesParallelIterator`].

use super::*;
use num_traits::Zero;
use std::iter::repeat_n;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;

#[cfg(feature = "ndarray")]
use ndarray::{ArrayRef, Dimension};

/// Raw pointer + length + stride triplet backing a strided slice view.
#[derive(Clone, Debug, Copy)]
struct StrideParts<T> {
    base: NonNull<T>,
    length: usize,
    stride: isize,
}

/// Owned strided slice view parameterised by a lifetime marker `L`.
pub struct StridedSliceBase<L, T = <L as Data>::Elem>
where
    L: Data<Elem = T>,
{
    parts: StrideParts<T>,
    _member: SliceLifetime<L>,
}

/// Non-owning strided view into a contiguous or strided memory region.
///
/// Provides indexed access and iteration over `length` elements spaced `stride`
/// elements apart in memory.  The stride may be 1 (contiguous) or larger.
///
/// This type is used internally by lane iterators so that a single iterator
/// implementation can handle both row-major flat slices and ndarray's arbitrary
/// memory layouts.
#[repr(transparent)]
#[derive(Clone, Debug, Copy)]
pub struct StridedSliceRef<T>(StrideParts<T>);

impl<T> StridedSliceRef<T> {
    /// Return a raw read-only pointer to the first element.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.0.base.as_ptr()
    }

    /// Return a raw mutable pointer to the first element.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.0.base.as_ptr()
    }

    /// Number of elements in this strided view.
    #[inline]
    pub fn len(&self) -> usize {
        self.0.length
    }

    /// Whether or not this strided slice has any elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.length == 0
    }

    /// Return a reference to the element at `index`, or `None` if out of bounds.
    #[inline]
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.0.length {
            None
        } else {
            Some(unsafe { self.get_unchecked(index) })
        }
    }

    /// Return a reference to the element at `index` without bounds checking.
    ///
    /// # Safety
    /// `index` must be less than `self.len()`.
    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> &T {
        // SAFETY: index must be within bounds.
        unsafe { &*self.as_ptr().offset(index as isize * self.0.stride) }
    }

    /// Return a mutable reference to the element at `index`, or `None` if out of bounds.
    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.0.length {
            None
        } else {
            Some(unsafe { self.get_unchecked_mut(index) })
        }
    }

    #[inline]
    /// Return a mutable reference to element `index` without bounds checking.
    ///
    /// # Safety
    /// `index` must be less than `self.len()`.
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        // SAFETY: index must be within bounds.
        unsafe { &mut *self.as_mut_ptr().offset(index as isize * self.0.stride) }
    }

    #[inline]
    /// Split the strided slice into two parts at the specified location
    ///
    /// # Panics
    /// If mid > len.
    pub fn split_at(&self, mid: usize) -> (StridedSlice<'_, T>, StridedSlice<'_, T>) {
        assert!(mid <= self.len(), "mid > len");
        (
            StridedSlice {
                parts: StrideParts {
                    base: self.0.base,
                    length: mid,
                    stride: self.0.stride,
                },
                _member: SliceLifetime {
                    _member: PhantomData,
                },
            },
            StridedSlice {
                parts: StrideParts {
                    base: unsafe { self.0.base.offset(mid as isize * self.0.stride) },
                    length: self.0.length - mid,
                    stride: self.0.stride,
                },
                _member: SliceLifetime {
                    _member: PhantomData,
                },
            },
        )
    }

    #[inline]
    /// Split the mutable strided slice into two parts at the specified location
    ///
    /// # Panics
    /// If mid > len.
    pub fn split_at_mut(&mut self, mid: usize) -> (StridedSliceMut<'_, T>, StridedSliceMut<'_, T>) {
        assert!(mid <= self.len(), "mid > len");
        (
            StridedSliceMut {
                parts: StrideParts {
                    base: self.0.base,
                    length: mid,
                    stride: self.0.stride,
                },
                _member: SliceLifetime {
                    _member: PhantomData,
                },
            },
            StridedSliceMut {
                parts: StrideParts {
                    base: unsafe { self.0.base.offset(mid as isize * self.0.stride) },
                    length: self.0.length - mid,
                    stride: self.0.stride,
                },
                _member: SliceLifetime {
                    _member: PhantomData,
                },
            },
        )
    }

    /// Return `true` if the elements in this strided view are contiguous in memory (i.e. stride is 1).
    #[inline(always)]
    pub fn is_contiguous(&self) -> bool {
        self.0.stride == 1
    }

    /// Iterate over elements in this strided view (read-only).
    #[inline]
    pub fn iter(&self) -> Iter<'_, T> {
        let start = self.0.base;
        let end = unsafe {
            start
                .as_ptr()
                .offset(self.0.stride * self.0.length as isize)
        };
        Iter {
            ptr: start,
            end,
            stride: self.0.stride,
            _member: PhantomData,
        }
    }

    /// Iterate mutably over elements in this strided view.
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        let start = self.0.base;
        let end = unsafe {
            start
                .as_ptr()
                .offset(self.0.stride * self.0.length as isize)
        };
        IterMut {
            ptr: start,
            end,
            stride: self.0.stride,
            _member: PhantomData,
        }
    }
}

impl<T: Clone> StridedSliceRef<T> {
    /// Deinterleave a strided lane into even- and odd-indexed flat buffers.
    ///
    /// Equivalent to [`crate::utils::deinterleave`] but reads from a [`StridedSliceRef`] instead of a plain
    /// slice; takes a fast path when the view happens to be contiguous.
    #[inline(always)]
    #[track_caller]
    pub fn deinterleave(&self, evens: &mut [T], odds: &mut [T]) {
        let nx = self.len();
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

        match TryInto::<&[T]>::try_into(self) {
            Ok(x) => {
                let (xc, rem) = x.as_chunks();
                xc.iter()
                    .zip(evens.iter_mut().zip(odds))
                    .for_each(|([xe, xo], (e, o))| {
                        *e = xe.clone();
                        *o = xo.clone();
                    });
                if !rem.is_empty() {
                    *evens.last_mut().unwrap() = rem.first().unwrap().clone();
                }
            }
            Err(x) => {
                (0..nx)
                    .step_by(2)
                    .zip(evens.iter_mut().zip(odds))
                    .for_each(|(i, (e, o))| {
                        // SAFETY: Lengths checked above to be valid.
                        *e = unsafe { x.get_unchecked(i) }.clone();
                        *o = unsafe { x.get_unchecked(i + 1) }.clone();
                    });
                if n_e > n_o {
                    // SAFETY: Lengths checked above to be valid.
                    *evens.last_mut().unwrap() = unsafe { self.get_unchecked(nx - 1) }.clone()
                }
            }
        }
    }

    /// Strided variant of [`crate::utils::interleave`]: write interleaved values into a [`StridedSliceRef`].
    #[inline(always)]
    #[track_caller]
    pub fn interleave(&mut self, evens: &[T], odds: &[T]) {
        let n = self.len();
        let n_e = evens.len();
        let n_o = odds.len();

        assert_eq!(n / 2, n_o);
        assert_eq!(n.div_ceil(2), n_e);

        match TryInto::<&mut [T]>::try_into(self) {
            Ok(x) => {
                let (xc, rem) = x.as_chunks_mut();
                xc.iter_mut()
                    .zip(evens.iter().cloned().zip(odds.iter().cloned()))
                    .for_each(|([xe, xo], (e, o))| {
                        *xe = e;
                        *xo = o;
                    });
                if !rem.is_empty() {
                    *rem.first_mut().unwrap() = evens.last().unwrap().clone();
                }
            }
            Err(x) => {
                (0..n)
                    .step_by(2)
                    .zip(evens.iter().cloned().zip(odds.iter().cloned()))
                    .for_each(|(i, (e, o))| {
                        // SAFETY: Lengths checked above to be valid.
                        unsafe {
                            *x.get_unchecked_mut(i) = e;
                            *x.get_unchecked_mut(i + 1) = o;
                        }
                    });
                if n_e > n_o {
                    // SAFETY: Lengths checked above to be valid.
                    unsafe { *x.get_unchecked_mut(n - 1) = evens.last().unwrap().clone() }
                }
            }
        }
    }

    /// Split `self` into a leading `first` segment and a trailing `second` segment, skipping the gap.
    ///
    /// `second` is taken from the tail of `self`, not from immediately after `first`.  This is the
    /// inverse of [`StridedSliceRef::stack`].
    #[inline(always)]
    #[track_caller]
    pub fn split(&self, first: &mut [T], second: &mut [T]) {
        let nf = first.len();
        let ns = second.len();
        let nx = self.len();
        assert!(
            nf + ns <= nx,
            "invalid lengths for slice stack, first: {nf}, second: {ns}, third: {nx}"
        );
        let n_mid = nx - (nf + ns);

        match TryInto::<&[T]>::try_into(self) {
            Ok(x) => {
                let (xf, xe) = x.split_at(nf);
                let (_, xs) = xe.split_at(n_mid);
                xf.iter().cloned().zip(first).for_each(|(x, v)| *v = x);
                xs.iter().cloned().zip(second).for_each(|(x, v)| *v = x);
            }
            Err(x) => {
                first.iter_mut().enumerate().for_each(|(i, v)| {
                    // SAFETY: Lengths verified above to be valid
                    *v = unsafe { x.get_unchecked(i) }.clone();
                });
                second.iter_mut().enumerate().for_each(|(i, v)| {
                    // SAFETY: Lengths verified above to be valid
                    *v = unsafe { x.get_unchecked(i + nf + n_mid) }.clone();
                });
            }
        }
    }

    /// Fill the slice `sink` with cloned elements of `self`.
    #[inline(always)]
    #[track_caller]
    pub fn pour_into(&self, sink: &mut [T]) {
        let n = self.len();
        let no = sink.len();
        assert!(
            no <= n,
            "Output slice with length {no} too long for strided slice with length {n}."
        );
        match TryInto::<&[T]>::try_into(self) {
            Ok(source) => {
                source.iter().cloned().zip(sink).for_each(|(a, b)| *b = a);
            }
            Err(source) => {
                source.iter().cloned().zip(sink).for_each(|(a, b)| *b = a);
            }
        }
    }
}

impl<T: Clone + Zero> StridedSliceRef<T> {
    /// Write `first` at the start of `self` and `second` at the very end, zero-filling the gap.
    ///
    /// Unlike a simple concatenation, the second half is placed at the tail of `out` rather than
    /// immediately after `first`.
    #[inline(always)]
    #[track_caller]
    pub fn stack(&mut self, first: &[T], second: &[T]) {
        // stacks first and second into self, but with the second half at the very end of out, instead of immediately after the first half.
        let nf = first.len();
        let ns = second.len();
        let n = self.len();
        assert!(
            nf + ns <= n,
            "invalid lengths for slice stack, first: {nf}, second: {ns}, third: {n}",
        );
        let n_mid = n - (nf + ns);

        match TryInto::<&mut [T]>::try_into(self) {
            Ok(x) => {
                let (xf, xe) = x.split_at_mut(nf);
                let (xm, xs) = xe.split_at_mut(n_mid);
                xf.iter_mut().zip(first).for_each(|(x, v)| *x = v.clone());
                xm.iter_mut().for_each(|v| *v = T::zero());
                xs.iter_mut().zip(second).for_each(|(x, v)| *x = v.clone());
            }
            Err(x) => {
                x.iter_mut()
                    .zip(
                        first
                            .iter()
                            .cloned()
                            .chain(repeat_n(T::zero(), n_mid))
                            .chain(second.iter().cloned()),
                    )
                    .for_each(|(a, b)| *a = b);
            }
        }
    }

    /// Fill `self` with cloned elements from slice `source`, filling the leftover with zero values.
    #[inline(always)]
    #[track_caller]
    pub fn fill_from(&mut self, source: &[T]) {
        let n = self.len();
        let no = source.len();
        assert!(
            no <= n,
            "Output slice with length {no} too long for strided slice with length {n}."
        );

        match TryInto::<&mut [T]>::try_into(self) {
            Ok(sink) => {
                let (sink, tail) = sink.split_at_mut(no);
                source.iter().cloned().zip(sink).for_each(|(a, b)| *b = a);
                tail.fill(T::zero());
            }
            Err(sink) => {
                let mut sink = sink.into_iter();
                source
                    .iter()
                    .cloned()
                    .zip(sink.by_ref())
                    .for_each(|(a, b)| *b = a);
                sink.for_each(|v| *v = T::zero());
            }
        }
    }
}

impl<'a, T> TryFrom<&'a StridedSliceRef<T>> for &[T] {
    type Error = &'a StridedSliceRef<T>;

    #[inline]
    fn try_from(value: &'a StridedSliceRef<T>) -> Result<Self, Self::Error> {
        if value.is_contiguous() {
            unsafe { Ok(std::slice::from_raw_parts(value.as_ptr(), value.len())) }
        } else {
            Err(value)
        }
    }
}

impl<'a, T> TryFrom<&'a mut StridedSliceRef<T>> for &'a mut [T] {
    type Error = &'a mut StridedSliceRef<T>;

    #[inline]
    fn try_from(value: &'a mut StridedSliceRef<T>) -> Result<Self, Self::Error> {
        if value.is_contiguous() {
            // SAFETY: The construction guarantees that the data is valid for `length` elements
            // and we checked that the stride is 1, so this is a valid slice.
            unsafe {
                Ok(std::slice::from_raw_parts_mut(
                    value.as_mut_ptr(),
                    value.len(),
                ))
            }
        } else {
            Err(value)
        }
    }
}

impl<'a, T> IntoIterator for &'a StridedSliceRef<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T> IntoIterator for &'a mut StridedSliceRef<T> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

/// Read-only strided slice view with an explicit lifetime.
pub type StridedSlice<'a, T> = StridedSliceBase<SliceLifetime<&'a T>, T>;
/// Mutable strided slice view with an explicit lifetime.
pub type StridedSliceMut<'a, T> = StridedSliceBase<SliceLifetime<&'a mut T>, T>;

impl<'a, T> StridedSlice<'a, T> {
    /// Create a strided view over `slice` sampling every `stride` elements.
    pub fn from_slice(slice: &'a [T], stride: usize) -> Self {
        assert_ne!(slice.len(), 0);
        Self {
            // SAFETY: slice length > 0 so ptr is non-null.
            parts: StrideParts {
                base: unsafe { NonNull::new_unchecked(slice.as_ptr() as *mut T) },
                length: slice.len().div_ceil(stride),
                stride: stride as isize,
            },
            _member: SliceLifetime {
                _member: PhantomData,
            },
        }
    }
}

impl<'a, T> StridedSliceMut<'a, T> {
    /// Create a mutable strided view over `slice` sampling every `stride` elements.
    pub fn from_mut_slice(slice: &'a mut [T], stride: usize) -> Self {
        assert_ne!(slice.len(), 0);

        // SAFETY: slice length > 0 so ptr is non-null.
        Self {
            parts: StrideParts {
                base: unsafe { NonNull::new_unchecked(slice.as_ptr() as *mut T) },
                length: slice.len().div_ceil(stride),
                stride: stride as isize,
            },
            _member: SliceLifetime {
                _member: PhantomData,
            },
        }
    }
}

macro_rules! implement_strided_iter {
    ($name:ident -> $ptr:ty, $elem:ty, {$( $mut_:tt )?}, $into_ref:ident) => {
        /// Iterator over a [`StridedSliceRef`] that traverses elements with a fixed stride.
        pub struct $name<'a, T> {
            ptr: NonNull<T>,
            end: $ptr,
            stride: isize,
            _member: PhantomData<$elem>,
        }

        impl<'a, T> $name<'a, T> {
            #[inline]
            unsafe fn next_unchecked(&mut self) -> $elem {
                // SAFETY: The caller promised there's at least one more item.
                unsafe { self.post_inc_start(1).$into_ref() }
            }

            #[inline]
            unsafe fn next_back_unchecked(&mut self) -> $elem {
                // SAFETY: the caller promised it's not empty, so
                // the offsetting is in-bounds and there's an element to return.
                unsafe { self.pre_dec_end(1).$into_ref() }
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

        impl<T> ExactSizeIterator for $name<'_, T> {
            #[inline(always)]
            fn len(&self) -> usize {
                let end = unsafe { std::mem::transmute::<*const T, NonNull<T>>(self.end) };
                let offset = unsafe { end.offset_from(self.ptr) };
                // if stride is negative, end address will be before start address
                // and offset will also be negative, thus cast to usize is valid.
                (offset / self.stride) as usize
            }
        }

        impl<'a, T> Iterator for $name<'a, T> {
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
            // faster to compile, according to slice iter source code.
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

        impl<'a, T> DoubleEndedIterator for $name<'a, T> {
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
                // SAFETY: We are in bounds.
                unsafe {
                    self.pre_dec_end(n);
                    Some(self.next_back_unchecked())
                }
            }
        }

        impl<T> std::iter::FusedIterator for $name<'_, T> {}
    };
}

implement_strided_iter!(Iter -> *const T, &'a T, {}, as_ref);
implement_strided_iter!(IterMut -> *mut T, &'a mut T, {mut}, as_mut);

impl<L, T> Deref for StridedSliceBase<L, T>
where
    L: Data<Elem = T>,
{
    type Target = StridedSliceRef<T>;

    fn deref(&self) -> &Self::Target {
        // SAFETY:
        // - The pointer is aligned because neither type uses repr(align)
        // - It is "dereferencable" because it comes from a reference
        // - For the same reason, it is initialized
        let Self { parts, _member } = self;
        let ptr = (parts as *const StrideParts<T>) as *const StridedSliceRef<T>;
        unsafe { &*ptr }
    }
}

impl<L, T> DerefMut for StridedSliceBase<L, T>
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
        let ptr = (parts as *mut StrideParts<T>) as *mut StridedSliceRef<T>;
        unsafe { &mut *ptr }
    }
}

macro_rules! implement_lane_iter {
    ($name:ident -> $ptr:ty, $memb:ty, $elem:ty, {$( $mut_:tt )?}, $into_ref:ident) => {
        /// Iterator that yields successive 1-D lanes of an N-dimensional array.
        pub struct $name<'a, T> {
            pub(crate) base: NonNull<T>,
            pub(crate) arr_info: ArrayInfo,
            pub(crate) front_pos: Vec<usize>,
            pub(crate) front_offset: isize,
            pub(crate) rear_offset: isize,
            pub(crate) rear_pos: Vec<usize>,
            pub(crate) remaining: usize,
            pub(crate) _member: PhantomData<$memb>,
        }

        unsafe impl<T: Send> Send for $name<'_, T> {}
        unsafe impl<T: Sync> Sync for $name<'_, T> {}

        impl<'a, T> $name<'a, T> {
            /// Create a lane iterator over the flat slice `arr` interpreted as an
            /// N-dimensional array with the given `shape`, iterating lanes along `axis`.
            #[track_caller]
            pub(crate) fn from_slice(arr: &'a $( $mut_ )? [T], shape: &[usize], axis: usize) -> Self {
                let (ptr, arr_info) = lane_parts_from_slice(arr, shape, axis);
                Self::new(ptr, arr_info)
            }

            /// Create a lane iterator over the leading `sub_shape` elements of `arr`.
            #[track_caller]
            pub(crate) fn from_sub_slice(
                arr: &'a $( $mut_ )? [T],
                shape: &[usize],
                sub_shape: &[usize],
                axis: usize,
            ) -> Self {
                let (ptr, arr_info) = lane_parts_from_sub_slice(arr, shape, sub_shape, axis);
                Self::new(ptr, arr_info)
            }

            /// Create a lane iterator from an ndarray `ArrayRef`, using its strides.
            #[cfg(feature="ndarray")]
            #[track_caller]
            pub(crate) fn from_ndarray<D: Dimension>(
                arr: &'a $( $mut_ )? ArrayRef<T, D>,
                shape: &[usize],
                axis: usize,
            ) -> Self
            {
                let (ptr, arr_info) = lane_parts_from_ndarray(arr, shape, axis);
                Self::new(ptr, arr_info)
            }


            fn new(base: NonNull<T>, arr_info: ArrayInfo) -> Self {
                let remaining = arr_info.n_lanes();
                let front_offset = 0;
                let front_pos = arr_info.get_position_at(0);

                let rear_pos = arr_info.get_position_at(remaining);
                let rear_offset = arr_info.get_offset_at(&rear_pos);

                Self {
                    base,
                    arr_info,
                    front_offset,
                    front_pos,
                    rear_offset,
                    rear_pos,
                    remaining,
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
        }

        impl<'a, T> ExactSizeIterator for $name<'a, T> {
            #[inline(always)]
            fn len(&self) -> usize{
                self.remaining
            }
        }

        impl<'a ,T> Iterator for $name<'a ,T>{
            type Item = $elem;

            #[inline]
            fn next(&mut self) -> Option<Self::Item>{
                if self.remaining == 0{
                    return None
                }
                // already checked to ensure there is at least 1 remaining item.
                let base = unsafe{self.post_inc_start(1)};
                Some(
                    Self::Item{
                        parts: StrideParts {
                            base,
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

        impl<'a, T> DoubleEndedIterator for $name<'a, T> {
            #[inline]
            fn next_back(&mut self) -> Option<Self::Item> {
                if self.remaining == 0 {
                    return None;
                }
                // already checked to ensure there is at least 1 remaining item.
                let base = unsafe{self.pre_dec_end(1)};
                Some(
                    Self::Item{
                        parts: StrideParts {
                            base,
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

        impl<T> std::iter::FusedIterator for $name<'_, T> {}
    };
}

implement_lane_iter!(IterLanes -> *const T, &'a T, StridedSlice<'a, T>, {}, as_ref);
implement_lane_iter!(IterLanesMut -> *mut T, &'a mut T, StridedSliceMut<'a, T>, {mut}, as_mut);

unsafe impl<L: Sync + Data> Send for StridedSliceBase<L> {}
unsafe impl<T: Sync> Sync for StridedSliceRef<T> {}
unsafe impl<T: Send> Send for StridedSliceRef<T> {}

#[cfg(feature = "rayon")]
/// Rayon-parallel lane iterators for flat slices and ndarray arrays.
pub mod parallel {
    use super::*;

    use rayon::iter::plumbing::{Consumer, Producer, ProducerCallback, UnindexedConsumer, bridge};
    pub use rayon::iter::{IndexedParallelIterator, ParallelIterator};

    macro_rules! implement_lane_par_iter {
        ($par_name:ident, $prod_name:ident, $memb:ty, $item:ident, $into_iter:ident, {$( $mut_:tt )?}) => {
            /// Rayon parallel iterator over 1-D lanes of an N-dimensional array.
            pub struct $par_name<'a, T> {
                pub(crate) base: NonNull<T>,
                pub(crate) arr_info: ArrayInfo,
                pub(crate) start: usize,
                pub(crate) end: usize,
                pub(crate) _member: PhantomData<$memb>,
            }
            unsafe impl<T: Send> Send for $par_name<'_, T> {}
            unsafe impl<T: Sync> Sync for $par_name<'_, T> {}

            impl<'a, T> $par_name<'a, T> {
                /// Construct from a flat slice with the given `shape`, iterating lanes along `axis`.
                pub fn from_slice(arr: &'a $( $mut_ )? [T], shape: &[usize], axis: usize) -> Self {
                    let (ptr, arr_info) = lane_parts_from_slice(arr, shape, axis);
                    Self::new(ptr, arr_info)
                }

                /// Construct from a sub-region of a flat slice.
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
                #[cfg(feature="ndarray")]
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
                    let end = arr_info.n_lanes();
                    Self {
                        base,
                        arr_info,
                        start: 0,
                        end,
                        _member: PhantomData,
                    }
                }
            }

            impl<'a, T: Sync + Send> ParallelIterator for $par_name<'a, T> {
                type Item = $item<'a, T>;
                fn drive_unindexed<C>(self, consumer: C) -> C::Result
                where
                    C: UnindexedConsumer<Self::Item>,
                {
                    bridge(self, consumer)
                }
            }

            impl<'a, T: Sync + Send> IndexedParallelIterator for $par_name<'a, T> {
                fn drive<C>(self, consumer: C) -> C::Result
                where
                    C: Consumer<Self::Item>,
                {
                    bridge(self, consumer)
                }


                #[inline(always)]
                fn len(&self) -> usize {
                    self.end - self.start
                }

                fn with_producer<CB: ProducerCallback<Self::Item>>(
                    self,
                    callback: CB,
                ) -> CB::Output {
                    callback.callback($prod_name {
                        base: self.base,
                        arr_info: &self.arr_info,
                        start: self.start,
                        end: self.end,
                        _member: PhantomData,
                    })
                }
            }

            struct $prod_name<'a, 'b, T> {
                base: NonNull<T>,
                arr_info: &'b ArrayInfo,
                start: usize,
                end: usize,
                _member: PhantomData<$memb>,
            }

            unsafe impl<'a, 'b, T: Send> Send for $prod_name<'a, 'b, T> {}

            impl<'a, 'b, T: Send + Sync> Producer for $prod_name<'a, 'b, T> {
                type Item = $item<'a, T>;
                type IntoIter = $into_iter<'a, T>;

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
                    let index = self.start + index;
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

    implement_lane_par_iter!(
        ParIterLanes,
        IterLanesProducer,
        &'a T,
        StridedSlice,
        IterLanes,
        {}
    );

    implement_lane_par_iter!(
        ParIterLanesMut,
        IterLanesMutProducer,
        &'a mut T,
        StridedSliceMut,
        IterLanesMut,
        {}
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    use itertools::Itertools;
    use rstest::rstest;

    #[rstest]
    fn test_strided_iter(
        #[values(51, 52, 62, 63, 64)] n: usize,
        #[values(1, 2, 3, 4, 5)] step: usize,
    ) {
        let data = (0..n).collect::<Vec<_>>();
        let slice = StridedSliceBase::from_slice(&data, step);

        assert_eq!(slice.get(0), Some(&data[0]));
        assert_eq!(slice.get(1), Some(&data[step]));

        let collected: Vec<_> = slice.iter().cloned().collect();
        let expected = data.into_iter().step_by(step).collect::<Vec<_>>();
        assert_eq!(collected, expected);

        let mut data = (0..n).collect::<Vec<_>>();
        let mut slice = StridedSliceMut::from_mut_slice(&mut data, step);
        slice.iter_mut().for_each(|v| *v *= 10);

        let collected: Vec<_> = slice.iter().cloned().collect();
        let expected = data.into_iter().step_by(step).collect::<Vec<_>>();
        assert_eq!(collected, expected);
    }

    #[test]
    fn test_strided_deref() {
        let n = 80;
        let step = 2;
        let data = (0..n).collect::<Vec<_>>();

        let slice = StridedSliceBase::from_slice(&data, step);

        fn sum_slice(slice: &StridedSliceRef<usize>) -> usize {
            slice.iter().sum()
        }

        let expected = data.iter().step_by(step).sum::<usize>();
        assert_eq!(sum_slice(&slice), expected);

        let mut data = (0..n).collect::<Vec<_>>();
        let mut slice = StridedSliceMut::from_mut_slice(&mut data, step);
        slice.iter_mut().for_each(|v| *v *= 10);
        let actual = sum_slice(&slice);

        let expected = data.iter().step_by(step).sum::<usize>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_strided_mut_deref() {
        let n = 80;
        let step = 2;
        let mut data = (0..n).collect::<Vec<_>>();

        let mut slice = StridedSliceBase::from_mut_slice(&mut data, step);

        fn double_slice(slice: &mut StridedSliceRef<usize>) {
            slice.iter_mut().for_each(|v| *v *= 2);
        }

        double_slice(&mut slice);

        let collected: Vec<_> = slice.iter().cloned().collect();
        let expected = data.into_iter().step_by(step).collect::<Vec<_>>();
        assert_eq!(collected, expected);
    }

    #[test]
    fn test_strided_as_slice() {
        let n = 80;
        let step = 1;
        let data = (0..n).collect::<Vec<_>>();

        let slice = StridedSliceBase::from_slice(&data, step);

        let expected = data.iter().sum::<usize>();

        fn sum_slice(slice: &StridedSliceRef<usize>) -> usize {
            let slice: &[_] = slice.try_into().unwrap(); // step size in this test is 1
            slice.iter().sum()
        }

        assert_eq!(sum_slice(&slice), expected);
    }

    #[test]
    fn test_strided_as_mut_slice_checked() {
        let n = 80;
        let step = 1;
        let mut data = (0..n).collect::<Vec<_>>();

        let mut slice = StridedSliceBase::from_mut_slice(&mut data, step);

        fn double_slice(slice: &mut StridedSliceRef<usize>) {
            let slice: &mut [_] = slice.try_into().unwrap();
            slice.iter_mut().for_each(|v| *v *= 2);
        }

        double_slice(&mut slice);
        let expected = (0..n).map(|v| v * 2).collect::<Vec<_>>();
        assert_eq!(data, expected);
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
    fn test_strided_lane_iter(#[case] dim: usize, #[case] axis: usize, #[values(4)] n: usize) {
        let shape = (0..dim).map(|i| n + i).collect_vec();
        let n_t = shape.iter().product();
        let arr = (0..n_t).collect::<Vec<_>>();

        let strides = stride_from_shape(&shape);
        let mut shape_sub = shape.clone();
        let _ = shape_sub.remove(axis);
        let mut stride_sub = strides.clone();
        let _ = stride_sub.remove(axis);

        let n_lanes_expected: usize = shape_sub.iter().product();

        let lane_iter = IterLanes::from_slice(&arr, &shape, axis);
        assert_eq!(lane_iter.len(), n_lanes_expected);

        let actual = lane_iter
            .map(|slc| slc.iter().map(|v| *v).collect_vec())
            .concat();
        let expected = (0..n_lanes_expected)
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
            .concat();

        assert_eq!(actual, expected);
    }

    #[inline]
    fn dot_product(v1: &[usize], v2: &[usize]) -> usize {
        v1.iter()
            .zip(v2)
            .fold(0, |acc, (v1, v2)| acc + v1.clone() * v2.clone())
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
    fn test_strided_lane_iter_mut(
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

        let lane_iter = IterLanesMut::from_slice(&mut arr, &shape, axis);
        assert_eq!(lane_iter.len(), n_lanes_expected);

        lane_iter.enumerate().for_each(|(i_lane, mut slc)| {
            slc.iter_mut().for_each(|v| {
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
        fn test_strided_lane_par_iter(
            #[case] dim: usize,
            #[case] axis: usize,
            #[values(4, 5, 6)] n: usize,
        ) {
            let shape = (0..dim).map(|i| n + i).collect_vec();
            let n_t = shape.iter().product();
            let arr = (0..n_t).collect::<Vec<_>>();

            let strides = stride_from_shape(&shape);
            let mut shape_sub = shape.clone();
            let _ = shape_sub.remove(axis);
            let mut stride_sub = strides.clone();
            let _ = stride_sub.remove(axis);

            let n_lanes_expected: usize = shape_sub.iter().product();

            let lane_iter = ParIterLanes::from_slice(&arr, &shape, axis);
            assert_eq!(
                lane_iter.len(),
                n_lanes_expected,
                "Incorrect number of lanes in the iterator."
            );

            let actual = lane_iter
                .map(|slc| slc.iter().map(|v| *v).collect_vec())
                .collect::<Vec<_>>();
            let actual = actual.concat();

            let expected = (0..n_lanes_expected)
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
                .concat();

            assert_eq!(actual.len(), expected.len());
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
        fn test_strided_lane_par_iter_mut(
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

            let lane_iter = ParIterLanesMut::from_slice(&mut arr, &shape, axis);
            assert_eq!(lane_iter.len(), n_lanes_expected);

            lane_iter.enumerate().for_each(|(i_lane, mut slc)| {
                slc.iter_mut().for_each(|v| {
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
            slc.interleave(&s, &d);
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
            slc.deinterleave(&mut s, &mut d);

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
            slc.stack(&first, &second);
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
            slc.split(&mut first, &mut second);

            assert_eq!(first, (i..ns + i).collect_vec());
            assert_eq!(second, (ns + i..n + i).collect_vec());
        }
    }

    #[rstest]
    fn test_clone_slice_to_strided(
        #[values(10, 11, 12)] n0: usize,
        #[values(10, 11, 12)] n1: usize,
        #[values(1, 2, 3)] stride: usize,
    ) {
        let n_min = n0.min(n1);
        let inp = (1..n_min + 1).collect_vec();
        let mut out = vec![0; n1 * stride];

        let mut out_strided = StridedSliceMut::from_mut_slice(&mut out, stride);
        out_strided.fill_from(&inp);

        let output = out.iter().step_by(stride).cloned().collect_vec();

        assert_eq!(output[..n_min], inp[..n_min]);
    }

    #[rstest]
    fn test_clone_strided_to_slice(
        #[values(10, 11, 12)] n0: usize,
        #[values(10, 11, 12)] n1: usize,
        #[values(1, 2, 3)] stride: usize,
    ) {
        let inp = (1..n0 * stride + 1).collect_vec();
        let inp_strided = StridedSlice::from_slice(&inp, stride);

        let n_min = n0.min(n1);
        let mut out = vec![0; n_min];

        inp_strided.pour_into(&mut out);

        let out_ref = inp.iter().step_by(stride).cloned().collect_vec();

        assert_eq!(out[..n_min], out_ref[..n_min]);
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
        out_strided.stack(&in_1, &in_2);

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
        inp_strided.split(out_f, out_s);

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
}
