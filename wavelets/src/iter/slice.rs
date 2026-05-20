//! Strided slice views and lane iterators over flat multi-dimensional arrays.
//!
//! This module provides [`StridedSliceRef`] — a lightweight non-owning view into a
//! strided region of memory — and the concrete iterator types returned by
//! [`super::LanesIterator`] and [`super::parallel::LanesParallelIterator`].

use crate::utils::stride_from_shape;
use std::ops::{ControlFlow, Deref, DerefMut};
use std::{marker::PhantomData, ptr::NonNull};

#[cfg(feature = "ndarray")]
use ndarray::{ArrayRef, Dimension};

#[inline]
pub(crate) fn unravel(flat_index: usize, shape: &[usize]) -> Vec<usize> {
    let n_max: usize = shape.iter().product();
    assert!(
        flat_index <= n_max,
        "Flat index is beyond the end of the array."
    );

    // a special case for flat_index == n_max to return an unraveled index that points
    // one past the last item.
    // i.e. it looks like (n0-1, n1 -1, n2)
    // so it would need to be pre retreated by one **before** it is valid.
    if flat_index == n_max {
        let mut inds = shape.iter().map(|n| n - 1).collect::<Vec<_>>();
        if let Some(last) = inds.last_mut()
            && let Some(n_last) = shape.last()
        {
            *last = *n_last;
        }
        return inds;
    }
    let mut inds = vec![0; shape.len()];
    let mut flat_index = flat_index;
    inds.iter_mut()
        .zip(shape.iter())
        .rev()
        .for_each(|(i_dir, n_dir)| {
            *i_dir = flat_index % n_dir;
            flat_index /= n_dir;
        });
    inds
}

/// Marker trait for readable strided data containers.
///
/// # Safety
/// Implementors must guarantee that the underlying memory is valid for reads for the
/// declared element type.
pub unsafe trait Data: Sized {
    /// The element type stored in the container.
    type Elem;
}

/// Marker trait for writable strided data containers.
///
/// # Safety
/// Implementors must guarantee that the underlying memory is valid for reads and writes
/// for the declared element type, and that no other reference aliases the same memory.
pub unsafe trait DataMut: Data {}

/// Phantom type used to attach lifetime information to strided slice views.
pub struct SliceLifetime<T> {
    _member: PhantomData<T>,
}

unsafe impl<'a, T> Data for SliceLifetime<&'a T> {
    type Elem = T;
}

unsafe impl<'a, T> Data for SliceLifetime<&'a mut T> {
    type Elem = T;
}

unsafe impl<'a, T> DataMut for SliceLifetime<&'a mut T> {}

/// Raw pointer + length + stride triplet backing a strided slice view.
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
pub struct StridedSliceRef<T>(StrideParts<T>);
impl<T> StridedSliceRef<T> {
    /// Return a raw read-only pointer to the first element.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.0.base.as_ptr()
    }

    /// Return a raw mutable pointer to the first element.
    #[inline]
    pub fn as_ptr_mut(&mut self) -> *mut T {
        self.0.base.as_ptr()
    }

    /// Number of elements in this strided view.
    #[inline]
    pub fn len(&self) -> usize {
        self.0.length
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
        unsafe { &mut *self.as_ptr_mut().offset(index as isize * self.0.stride) }
    }

    /// Return the underlying data as a regular `&[T]` slice, if the stride is 1.
    ///
    /// Returns `None` when the stride is greater than 1 (non-contiguous layout).
    #[inline]
    pub fn as_slice(&self) -> Option<&[T]> {
        if self.0.stride == 1 {
            // SAFETY: The construction guarantees that the data is valid for `length` elements
            // and we checked that the stride is 1, so this is a valid slice.
            unsafe { Some(std::slice::from_raw_parts(self.as_ptr(), self.len())) }
        } else {
            None
        }
    }

    /// Return the underlying data as a mutable `&mut [T]` slice, if the stride is 1.
    ///
    /// Returns `None` when the stride is greater than 1.
    #[inline]
    pub fn as_slice_mut(&mut self) -> Option<&mut [T]> {
        if self.0.stride == 1 {
            // SAFETY: The construction guarantees that the data is valid for `length` elements
            // and we checked that the stride is 1, so this is a valid slice.
            unsafe {
                Some(std::slice::from_raw_parts_mut(
                    self.as_ptr_mut(),
                    self.len(),
                ))
            }
        } else {
            None
        }
    }

    /// Iterate over elements in this strided view (read-only).
    #[inline]
    pub fn iter(&self) -> StridedIter<'_, T> {
        let start = self.0.base;
        let end = unsafe {
            start
                .as_ptr()
                .offset(self.0.stride * self.0.length as isize)
        };
        StridedIter {
            ptr: start,
            end,
            stride: self.0.stride,
            _member: PhantomData,
        }
    }

    /// Iterate mutably over elements in this strided view.
    #[inline]
    pub fn iter_mut(&mut self) -> StridedIterMut<'_, T> {
        let start = self.0.base;
        let end = unsafe {
            start
                .as_ptr()
                .offset(self.0.stride * self.0.length as isize)
        };
        StridedIterMut {
            ptr: start,
            end,
            stride: self.0.stride,
            _member: PhantomData,
        }
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
                length: (slice.len() + stride - 1) / stride,
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
    pub fn from_slice_mut(slice: &'a mut [T], stride: usize) -> Self {
        assert_ne!(slice.len(), 0);

        // SAFETY: slice length > 0 so ptr is non-null.
        Self {
            parts: StrideParts {
                base: unsafe { NonNull::new_unchecked(slice.as_ptr() as *mut T) },
                length: (slice.len() + stride - 1) / stride,
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

implement_strided_iter!(StridedIter -> *const T, &'a T, {}, as_ref);
implement_strided_iter!(StridedIterMut -> *mut T, &'a mut T, {mut}, as_mut);

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
    pub fn from_slice_mut(slice: &'a [T], shape: &[usize], ax: usize, ind: usize) -> Self {
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

#[derive(Clone, Debug)]
struct ArrayInfo {
    shape: Vec<usize>,
    stride: Vec<isize>,
    lane_length: usize,
    lane_stride: isize,
}

impl ArrayInfo {
    fn new(shape: &[usize], stride: &[isize], axis: usize) -> Self {
        assert!(
            axis < shape.len(),
            "Specified axis exceeds shape dimensions"
        );
        assert_eq!(
            stride.len(),
            shape.len(),
            "Shape and stride should have the same length."
        );
        let mut stride = stride.to_owned();
        let mut shape = shape.to_owned();

        let lane_length = shape.remove(axis);
        let lane_stride = stride.remove(axis);

        Self {
            shape,
            stride,
            lane_length,
            lane_stride,
        }
    }
    #[inline(always)]
    fn n_lanes(&self) -> usize {
        self.shape.iter().product()
    }

    #[inline(always)]
    fn get_position_at(&self, i: usize) -> Vec<usize> {
        unravel(i, &self.shape)
    }

    #[inline(always)]
    fn get_offset_at(&self, pos: &[usize]) -> isize {
        pos.iter()
            .zip(self.stride.iter())
            .fold(0, |acc, (i, step)| acc + *i as isize * step)
    }

    #[inline(always)]
    fn advance_position_and_offset(&self, pos: &mut [usize], offset: &mut isize) {
        let _ = self
            .stride
            .iter()
            .zip(self.shape.iter())
            .zip(pos)
            .rev()
            .try_for_each(|((str, shp), pos)| {
                *offset += *str;
                *pos += 1;
                if *pos < *shp {
                    return ControlFlow::Break(());
                };
                *pos = 0;
                *offset -= *shp as isize * str;
                ControlFlow::Continue(())
            });
    }

    #[inline(always)]
    fn retreat_position_and_offset(&self, pos: &mut [usize], offset: &mut isize) {
        let _ = self
            .stride
            .iter()
            .zip(self.shape.iter())
            .zip(pos)
            .rev()
            .try_for_each(|((str, shp), pos)| {
                if *pos == 0 {
                    *pos = *shp - 1;
                    *offset += *pos as isize * str;
                    return ControlFlow::Continue(());
                } else {
                    *pos -= 1;
                    *offset -= *str;
                    return ControlFlow::Break(());
                }
            });
    }
}

fn lane_parts_from_slice<T>(arr: &[T], shape: &[usize], axis: usize) -> (NonNull<T>, ArrayInfo) {
    lane_parts_from_sub_slice(arr, shape, shape, axis)
}

fn lane_parts_from_sub_slice<T>(
    arr: &[T],
    shape: &[usize],
    sub_shape: &[usize],
    axis: usize,
) -> (NonNull<T>, ArrayInfo) {
    let n = arr.len();
    assert_ne!(
        n, 0,
        "Attempted to create a lane iterator from an empty slice."
    );
    let n_items: usize = shape.iter().product();
    assert_eq!(
        n, n_items,
        "array length must be consistent with the shape. Shape suggests {n_items}, but slice had {n} items."
    );
    assert_eq!(
        shape.len(),
        sub_shape.len(),
        "shape length, {}, and sub_shape length, {}, must be equal",
        shape.len(),
        sub_shape.len()
    );
    assert!(
        sub_shape.iter().zip(shape.iter()).all(|(n1, n2)| n1 <= n2),
        "sub_shape: {:?}, must be equal to our smaller than shape, {:?}",
        sub_shape,
        shape,
    );
    assert!(
        axis < shape.len(),
        "axis: {axis} is out of bounds for dimension size of {}",
        shape.len()
    );

    let stride = stride_from_shape(shape)
        .into_iter()
        .map(|s| s as isize)
        .collect::<Vec<_>>();
    // SAFETY: slice length > 0 so ptr is NonNull.
    let ptr = unsafe { NonNull::new_unchecked(arr.as_ptr() as *mut T) };
    (ptr, ArrayInfo::new(sub_shape, &stride, axis))
}

#[cfg(feature = "ndarray")]
fn lane_parts_from_ndarray<T, D: Dimension>(
    arr: &ArrayRef<T, D>,
    sub_shape: &[usize],
    axis: usize,
) -> (NonNull<T>, ArrayInfo) {
    assert_ne!(
        arr.len(),
        0,
        "Cannot create a lane iterator from an empty ndarray."
    );
    let ndim = arr.ndim();
    assert!(
        axis < ndim,
        "axis: {axis} is out of bounds for dimension size of {ndim}",
    );
    assert_eq!(
        sub_shape.len(),
        arr.ndim(),
        "shape.len(), {}, is not equal to arr.ndim(), {ndim}",
        sub_shape.len(),
    );

    assert!(
        sub_shape.iter().zip(arr.shape()).all(|(n, m)| n <= m),
        "requested shape, {:?} must all be <= arr.shape(), {:?}.",
        sub_shape,
        arr.shape(),
    );

    // SAFETY: Array is not empty, so pointer to first element is gauranteed non-null.
    let ptr = unsafe { NonNull::new_unchecked(arr.as_ptr() as *mut T) };
    (ptr, ArrayInfo::new(sub_shape, arr.strides(), axis))
}

macro_rules! implement_lane_iter {
    ($name:ident -> $ptr:ty, $memb:ty, $elem:ty, {$( $mut_:tt )?}, $into_ref:ident) => {
        /// Iterator that yields successive 1-D lanes of an N-dimensional array.
        pub struct $name<'a, T> {
            base: NonNull<T>,
            arr_info: ArrayInfo,
            front_pos: Vec<usize>,
            front_offset: isize,
            rear_offset: isize,
            rear_pos: Vec<usize>,
            remaining: usize,
            _member: PhantomData<$memb>,
        }

        unsafe impl<T: Send> Send for $name<'_, T> {}
        unsafe impl<T: Sync> Sync for $name<'_, T> {}

        impl<'a, T> $name<'a, T> {
            /// Create a lane iterator over the flat slice `arr` interpreted as an
            /// N-dimensional array with the given `shape`, iterating lanes along `axis`.
            pub fn from_slice(arr: &'a $( $mut_ )? [T], shape: &[usize], axis: usize) -> Self {
                let (ptr, arr_info) = lane_parts_from_slice(arr, shape, axis);
                Self::new(ptr, arr_info)
            }

            /// Create a lane iterator over the leading `sub_shape` elements of `arr`.
            pub fn from_sub_slice(
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
                let remaining = arr_info.n_lanes();
                let front_offset = 0;
                let front_pos = arr_info.get_position_at(0);

                let rear_pos = arr_info.get_position_at(remaining);
                let rear_offset = arr_info.get_offset_at(&rear_pos);

                Self {
                    base: base,
                    arr_info,
                    front_offset: front_offset,
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

implement_lane_iter!(LaneSliceIter -> *const T, &'a T, StridedSlice<'a, T>, {}, as_ref);
implement_lane_iter!(LaneSliceIterMut -> *mut T, &'a mut T, StridedSliceMut<'a, T>, {mut}, as_mut);

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

macro_rules! implement_chunk_strided_iter {
    ($name:ident -> $ptr:ty, $elem:ty, {$( $mut_:tt )?}, $into_ref:ident) => {
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

implement_chunk_strided_iter!(ChunkStridedIter -> *const T, [&'a T; N], {}, as_ref);
implement_chunk_strided_iter!(ChunkStridedIterMut -> *mut T, [&'a mut T; N], {}, as_mut);

macro_rules! implement_continuous_chunk_strided_iter {
    ($name:ident -> $ptr:ty, $elem:ty, {$( $mut_:tt )?}, $into_ref:ident) => {
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

implement_continuous_chunk_strided_iter!(ContiguousChunkStridedIter -> *const T, &'a [T; N], {}, as_ref);
implement_continuous_chunk_strided_iter!(ContiguousChunkStridedIterMut -> *mut T, &'a mut [T; N], {}, as_mut);

impl<T, const N: usize> ChunkStridedSliceRef<T, N> {
    /// Return a const pointer to the base element of this view.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.0.base.as_ptr()
    }

    /// Return a mut pointer to the base element of this view.
    #[inline]
    pub fn as_ptr_mut(&mut self) -> *mut T {
        self.0.base.as_ptr()
    }

    /// Number of positions (rows) in the view.  Each position holds `N` elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.0.length
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
                .as_ptr_mut()
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

    /// Reinterpret the view as a slice of `[T; N]` chunks, or `None` if the memory is not
    /// fully contiguous.
    #[inline]
    pub fn as_chunks(&self) -> Option<&[[T; N]]> {
        if self.is_contiguous() {
            // SAFETY: I'm gauranteed to point to a contiguous set of values.
            let slc = unsafe { std::slice::from_raw_parts(self.as_ptr(), self.0.length * N) };
            // SAFETY: slc is gauranteed to be divisible by N, so there will be no remainder slice.
            unsafe { Some(slc.as_chunks_unchecked()) }
        } else {
            None
        }
    }

    /// Reinterpret the view as a mutable slice of `[T; N]` chunks, or `None` if not fully
    /// contiguous.
    #[inline]
    pub fn as_chunks_mut(&mut self) -> Option<&mut [[T; N]]> {
        if self.is_contiguous() {
            // SAFETY: I'm gauranteed to point to a contiguous set of values.
            let slc =
                unsafe { std::slice::from_raw_parts_mut(self.as_ptr_mut(), self.0.length * N) };

            // SAFETY: slc is gauranteed to be divisible by N, so there will be no remainder slice.
            unsafe { Some(slc.as_chunks_unchecked_mut()) }
        } else {
            None
        }
    }

    /// Return an iterator that yields `N`-element arrays at each position.
    #[inline]
    pub fn iter(&self) -> ChunkStridedIter<'_, T, N> {
        let start = self.0.base;
        let end = unsafe {
            start
                .as_ptr()
                .offset(self.0.stride * self.0.length as isize)
        };
        ChunkStridedIter {
            ptr: start,
            end,
            offsets: self.0.offsets,
            stride: self.0.stride,
            _member: PhantomData,
        }
    }

    /// Return an iterator over contiguous `&[T; N]` slices when the chunk is contiguous,
    /// or `None` otherwise.
    #[inline]
    pub fn slices(&self) -> Option<ContiguousChunkStridedIter<'_, T, N>> {
        if self.is_chunk_contiguous() {
            let start = unsafe { self.0.base.offset(self.0.offsets[0]) };
            let end = unsafe {
                start
                    .as_ptr()
                    .offset(self.0.stride * self.0.length as isize)
            };
            Some(ContiguousChunkStridedIter {
                ptr: start,
                end,
                stride: self.0.stride,
                _member: PhantomData,
            })
        } else {
            None
        }
    }

    /// Return a mutable iterator that yields `N`-element arrays at each position.
    #[inline]
    pub fn iter_mut(&mut self) -> ChunkStridedIterMut<'_, T, N> {
        let start = self.0.base;
        let end = unsafe {
            start
                .as_ptr()
                .offset(self.0.stride * self.0.length as isize)
        };
        ChunkStridedIterMut {
            ptr: start,
            end,
            offsets: self.0.offsets,
            stride: self.0.stride,
            _member: PhantomData,
        }
    }

    /// Return a mutable iterator over contiguous `&mut [T; N]` slices when contiguous, or `None`.
    #[inline]
    pub fn slices_mut(&mut self) -> Option<ContiguousChunkStridedIterMut<'_, T, N>> {
        if self.is_chunk_contiguous() {
            let start = unsafe { self.0.base.offset(self.0.offsets[0]) };
            let end = unsafe {
                start
                    .as_ptr()
                    .offset(self.0.stride * self.0.length as isize)
            };
            Some(ContiguousChunkStridedIterMut {
                ptr: start,
                end,
                stride: self.0.stride,
                _member: PhantomData,
            })
        } else {
            None
        }
    }
}

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
            pub fn from_slice(arr: &'a $( $mut_ )? [T], shape: &[usize], axis: usize) -> Self {
                let (ptr, arr_info) = lane_parts_from_slice(arr,  shape, axis);
                Self::new(ptr, arr_info)
            }

            /// Construct from a sub-region of a flat slice: the outer layout is `shape`, but
            /// only the first `sub_shape[axis]` elements along `axis` are iterated.
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

implement_lane_chunk_iter!(LaneChunkSliceIter -> *const T, &'a T, ChunkStridedSlice<'a, T, N>, {LaneSliceIter}, {}, as_ref);
implement_lane_chunk_iter!(LaneChunkSliceIterMut -> *mut T, &'a mut T, ChunkStridedSliceMut<'a, T, N>, {LaneSliceIterMut},  {mut}, as_mut);

unsafe impl<L: Sync + Data> Send for StridedSliceBase<L> {}
unsafe impl<T: Sync> Sync for StridedSliceRef<T> {}
unsafe impl<T: Send> Send for StridedSliceRef<T> {}

unsafe impl<L: Sync + Data, const N: usize> Send for ChunkStridedSliceBase<L, N> {}
unsafe impl<T: Sync, const N: usize> Sync for ChunkStridedSliceRef<T, N> {}
unsafe impl<T: Send, const N: usize> Send for ChunkStridedSliceRef<T, N> {}

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
                base: NonNull<T>,
                arr_info: ArrayInfo,
                start: usize,
                end: usize,
                _member: PhantomData<$memb>,
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
        LaneSliceParIter,
        LaneSliceIterProducer,
        &'a T,
        StridedSlice,
        LaneSliceIter,
        {}
    );

    implement_lane_par_iter!(
        LaneSliceParIterMut,
        LaneSliceIterMutProducer,
        &'a mut T,
        StridedSliceMut,
        LaneSliceIterMut,
        {}
    );

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
        LaneChunkSliceParIter,
        LaneChunkSliceIterProducer,
        &'a T,
        ChunkStridedSlice,
        LaneChunkSliceIter,
        { LaneSliceParIter },
        {}
    );

    implement_lane_chunk_par_iter!(
        LaneChunkSliceParIterMut,
        LaneChunkSliceIterMutProducer,
        &'a mut T,
        ChunkStridedSliceMut,
        LaneChunkSliceIterMut,
        { LaneSliceParIterMut },
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
        let mut slice = StridedSliceMut::from_slice_mut(&mut data, step);
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
        let mut slice = StridedSliceMut::from_slice_mut(&mut data, step);
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

        let mut slice = StridedSliceBase::from_slice_mut(&mut data, step);

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
            let slice = slice.as_slice().unwrap(); // step size in this test is 1
            slice.iter().sum()
        }

        assert_eq!(sum_slice(&slice), expected);
    }

    #[test]
    fn test_strided_as_slice_mut() {
        let n = 80;
        let step = 1;
        let mut data = (0..n).collect::<Vec<_>>();

        let mut slice = StridedSliceBase::from_slice_mut(&mut data, step);

        fn double_slice(slice: &mut StridedSliceRef<usize>) {
            let slice = slice.as_slice_mut().unwrap();
            slice.iter_mut().for_each(|v| *v *= 2);
        }

        double_slice(&mut slice);
        let expected = (0..n).map(|v| v * 2).collect::<Vec<_>>();
        assert_eq!(data, expected);
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
                    ChunkStridedSliceMut::<_, N>::from_slice_mut(&mut data_mut, &shape, ax, ind);

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
                    ChunkStridedSliceMut::<_, N>::from_slice_mut(&mut data_mut, &shape, ax, ind);

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
                    ChunkStridedSliceMut::<_, N>::from_slice_mut(&mut data, &shape, ax, ind);
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
                    ChunkStridedSliceMut::<_, N>::from_slice_mut(&mut data_mut, &shape, ax, ind);

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

        let chunks = slice
            .as_chunks()
            .expect("The chunk points to a slice of the same size");

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

        let mut slice = ChunkStridedSliceMut::<_, N>::from_slice_mut(&mut data_mut, &shape, 0, 0);

        assert_eq!(slice.len(), n);

        let chunks = slice
            .as_chunks_mut()
            .expect("The chunk points to a slice of the same size");

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

        let chunks = slice.slices();
        if let Some(chunks) = chunks {
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
            ChunkStridedSliceMut::<_, N>::from_slice_mut(&mut data_mut, &shape, ax, ind);

        assert_eq!(slice.len(), n);

        let chunks = slice.slices_mut();
        if let Some(chunks) = chunks {
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

        let lane_iter = LaneSliceIter::from_slice(&arr, &shape, axis);
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

        let lane_iter = LaneSliceIterMut::from_slice(&mut arr, &shape, axis);
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

        let chunks = LaneChunkSliceIter::<_, N>::from_slice(&arr, &shape, axis);
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

        let chunks = LaneChunkSliceIterMut::<_, N>::from_slice(&mut arr, &shape, axis);
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

            let lane_iter = LaneSliceParIter::from_slice(&arr, &shape, axis);
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
        fn test_strided_lane_paaszr_iter_mut(
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

            let lane_iter = LaneSliceParIterMut::from_slice(&mut arr, &shape, axis);
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

            let chunks = LaneChunkSliceParIter::<_, N>::from_slice(&arr, &shape, axis);
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
}
