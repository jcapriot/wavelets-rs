//! N-dimensional lane iteration over flat slices and ndarray arrays.
//!
//! A *lane* is a 1-D slice through an N-dimensional array along a single axis —
//! equivalent to a row, column, or fibre depending on which axis is chosen.
//! These iterators are used internally by the wavelet drivers to apply 1-D
//! transforms independently along each axis of a multi-dimensional array.

pub mod chunk_strided_slice;
pub mod strided_slice;
#[cfg(feature = "ndarray")]
use ndarray::{ArrayRef, Dimension};
use num_traits::Zero;
use std::marker::PhantomData;
use std::ops::ControlFlow;
use std::ptr::NonNull;

use crate::{ChunkWidth, utils::stride_from_shape};
use chunk_strided_slice::{IterLaneChunks, IterLaneChunksMut};
use strided_slice::{IterLanes, IterLanesMut};

pub use chunk_strided_slice::ChunkStridedSliceRef;
pub use strided_slice::StridedSliceRef;

#[inline]
#[track_caller]
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

unsafe impl<T> Data for SliceLifetime<&T> {
    type Elem = T;
}

unsafe impl<T> Data for SliceLifetime<&mut T> {
    type Elem = T;
}

unsafe impl<T> DataMut for SliceLifetime<&mut T> {}

#[derive(Clone, Debug)]
pub(crate) struct ArrayInfo {
    shape: Vec<usize>,
    stride: Vec<isize>,
    lane_length: usize,
    lane_stride: isize,
}

impl ArrayInfo {
    #[track_caller]
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
                    ControlFlow::Continue(())
                } else {
                    *pos -= 1;
                    *offset -= *str;
                    ControlFlow::Break(())
                }
            });
    }
}

#[track_caller]
fn lane_parts_from_slice<T>(arr: &[T], shape: &[usize], axis: usize) -> (NonNull<T>, ArrayInfo) {
    lane_parts_from_sub_slice(arr, shape, shape, axis)
}

#[track_caller]
fn lane_parts_from_sub_slice<T>(
    arr: &[T],
    shape: &[usize],
    sub_shape: &[usize],
    axis: usize,
) -> (NonNull<T>, ArrayInfo) {
    let n = arr.len();
    assert!(
        !arr.is_empty(),
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
#[track_caller]
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

/// Iterate over 1-D lanes of an N-dimensional array along a chosen axis.
///
/// A *lane* is a 1-D slice along one axis while all other indices are held
/// fixed.  Implementations are provided for `[T]` (flat row-major slices)
/// and, when the `ndarray` feature is enabled, for `ndarray::ArrayRef<T, D>`.
pub trait LanesIterator {
    /// Element type stored in the array.
    type Item;

    /// Iterate over all lanes along `axis` of an array with the given `shape`.
    ///
    /// # Panics
    ///
    /// Panics if `axis >= shape.len()`, if the slice is empty, or if the slice length
    /// does not equal `shape.iter().product()`.
    fn iter_lanes<'a>(&'a self, shape: &[usize], axis: usize) -> IterLanes<'a, Self::Item>;

    /// Mutably iterate over all lanes along `axis`.
    ///
    /// # Panics
    ///
    /// Same conditions as [`iter_lanes`](Self::iter_lanes).
    fn iter_lanes_mut<'a>(
        &'a mut self,
        shape: &[usize],
        axis: usize,
    ) -> IterLanesMut<'a, Self::Item>;

    /// Iterate over fixed-width chunks of lanes along `axis`.
    ///
    /// Each item is a group of `N` consecutive elements within a lane.
    ///
    /// # Panics
    ///
    /// Same conditions as [`iter_lanes`](Self::iter_lanes).
    fn iter_lane_chunks<'a, const N: usize>(
        &'a self,
        shape: &[usize],
        axis: usize,
    ) -> (IterLaneChunks<'a, Self::Item, N>, IterLanes<'a, Self::Item>);

    /// Mutably iterate over fixed-width chunks of lanes along `axis`.
    ///
    /// # Panics
    ///
    /// Same conditions as [`iter_lanes`](Self::iter_lanes).
    fn iter_lane_chunks_mut<'a, const N: usize>(
        &'a mut self,
        shape: &[usize],
        axis: usize,
    ) -> (
        IterLaneChunksMut<'a, Self::Item, N>,
        IterLanesMut<'a, Self::Item>,
    );

    /// Iterate over lanes of a sub-region defined by `sub_shape`.
    ///
    /// Lanes are taken from the first `sub_shape[axis]` elements along `axis`,
    /// with the outer shape given by `shape`.
    ///
    /// # Panics
    ///
    /// Panics if `axis >= shape.len()`, if `shape.len() != sub_shape.len()`, if any
    /// `sub_shape[i] > shape[i]`, if the slice is empty, or if the slice length does
    /// not equal `shape.iter().product()`.
    fn iter_lanes_sub<'a>(
        &'a self,
        shape: &[usize],
        sub_shape: &[usize],
        axis: usize,
    ) -> IterLanes<'a, Self::Item>;

    /// Mutably iterate over lanes of a sub-region.
    ///
    /// # Panics
    ///
    /// Same conditions as [`iter_lanes_sub`](Self::iter_lanes_sub).
    fn iter_lanes_sub_mut<'a>(
        &'a mut self,
        shape: &[usize],
        sub_shape: &[usize],
        axis: usize,
    ) -> IterLanesMut<'a, Self::Item>;

    /// Iterate over fixed-width chunks of lanes within a sub-region.
    ///
    /// # Panics
    ///
    /// Same conditions as [`iter_lanes_sub`](Self::iter_lanes_sub).
    fn iter_lane_chunks_sub<'a, const N: usize>(
        &'a self,
        shape: &[usize],
        sub_shape: &[usize],
        axis: usize,
    ) -> (IterLaneChunks<'a, Self::Item, N>, IterLanes<'a, Self::Item>);

    /// Mutably iterate over fixed-width chunks of lanes within a sub-region.
    ///
    /// # Panics
    ///
    /// Same conditions as [`iter_lanes_sub`](Self::iter_lanes_sub).
    fn iter_lane_chunks_sub_mut<'a, const N: usize>(
        &'a mut self,
        shape: &[usize],
        sub_shape: &[usize],
        axis: usize,
    ) -> (
        IterLaneChunksMut<'a, Self::Item, N>,
        IterLanesMut<'a, Self::Item>,
    );

    /// Return the axis index with the smallest stride (most cache-friendly to
    /// iterate over for the given `shape`).
    fn min_stride_axis(&self, shape: &[usize]) -> usize;

    /// Return whether a lane along the request axis will be contiguous.
    fn is_ax_contiguous(&self, ax: usize, shape: &[usize]) -> bool;
}

impl<T> LanesIterator for [T] {
    type Item = T;
    #[track_caller]
    fn iter_lanes<'a>(&'a self, shape: &[usize], axis: usize) -> IterLanes<'a, Self::Item> {
        IterLanes::from_slice(self, shape, axis)
    }
    #[track_caller]
    fn iter_lanes_mut<'a>(
        &'a mut self,
        shape: &[usize],
        axis: usize,
    ) -> IterLanesMut<'a, Self::Item> {
        IterLanesMut::from_slice(self, shape, axis)
    }

    #[track_caller]
    fn iter_lane_chunks<'a, const N: usize>(
        &'a self,
        shape: &[usize],
        axis: usize,
    ) -> (IterLaneChunks<'a, Self::Item, N>, IterLanes<'a, Self::Item>) {
        IterLaneChunks::from_slice(self, shape, axis)
    }

    #[track_caller]
    fn iter_lane_chunks_mut<'a, const N: usize>(
        &'a mut self,
        shape: &[usize],
        axis: usize,
    ) -> (
        IterLaneChunksMut<'a, Self::Item, N>,
        IterLanesMut<'a, Self::Item>,
    ) {
        IterLaneChunksMut::from_slice(self, shape, axis)
    }

    #[track_caller]
    fn iter_lanes_sub<'a>(
        &'a self,
        shape: &[usize],
        sub_shape: &[usize],
        axis: usize,
    ) -> IterLanes<'a, Self::Item> {
        IterLanes::from_sub_slice(self, shape, sub_shape, axis)
    }
    #[track_caller]
    fn iter_lanes_sub_mut<'a>(
        &'a mut self,
        shape: &[usize],
        sub_shape: &[usize],
        axis: usize,
    ) -> IterLanesMut<'a, Self::Item> {
        IterLanesMut::from_sub_slice(self, shape, sub_shape, axis)
    }

    #[track_caller]
    fn iter_lane_chunks_sub<'a, const N: usize>(
        &'a self,
        shape: &[usize],
        sub_shape: &[usize],
        axis: usize,
    ) -> (IterLaneChunks<'a, Self::Item, N>, IterLanes<'a, Self::Item>) {
        IterLaneChunks::from_sub_slice(self, shape, sub_shape, axis)
    }

    #[track_caller]
    fn iter_lane_chunks_sub_mut<'a, const N: usize>(
        &'a mut self,
        shape: &[usize],
        sub_shape: &[usize],
        axis: usize,
    ) -> (
        IterLaneChunksMut<'a, Self::Item, N>,
        IterLanesMut<'a, Self::Item>,
    ) {
        IterLaneChunksMut::from_sub_slice(self, shape, sub_shape, axis)
    }

    fn min_stride_axis(&self, shape: &[usize]) -> usize {
        if !shape.is_empty() {
            shape.len() - 1
        } else {
            0
        }
    }

    #[inline]
    fn is_ax_contiguous(&self, ax: usize, shape: &[usize]) -> bool {
        ax + 1 == shape.len()
    }
}

#[cfg(feature = "ndarray")]
impl<T, D: ::ndarray::Dimension> LanesIterator for ArrayRef<T, D> {
    type Item = T;
    #[track_caller]
    fn iter_lanes<'a>(&'a self, shape: &[usize], axis: usize) -> IterLanes<'a, Self::Item> {
        IterLanes::from_ndarray(self, shape, axis)
    }
    #[track_caller]
    fn iter_lanes_mut<'a>(
        &'a mut self,
        shape: &[usize],
        axis: usize,
    ) -> IterLanesMut<'a, Self::Item> {
        IterLanesMut::from_ndarray(self, shape, axis)
    }

    #[track_caller]
    fn iter_lane_chunks<'a, const N: usize>(
        &'a self,
        shape: &[usize],
        axis: usize,
    ) -> (IterLaneChunks<'a, Self::Item, N>, IterLanes<'a, Self::Item>) {
        IterLaneChunks::from_ndarray(self, shape, axis)
    }

    #[track_caller]
    fn iter_lane_chunks_mut<'a, const N: usize>(
        &'a mut self,
        shape: &[usize],
        axis: usize,
    ) -> (
        IterLaneChunksMut<'a, Self::Item, N>,
        IterLanesMut<'a, Self::Item>,
    ) {
        IterLaneChunksMut::from_ndarray(self, shape, axis)
    }

    #[track_caller]
    fn iter_lanes_sub<'a>(
        &'a self,
        _shape: &[usize],
        sub_shape: &[usize],
        axis: usize,
    ) -> IterLanes<'a, Self::Item> {
        IterLanes::from_ndarray(self, sub_shape, axis)
    }

    #[track_caller]
    fn iter_lanes_sub_mut<'a>(
        &'a mut self,
        _shape: &[usize],
        sub_shape: &[usize],
        axis: usize,
    ) -> IterLanesMut<'a, Self::Item> {
        IterLanesMut::from_ndarray(self, sub_shape, axis)
    }

    #[track_caller]
    fn iter_lane_chunks_sub<'a, const N: usize>(
        &'a self,
        _shape: &[usize],
        sub_shape: &[usize],
        axis: usize,
    ) -> (IterLaneChunks<'a, Self::Item, N>, IterLanes<'a, Self::Item>) {
        IterLaneChunks::from_ndarray(self, sub_shape, axis)
    }

    #[track_caller]
    fn iter_lane_chunks_sub_mut<'a, const N: usize>(
        &'a mut self,
        _shape: &[usize],
        sub_shape: &[usize],
        axis: usize,
    ) -> (
        IterLaneChunksMut<'a, Self::Item, N>,
        IterLanesMut<'a, Self::Item>,
    ) {
        IterLaneChunksMut::from_ndarray(self, sub_shape, axis)
    }

    fn min_stride_axis(&self, _shape: &[usize]) -> usize {
        // copy input into the output
        let (min_axis, _) = self
            .strides()
            .iter()
            .cloned()
            .enumerate()
            .reduce(|acc, v| if v.1.abs() < acc.1.abs() { v } else { acc })
            .unwrap_or((0, 0));

        min_axis
    }

    #[inline]
    fn is_ax_contiguous(&self, ax: usize, _shape: &[usize]) -> bool {
        self.strides().get(ax).map(|v| *v == 1).unwrap_or(false)
    }
}

pub(crate) fn copy_over<T, L, const N: usize>(
    input: &L,
    output: &mut L,
    in_shape: &[usize],
    out_shape: &[usize],
) where
    L: LanesIterator<Item = T> + ?Sized,
    T: Clone + Zero + ChunkWidth<T, N>,
{
    // copy input into output
    let min_axis = output.min_stride_axis(out_shape);
    let (in_lanes, out_lanes) = if input.is_ax_contiguous(min_axis, in_shape)
        || output.is_ax_contiguous(min_axis, out_shape)
    {
        (
            input.iter_lanes_sub(in_shape, out_shape, min_axis),
            output.iter_lanes_mut(out_shape, min_axis),
        )
    } else {
        let (in_chunks, in_rem) = input.iter_lane_chunks_sub::<N>(in_shape, out_shape, min_axis);
        let (out_chunks, out_rem) = output.iter_lane_chunks_mut::<N>(out_shape, min_axis);

        out_chunks.zip(in_chunks).for_each(|(mut o, i)| {
            o.iter_mut().zip(i.iter()).for_each(|(o, i)| {
                o.into_iter()
                    .zip(i.into_iter().cloned())
                    .for_each(|(o, i)| {
                        *o = i;
                    });
            });
        });
        (in_rem, out_rem)
    };
    out_lanes.zip(in_lanes).for_each(|(mut o, i)| {
        o.iter_mut()
            .zip(i.iter().cloned())
            .for_each(|(o, i)| *o = i);
    });
}

#[cfg(feature = "rayon")]
/// Parallel lane iteration using Rayon.
///
/// Provides the same interface as [`LanesIterator`] but returns Rayon parallel
/// iterators so that the caller can process independent lanes on multiple threads.
pub mod parallel {
    use super::chunk_strided_slice::parallel::{ParIterLaneChunks, ParIterLaneChunksMut};
    use super::strided_slice::parallel::{ParIterLanes, ParIterLanesMut};
    use super::*;

    /// Iterate over 1-D lanes of an N-dimensional array along a chosen axis in parallel.
    ///
    /// A *lane* is a 1-D slice along one axis while all other indices are held
    /// fixed.  Implementations are provided for `[T]` (flat row-major slices)
    /// and, when the `ndarray` feature is enabled, for `ndarray::ArrayRef<T, D>`.
    pub trait LanesParallelIterator: LanesIterator {
        /// Iterate over all lanes along `axis` of an array with the given `shape`.
        ///
        /// # Panics
        ///
        /// Panics if `axis >= shape.len()`, if the slice is empty, or if the slice length
        /// does not equal `shape.iter().product()`.
        fn par_iter_lanes<'a>(
            &'a self,
            shape: &[usize],
            axis: usize,
        ) -> ParIterLanes<'a, Self::Item>;

        /// Mutably iterate over all lanes along `axis`.
        ///
        /// # Panics
        ///
        /// Same conditions as [`par_iter_lanes`](Self::par_iter_lanes).
        fn par_iter_lanes_mut<'a>(
            &'a mut self,
            shape: &[usize],
            axis: usize,
        ) -> ParIterLanesMut<'a, Self::Item>;

        /// Iterate over SIMD-width chunks of lanes along `axis`.
        ///
        /// Each item is a group of `N` consecutive elements within a lane.
        ///
        /// # Panics
        ///
        /// Same conditions as [`par_iter_lanes`](Self::par_iter_lanes).
        fn par_iter_lane_chunks<'a, const N: usize>(
            &'a self,
            shape: &[usize],
            axis: usize,
        ) -> (
            ParIterLaneChunks<'a, Self::Item, N>,
            ParIterLanes<'a, Self::Item>,
        );

        /// Mutably iterate over SIMD-width chunks of lanes along `axis`.
        ///
        /// # Panics
        ///
        /// Same conditions as [`par_iter_lanes`](Self::par_iter_lanes).
        fn par_iter_lane_chunks_mut<'a, const N: usize>(
            &'a mut self,
            shape: &[usize],
            axis: usize,
        ) -> (
            ParIterLaneChunksMut<'a, Self::Item, N>,
            ParIterLanesMut<'a, Self::Item>,
        );

        /// Iterate over lanes of a sub-region defined by `sub_shape`.
        ///
        /// Lanes are taken from the first `sub_shape[axis]` elements along `axis`,
        /// with the outer shape given by `shape`.
        ///
        /// # Panics
        ///
        /// Panics if `axis >= shape.len()`, if `shape.len() != sub_shape.len()`, if any
        /// `sub_shape[i] > shape[i]`, if the slice is empty, or if the slice length does
        /// not equal `shape.iter().product()`.
        fn par_iter_lanes_sub<'a>(
            &'a self,
            shape: &[usize],
            sub_shape: &[usize],
            axis: usize,
        ) -> ParIterLanes<'a, Self::Item>;

        /// Mutably iterate over lanes of a sub-region.
        ///
        /// # Panics
        ///
        /// Same conditions as [`par_iter_lanes_sub`](Self::par_iter_lanes_sub).
        fn par_iter_lanes_sub_mut<'a>(
            &'a mut self,
            shape: &[usize],
            sub_shape: &[usize],
            axis: usize,
        ) -> ParIterLanesMut<'a, Self::Item>;

        /// Iterate over SIMD-width chunks of lanes within a sub-region.
        ///
        /// # Panics
        ///
        /// Same conditions as [`par_iter_lanes_sub`](Self::par_iter_lanes_sub).
        fn par_iter_lane_chunks_sub<'a, const N: usize>(
            &'a self,
            shape: &[usize],
            sub_shape: &[usize],
            axis: usize,
        ) -> (
            ParIterLaneChunks<'a, Self::Item, N>,
            ParIterLanes<'a, Self::Item>,
        );

        /// Mutably iterate over SIMD-width chunks of lanes within a sub-region.
        ///
        /// # Panics
        ///
        /// Same conditions as [`par_iter_lanes_sub`](Self::par_iter_lanes_sub).
        fn par_iter_lane_chunks_sub_mut<'a, const N: usize>(
            &'a mut self,
            shape: &[usize],
            sub_shape: &[usize],
            axis: usize,
        ) -> (
            ParIterLaneChunksMut<'a, Self::Item, N>,
            ParIterLanesMut<'a, Self::Item>,
        );
    }

    impl<T> LanesParallelIterator for [T] {
        #[track_caller]
        fn par_iter_lanes<'a>(
            &'a self,
            shape: &[usize],
            axis: usize,
        ) -> ParIterLanes<'a, Self::Item> {
            ParIterLanes::from_slice(self, shape, axis)
        }
        #[track_caller]
        fn par_iter_lanes_mut<'a>(
            &'a mut self,
            shape: &[usize],
            axis: usize,
        ) -> ParIterLanesMut<'a, Self::Item> {
            ParIterLanesMut::from_slice(self, shape, axis)
        }

        #[track_caller]
        fn par_iter_lane_chunks<'a, const N: usize>(
            &'a self,
            shape: &[usize],
            axis: usize,
        ) -> (
            ParIterLaneChunks<'a, Self::Item, N>,
            ParIterLanes<'a, Self::Item>,
        ) {
            ParIterLaneChunks::from_slice(self, shape, axis)
        }

        #[track_caller]
        fn par_iter_lane_chunks_mut<'a, const N: usize>(
            &'a mut self,
            shape: &[usize],
            axis: usize,
        ) -> (
            ParIterLaneChunksMut<'a, Self::Item, N>,
            ParIterLanesMut<'a, Self::Item>,
        ) {
            ParIterLaneChunksMut::from_slice(self, shape, axis)
        }

        #[track_caller]
        fn par_iter_lanes_sub<'a>(
            &'a self,
            shape: &[usize],
            sub_shape: &[usize],
            axis: usize,
        ) -> ParIterLanes<'a, Self::Item> {
            ParIterLanes::from_sub_slice(self, shape, sub_shape, axis)
        }
        #[track_caller]
        fn par_iter_lanes_sub_mut<'a>(
            &'a mut self,
            shape: &[usize],
            sub_shape: &[usize],
            axis: usize,
        ) -> ParIterLanesMut<'a, Self::Item> {
            ParIterLanesMut::from_sub_slice(self, shape, sub_shape, axis)
        }

        #[track_caller]
        fn par_iter_lane_chunks_sub<'a, const N: usize>(
            &'a self,
            shape: &[usize],
            sub_shape: &[usize],
            axis: usize,
        ) -> (
            ParIterLaneChunks<'a, Self::Item, N>,
            ParIterLanes<'a, Self::Item>,
        ) {
            ParIterLaneChunks::from_sub_slice(self, shape, sub_shape, axis)
        }

        #[track_caller]
        fn par_iter_lane_chunks_sub_mut<'a, const N: usize>(
            &'a mut self,
            shape: &[usize],
            sub_shape: &[usize],
            axis: usize,
        ) -> (
            ParIterLaneChunksMut<'a, Self::Item, N>,
            ParIterLanesMut<'a, Self::Item>,
        ) {
            ParIterLaneChunksMut::from_sub_slice(self, shape, sub_shape, axis)
        }
    }

    #[cfg(feature = "ndarray")]
    impl<T, D: ::ndarray::Dimension> LanesParallelIterator for ArrayRef<T, D> {
        #[track_caller]
        fn par_iter_lanes<'a>(
            &'a self,
            shape: &[usize],
            axis: usize,
        ) -> ParIterLanes<'a, Self::Item> {
            ParIterLanes::from_ndarray(self, shape, axis)
        }
        #[track_caller]
        fn par_iter_lanes_mut<'a>(
            &'a mut self,
            shape: &[usize],
            axis: usize,
        ) -> ParIterLanesMut<'a, Self::Item> {
            ParIterLanesMut::from_ndarray(self, shape, axis)
        }

        #[track_caller]
        fn par_iter_lane_chunks<'a, const N: usize>(
            &'a self,
            shape: &[usize],
            axis: usize,
        ) -> (
            ParIterLaneChunks<'a, Self::Item, N>,
            ParIterLanes<'a, Self::Item>,
        ) {
            ParIterLaneChunks::from_ndarray(self, shape, axis)
        }

        #[track_caller]
        fn par_iter_lane_chunks_mut<'a, const N: usize>(
            &'a mut self,
            shape: &[usize],
            axis: usize,
        ) -> (
            ParIterLaneChunksMut<'a, Self::Item, N>,
            ParIterLanesMut<'a, Self::Item>,
        ) {
            ParIterLaneChunksMut::from_ndarray(self, shape, axis)
        }

        #[track_caller]
        fn par_iter_lanes_sub<'a>(
            &'a self,
            _shape: &[usize],
            sub_shape: &[usize],
            axis: usize,
        ) -> ParIterLanes<'a, Self::Item> {
            ParIterLanes::from_ndarray(self, sub_shape, axis)
        }
        #[track_caller]
        fn par_iter_lanes_sub_mut<'a>(
            &'a mut self,
            _shape: &[usize],
            sub_shape: &[usize],
            axis: usize,
        ) -> ParIterLanesMut<'a, Self::Item> {
            ParIterLanesMut::from_ndarray(self, sub_shape, axis)
        }

        #[track_caller]
        fn par_iter_lane_chunks_sub<'a, const N: usize>(
            &'a self,
            _shape: &[usize],
            sub_shape: &[usize],
            axis: usize,
        ) -> (
            ParIterLaneChunks<'a, Self::Item, N>,
            ParIterLanes<'a, Self::Item>,
        ) {
            ParIterLaneChunks::from_ndarray(self, sub_shape, axis)
        }

        #[track_caller]
        fn par_iter_lane_chunks_sub_mut<'a, const N: usize>(
            &'a mut self,
            _shape: &[usize],
            sub_shape: &[usize],
            axis: usize,
        ) -> (
            ParIterLaneChunksMut<'a, Self::Item, N>,
            ParIterLanesMut<'a, Self::Item>,
        ) {
            ParIterLaneChunksMut::from_ndarray(self, sub_shape, axis)
        }
    }

    pub(crate) fn copy_over<T, L, const N: usize>(
        input: &L,
        output: &mut L,
        in_shape: &[usize],
        out_shape: &[usize],
    ) where
        L: LanesParallelIterator<Item = T> + ?Sized,
        T: Clone + Zero + ChunkWidth<T, N> + Send + Sync,
    {
        use rayon::iter::{IndexedParallelIterator, ParallelIterator};
        // copy input into output
        let min_axis = output.min_stride_axis(out_shape);
        let (in_chunks, in_rem) =
            input.par_iter_lane_chunks_sub::<N>(in_shape, out_shape, min_axis);
        let (out_chunks, out_rem) = output.par_iter_lane_chunks_mut::<N>(out_shape, min_axis);

        out_chunks.zip(in_chunks).for_each(|(mut o, i)| {
            o.iter_mut().zip(i.iter()).for_each(|(o, i)| {
                o.into_iter()
                    .zip(i.into_iter().cloned())
                    .for_each(|(o, i)| {
                        *o = i;
                    });
            });
        });
        out_rem.zip(in_rem).for_each(|(mut o, i)| {
            o.iter_mut()
                .zip(i.iter().cloned())
                .for_each(|(o, i)| *o = i);
        });
    }
}
