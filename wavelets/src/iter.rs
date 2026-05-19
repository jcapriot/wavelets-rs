//! N-dimensional lane iteration over flat slices and ndarray arrays.
//!
//! A *lane* is a 1-D slice through an N-dimensional array along a single axis —
//! equivalent to a row, column, or fibre depending on which axis is chosen.
//! These iterators are used internally by the wavelet drivers to apply 1-D
//! transforms independently along each axis of a multi-dimensional array.

pub mod slice;
use slice::{LaneChunkSliceIter, LaneChunkSliceIterMut, LaneSliceIter, LaneSliceIterMut};

macro_rules! implement_lanes_iterator {
    ($trait:ident, $lane_iter:ident, $lane_iter_mut:ident, $chunk_iter:ident, $chunk_iter_mut:ident) => {
        /// Iterate over 1-D lanes of an N-dimensional array along a chosen axis.
        ///
        /// A *lane* is a 1-D slice along one axis while all other indices are held
        /// fixed.  Implementations are provided for `[T]` (flat row-major slices)
        /// and, when the `ndarray` feature is enabled, for `ndarray::ArrayRef<T, D>`.
        pub trait $trait {
            /// Element type stored in the array.
            type Item;

            /// Iterate over all lanes along `axis` of an array with the given `shape`.
            fn iter_lanes<'a>(&'a self, shape: &[usize], axis: usize)
            -> $lane_iter<'a, Self::Item>;

            /// Mutably iterate over all lanes along `axis`.
            fn iter_lanes_mut<'a>(
                &'a mut self,
                shape: &[usize],
                axis: usize,
            ) -> $lane_iter_mut<'a, Self::Item>;

            /// Iterate over SIMD-width chunks of lanes along `axis`.
            ///
            /// Each item is a group of `N` consecutive elements within a lane.
            fn iter_lane_chunks<'a, const N: usize>(
                &'a self,
                shape: &[usize],
                axis: usize,
            ) -> $chunk_iter<'a, Self::Item, N>;

            /// Mutably iterate over SIMD-width chunks of lanes along `axis`.
            fn iter_lane_chunks_mut<'a, const N: usize>(
                &'a mut self,
                shape: &[usize],
                axis: usize,
            ) -> $chunk_iter_mut<'a, Self::Item, N>;

            /// Iterate over lanes of a sub-region defined by `sub_shape`.
            ///
            /// Lanes are taken from the first `sub_shape[axis]` elements along `axis`,
            /// with the outer shape given by `shape`.
            fn iter_lanes_sub<'a>(
                &'a self,
                shape: &[usize],
                sub_shape: &[usize],
                axis: usize,
            ) -> $lane_iter<'a, Self::Item>;

            /// Mutably iterate over lanes of a sub-region.
            fn iter_lanes_sub_mut<'a>(
                &'a mut self,
                shape: &[usize],
                sub_shape: &[usize],
                axis: usize,
            ) -> $lane_iter_mut<'a, Self::Item>;

            /// Iterate over SIMD-width chunks of lanes within a sub-region.
            fn iter_lane_chunks_sub<'a, const N: usize>(
                &'a self,
                shape: &[usize],
                sub_shape: &[usize],
                axis: usize,
            ) -> $chunk_iter<'a, Self::Item, N>;

            /// Mutably iterate over SIMD-width chunks of lanes within a sub-region.
            fn iter_lane_chunks_sub_mut<'a, const N: usize>(
                &'a mut self,
                shape: &[usize],
                sub_shape: &[usize],
                axis: usize,
            ) -> $chunk_iter_mut<'a, Self::Item, N>;

            /// Return the axis index with the smallest stride (most cache-friendly to
            /// iterate over for the given `shape`).
            fn min_stride_axis(&self, shape: &[usize]) -> usize;
        }
        impl<T> $trait for [T] {
            type Item = T;
            fn iter_lanes<'a>(
                &'a self,
                shape: &[usize],
                axis: usize,
            ) -> $lane_iter<'a, Self::Item> {
                $lane_iter::from_slice(self, shape, axis)
            }
            fn iter_lanes_mut<'a>(
                &'a mut self,
                shape: &[usize],
                axis: usize,
            ) -> $lane_iter_mut<'a, Self::Item> {
                $lane_iter_mut::from_slice(self, shape, axis)
            }

            fn iter_lane_chunks<'a, const N: usize>(
                &'a self,
                shape: &[usize],
                axis: usize,
            ) -> $chunk_iter<'a, Self::Item, N> {
                $chunk_iter::from_slice(self, shape, axis)
            }

            fn iter_lane_chunks_mut<'a, const N: usize>(
                &'a mut self,
                shape: &[usize],
                axis: usize,
            ) -> $chunk_iter_mut<'a, Self::Item, N> {
                $chunk_iter_mut::from_slice(self, shape, axis)
            }

            fn iter_lanes_sub<'a>(
                &'a self,
                shape: &[usize],
                sub_shape: &[usize],
                axis: usize,
            ) -> $lane_iter<'a, Self::Item> {
                $lane_iter::from_sub_slice(self, shape, sub_shape, axis)
            }
            fn iter_lanes_sub_mut<'a>(
                &'a mut self,
                shape: &[usize],
                sub_shape: &[usize],
                axis: usize,
            ) -> $lane_iter_mut<'a, Self::Item> {
                $lane_iter_mut::from_sub_slice(self, shape, sub_shape, axis)
            }

            fn iter_lane_chunks_sub<'a, const N: usize>(
                &'a self,
                shape: &[usize],
                sub_shape: &[usize],
                axis: usize,
            ) -> $chunk_iter<'a, Self::Item, N> {
                $chunk_iter::from_sub_slice(self, shape, sub_shape, axis)
            }

            fn iter_lane_chunks_sub_mut<'a, const N: usize>(
                &'a mut self,
                shape: &[usize],
                sub_shape: &[usize],
                axis: usize,
            ) -> $chunk_iter_mut<'a, Self::Item, N> {
                $chunk_iter_mut::from_sub_slice(self, shape, sub_shape, axis)
            }

            fn min_stride_axis(&self, shape: &[usize]) -> usize {
                if shape.len() > 0 { shape.len() - 1 } else { 0 }
            }
        }

        #[cfg(feature = "ndarray")]
        impl<T, D: ::ndarray::Dimension> $trait for ::ndarray::ArrayRef<T, D> {
            type Item = T;
            fn iter_lanes<'a>(
                &'a self,
                shape: &[usize],
                axis: usize,
            ) -> $lane_iter<'a, Self::Item> {
                $lane_iter::from_ndarray(self, shape, axis)
            }
            fn iter_lanes_mut<'a>(
                &'a mut self,
                shape: &[usize],
                axis: usize,
            ) -> $lane_iter_mut<'a, Self::Item> {
                $lane_iter_mut::from_ndarray(self, shape, axis)
            }

            fn iter_lane_chunks<'a, const N: usize>(
                &'a self,
                shape: &[usize],
                axis: usize,
            ) -> $chunk_iter<'a, Self::Item, N> {
                $chunk_iter::from_ndarray(self, shape, axis)
            }

            fn iter_lane_chunks_mut<'a, const N: usize>(
                &'a mut self,
                shape: &[usize],
                axis: usize,
            ) -> $chunk_iter_mut<'a, Self::Item, N> {
                $chunk_iter_mut::from_ndarray(self, shape, axis)
            }

            fn iter_lanes_sub<'a>(
                &'a self,
                _shape: &[usize],
                sub_shape: &[usize],
                axis: usize,
            ) -> $lane_iter<'a, Self::Item> {
                $lane_iter::from_ndarray(self, sub_shape, axis)
            }
            fn iter_lanes_sub_mut<'a>(
                &'a mut self,
                _shape: &[usize],
                sub_shape: &[usize],
                axis: usize,
            ) -> $lane_iter_mut<'a, Self::Item> {
                $lane_iter_mut::from_ndarray(self, sub_shape, axis)
            }

            fn iter_lane_chunks_sub<'a, const N: usize>(
                &'a self,
                _shape: &[usize],
                sub_shape: &[usize],
                axis: usize,
            ) -> $chunk_iter<'a, Self::Item, N> {
                $chunk_iter::from_ndarray(self, sub_shape, axis)
            }

            fn iter_lane_chunks_sub_mut<'a, const N: usize>(
                &'a mut self,
                _shape: &[usize],
                sub_shape: &[usize],
                axis: usize,
            ) -> $chunk_iter_mut<'a, Self::Item, N> {
                $chunk_iter_mut::from_ndarray(self, sub_shape, axis)
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
        }
    };
}

implement_lanes_iterator!(
    LanesIterator,
    LaneSliceIter,
    LaneSliceIterMut,
    LaneChunkSliceIter,
    LaneChunkSliceIterMut
);

#[cfg(feature = "rayon")]
/// Parallel lane iteration using Rayon.
///
/// Provides the same interface as [`LanesIterator`] but returns Rayon parallel
/// iterators so that the caller can process independent lanes on multiple threads.
pub mod parallel {
    use super::slice::parallel::{
        LaneChunkSliceParIter, LaneChunkSliceParIterMut, LaneSliceParIter, LaneSliceParIterMut,
    };
    implement_lanes_iterator!(
        LanesParallelIterator,
        LaneSliceParIter,
        LaneSliceParIterMut,
        LaneChunkSliceParIter,
        LaneChunkSliceParIterMut
    );
}
