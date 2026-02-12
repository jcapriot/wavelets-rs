pub mod slice;
use slice::{LaneChunkSliceIter, LaneChunkSliceIterMut, LaneSliceIter, LaneSliceIterMut};

macro_rules! implement_lanes_iterator {
    ($trait:ident, $lane_iter:ident, $lane_iter_mut:ident, $chunk_iter:ident, $chunk_iter_mut:ident) => {
        pub trait $trait {
            type Item;

            fn iter_lanes<'a>(&'a self, shape: &[usize], axis: usize)
            -> $lane_iter<'a, Self::Item>;
            fn iter_lanes_mut<'a>(
                &'a mut self,
                shape: &[usize],
                axis: usize,
            ) -> $lane_iter_mut<'a, Self::Item>;
            fn iter_lane_chunks<'a, const N: usize>(
                &'a self,
                shape: &[usize],
                axis: usize,
            ) -> $chunk_iter<'a, Self::Item, N>;
            fn iter_lane_chunks_mut<'a, const N: usize>(
                &'a mut self,
                shape: &[usize],
                axis: usize,
            ) -> $chunk_iter_mut<'a, Self::Item, N>;

            fn iter_lanes_sub<'a>(
                &'a self,
                shape: &[usize],
                sub_shape: &[usize],
                axis: usize,
            ) -> $lane_iter<'a, Self::Item>;
            fn iter_lanes_sub_mut<'a>(
                &'a mut self,
                shape: &[usize],
                sub_shape: &[usize],
                axis: usize,
            ) -> $lane_iter_mut<'a, Self::Item>;
            fn iter_lane_chunks_sub<'a, const N: usize>(
                &'a self,
                shape: &[usize],
                sub_shape: &[usize],
                axis: usize,
            ) -> $chunk_iter<'a, Self::Item, N>;
            fn iter_lane_chunks_sub_mut<'a, const N: usize>(
                &'a mut self,
                shape: &[usize],
                sub_shape: &[usize],
                axis: usize,
            ) -> $chunk_iter_mut<'a, Self::Item, N>;

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
