pub mod slice;
//pub mod slice_old;

use slice::*;

pub trait LanesIterator {
    type Item;

    fn iter_lanes<'a>(&'a self, shape: &[usize], axis: usize) -> LaneSliceIter<'a, Self::Item>;
    fn iter_lanes_mut<'a>(
        &'a mut self,
        shape: &[usize],
        axis: usize,
    ) -> LaneSliceIterMut<'a, Self::Item>;
    fn iter_lane_chunks<'a, const N: usize>(
        &'a self,
        shape: &[usize],
        axis: usize,
    ) -> LaneChunkSliceIter<'a, Self::Item, N>;
    fn iter_lane_chunks_mut<'a, const N: usize>(
        &'a mut self,
        shape: &[usize],
        axis: usize,
    ) -> LaneChunkSliceIterMut<'a, Self::Item, N>;

    fn iter_lanes_strided<'a>(
        &'a self,
        shape: &[usize],
        stride: &[usize],
        axis: usize,
    ) -> LaneSliceIter<'a, Self::Item>;
    fn iter_lanes_mut_strided<'a>(
        &'a mut self,
        shape: &[usize],
        stride: &[usize],
        axis: usize,
    ) -> LaneSliceIterMut<'a, Self::Item>;
    fn iter_lane_chunks_strided<'a, const N: usize>(
        &'a self,
        shape: &[usize],
        stride: &[usize],
        axis: usize,
    ) -> LaneChunkSliceIter<'a, Self::Item, N>;
    fn iter_lane_chunks_mut_strided<'a, const N: usize>(
        &'a mut self,
        shape: &[usize],
        stride: &[usize],
        axis: usize,
    ) -> LaneChunkSliceIterMut<'a, Self::Item, N>;
}
impl<T> LanesIterator for [T] {
    type Item = T;
    fn iter_lanes<'a>(&'a self, shape: &[usize], axis: usize) -> LaneSliceIter<'a, Self::Item> {
        LaneSliceIter::from_slice(self, shape, axis)
    }
    fn iter_lanes_mut<'a>(
        &'a mut self,
        shape: &[usize],
        axis: usize,
    ) -> LaneSliceIterMut<'a, Self::Item> {
        LaneSliceIterMut::from_slice(self, shape, axis)
    }

    fn iter_lane_chunks<'a, const N: usize>(
        &'a self,
        shape: &[usize],
        axis: usize,
    ) -> LaneChunkSliceIter<'a, Self::Item, N> {
        LaneChunkSliceIter::from_slice(self, shape, axis)
    }

    fn iter_lane_chunks_mut<'a, const N: usize>(
        &'a mut self,
        shape: &[usize],
        axis: usize,
    ) -> LaneChunkSliceIterMut<'a, Self::Item, N> {
        LaneChunkSliceIterMut::from_slice(self, shape, axis)
    }

    fn iter_lanes_strided<'a>(
        &'a self,
        shape: &[usize],
        stride: &[usize],
        axis: usize,
    ) -> LaneSliceIter<'a, Self::Item> {
        LaneSliceIter::from_slice_with_stride(self, shape, stride, axis)
    }
    fn iter_lanes_mut_strided<'a>(
        &'a mut self,
        shape: &[usize],
        stride: &[usize],
        axis: usize,
    ) -> LaneSliceIterMut<'a, Self::Item> {
        LaneSliceIterMut::from_slice_with_stride(self, shape, stride, axis)
    }

    fn iter_lane_chunks_strided<'a, const N: usize>(
        &'a self,
        shape: &[usize],
        stride: &[usize],
        axis: usize,
    ) -> LaneChunkSliceIter<'a, Self::Item, N> {
        LaneChunkSliceIter::from_slice_with_stride(self, shape, stride, axis)
    }

    fn iter_lane_chunks_mut_strided<'a, const N: usize>(
        &'a mut self,
        shape: &[usize],
        stride: &[usize],
        axis: usize,
    ) -> LaneChunkSliceIterMut<'a, Self::Item, N> {
        LaneChunkSliceIterMut::from_slice_with_stride(self, shape, stride, axis)
    }
}

#[cfg(feature = "rayon")]
pub mod parallel {
    use super::slice::parallel::*;
    pub trait ParallelLanesIterator {
        type Item;

        fn iter_lanes<'a>(
            &'a self,
            shape: &[usize],
            axis: usize,
        ) -> LaneSliceParIter<'a, Self::Item>;
        fn iter_lanes_mut<'a>(
            &'a mut self,
            shape: &[usize],
            axis: usize,
        ) -> LaneSliceParIterMut<'a, Self::Item>;
        fn iter_lane_chunks<'a, const N: usize>(
            &'a self,
            shape: &[usize],
            axis: usize,
        ) -> LaneChunkSliceParIter<'a, Self::Item, N>;
        fn iter_lane_chunks_mut<'a, const N: usize>(
            &'a mut self,
            shape: &[usize],
            axis: usize,
        ) -> LaneChunkSliceParIterMut<'a, Self::Item, N>;

        fn iter_lanes_strided<'a>(
            &'a self,
            shape: &[usize],
            stride: &[usize],
            axis: usize,
        ) -> LaneSliceParIter<'a, Self::Item>;
        fn iter_lanes_mut_strided<'a>(
            &'a mut self,
            shape: &[usize],
            stride: &[usize],
            axis: usize,
        ) -> LaneSliceParIterMut<'a, Self::Item>;
        fn iter_lane_chunks_strided<'a, const N: usize>(
            &'a self,
            shape: &[usize],
            stride: &[usize],
            axis: usize,
        ) -> LaneChunkSliceParIter<'a, Self::Item, N>;
        fn iter_lane_chunks_mut_strided<'a, const N: usize>(
            &'a mut self,
            shape: &[usize],
            stride: &[usize],
            axis: usize,
        ) -> LaneChunkSliceParIterMut<'a, Self::Item, N>;
    }
    impl<T> ParallelLanesIterator for [T] {
        type Item = T;
        fn iter_lanes<'a>(
            &'a self,
            shape: &[usize],
            axis: usize,
        ) -> LaneSliceParIter<'a, Self::Item> {
            LaneSliceParIter::from_slice(self, shape, axis)
        }
        fn iter_lanes_mut<'a>(
            &'a mut self,
            shape: &[usize],
            axis: usize,
        ) -> LaneSliceParIterMut<'a, Self::Item> {
            LaneSliceParIterMut::from_slice(self, shape, axis)
        }

        fn iter_lane_chunks<'a, const N: usize>(
            &'a self,
            shape: &[usize],
            axis: usize,
        ) -> LaneChunkSliceParIter<'a, Self::Item, N> {
            LaneChunkSliceParIter::from_slice(self, shape, axis)
        }

        fn iter_lane_chunks_mut<'a, const N: usize>(
            &'a mut self,
            shape: &[usize],
            axis: usize,
        ) -> LaneChunkSliceParIterMut<'a, Self::Item, N> {
            LaneChunkSliceParIterMut::from_slice(self, shape, axis)
        }

        fn iter_lanes_strided<'a>(
            &'a self,
            shape: &[usize],
            stride: &[usize],
            axis: usize,
        ) -> LaneSliceParIter<'a, Self::Item> {
            LaneSliceParIter::from_slice_with_stride(self, shape, stride, axis)
        }
        fn iter_lanes_mut_strided<'a>(
            &'a mut self,
            shape: &[usize],
            stride: &[usize],
            axis: usize,
        ) -> LaneSliceParIterMut<'a, Self::Item> {
            LaneSliceParIterMut::from_slice_with_stride(self, shape, stride, axis)
        }

        fn iter_lane_chunks_strided<'a, const N: usize>(
            &'a self,
            shape: &[usize],
            stride: &[usize],
            axis: usize,
        ) -> LaneChunkSliceParIter<'a, Self::Item, N> {
            LaneChunkSliceParIter::from_slice_with_stride(self, shape, stride, axis)
        }

        fn iter_lane_chunks_mut_strided<'a, const N: usize>(
            &'a mut self,
            shape: &[usize],
            stride: &[usize],
            axis: usize,
        ) -> LaneChunkSliceParIterMut<'a, Self::Item, N> {
            LaneChunkSliceParIterMut::from_slice_with_stride(self, shape, stride, axis)
        }
    }
}
