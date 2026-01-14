use ndarray::{
    ArrayRef, ArrayView, ArrayView1, ArrayViewMut, ArrayViewMut1, Axis, Dimension, Ix1, RemoveAxis,
    ShapeBuilder, StrideShape,
};

pub struct LaneChunksExact<'a, A, D, const N: usize> {
    base: ArrayView<'a, A, D>,
    inner_shape: StrideShape<Ix1>,
    bounds: (usize, usize),
}

#[inline]
fn unravel<D: Dimension>(flat_index: usize, shape: D) -> D {
    let mut inds: D = D::zeros(shape.ndim());

    let mut flat_index = flat_index;
    inds.slice_mut()
        .iter_mut()
        .zip(shape.slice().iter())
        .rev()
        .for_each(|(i_dir, n_dir)| {
            *i_dir = flat_index % n_dir;
            flat_index /= n_dir;
        });
    inds
}

impl<'a, A, D: Dimension, const N: usize> LaneChunksExact<'a, A, D, N> {
    fn get_chunk(&self, ind: usize) -> [ArrayView1<'a, A>; N] {
        let ind = N * ind;
        let shape = self.base.raw_dim();

        let mut inds = unravel(ind, shape);
        std::array::from_fn(|_| {
            let ptr = self.base.get(inds.clone()).unwrap();
            //advance inds;
            for (i_d, n_d) in inds
                .slice_mut()
                .iter_mut()
                .zip(self.base.shape().iter())
                .rev()
            {
                *i_d += 1;
                if i_d == n_d {
                    *i_d = 0
                } else {
                    break;
                }
            }
            unsafe { ArrayView1::from_shape_ptr(self.inner_shape, ptr) }
        })
    }
}

impl<'a, A, D: Dimension, const N: usize> Iterator for LaneChunksExact<'a, A, D, N> {
    type Item = [ArrayView1<'a, A>; N];

    fn next(&mut self) -> Option<Self::Item> {
        if self.bounds.0 == self.bounds.1 {
            None
        } else {
            let chunk = self.get_chunk(self.bounds.0);
            self.bounds.0 += 1;
            Some(chunk)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.bounds.1 - self.bounds.0;
        (len, Some(len))
    }
}

impl<'a, const N: usize, A, D: Dimension> ExactSizeIterator for LaneChunksExact<'a, A, D, N> {}
impl<'a, const N: usize, A, D: Dimension> DoubleEndedIterator for LaneChunksExact<'a, A, D, N> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.bounds.0 == self.bounds.1 {
            None
        } else {
            self.bounds.1 -= 1;
            let chunk = self.get_chunk(self.bounds.1);
            Some(chunk)
        }
    }
}

pub struct LaneChunksExactMut<'a, A, D, const N: usize> {
    base: ArrayViewMut<'a, A, D>,
    inner_shape: StrideShape<Ix1>,
    bounds: (usize, usize),
}

impl<'a, A, D: Dimension, const N: usize> LaneChunksExactMut<'a, A, D, N> {
    fn get_chunk(&mut self, ind: usize) -> [ArrayViewMut1<'a, A>; N] {
        let ind = N * ind;
        let shape = self.base.raw_dim();

        let mut inds = unravel(ind, shape.clone());
        std::array::from_fn(|_| {
            let ptr = self.base.get_mut(inds.clone()).unwrap();
            //advance inds;
            for (i_d, n_d) in inds.slice_mut().iter_mut().zip(shape.slice()).rev() {
                *i_d += 1;
                if i_d == n_d {
                    *i_d = 0
                } else {
                    break;
                }
            }
            unsafe { ArrayViewMut1::from_shape_ptr(self.inner_shape, ptr) }
        })
    }
}

impl<'a, A, D: Dimension, const N: usize> Iterator for LaneChunksExactMut<'a, A, D, N> {
    type Item = [ArrayViewMut1<'a, A>; N];

    fn next(&mut self) -> Option<Self::Item> {
        if self.bounds.0 == self.bounds.1 {
            None
        } else {
            let chunk = self.get_chunk(self.bounds.0);
            self.bounds.0 += 1;
            Some(chunk)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.bounds.1 - self.bounds.0;
        (len, Some(len))
    }
}

impl<'a, const N: usize, A, D: Dimension> ExactSizeIterator for LaneChunksExactMut<'a, A, D, N> {}
impl<'a, const N: usize, A, D: Dimension> DoubleEndedIterator for LaneChunksExactMut<'a, A, D, N> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.bounds.0 == self.bounds.1 {
            None
        } else {
            self.bounds.1 -= 1;
            let chunk = self.get_chunk(self.bounds.1);
            Some(chunk)
        }
    }
}

pub trait LaneChunksExactIterator<A, D: Dimension> {
    fn lane_chunks_iter<const N: usize>(
        &self,
        axis: Axis,
    ) -> (
        LaneChunksExact<'_, A, D::Smaller, N>,
        LaneChunksExact<'_, A, D::Smaller, 1>,
    );
}

impl<A, D: Dimension + RemoveAxis> LaneChunksExactIterator<A, D> for ArrayRef<A, D> {
    fn lane_chunks_iter<const N: usize>(
        &self,
        axis: Axis,
    ) -> (
        LaneChunksExact<'_, A, D::Smaller, N>,
        LaneChunksExact<'_, A, D::Smaller, 1>,
    ) {
        let shape = (self.len_of(axis),).strides((usize::try_from(self.stride_of(axis))
            .expect("Cannot iterate lanes with negative strides."),));
        let view = self.view();

        let sub_arr = view.remove_axis(axis);
        let n_chunks = sub_arr.len() / N;
        let end = sub_arr.len();

        (
            LaneChunksExact {
                base: sub_arr.clone(),
                inner_shape: shape,
                bounds: (0, n_chunks),
            },
            LaneChunksExact {
                base: sub_arr,
                inner_shape: shape,
                bounds: (n_chunks * N, end),
            },
        )
    }
}

pub trait LaneChunksExactIteratorMut<A, D: Dimension> {
    fn lane_chunks_iter_mut<const N: usize>(
        &mut self,
        axis: Axis,
    ) -> (
        LaneChunksExactMut<'_, A, D::Smaller, N>,
        LaneChunksExactMut<'_, A, D::Smaller, 1>,
    );
}

impl<A, D: Dimension + RemoveAxis> LaneChunksExactIteratorMut<A, D> for ArrayRef<A, D> {
    fn lane_chunks_iter_mut<const N: usize>(
        &mut self,
        axis: Axis,
    ) -> (
        LaneChunksExactMut<'_, A, D::Smaller, N>,
        LaneChunksExactMut<'_, A, D::Smaller, 1>,
    ) {
        let shape = (self.len_of(axis),).strides((usize::try_from(self.stride_of(axis))
            .expect("Cannot iterate lanes with negative strides."),));
        let view = self.view_mut();

        let mut sub_arr = view.remove_axis(axis);
        let mut strides = D::Smaller::zeros(sub_arr.ndim());
        strides
            .slice_mut()
            .iter_mut()
            .zip(sub_arr.strides().iter())
            .for_each(|(sc, s)| {
                *sc = usize::try_from(*s)
                    .expect("Cannot handle negative strides for mutable lane chunk iterator.")
            });
        let sub_shape = sub_arr.raw_dim().strides(strides);
        let sub_arr_clone;
        // need to clone the mutable sub array for the remainder iterator
        // which is gauranteed not to overlap with the chunk iterator
        unsafe {
            sub_arr_clone = ArrayViewMut::from_shape_ptr(sub_shape, sub_arr.as_mut_ptr());
        }

        let n_chunks = sub_arr.len() / N;
        let rem_start = n_chunks * N;
        let end = sub_arr.len();

        (
            LaneChunksExactMut {
                base: sub_arr,
                inner_shape: shape,
                bounds: (0, n_chunks),
            },
            LaneChunksExactMut {
                base: sub_arr_clone,
                inner_shape: shape,
                bounds: (rem_start, end),
            },
        )
    }
}

pub mod par {
    use super::*;
    use rayon::iter::plumbing::{Consumer, Producer, ProducerCallback, UnindexedConsumer, bridge};
    pub use rayon::iter::{IndexedParallelIterator, ParallelIterator};

    pub struct ParLaneChunksExact<'a, A, D, const N: usize> {
        base: ArrayView<'a, A, D>,
        inner_shape: StrideShape<Ix1>,
        bounds: (usize, usize),
    }

    impl<'a, const N: usize, A: Sync, D: Dimension> ParallelIterator
        for ParLaneChunksExact<'a, A, D, N>
    {
        type Item = [ArrayView1<'a, A>; N];
        fn drive_unindexed<C>(self, consumer: C) -> C::Result
        where
            C: UnindexedConsumer<Self::Item>,
        {
            bridge(self, consumer)
        }
    }

    impl<'a, const N: usize, A: Sync, D: Dimension> IndexedParallelIterator
        for ParLaneChunksExact<'a, A, D, N>
    {
        fn drive<C>(self, consumer: C) -> C::Result
        where
            C: Consumer<Self::Item>,
        {
            bridge(self, consumer)
        }

        fn len(&self) -> usize {
            self.bounds.1 - self.bounds.0
        }

        fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
            callback.callback(LaneChunksExactProducer {
                base: self.base,
                inner_shape: self.inner_shape,
                bounds: self.bounds,
            })
        }
    }

    struct LaneChunksExactProducer<'a, A: Sync, D: Dimension, const N: usize> {
        base: ArrayView<'a, A, D>,
        inner_shape: StrideShape<Ix1>,
        bounds: (usize, usize),
    }

    impl<'a, const N: usize, A: Sync, D: Dimension> Producer for LaneChunksExactProducer<'a, A, D, N> {
        type Item = [ArrayView1<'a, A>; N];
        type IntoIter = LaneChunksExact<'a, A, D, N>;

        fn into_iter(self) -> Self::IntoIter {
            LaneChunksExact {
                base: self.base,
                inner_shape: self.inner_shape,
                bounds: self.bounds,
            }
        }

        fn split_at(self, index: usize) -> (Self, Self) {
            let split = self.bounds.0 + index;
            let bounds_left = (self.bounds.0, split);
            let bounds_right = (split, self.bounds.1);
            (
                Self {
                    base: self.base.clone(),
                    inner_shape: self.inner_shape,
                    bounds: bounds_left,
                },
                Self {
                    base: self.base,
                    inner_shape: self.inner_shape,
                    bounds: bounds_right,
                },
            )
        }
    }

    pub struct ParLaneChunksExactMut<'a, A, D, const N: usize> {
        base: ArrayViewMut<'a, A, D>,
        outer_shape: StrideShape<D>,
        inner_shape: StrideShape<Ix1>,
        bounds: (usize, usize),
    }

    impl<'a, const N: usize, A: Sync + Send, D: Dimension> ParallelIterator
        for ParLaneChunksExactMut<'a, A, D, N>
    {
        type Item = [ArrayViewMut1<'a, A>; N];
        fn drive_unindexed<C>(self, consumer: C) -> C::Result
        where
            C: UnindexedConsumer<Self::Item>,
        {
            bridge(self, consumer)
        }
    }

    impl<'a, const N: usize, A: Sync + Send, D: Dimension> IndexedParallelIterator
        for ParLaneChunksExactMut<'a, A, D, N>
    {
        fn drive<C>(self, consumer: C) -> C::Result
        where
            C: Consumer<Self::Item>,
        {
            bridge(self, consumer)
        }

        fn len(&self) -> usize {
            self.bounds.1 - self.bounds.0
        }

        fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
            callback.callback(LaneChunksExactMutProducer {
                base: self.base,
                outer_shape: self.outer_shape,
                inner_shape: self.inner_shape,
                bounds: self.bounds,
            })
        }
    }

    struct LaneChunksExactMutProducer<'a, A: Sync, D: Dimension, const N: usize> {
        base: ArrayViewMut<'a, A, D>,
        outer_shape: StrideShape<D>,
        inner_shape: StrideShape<Ix1>,
        bounds: (usize, usize),
    }

    impl<'a, const N: usize, A: Sync + Send, D: Dimension> Producer
        for LaneChunksExactMutProducer<'a, A, D, N>
    {
        type Item = [ArrayViewMut1<'a, A>; N];
        type IntoIter = LaneChunksExactMut<'a, A, D, N>;

        fn into_iter(self) -> Self::IntoIter {
            LaneChunksExactMut {
                base: self.base,
                inner_shape: self.inner_shape,
                bounds: self.bounds,
            }
        }

        fn split_at(mut self, index: usize) -> (Self, Self) {
            let split = self.bounds.0 + index;
            let bounds_left = (self.bounds.0, split);
            let bounds_right = (split, self.bounds.1);

            let base_clone;
            // need to clone the mutable sub array for the remainder iterator
            // which is gauranteed not to overlap with the chunk iterator
            unsafe {
                base_clone =
                    ArrayViewMut::from_shape_ptr(self.outer_shape.clone(), self.base.as_mut_ptr());
            }
            (
                Self {
                    base: base_clone,
                    outer_shape: self.outer_shape.clone(),
                    inner_shape: self.inner_shape,
                    bounds: bounds_left,
                },
                Self {
                    base: self.base,
                    outer_shape: self.outer_shape,
                    inner_shape: self.inner_shape,
                    bounds: bounds_right,
                },
            )
        }
    }

    pub trait ParLaneChunksExactIterator<A, D: Dimension> {
        fn lane_chunks_par_iter<const N: usize>(
            &self,
            axis: Axis,
        ) -> (
            ParLaneChunksExact<'_, A, D::Smaller, N>,
            ParLaneChunksExact<'_, A, D::Smaller, 1>,
        );
    }

    impl<A, D: Dimension + RemoveAxis> ParLaneChunksExactIterator<A, D> for ArrayRef<A, D> {
        fn lane_chunks_par_iter<const N: usize>(
            &self,
            axis: Axis,
        ) -> (
            ParLaneChunksExact<'_, A, D::Smaller, N>,
            ParLaneChunksExact<'_, A, D::Smaller, 1>,
        ) {
            let shape = (self.len_of(axis),).strides((usize::try_from(self.stride_of(axis))
                .expect("Cannot iterate lanes with negative strides."),));
            let view = self.view();

            let sub_arr = view.remove_axis(axis);
            let n_chunks = sub_arr.len() / N;
            let end = sub_arr.len();

            (
                ParLaneChunksExact {
                    base: sub_arr.clone(),
                    inner_shape: shape,
                    bounds: (0, n_chunks),
                },
                ParLaneChunksExact {
                    base: sub_arr,
                    inner_shape: shape,
                    bounds: (n_chunks * N, end),
                },
            )
        }
    }

    pub trait ParLaneChunksExactIteratorMut<A, D: Dimension> {
        fn lane_chunks_par_iter_mut<const N: usize>(
            &mut self,
            axis: Axis,
        ) -> (
            ParLaneChunksExactMut<'_, A, D::Smaller, N>,
            ParLaneChunksExactMut<'_, A, D::Smaller, 1>,
        );
    }

    impl<A, D: Dimension + RemoveAxis> ParLaneChunksExactIteratorMut<A, D> for ArrayRef<A, D> {
        fn lane_chunks_par_iter_mut<const N: usize>(
            &mut self,
            axis: Axis,
        ) -> (
            ParLaneChunksExactMut<'_, A, D::Smaller, N>,
            ParLaneChunksExactMut<'_, A, D::Smaller, 1>,
        ) {
            let shape = (self.len_of(axis),).strides((usize::try_from(self.stride_of(axis))
                .expect("Cannot iterate lanes with negative strides."),));
            let view = self.view_mut();

            let mut sub_arr = view.remove_axis(axis);
            let mut strides = D::Smaller::zeros(sub_arr.ndim());
            strides
                .slice_mut()
                .iter_mut()
                .zip(sub_arr.strides().iter())
                .for_each(|(sc, s)| {
                    *sc = usize::try_from(*s)
                        .expect("Cannot handle negative strides for mutable lane chunk iterator.")
                });
            let sub_shape = sub_arr.raw_dim().strides(strides);
            let sub_arr_clone;
            // need to clone the mutable sub array for the remainder iterator
            // which is gauranteed not to overlap with the chunk iterator
            unsafe {
                sub_arr_clone =
                    ArrayViewMut::from_shape_ptr(sub_shape.clone(), sub_arr.as_mut_ptr());
            }
            let n_chunks = sub_arr.len() / N;
            let end = sub_arr.len();

            (
                ParLaneChunksExactMut {
                    base: sub_arr_clone,
                    outer_shape: sub_shape.clone(),
                    inner_shape: shape,
                    bounds: (0, n_chunks),
                },
                ParLaneChunksExactMut {
                    base: sub_arr,
                    outer_shape: sub_shape,
                    inner_shape: shape,
                    bounds: (n_chunks * N, end),
                },
            )
        }
    }
}
