use std::{marker::PhantomData, ops::ControlFlow};

#[inline]
pub fn unravel_array<const N: usize>(flat_index: usize, shape: &[usize; N]) -> [usize; N]{
    let mut inds = [0; N];

    let mut flat_index = flat_index;
    inds.iter_mut()
        .zip(shape.iter())
        .rev()
        .for_each(|(i_dir, n_dir )|{
            *i_dir = flat_index % n_dir;
            flat_index /= n_dir;
    });
    inds
}

#[inline]
pub fn unravel(flat_index: usize, shape: &[usize]) -> Vec<usize>{
    let mut inds = vec![0; shape.len()];

    let mut flat_index = flat_index;
    inds.iter_mut()
        .zip(shape.iter())
        .rev()
        .for_each(|(i_dir, n_dir )|{
            *i_dir = flat_index % n_dir;
            flat_index /= n_dir;
    });
    inds
}

fn stride_from_shape(shape: &[usize]) -> Vec<usize>{
    let mut stride = vec![1; shape.len()];
    for i in (0..shape.len()-1).rev(){
        stride[i] = stride[i + 1] * shape[i + 1];
    }
    stride
}

pub struct StridedSlice<'a, T>{
    base: *const T,
    length: usize,
    stride: usize,
    _member: PhantomData<&'a T>,
}

impl<'a, T> StridedSlice<'a, T>{
    #[inline]
    pub fn get(&self, index: usize) -> Option<&'a T>{
        if index >= self.length{ None}
        else{
            Some(
                unsafe{self.get_unchecked(index) }
            )
        }
    }

    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> &'a T{
        unsafe{
            &* self.base.add(index * self.stride)
        }
    }

    #[inline]
    pub fn len(&self) -> usize{
        self.length
    }

    #[inline]
    pub fn iter(&self) -> StridedIter<'a, T>{
        let end = unsafe{ self.base.add(self.stride * self.length)};
        StridedIter{
            start: self.base,
            end: end,
            stride: self.stride,
            _member: PhantomData
        }
    }
}

pub struct StridedIter<'a, T>{
    start: *const T,
    end: *const T,
    stride: usize,
    _member: PhantomData<&'a T>,
}

impl<'a, T> Iterator for StridedIter<'a, T>{
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item>{
        if self.start == self.end {None}
        else{
            let address = self.start;
            unsafe{
                self.start = self.start.add(self.stride);
                Some(&*address)
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = unsafe{self.end.offset_from_unsigned(self.start)};
        (len, Some(len))
    }
}

impl<'a, T> ExactSizeIterator for StridedIter<'a, T>{}
impl<'a, T> DoubleEndedIterator for StridedIter<'a, T>{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item>{
        if self.start == self.end {None}
        else{
            unsafe{
                self.end = self.end.sub(self.stride);
                Some(&*self.end)
            }
        }

    }
}
pub struct ChunkStridedSlice<'a, T, const N: usize>{
    base: *const T,
    offsets: [usize; N],
    length: usize,
    stride: usize,
    _member: PhantomData<&'a T>,
}
impl<'a, T, const N: usize> ChunkStridedSlice<'a, T, N>{
    #[inline]
    pub fn len(&self) -> usize{
        self.length
    }

    #[inline]
    pub fn iter(&'a self) -> ChunkStridedSliceIter<'a, T, N>{
        let end_offset = self.length * self.stride;
        let end = unsafe{self.base.add(end_offset)};
        ChunkStridedSliceIter {
            start: self.base,
            end: end,
            offsets: &self.offsets,
            stride:self.stride,
            _member: PhantomData
        }
    }

    #[inline]
    pub fn get(&self, index:(usize, usize)) -> Option<&T>{
        assert!(index.0 < self.length);
        assert!(index.1 < N);
        Some(unsafe{self.get_unchecked(index)})
    }

    #[inline]
    pub unsafe fn get_unchecked(&self, index:(usize, usize)) -> &T{
        unsafe{
            let slice_target = self.base.add(
                *self.offsets.get_unchecked(index.1)
            );
            let target = slice_target.add(self.stride * index.0);
            & *target
        }
    }
}

pub struct AlongChunkIter<'a, T, const N: usize>{
    pointer: *const T,
    offsets: &'a [usize; N],
    ind: usize,
    _member: PhantomData<&'a T>
}

impl<'a, T, const N: usize> Iterator for AlongChunkIter<'a, T, N>{
    type Item = &'a T;
    #[inline]
    fn next(&mut self) -> Option<Self::Item>{
        if self.ind == N {None}
        else{
            let offset = unsafe{*self.offsets.get_unchecked(self.ind)};
            let target = unsafe{ & *self.pointer.add(offset)};
            self.ind += 1;
            Some(target)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = N - self.ind;
        (len, Some(len))
    }
}
impl<'a, T, const N: usize> ExactSizeIterator for AlongChunkIter<'a, T, N>{}

pub struct ChunkStridedSliceIter<'a, T, const N: usize>{
    start: *const T,
    end: *const T,
    offsets: &'a [usize; N],
    stride: usize,
    _member: PhantomData<&'a T>,
}

impl<'a, T, const N: usize> Iterator for ChunkStridedSliceIter<'a, T, N>{
    type Item = AlongChunkIter<'a, T, N>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item>{
        if self.start == self.end {None}
        else{
            let items = AlongChunkIter{
                pointer: self.start,
                offsets: self.offsets,
                ind: 0,
                _member: PhantomData
            };
            self.start = unsafe{self.start.add(self.stride)};
            Some(items)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = unsafe{self.end.offset_from_unsigned(self.start)} / self.stride;
        (len, Some(len))
    }
}
impl<'a, T, const N: usize> ExactSizeIterator for ChunkStridedSliceIter<'a, T, N>{}
impl<'a, T, const N: usize> DoubleEndedIterator for ChunkStridedSliceIter<'a, T, N>{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start == self.end {None}
        else{
            self.end = unsafe{self.end.sub(self.stride)};
            let items = AlongChunkIter{
                pointer: self.end,
                offsets: self.offsets,
                ind: 0,
                _member: PhantomData
            };
            Some(items)
        }
    }
}


#[derive(Clone)]
struct ArrayInfo{
    shape: Vec<usize>,
    stride: Vec<usize>,
    lane_length: usize,
    lane_stride: usize,
}

pub struct LaneSliceIter<'a, T>{
    base: *const T,
    arr_info: ArrayInfo,
    front_pos: Vec<usize>,
    front_offset: usize,
    rear_offset: usize,
    rear_pos: Vec<usize>,
    remaining: usize,
    _member: PhantomData<&'a T>
}

impl<'a, T> LaneSliceIter<'a, T>{

    pub fn from_slice(arr: &'a [T], shape: &[usize], axis: usize) -> Self{
        assert_eq!(arr.len(), shape.iter().product());
        assert!(axis < shape.len());

        let mut stride = stride_from_shape(shape);
        let mut shape = shape.to_owned();

        let lane_length = shape.remove(axis);
        let lane_stride = stride.remove(axis);
        let n_lanes = shape.iter().product();

        let front_pos = vec![0; shape.len()];

        let rear_pos: Vec<_> = shape.iter().map(|i| i - 1).collect();
        let rear_offset = rear_pos.iter().zip(stride.iter()).map(|(v1, v2)| v1 * v2).sum();

        LaneSliceIter {
            base: arr.as_ptr(),
            arr_info: ArrayInfo { shape, stride, lane_length, lane_stride},
            front_offset: 0,
            front_pos,
            rear_offset,
            rear_pos,
            remaining: n_lanes,
            _member: PhantomData
        }
    }
}

impl<'a, T> Iterator for LaneSliceIter<'a, T>{
    type Item = StridedSlice<'a, T>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item>{
        if self.remaining == 0 {None}
        else{
            self.remaining -= 1;
            let ArrayInfo{shape, stride, lane_length, lane_stride} = &self.arr_info;
            
            let slice = 
                StridedSlice{
                    base: unsafe{self.base.add(self.front_offset)},
                    length: *lane_length,
                    stride: *lane_stride,
                    _member: PhantomData
            };

            let _ = stride.iter()
                .zip(shape.iter())
                .zip(self.front_pos.iter_mut())
                .rev().try_for_each(|((str, shp), pos)|{
                    self.front_offset += *str;
                    *pos += 1;
                    if *pos < *shp  { return ControlFlow::Break(())};
                    *pos = 0;
                    self.front_offset -= shp * str;
                    ControlFlow::Continue(())
                });
            Some(slice)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}
impl<'a, T> ExactSizeIterator for LaneSliceIter<'a, T>{}
impl<'a, T> DoubleEndedIterator for LaneSliceIter<'a, T>{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {None}
        else{
            self.remaining -= 1;
            let ArrayInfo{shape, stride, lane_length, lane_stride} = &self.arr_info;

            let slice = StridedSlice{
                    base: unsafe{self.base.add(self.rear_offset)},
                    length: *lane_length,
                    stride: *lane_stride,
                    _member: PhantomData
            };

            let _ = stride.iter()
                .zip(shape.iter())
                .zip(self.rear_pos.iter_mut())
                .rev().try_for_each(|((str, shp), pos)|{
                    if *pos == 0{
                        *pos = *shp - 1;
                        self.rear_offset += *pos * str;
                        return ControlFlow::Continue(())
                    }else{
                        *pos -= 1;
                        self.rear_offset -= *str;
                        return ControlFlow::Break(())
                    }
                });
            Some(slice)
        }
    }
}

pub struct LaneSliceChunkIter<'a, T, const N: usize>{
    base: *const T,
    arr_info: ArrayInfo,
    front_pos: Vec<usize>,
    front_offset: usize,
    rear_pos: Vec<usize>,
    rear_offset: usize,
    remaining: usize,
    _member: PhantomData<&'a T>
}

impl<'a, T, const N:usize> LaneSliceChunkIter<'a, T, N>{

    pub fn from_slice(arr: &'a [T], shape: &[usize], axis: usize) -> (Self, LaneSliceIter<'a, T>){
        assert_eq!(arr.len(), shape.iter().product());
        assert!(axis < shape.len());

        let mut stride = stride_from_shape(shape);
        let mut shape = shape.to_owned();
        let lane_length = shape.remove(axis);
        let lane_stride = stride.remove(axis);
        let n_lanes: usize = shape.iter().product();

        let n_remainder = n_lanes % N;
        let n_chunkable = n_lanes - n_remainder;

        let front_pos = vec![0; shape.len()];
        let front_offset = 0;

        let rear_pos = unravel(n_chunkable - 1, &shape);
        let rear_offset = rear_pos.iter().zip(stride.iter()).map(|(v1, v2)| v1 * v2).sum();

        let pos_rem = unravel(n_chunkable, &shape);
        let offset_rem = pos_rem.iter().zip(stride.iter()).map(|(v1, v2)| v1 * v2).sum();

        let rear_rem_pos: Vec<_> = shape.iter().map(|i| i - 1).collect();
        let rear_rem_offset = rear_rem_pos.iter().zip(stride.iter()).map(|(v1, v2)| v1 * v2).sum();

        let arr_info = ArrayInfo{
            shape: shape.clone(),
            stride: stride.clone(),
            lane_length,
            lane_stride,
        };

        (
            Self {
                base: arr.as_ptr(),
                arr_info: arr_info.clone(),
                front_pos,
                front_offset,
                rear_pos,
                rear_offset,
                remaining: n_chunkable,
                _member: PhantomData
            },
            LaneSliceIter{
                base: arr.as_ptr(),
                arr_info: arr_info,
                front_pos: pos_rem,
                front_offset: offset_rem,
                rear_pos: rear_rem_pos,
                rear_offset: rear_rem_offset,
                remaining: n_remainder,
                _member: PhantomData
            }
        )
    }

}

impl<'a, T, const N:usize> Iterator for LaneSliceChunkIter<'a, T, N>{
    type Item = ChunkStridedSlice<'a, T, N>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item>{
        if self.remaining == 0 {None}
        else{
            self.remaining -= N;
            let ArrayInfo{shape, stride, lane_length, lane_stride} = &self.arr_info;

            let offsets = std::array::from_fn(|_| {
                let off = self.front_offset;
                let _ = stride.iter()
                    .zip(shape.iter())
                    .zip(self.front_pos.iter_mut())
                    .rev().try_for_each(|((str, shp), pos)|{
                        self.front_offset += *str;
                        *pos += 1;
                        if *pos < *shp  { return ControlFlow::Break(())};
                        *pos = 0;
                        self.front_offset -= shp * str;
                        ControlFlow::Continue(())
                    });
                off
            });

            Some(ChunkStridedSlice{
                    base: self.base,
                    offsets: offsets,
                    length: *lane_length,
                    stride: *lane_stride,
                    _member: PhantomData,
            })
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.remaining/N;
        (len, Some(len))
    }
}

impl<'a, T, const N:usize> ExactSizeIterator for LaneSliceChunkIter<'a, T, N>{}
impl<'a, T, const N:usize> DoubleEndedIterator for LaneSliceChunkIter<'a, T, N>{
    
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item>{
        if self.remaining == 0 {None}
        else{
            self.remaining -= N;
            let ArrayInfo{shape, stride, lane_length, lane_stride} = &self.arr_info;

            let offsets = std::array::from_fn(|_| {
                let off = self.rear_offset;
                let _ = stride.iter()
                .zip(shape.iter())
                .zip(self.rear_pos.iter_mut())
                .rev().try_for_each(|((str, shp), pos)|{
                    if *pos == 0{
                        *pos = *shp - 1;
                        self.rear_offset += *pos * str;
                        return ControlFlow::Continue(())
                    }else{
                        *pos -= 1;
                        self.rear_offset -= *str;
                        return ControlFlow::Break(())
                    }
                });
                off
            });

            Some(ChunkStridedSlice{
                    base: self.base,
                    offsets: offsets,
                    length: *lane_length,
                    stride: *lane_stride,
                    _member: PhantomData,
            })
        }
    }
}
//  Mutable strided slices and chunks of strided slices

pub struct MutStridedSlice<'a, T>{
    base: *mut T,
    length: usize,
    stride: usize,
    _member: PhantomData<&'a T>,
}

impl<'a, T> MutStridedSlice<'a, T>{
    #[inline]
    pub fn get(&self, index: usize) -> Option<&'a T>{
        if index >= self.length{ None}
        else{
            Some(
                unsafe{self.get_unchecked(index) }
            )
        }
    }

    #[inline]
    pub unsafe fn get_unchecked(&self, index: usize) -> &'a T{
        unsafe{
            &* self.base.add(index * self.stride)
        }
    }

    #[inline]
    pub fn get_mut(&mut self, index: usize) -> Option<&'a mut T>{
        if index >= self.length{ None}
        else{
            Some(
                unsafe{self.get_unchecked_mut(index) }
            )
        }
    }

    #[inline]
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &'a mut T{
        unsafe{
            &mut * self.base.add(index * self.stride)
        }
    }

    #[inline]
    pub fn len(&self) -> usize{
        self.length
    }

    #[inline]
    pub fn iter(&self) -> StridedIter<'a, T>{
        let end = unsafe{ self.base.add(self.stride * self.length)};
        StridedIter{
            start: self.base,
            end: end,
            stride: self.stride,
            _member: PhantomData
        }
    }

    #[inline]
    pub fn iter_mut(&mut self) -> MutStridedIter<'a, T>{
        let end = unsafe{ self.base.add(self.stride * self.length)};
        MutStridedIter {
            start: self.base as *mut T,
            end: end as *mut T,
            stride: self.stride,
            _member: PhantomData
        }
    }
}

pub struct MutStridedIter<'a, T>{
    start: *mut T,
    end: *mut T,
    stride: usize,
    _member: PhantomData<&'a T>,
}

impl<'a, T> Iterator for MutStridedIter<'a, T>{
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item>{
        if self.start == self.end {None}
        else{
            let address = self.start;
            unsafe{
                self.start = self.start.add(self.stride);
                Some(&mut *address)
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = unsafe{self.end.offset_from_unsigned(self.start)};
        (len, Some(len))
    }
}

impl<'a, T> ExactSizeIterator for MutStridedIter<'a, T>{}
impl<'a, T> DoubleEndedIterator for MutStridedIter<'a, T>{

    #[inline]
    fn next_back(&mut self) -> Option<Self::Item>{
        if self.start == self.end {None}
        else{
            unsafe{
                self.end = self.end.sub(self.stride);
                Some(&mut *self.end)
            }
        }

    }
}

pub struct MutChunkStridedSlice<'a, T, const N: usize>{
    base: *mut T,
    offsets: [usize; N],
    length: usize,
    stride: usize,
    _member: PhantomData<&'a T>,
}
impl<'a, T, const N: usize> MutChunkStridedSlice<'a, T, N>{
    #[inline]
    pub fn len(&self) -> usize{
        self.length
    }

    #[inline]
    pub fn iter(&'a self) -> ChunkStridedSliceIter<'a, T, N>{
        let end_offset = self.length * self.stride;
        let end = unsafe{self.base.add(end_offset)};
        ChunkStridedSliceIter {
            start: self.base,
            end: end,
            offsets: &self.offsets,
            stride:self.stride,
            _member: PhantomData
        }
    }

    #[inline]
    pub fn iter_mut(&'a mut self) -> MutChunkStridedSliceIter<'a, T, N>{
        let end_offset = self.length * self.stride;
        let end = unsafe{self.base.add(end_offset)};
        MutChunkStridedSliceIter {
            start: self.base,
            end: end,
            offsets: &self.offsets,
            stride:self.stride,
            _member: PhantomData
        }
    }

    #[inline]
    pub fn get(&self, index:(usize, usize)) -> Option<&T>{
        assert!(index.0 < self.length);
        assert!(index.1 < N);
        Some(unsafe{self.get_unchecked(index)})
    }

    #[inline]
    pub unsafe fn get_unchecked(&self, index:(usize, usize)) -> &T{
        unsafe{
            let slice_target = self.base.add(
                *self.offsets.get_unchecked(index.1)
            );
            let target = slice_target.add(self.stride * index.0);
            & *target
        }
    }
}

pub struct MutAlongChunkIter<'a, T, const N: usize>{
    pointer: *mut T,
    offsets: &'a [usize; N],
    ind: usize,
    _member: PhantomData<&'a T>
}

impl<'a, T, const N: usize> Iterator for MutAlongChunkIter<'a, T, N>{
    type Item = &'a mut T;
    
    #[inline]
    fn next(&mut self) -> Option<Self::Item>{
        if self.ind == N {None}
        else{
            let target = unsafe{ &mut *self.pointer.add(*self.offsets.get_unchecked(self.ind))};
            self.ind += 1;
            Some(target)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = N - self.ind;
        (len, Some(len))
    }
}
impl<'a, T, const N: usize> ExactSizeIterator for MutAlongChunkIter<'a, T, N>{}

pub struct MutChunkStridedSliceIter<'a, T, const N: usize>{
    start: *mut T,
    end: *mut T,
    offsets: &'a [usize; N],
    stride: usize,
    _member: PhantomData<&'a T>,
}

impl<'a, T, const N: usize> Iterator for MutChunkStridedSliceIter<'a, T, N>{
    type Item = MutAlongChunkIter<'a, T, N>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item>{
        if self.start == self.end {None}
        else{
            let items = MutAlongChunkIter{
                pointer: self.start,
                offsets: self.offsets,
                ind: 0,
                _member: PhantomData
            };
            self.start = unsafe{self.start.add(self.stride)};
            Some(items)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = unsafe{self.end.offset_from_unsigned(self.start)} / self.stride;
        (len, Some(len))
    }
}
impl<'a, T, const N: usize> ExactSizeIterator for MutChunkStridedSliceIter<'a, T, N>{}
impl<'a, T, const N: usize> DoubleEndedIterator for MutChunkStridedSliceIter<'a, T, N>{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start == self.end {None}
        else{
            self.end = unsafe{self.end.sub(self.stride)};
            let items = MutAlongChunkIter{
                pointer: self.end,
                offsets: self.offsets,
                ind: 0,
                _member: PhantomData
            };
            Some(items)
        }
    }
}

pub struct MutLaneSliceIter<'a, T>{
    base: *mut T,
    arr_info: ArrayInfo,
    front_pos: Vec<usize>,
    front_offset: usize,
    rear_offset: usize,
    rear_pos: Vec<usize>,
    remaining: usize,
    _member: PhantomData<&'a T>
}

impl<'a, T> MutLaneSliceIter<'a, T>{

    pub fn from_slice_mut(arr: &'a mut [T], shape: &[usize], axis: usize) -> Self{
        assert_eq!(arr.len(), shape.iter().product());
        assert!(axis < shape.len());

        let mut stride = stride_from_shape(shape);
        let mut shape = shape.to_owned();
        let lane_length = shape.remove(axis);
        let lane_stride = stride.remove(axis);
        let n_lanes = shape.iter().product();

        let front_pos = vec![0; shape.len()];

        let rear_pos: Vec<_> = shape.iter().map(|i| i - 1).collect();
        let rear_offset = rear_pos.iter().zip(stride.iter()).map(|(v1, v2)| v1 * v2).sum();

        MutLaneSliceIter {
            base: arr.as_mut_ptr(),
            arr_info: ArrayInfo { shape, stride, lane_length, lane_stride},
            front_offset: 0,
            front_pos,
            rear_offset,
            rear_pos,
            remaining: n_lanes,
            _member: PhantomData
        }
    }
}

impl<'a, T> Iterator for MutLaneSliceIter<'a, T>{
    type Item = MutStridedSlice<'a, T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item>{
        if self.remaining == 0 {return None}
        else{
            self.remaining -= 1;
            let ArrayInfo{shape, stride, lane_length, lane_stride} = &self.arr_info;

            let slice = MutStridedSlice{
                    base: unsafe{self.base.add(self.front_offset)},
                    length: *lane_length,
                    stride: *lane_stride,
                    _member: PhantomData
            };

            let _ = stride.iter()
                .zip(shape.iter())
                .zip(self.front_pos.iter_mut())
                .rev().try_for_each(|((str, shp), pos)|{
                    self.front_offset += *str;
                    *pos += 1;
                    if *pos < *shp  { return ControlFlow::Break(())};
                    *pos = 0;
                    self.front_offset -= shp * str;
                    ControlFlow::Continue(())
                });
            Some(slice)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}
impl<'a, T> ExactSizeIterator for MutLaneSliceIter<'a, T>{}
impl<'a, T> DoubleEndedIterator for MutLaneSliceIter<'a, T>{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {None}
        else{
            self.remaining -= 1;
            let ArrayInfo{shape, stride, lane_length, lane_stride} = &self.arr_info;

            let slice = MutStridedSlice{
                    base: unsafe{self.base.add(self.rear_offset)},
                    length: *lane_length,
                    stride: *lane_stride,
                    _member: PhantomData
            };

            let _ = stride.iter()
                .zip(shape.iter())
                .zip(self.rear_pos.iter_mut())
                .rev().try_for_each(|((str, shp), pos)|{
                    if *pos == 0{
                        *pos = *shp - 1;
                        self.rear_offset += *pos * str;
                        return ControlFlow::Continue(())
                    }else{
                        *pos -= 1;
                        self.rear_offset -= *str;
                        return ControlFlow::Break(())
                    }
                });
            Some(slice)
        }
    }
}

pub struct MutLaneSliceChunkIter<'a, T, const N: usize>{
    base: *mut T,
    arr_info: ArrayInfo,
    front_pos: Vec<usize>,
    front_offset: usize,
    rear_pos: Vec<usize>,
    rear_offset: usize,
    remaining: usize,
    _member: PhantomData<&'a T>
}

impl<'a, T, const N:usize> MutLaneSliceChunkIter<'a, T, N>{

    pub fn from_slice_mut(arr: &'a mut [T], shape: &[usize], axis: usize) -> (Self, MutLaneSliceIter<'a, T>){
        assert_eq!(arr.len(), shape.iter().product());
        assert!(axis < shape.len());

        let mut stride = stride_from_shape(shape);

        let mut shape = shape.to_owned();
        let lane_length = shape.remove(axis);
        let lane_stride = stride.remove(axis);
        let n_lanes: usize = shape.iter().product();

        let n_remainder = n_lanes % N;
        let n_chunkable = n_lanes - n_remainder;

        let front_pos = vec![0; shape.len()];
        let front_offset = 0;

        let rear_pos = unravel(n_chunkable - 1, &shape);
        let rear_offset = rear_pos.iter().zip(stride.iter()).map(|(v1, v2)| v1 * v2).sum();

        let pos_rem = unravel(n_chunkable, &shape);
        let offset_rem = pos_rem.iter().zip(stride.iter()).map(|(v1, v2)| v1 * v2).sum();

        let rear_rem_pos: Vec<_> = shape.iter().map(|i| i - 1).collect();
        let rear_rem_offset = rear_rem_pos.iter().zip(stride.iter()).map(|(v1, v2)| v1 * v2).sum();

        let arr_info = ArrayInfo{
            shape: shape.clone(), stride: stride.clone(), lane_length, lane_stride
        };

        (
            Self {
                base: arr.as_mut_ptr(),
                arr_info: arr_info.clone(),
                front_pos,
                front_offset,
                rear_pos,
                rear_offset,
                remaining: n_chunkable,
                _member: PhantomData
            },
            MutLaneSliceIter{
                base: arr.as_mut_ptr(),
                arr_info,
                front_pos: pos_rem,
                front_offset: offset_rem,
                rear_pos: rear_rem_pos,
                rear_offset: rear_rem_offset,
                remaining: n_remainder,
                _member: PhantomData
            }
        )
    }
}
impl<'a, T, const N:usize> Iterator for MutLaneSliceChunkIter<'a, T, N>{
    type Item = MutChunkStridedSlice<'a, T, N>;
    #[inline]
    fn next(&mut self) -> Option<Self::Item>{
        if self.remaining == 0 {None}
        else{
            self.remaining -= N;
            let ArrayInfo{shape, stride, lane_length, lane_stride} = &self.arr_info;

            let offsets = std::array::from_fn(|_| {
                let off = self.front_offset;
                let _ = stride.iter()
                    .zip(shape.iter())
                    .zip(self.front_pos.iter_mut())
                    .rev().try_for_each(|((str, shp), pos)|{
                        self.front_offset += *str;
                        *pos += 1;
                        if *pos < *shp  { return ControlFlow::Break(())};
                        *pos = 0;
                        self.front_offset -= shp * str;
                        ControlFlow::Continue(())
                    });
                off
            });

            Some(Self::Item{
                    base: self.base,
                    offsets: offsets,
                    length: *lane_length,
                    stride: *lane_stride,
                    _member: PhantomData,
            })
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.remaining/N;
        (len, Some(len))
    }
}
impl<'a, T, const N:usize> ExactSizeIterator for MutLaneSliceChunkIter<'a, T, N>{}
impl<'a, T, const N:usize> DoubleEndedIterator for MutLaneSliceChunkIter<'a, T, N>{
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item>{
        if self.remaining == 0 {None}
        else{
            self.remaining -= N;
            let ArrayInfo{shape, stride, lane_length, lane_stride} = &self.arr_info;

            let offsets = std::array::from_fn(|_| {
                let off = self.rear_offset;
                let _ = stride.iter()
                .zip(shape.iter())
                .zip(self.rear_pos.iter_mut())
                .rev().try_for_each(|((str, shp), pos)|{
                    if *pos == 0{
                        *pos = *shp - 1;
                        self.rear_offset += *pos * str;
                        return ControlFlow::Continue(())
                    }else{
                        *pos -= 1;
                        self.rear_offset -= *str;
                        return ControlFlow::Break(())
                    }
                });
                off
            });

            Some(Self::Item{
                    base: self.base,
                    offsets: offsets,
                    length: *lane_length,
                    stride: *lane_stride,
                    _member: PhantomData,
            })
        }
    }
}

pub trait LanesIterator<T>{
    fn iter_lanes<'a> (&'a self, shape: &[usize], axis: usize) -> LaneSliceIter<'a, T>;
    fn iter_lanes_mut<'a>(&'a mut self, shape: &[usize], aixs:usize) -> MutLaneSliceIter<'a, T>;
    fn iter_lane_chunks<'a, const N: usize> (&'a self, shape: &[usize], axis: usize) -> (
        LaneSliceChunkIter<'a, T, N>,
        LaneSliceIter<'a, T>,
    );
    fn iter_lane_chunks_mut<'a, const N: usize>(&'a mut self, shape: &[usize], aixs:usize) -> (
        MutLaneSliceChunkIter<'a, T, N>,
        MutLaneSliceIter<'a, T>
    );
}
impl<T> LanesIterator<T> for [T]{
    fn iter_lanes<'a> (&'a self, shape: &[usize], axis: usize) -> LaneSliceIter<'a, T> {
        LaneSliceIter::from_slice(self, shape, axis)
    }
    fn iter_lanes_mut<'a>(&'a mut self, shape: &[usize], axis: usize) -> MutLaneSliceIter<'a, T>{
        MutLaneSliceIter::from_slice_mut(self, shape, axis)
    }

    fn iter_lane_chunks<'a, const N:usize>(&'a self, shape: &[usize], axis: usize) -> (
        LaneSliceChunkIter<'a, T, N>,
        LaneSliceIter<'a, T>,
    ){
        LaneSliceChunkIter::from_slice(self, shape, axis)
    }

    fn iter_lane_chunks_mut<'a, const N:usize>(&'a mut self, shape: &[usize], axis: usize) -> (
        MutLaneSliceChunkIter<'a, T, N>,
        MutLaneSliceIter<'a, T>,
    ){
        MutLaneSliceChunkIter::from_slice_mut(self, shape, axis)
    }
}

pub mod parallel{

    pub use rayon::iter::{ParallelIterator, IndexedParallelIterator};
    use rayon::iter::plumbing::{UnindexedConsumer, Consumer, bridge, ProducerCallback, Producer};

    use super::*;

    unsafe impl<'a, T> Send for StridedSlice<'a, T>{}

    pub struct LaneSliceParIter<'a, T>{
        base: *const T,
        arr_info: ArrayInfo,
        start: usize,
        stop: usize,
        _member: PhantomData<&'a T>
    }
    unsafe impl<'a, T> Send for LaneSliceParIter<'a, T>{}

    impl<'a, T> LaneSliceParIter<'a, T>{
        pub fn from_slice(arr: &'a [T], shape: &[usize], axis: usize) -> Self{
            assert_eq!(arr.len(), shape.iter().product());
            assert!(axis < shape.len());

            let mut stride = stride_from_shape(shape);
            let mut shape = shape.to_owned();
            let lane_stride = stride.remove(axis);
            let lane_length = shape.remove(axis);

            let n_lanes = shape.iter().product();

            Self {
                base: arr.as_ptr(),
                arr_info: ArrayInfo { shape, stride, lane_length, lane_stride },
                start: 0,
                stop: n_lanes,
                _member: PhantomData
            }
        }
    }

    impl<'a, T> ParallelIterator for LaneSliceParIter<'a, T>{
        type Item = StridedSlice<'a, T>;
        fn drive_unindexed<C>(self, consumer: C) -> C::Result
        where
            C: UnindexedConsumer<Self::Item>
        {
            bridge(self, consumer)
        }
    }

    impl<'a, T> IndexedParallelIterator for LaneSliceParIter<'a, T>{
        fn drive<C>(self, consumer: C) -> C::Result
        where
            C: Consumer<Self::Item>,
        {
            bridge(self, consumer)
        }

        fn len(&self) -> usize{
            self.stop - self.start
        }

        fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
            callback.callback(
                LaneSliceParIter{..self}
            )
        }
    }

    impl<'a, T> Producer for LaneSliceParIter<'a, T>{
        type Item = StridedSlice<'a, T>;
        type IntoIter = LaneSliceIter<'a, T>;

        fn into_iter(self) -> Self::IntoIter{
            let arr_info = self.arr_info.clone();
            let front_pos = unravel(self.start, &arr_info.shape);
            let front_offset = front_pos.iter().zip(arr_info.stride.iter()).map(|(v1, v2)| v1 * v2).sum();
            let rear_pos = unravel(self.stop - 1, &arr_info.shape);
            let rear_offset = rear_pos.iter().zip(arr_info.stride.iter()).map(|(v1, v2)| v1 * v2).sum();

            LaneSliceIter{
                base: self.base,
                arr_info,
                front_pos,
                front_offset,
                rear_pos,
                rear_offset,
                remaining: self.stop - self.start,
                _member: PhantomData
            }

        }

        fn split_at(self, index: usize) -> (Self, Self) {
            let split = self.start + index;
            (
                Self{
                    start: self.start,
                    stop: split,
                    arr_info: self.arr_info.clone(),
                    ..self
                },
                Self{
                    start: split,
                    stop: self.stop,
                    ..self
                }
            )
        }
    }

    unsafe impl<'a, T, const N: usize> Send for ChunkStridedSlice<'a, T, N>{}

    pub struct LaneSliceChunkParIter<'a, T, const N: usize>{
        base: *const T,
        arr_info: ArrayInfo,
        start: usize,
        stop: usize,
        _member: PhantomData<&'a T>
    }
    unsafe impl<'a, T, const N: usize> Send for LaneSliceChunkParIter<'a, T, N>{}

    impl<'a, T, const N: usize> LaneSliceChunkParIter<'a, T, N>{

        pub fn from_slice(arr: &'a [T], shape: &[usize], axis: usize) -> (Self, LaneSliceParIter<'a, T>){
            assert_eq!(arr.len(), shape.iter().product());
            assert!(axis < shape.len());

            let mut stride = stride_from_shape(shape);
            let mut shape = shape.to_owned();
            let lane_stride = stride.remove(axis);
            let lane_length = shape.remove(axis);

            let n_lanes = shape.iter().product();
            let n_rem = n_lanes % N;
            let n_chunkable = n_lanes - n_rem;
            let arr_info = ArrayInfo { shape, stride, lane_length, lane_stride };

            (
                Self {
                    base: arr.as_ptr(),
                    arr_info: arr_info.clone(),
                    start: 0,
                    stop: n_chunkable,
                    _member: PhantomData
                },
                LaneSliceParIter{
                    base: arr.as_ptr(),
                    arr_info: arr_info.clone(),
                    start: n_chunkable,
                    stop: n_lanes,
                    _member: PhantomData
                }
            )
        }
    }

    impl<'a, T, const N: usize> ParallelIterator for LaneSliceChunkParIter<'a, T, N>{
        type Item = ChunkStridedSlice<'a, T, N>;
        fn drive_unindexed<C>(self, consumer: C) -> C::Result
        where
            C: UnindexedConsumer<Self::Item>
        {
            bridge(self, consumer)
        }
    }

    impl<'a, T, const N: usize> IndexedParallelIterator for LaneSliceChunkParIter<'a, T, N>{
        fn drive<C>(self, consumer: C) -> C::Result
        where
            C: Consumer<Self::Item>,
        {
            bridge(self, consumer)
        }

        fn len(&self) -> usize{
            (self.stop - self.start) / N
        }

        fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
            callback.callback(
                LaneSliceChunkParIter{..self}
            )
        }
    }

    impl<'a, T, const N: usize> Producer for LaneSliceChunkParIter<'a, T, N>{
        type Item = ChunkStridedSlice<'a, T, N>;
        type IntoIter = LaneSliceChunkIter<'a, T, N>;

        fn into_iter(self) -> Self::IntoIter{
            let arr_info = self.arr_info.clone();
            let front_pos = unravel(self.start, &arr_info.shape);
            let front_offset = front_pos.iter().zip(arr_info.stride.iter()).map(|(v1, v2)| v1 * v2).sum();
            let rear_pos = unravel(self.stop - 1, &arr_info.shape);
            let rear_offset = rear_pos.iter().zip(arr_info.stride.iter()).map(|(v1, v2)| v1 * v2).sum();

            Self::IntoIter{
                base: self.base,
                arr_info,
                front_pos,
                front_offset,
                rear_pos,
                rear_offset,
                remaining: self.stop - self.start,
                _member: PhantomData
            }
        }

        fn split_at(self, index: usize) -> (Self, Self) {
            let split = self.start + index * N;
            (
                Self{
                    start: self.start,
                    stop: split,
                    arr_info: self.arr_info.clone(),
                    ..self
                },
                Self{
                    start: split,
                    stop: self.stop,
                    ..self
                }
            )
        }
    }


    // Mutable versions

    unsafe impl<'a, T> Send for MutStridedSlice<'a, T>{}
    pub struct MutLaneSliceParIter<'a, T>{
        base: *mut T,
        arr_info: ArrayInfo,
        start: usize,
        stop: usize,
        _member: PhantomData<&'a T>
    }
    unsafe impl<'a, T> Send for MutLaneSliceParIter<'a, T>{}

    impl<'a, T> MutLaneSliceParIter<'a, T>{
        pub fn from_slice_mut(arr: &'a mut [T], shape: &[usize], axis: usize) -> Self{
            assert_eq!(arr.len(), shape.iter().product());
            assert!(axis < shape.len());

            let mut stride = stride_from_shape(shape);
            let mut shape = shape.to_owned();
            let lane_stride = stride.remove(axis);
            let lane_length = shape.remove(axis);

            let n_lanes = shape.iter().product();

            Self {
                base: arr.as_mut_ptr(),
                arr_info: ArrayInfo { shape, stride, lane_length, lane_stride },
                start: 0,
                stop: n_lanes,
                _member: PhantomData
            }
        }
    }

    impl<'a, T> ParallelIterator for MutLaneSliceParIter<'a, T>{
        type Item = MutStridedSlice<'a, T>;
        fn drive_unindexed<C>(self, consumer: C) -> C::Result
        where
            C: UnindexedConsumer<Self::Item>
        {
            bridge(self, consumer)
        }
    }

    impl<'a, T> IndexedParallelIterator for MutLaneSliceParIter<'a, T>{
        fn drive<C>(self, consumer: C) -> C::Result
        where
            C: Consumer<Self::Item>,
        {
            bridge(self, consumer)
        }

        fn len(&self) -> usize{
            self.stop - self.start
        }

        fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
            callback.callback(
                MutLaneSliceParIter{..self}
            )
        }
    }

    impl<'a, T> Producer for MutLaneSliceParIter<'a, T>{
        type Item = MutStridedSlice<'a, T>;
        type IntoIter = MutLaneSliceIter<'a, T>;

        fn into_iter(self) -> Self::IntoIter{
            let arr_info = self.arr_info.clone();
            let front_pos = unravel(self.start, &arr_info.shape);
            let front_offset = front_pos.iter().zip(arr_info.stride.iter()).map(|(v1, v2)| v1 * v2).sum();
            let rear_pos = unravel(self.stop - 1, &arr_info.shape);
            let rear_offset = rear_pos.iter().zip(arr_info.stride.iter()).map(|(v1, v2)| v1 * v2).sum();

            MutLaneSliceIter{
                base: self.base,
                arr_info,
                front_pos,
                front_offset,
                rear_pos,
                rear_offset,
                remaining: self.stop - self.start,
                _member: PhantomData
            }

        }

        fn split_at(self, index: usize) -> (Self, Self) {
            let split = self.start + index;
            (
                Self{
                    start: self.start,
                    stop: split,
                    arr_info: self.arr_info.clone(),
                    ..self
                },
                Self{
                    start: split,
                    stop: self.stop,
                    ..self
                }
            )
        }
    }


    unsafe impl<'a, T, const N: usize> Send for MutChunkStridedSlice<'a, T, N>{}

    pub struct MutLaneSliceChunkParIter<'a, T, const N: usize>{
        base: *mut T,
        arr_info: ArrayInfo,
        start: usize,
        stop: usize,
        _member: PhantomData<&'a T>
    }
    unsafe impl<'a, T, const N: usize> Send for MutLaneSliceChunkParIter<'a, T, N>{}

    impl<'a, T, const N: usize> MutLaneSliceChunkParIter<'a, T, N>{

        pub fn from_slice_mut(arr: &'a mut [T], shape: &[usize], axis: usize) -> (Self, MutLaneSliceParIter<'a, T>){
            assert_eq!(arr.len(), shape.iter().product());
            assert!(axis < shape.len());

            let mut stride = stride_from_shape(shape);
            let mut shape = shape.to_owned();
            let lane_stride = stride.remove(axis);
            let lane_length = shape.remove(axis);

            let n_lanes = shape.iter().product();
            let n_rem = n_lanes % N;
            let n_chunkable = n_lanes - n_rem;
            let arr_info = ArrayInfo { shape, stride, lane_length, lane_stride };

            (
                Self {
                    base: arr.as_mut_ptr(),
                    arr_info: arr_info.clone(),
                    start: 0,
                    stop: n_chunkable,
                    _member: PhantomData
                },
                MutLaneSliceParIter{
                    base: arr.as_mut_ptr(),
                    arr_info: arr_info.clone(),
                    start: n_chunkable,
                    stop: n_lanes,
                    _member: PhantomData
                }
            )
        }
    }

    impl<'a, T, const N: usize> ParallelIterator for MutLaneSliceChunkParIter<'a, T, N>{
        type Item = MutChunkStridedSlice<'a, T, N>;
        fn drive_unindexed<C>(self, consumer: C) -> C::Result
        where
            C: UnindexedConsumer<Self::Item>
        {
            bridge(self, consumer)
        }
    }

    impl<'a, T, const N: usize> IndexedParallelIterator for MutLaneSliceChunkParIter<'a, T, N>{
        fn drive<C>(self, consumer: C) -> C::Result
        where
            C: Consumer<Self::Item>,
        {
            bridge(self, consumer)
        }

        fn len(&self) -> usize{
            (self.stop - self.start) / N
        }

        fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
            callback.callback(
                Self{..self}
            )
        }
    }

    impl<'a, T, const N: usize> Producer for MutLaneSliceChunkParIter<'a, T, N>{
        type Item = MutChunkStridedSlice<'a, T, N>;
        type IntoIter = MutLaneSliceChunkIter<'a, T, N>;

        fn into_iter(self) -> Self::IntoIter{
            let arr_info = self.arr_info.clone();
            let front_pos = unravel(self.start, &arr_info.shape);
            let front_offset = front_pos.iter().zip(arr_info.stride.iter()).map(|(v1, v2)| v1 * v2).sum();
            let rear_pos = unravel(self.stop - 1, &arr_info.shape);
            let rear_offset = rear_pos.iter().zip(arr_info.stride.iter()).map(|(v1, v2)| v1 * v2).sum();

            Self::IntoIter{
                base: self.base,
                arr_info,
                front_pos,
                front_offset,
                rear_pos,
                rear_offset,
                remaining: self.stop - self.start,
                _member: PhantomData
            }
        }

        fn split_at(self, index: usize) -> (Self, Self) {
            let split = self.start + index * N;
            (
                Self{
                    start: self.start,
                    stop: split,
                    arr_info: self.arr_info.clone(),
                    ..self
                },
                Self{
                    start: split,
                    stop: self.stop,
                    ..self
                }
            )
        }
    }


    pub trait ParallelLanesIterator<T>{
        fn par_iter_lanes<'a> (&'a self, shape: &[usize], axis: usize) -> LaneSliceParIter<'a, T>;
        fn par_iter_lanes_mut<'a> (&'a mut self, shape: &[usize], axis: usize) -> MutLaneSliceParIter<'a, T>;

        fn par_iter_lane_chunks<'a, const N: usize> (&'a self, shape: &[usize], axis: usize) -> (
            LaneSliceChunkParIter<'a, T, N>,
            LaneSliceParIter<'a, T>,
        );
        fn par_iter_lane_chunks_mut<'a, const N: usize>(&'a mut self, shape: &[usize], axis:usize) -> (
            MutLaneSliceChunkParIter<'a, T, N>,
            MutLaneSliceParIter<'a, T>
        );
    }

    impl<T> ParallelLanesIterator<T> for [T]{
        fn par_iter_lanes<'a> (&'a self, shape: &[usize], axis: usize) -> LaneSliceParIter<'a, T>{
            LaneSliceParIter::from_slice(self, shape, axis)
        }
        fn par_iter_lanes_mut<'a> (&'a mut self, shape: &[usize], axis: usize) -> MutLaneSliceParIter<'a, T>{
            MutLaneSliceParIter::from_slice_mut(self, shape, axis)
        }
        fn par_iter_lane_chunks<'a, const N: usize> (&'a self, shape: &[usize], axis: usize) -> (
            LaneSliceChunkParIter<'a, T, N>,
            LaneSliceParIter<'a, T>,
        ){
            LaneSliceChunkParIter::from_slice(self, shape, axis)
        }
        fn par_iter_lane_chunks_mut<'a, const N: usize>(&'a mut self, shape: &[usize], axis:usize) -> (
            MutLaneSliceChunkParIter<'a, T, N>,
            MutLaneSliceParIter<'a, T>
        ){
            MutLaneSliceChunkParIter::from_slice_mut(self, shape, axis)
        }
    }
}

#[cfg(test)]
mod tests{
    use super::*;

    #[test]
    fn test_strided(){
        let arr = [1, 2, 3, 4, 5, 6];
        let strided = StridedSlice{
            base: arr.as_ptr(), stride:2, length: arr.len() / 2, _member:PhantomData
        };

        let collected = strided.iter().map(|ind| *ind).collect::<Vec<_>>();
        assert_eq!(collected, vec![1, 3, 5]);

        let strided = StridedSlice{
            base: arr[1..].as_ptr(), stride:2, length: arr.len()/2, _member:PhantomData
        };

        let collected = strided.iter().map(|ind| *ind).collect::<Vec<_>>();
        assert_eq!(collected, vec![2, 4, 6]);

    }

    #[test]
    fn test_strided_lane_iter_2d(){
        let arr = [1, 2, 3, 4, 5, 6];
        let shape = [3, 2];
        arr.iter_lanes(&shape, 0).enumerate().for_each( |(i_lane, lane)| {
            assert_eq!(lane.len(), shape[0]);
            let collected = lane.iter().map(|ind| *ind).collect::<Vec<_>>();
            if i_lane == 0{
                assert_eq!(collected, [1, 3, 5]);
            }else if i_lane == 1{
                assert_eq!(collected, [2, 4, 6]);
            }else{
                panic!("should only be two lanes along axis 0")
            }
        });


        arr.iter_lanes(&shape, 1).enumerate().for_each( |(i_lane, lane)| {
            assert_eq!(lane.len(), shape[1]);

            let collected = lane.iter().map(|ind| *ind).collect::<Vec<_>>();
            if i_lane == 0{
                assert_eq!(collected, [1, 2]);
            }else if i_lane == 1{
                assert_eq!(collected, [3, 4]);
            }else if i_lane == 2{
                assert_eq!(collected, [5, 6]);
            }else{
                panic!("should only be three lanes along axis 0")
            }
        });
    }


    #[test]
    fn test_strided_lane_iter_3d(){
        let shape = [3, 4, 5];
        let n_t = shape.iter().product();
        let arr = (0..n_t).collect::<Vec<_>>();

        arr.iter_lanes(&shape, 0).enumerate().for_each( |(i_lane, lane)| {
            assert_eq!(lane.len(), shape[0]);

            let collected = lane.iter().map(|ind| *ind).collect::<Vec<_>>();
            let base = vec![0 + i_lane, 20 + i_lane, 40 + i_lane];
            assert_eq!(collected, base);
        });

        arr.iter_lanes(&shape, 1).enumerate().for_each( |(i_lane, lane)| {
            assert_eq!(lane.len(), shape[1]);
            let i_back = i_lane % shape[2];
            let i_front = i_lane / shape[2];
            let ind = i_front * shape[1] * shape[2] + i_back;
            let collected = lane.iter().map(|ind| *ind).collect::<Vec<_>>();
            let base = vec![0 + ind, 5 + ind, 10 + ind,15 + ind];
            assert_eq!(collected, base);

        });

        arr.iter_lanes(&shape, 2).enumerate().for_each( |(i_lane, lane)| {
            assert_eq!(lane.len(), shape[2]);
            let offset = i_lane * shape[2];
            let collected = lane.iter().map(|ind| *ind).collect::<Vec<_>>();
            let base = (0..shape[2]).map(|i| i + offset).collect::<Vec<_>>();
            assert_eq!(collected, base);

        });
    }


    #[test]
    fn test_strided_lane_iter_4d(){
        let shape = [3, 4, 5, 6];
        let n_t = shape.iter().product();
        let arr = (0..n_t).collect::<Vec<_>>();
        let strides = (0..4).map(|i| shape.iter().skip(i + 1).product()).collect::<Vec<usize>>();

        for axis in 0..shape.len(){
            let shape_sub: [usize; 3] = (0..4).filter(|i| * i!= axis).map(|i| shape[i]).collect::<Vec<_>>().try_into().unwrap();
            arr.iter_lanes(&shape, axis).enumerate().for_each( |(i_lane, lane)| {
                assert_eq!(lane.len(), shape[axis]);

                let inds_sub = unravel(i_lane, &shape_sub);
                let offset: usize = strides.iter().
                    enumerate().filter(|(i, _)| *i != axis)
                    .zip(inds_sub)
                    .map(|((_, off), i_ax)| i_ax * off)
                    .sum();

                let collected = lane.iter().map(|ind| *ind).collect::<Vec<_>>();

                let base = (0..shape[axis]).map(|i| strides[axis] * i + offset).collect::<Vec<_>>();
                assert_eq!(collected, base);
            });
        }
    }


    #[test]
    fn test_strided_mut_2d(){
        let shape = [3, 2];
        let n_t = shape.iter().product();
        let mut arr = vec![0; n_t];

        for axis in 0..2{
            let mut index = 0;
            arr.iter_lanes_mut(&shape, axis).for_each( | mut lane| {
                assert_eq!(lane.len(), shape[axis]);
                lane.iter_mut().for_each(|v| {
                    *v = index;
                    index += 1
                });
            });

            let collected = arr.iter_lanes(&shape, axis).map(|lane|{
                lane.iter().map(|v| *v).collect::<Vec<_>>()
            }).collect::<Vec<_>>().concat();

            assert_eq!(collected, (0..n_t).collect::<Vec<_>>());
        }
    }

    #[test]
    fn test_strided_mut_3d(){
        let shape = [3, 4, 5];
        let n_t = shape.iter().product();
        let mut arr = vec![0; n_t];

        for axis in 0..shape.len(){
            let mut index = 0;
            arr.iter_lanes_mut(&shape, axis).for_each( | mut lane| {
                assert_eq!(lane.len(), shape[axis]);
                lane.iter_mut().for_each(|v| {
                    *v = index;
                    index += 1
                });
            });

            let collected = arr.iter_lanes(&shape, axis).map(|lane|{
                lane.iter().map(|v| *v).collect::<Vec<_>>()
            }).collect::<Vec<_>>().concat();

            assert_eq!(collected, (0..n_t).collect::<Vec<_>>());
        }
    }

    #[test]
    fn test_strided_mut_4d(){
        let shape = [3, 4, 5, 6];
        let n_t = shape.iter().product();
        let mut arr = vec![0; n_t];

        for axis in 0..shape.len(){
            let mut index = 0;
            arr.iter_lanes_mut(&shape, axis).for_each( | mut lane| {
                assert_eq!(lane.len(), shape[axis]);
                lane.iter_mut().for_each(|v| {
                    *v = index;
                    index += 1
                });
            });

            let collected = arr.iter_lanes(&shape, axis).map(|lane|{
                lane.iter().map(|v| *v).collect::<Vec<_>>()
            }).collect::<Vec<_>>().concat();

            assert_eq!(collected, (0..n_t).collect::<Vec<_>>());
        }
    }

    #[test]
    fn test_strided_mut(){
        let mut arr = [0; 6];
        let stride = 2;
        let mut strided = MutStridedSlice{
            base: arr.as_mut_ptr(), stride:stride, length:arr.len()/stride, _member:PhantomData
        };

        strided.iter_mut().for_each(|v| *v = 1);
        assert_eq!(arr, [1, 0, 1, 0, 1, 0]);

        let mut strided = MutStridedSlice{
            base: arr[1..].as_mut_ptr(), stride:stride, length:arr.len()/stride, _member:PhantomData
        };
        strided.iter_mut().for_each(|v| *v = 2);
        assert_eq!(arr, [1, 2, 1, 2, 1, 2]);

    }

    #[test]
    fn test_lane_chunks_2d(){

        const N: usize = 4;
        let shape = [6, 11];
        let n_total = shape.iter().product();
        let arr = (0..n_total).collect::<Vec<_>>();

        let (iter_chunks, iter_rem) = arr.iter_lane_chunks::<N>(&shape, 0);
        let n_chunks = iter_chunks.len();
        let n_rem = iter_rem.len();
        assert_eq!(n_chunks, shape[1] / N);
        assert_eq!(n_rem, shape[1] % N);

        for (i_chunk, chunk) in iter_chunks.enumerate(){
            assert_eq!(chunk.len(), shape[0]);
            for (i_row, row) in chunk.iter().enumerate(){
                assert_eq!(row.len(), N);
                let vals: [_; N] = row.map(|v| *v).collect::<Vec<_>>().try_into().unwrap();
                let goal = std::array::from_fn(|i_col|{
                    let col_ind = i_chunk * N + i_col;
                    col_ind + i_row * shape[1]
                });

                assert_eq!(vals, goal);
            }
        }
        for (i_lane, slice) in iter_rem.enumerate(){
            let col_ind = n_chunks * N + i_lane;
            assert_eq!(slice.len(), shape[0]);
            let val = slice.iter().map(|v| *v).collect::<Vec<_>>();
            let goal = (0..shape[0]).map(|i_row|{
                i_row * shape[1] + col_ind
            }).collect::<Vec<_>>();

            assert_eq!(val, goal);
        }

        let (iter_chunks, iter_rem) = arr.iter_lane_chunks::<N>(&shape, 1);
        let n_chunks = iter_chunks.len();
        let n_rem = iter_rem.len();
        assert_eq!(n_chunks, shape[0] / N);
        assert_eq!(n_rem, shape[0] % N);

        for (i_chunk, chunk) in iter_chunks.enumerate(){
            assert_eq!(chunk.len(), shape[1]);
            for (i_col, cols) in chunk.iter().enumerate(){
                assert_eq!(cols.len(), 4);
                let vals: [_; N] = cols.map(|v| *v).collect::<Vec<_>>().try_into().unwrap();
                let goal = std::array::from_fn(|i_row|{
                    let i_row = i_chunk * N + i_row;
                    i_col + i_row * shape[1]
                });

                assert_eq!(vals, goal);
            }
        }
        for (i_lane, slice) in iter_rem.enumerate(){
            let i_row = n_chunks * N + i_lane;
            assert_eq!(slice.len(), shape[1]);
            let val = slice.iter().map(|v| *v).collect::<Vec<_>>();
            let goal = (0..shape[1]).map(|i_col|{
                i_row * shape[1] + i_col
            }).collect::<Vec<_>>();

            assert_eq!(val, goal);
        }
    }


    #[test]
    fn test_lane_chunks_nd(){

        const N: usize = 4;
        let shape = [5, 6, 7, 8, 9];
        let n_total = shape.iter().product();
        let arr = (0..n_total).collect::<Vec<_>>();
        let strides = (0..shape.len()).map(|i| shape.iter().skip(i + 1).product()).collect::<Vec<usize>>();


        for ax in 0..shape.len(){
            println!("Axis: {ax}");
            let shape_sub: [usize; 4] = (0..5).filter(|i| * i!= ax).map(|i| shape[i]).collect::<Vec<_>>().try_into().unwrap();


            let n_along = shape[ax];
            let n_above: usize = shape.iter().take(ax).product();
            let n_below: usize = shape.iter().skip(ax + 1).product();
            let n_lanes = n_total / n_along;
            assert_eq!(n_above * n_below, n_lanes);

            let (iter_chunks, iter_rem) = arr.iter_lane_chunks::<N>(&shape, ax);
            let n_chunks = iter_chunks.len();
            let n_rem = iter_rem.len();

            assert_eq!(n_chunks, n_lanes / N);
            assert_eq!(n_rem, n_lanes % N);

            for (i_chunk, chunk) in iter_chunks.enumerate(){
                assert_eq!(chunk.len(), n_along);
                for (i_along, items) in chunk.iter().enumerate(){
                    assert_eq!(items.len(), N);
                    let vals = items.map(|v| *v).collect::<Vec<_>>();
                    let goal = (0..N).map(|i_l| {
                        let i_lane = i_chunk * N + i_l;
                        let inds_sub = unravel(i_lane, &shape_sub);
                        let offset: usize = strides.iter().
                            enumerate().filter(|(i, _)| *i != ax)
                            .zip(inds_sub)
                            .map(|((_, off), i_ax)| i_ax * off)
                            .sum();

                        strides[ax] * i_along + offset
                        }).collect::<Vec<_>>();

                    assert_eq!(vals, goal);

                }
            }

            for (i_rem, lane) in iter_rem.enumerate(){
                let i_lane = i_rem + n_chunks * N;
                assert_eq!(lane.len(), n_along);


                let inds_sub = unravel(i_lane, &shape_sub);
                let offset: usize = strides.iter().
                    enumerate().filter(|(i, _)| *i != ax)
                    .zip(inds_sub)
                    .map(|((_, off), i_ax)| i_ax * off)
                    .sum();

                let vals = lane.iter().map(|v| * v).collect::<Vec<_>>();
                let goal = (0..n_along).map(|i_along| {
                    strides[ax] * i_along + offset
                }).collect::<Vec<_>>();

                assert_eq!(vals, goal);

            }
        }

    }

    #[test]
    fn test_lane_chunks_mut_nd(){

        const N: usize = 4;
        let shape = [5, 6, 7, 8, 9];
        let n_total = shape.iter().product();
        let arr = (0..n_total).collect::<Vec<_>>();
        let strides = (0..shape.len()).map(|i| shape.iter().skip(i + 1).product()).collect::<Vec<usize>>();


        for ax in 0..shape.len(){

            let mut out = vec![0; n_total];
            let shape_sub: [usize; 4] = (0..5).filter(|i| * i!= ax).map(|i| shape[i]).collect::<Vec<_>>().try_into().unwrap();


            let n_along = shape[ax];
            let n_above: usize = shape.iter().take(ax).product();
            let n_below: usize = shape.iter().skip(ax + 1).product();
            let n_lanes = n_total / n_along;
            assert_eq!(n_above * n_below, n_lanes);

            let (iter_chunks, iter_rem) = out.iter_lane_chunks_mut::<N>(&shape, ax);
            let n_chunks = iter_chunks.len();
            let n_rem = iter_rem.len();

            assert_eq!(n_chunks, n_lanes / N);
            assert_eq!(n_rem, n_lanes % N);

            for (i_chunk, mut chunk) in iter_chunks.enumerate(){
                assert_eq!(chunk.len(), n_along);
                for (i_along, items) in chunk.iter_mut().enumerate(){
                    assert_eq!(items.len(), N);
                    items.enumerate().for_each(|(i_l, v)| {
                        let i_lane = i_chunk * N + i_l;
                        let inds_sub = unravel(i_lane, &shape_sub);

                        let offset: usize = strides.iter().
                            enumerate().filter(|(i, _)| *i != ax)
                            .zip(inds_sub)
                            .map(|((_, off), i_ax)| i_ax * off)
                            .sum();

                        *v = strides[ax] * i_along + offset;
                    });
                }
            }

            for (i_rem, mut lane) in iter_rem.enumerate(){
                let i_lane = i_rem + n_chunks * N;
                assert_eq!(lane.len(), n_along);

                let inds_sub = unravel(i_lane, &shape_sub);
                let offset: usize = strides.iter().
                    enumerate().filter(|(i, _)| *i != ax)
                    .zip(inds_sub)
                    .map(|((_, off), i_ax)| i_ax * off)
                    .sum();

                lane.iter_mut().enumerate()
                    .for_each(|(i_along, v)| {
                        *v = strides[ax] * i_along + offset;
                    })
            }

            assert_eq!(out, arr);
        }

    }

    mod par{
        use super::*;
        use super::parallel::*;

        #[test]
        fn test_lane_iter_4d(){
            let shape = [3, 4, 5, 6];
            let n_t = shape.iter().product();
            let arr = (0..n_t).collect::<Vec<_>>();
            let strides = (0..4).map(|i| shape.iter().skip(i + 1).product()).collect::<Vec<usize>>();

            for axis in 0..shape.len(){
                let shape_sub: [usize; 3] = (0..4).filter(|i| * i!= axis).map(|i| shape[i]).collect::<Vec<_>>().try_into().unwrap();
                arr.par_iter_lanes(&shape, axis).enumerate().for_each( |(i_lane, lane)| {
                    assert_eq!(lane.len(), shape[axis]);

                    let inds_sub = unravel(i_lane, &shape_sub);
                    let offset: usize = strides.iter().
                        enumerate().filter(|(i, _)| *i != axis)
                        .zip(inds_sub)
                        .map(|((_, off), i_ax)| i_ax * off)
                        .sum();

                    let collected = lane.iter().map(|ind| *ind).collect::<Vec<_>>();

                    let base = (0..shape[axis]).map(|i| strides[axis] * i + offset).collect::<Vec<_>>();
                    assert_eq!(collected, base);
                });
            }
        }

        #[test]
        fn test_lane_iter_mut_4d(){
            let shape = [3, 4, 5, 6];
            let n_t = shape.iter().product();
            let mut arr = vec![0; n_t];

            for axis in 0..shape.len(){
                arr.par_iter_lanes_mut(&shape, axis).enumerate().for_each( | (lane_ind, mut lane)| {
                    assert_eq!(lane.len(), shape[axis]);
                    let index = lane_ind * shape[axis];
                    lane.iter_mut().enumerate().for_each(|(ii, v)| {
                        *v = index + ii;
                    });
                });

                let collected = arr.iter_lanes(&shape, axis).map(|lane|{
                    lane.iter().map(|v| *v).collect::<Vec<_>>()
                }).collect::<Vec<_>>().concat();

                assert_eq!(collected, (0..n_t).collect::<Vec<_>>());
            }
        }

        #[test]
        fn test_lane_chunks_nd(){

            const N: usize = 4;
            let shape = [5, 6, 7, 8, 9];
            let n_total = shape.iter().product();
            let arr = (0..n_total).collect::<Vec<_>>();
            let strides = (0..shape.len()).map(|i| shape.iter().skip(i + 1).product()).collect::<Vec<usize>>();


            for ax in 0..shape.len(){
                println!("Axis: {ax}");
                let shape_sub: [usize; 4] = (0..5).filter(|i| * i!= ax).map(|i| shape[i]).collect::<Vec<_>>().try_into().unwrap();

                let lane_length = shape[ax];
                let n_lanes: usize = shape.iter().enumerate().filter(|(i, _)| *i != ax).map(|(_, v)| v).product();
                assert_eq!(n_lanes * lane_length, shape.iter().product());

                let (iter_chunks, iter_rem) = arr.par_iter_lane_chunks::<N>(&shape, ax);
                let n_chunks = iter_chunks.len();
                let n_rem = iter_rem.len();

                assert_eq!(n_chunks, n_lanes / N);
                assert_eq!(n_rem, n_lanes % N);

                iter_chunks.enumerate().for_each(|(i_chunk, chunk)| {
                    assert_eq!(chunk.len(), lane_length);
                    for (i_along, items) in chunk.iter().enumerate(){
                        assert_eq!(items.len(), N);
                        let vals = items.map(|v| *v).collect::<Vec<_>>();
                        let goal = (0..N).map(|i_l| {
                            let i_lane = i_chunk * N + i_l;
                            let inds_sub = unravel(i_lane, &shape_sub);
                            let offset: usize = strides.iter().
                                enumerate().filter(|(i, _)| *i != ax)
                                .zip(inds_sub)
                                .map(|((_, off), i_ax)| i_ax * off)
                                .sum();

                            strides[ax] * i_along + offset
                            }).collect::<Vec<_>>();

                        assert_eq!(vals, goal);

                    }
                });

                iter_rem.enumerate().for_each(|(i_rem, lane)|{
                    let i_lane = i_rem + n_chunks * N;
                    assert_eq!(lane.len(), lane_length);


                    let inds_sub = unravel(i_lane, &shape_sub);
                    let offset: usize = strides.iter().
                        enumerate().filter(|(i, _)| *i != ax)
                        .zip(inds_sub)
                        .map(|((_, off), i_ax)| i_ax * off)
                        .sum();

                    let vals = lane.iter().map(|v| * v).collect::<Vec<_>>();
                    let goal = (0..lane_length).map(|i_along| {
                        strides[ax] * i_along + offset
                    }).collect::<Vec<_>>();

                    assert_eq!(vals, goal);

                });
            }
        }
    
            #[test]
        fn test_lane_chunks_mut_nd(){

            const N: usize = 4;
            let shape = [5, 6, 7, 8, 9];
            let n_total = shape.iter().product();
            let arr = (0..n_total).collect::<Vec<_>>();
            let strides = (0..shape.len()).map(|i| shape.iter().skip(i + 1).product()).collect::<Vec<usize>>();


            for ax in 0..shape.len(){

                let mut out = vec![0; n_total];
                let shape_sub: [usize; 4] = (0..5).filter(|i| * i!= ax).map(|i| shape[i]).collect::<Vec<_>>().try_into().unwrap();


                let n_along = shape[ax];
                let n_above: usize = shape.iter().take(ax).product();
                let n_below: usize = shape.iter().skip(ax + 1).product();
                let n_lanes = n_total / n_along;
                assert_eq!(n_above * n_below, n_lanes);

                let (iter_chunks, iter_rem) = out.par_iter_lane_chunks_mut::<N>(&shape, ax);
                let n_chunks = iter_chunks.len();
                let n_rem = iter_rem.len();

                assert_eq!(n_chunks, n_lanes / N);
                assert_eq!(n_rem, n_lanes % N);

                iter_chunks.enumerate().for_each(|(i_chunk, mut chunk)| {
                    assert_eq!(chunk.len(), n_along);
                    for (i_along, items) in chunk.iter_mut().enumerate(){
                        assert_eq!(items.len(), N);
                        items.enumerate().for_each(|(i_l, v)| {
                            let i_lane = i_chunk * N + i_l;
                            let inds_sub = unravel(i_lane, &shape_sub);

                            let offset: usize = strides.iter().
                                enumerate().filter(|(i, _)| *i != ax)
                                .zip(inds_sub)
                                .map(|((_, off), i_ax)| i_ax * off)
                                .sum();

                            *v = strides[ax] * i_along + offset;
                        });
                    }
                });

                iter_rem.enumerate().for_each(|(i_rem, mut lane)|{
                    let i_lane = i_rem + n_chunks * N;
                    assert_eq!(lane.len(), n_along);

                    let inds_sub = unravel(i_lane, &shape_sub);
                    let offset: usize = strides.iter().
                        enumerate().filter(|(i, _)| *i != ax)
                        .zip(inds_sub)
                        .map(|((_, off), i_ax)| i_ax * off)
                        .sum();

                    lane.iter_mut().enumerate()
                        .for_each(|(i_along, v)| {
                            *v = strides[ax] * i_along + offset;
                        })
                });

                assert_eq!(out, arr);
            }

        }
    
    }
}