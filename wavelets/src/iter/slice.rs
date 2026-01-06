use std::marker::PhantomData;

#[inline]
pub fn unravel<const N: usize>(flat_index: usize, shape: &[usize; N]) -> [usize; N]{
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

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = unsafe{self.end.offset_from_unsigned(self.start)};
        (len, Some(len))
    }
}

impl<'a, T> ExactSizeIterator for StridedIter<'a, T>{}
impl<'a, T> DoubleEndedIterator for StridedIter<'a, T>{
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
    base: [*const T; N],
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
        let ends = std::array::from_fn(|i|{
            unsafe{self.base[i].add(end_offset)}
        });
        ChunkStridedSliceIter {
            starts:self.base,
            ends: ends,
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
            let slice_target = *self.base.get_unchecked(index.1);
            let target = slice_target.add(self.stride * index.0);
            & *target
        }
    }
}

pub struct AlongChunkIter<'a, T, const N: usize>{
    pointers: [*const T; N],
    start: usize,
    end: usize,
    _member: PhantomData<&'a T>
}

impl<'a, T, const N: usize> Iterator for AlongChunkIter<'a, T, N>{
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item>{
        if self.start == self.end {None}
        else{
            let target = unsafe{ & *self.pointers[self.start]};
            self.start += 1;
            Some(target)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.end - self.start;
        (len, Some(len))
    }
}
impl<'a, T, const N: usize> ExactSizeIterator for AlongChunkIter<'a, T, N>{}
impl<'a, T, const N: usize> DoubleEndedIterator for AlongChunkIter<'a, T, N>{

    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start == self.end {None}
        else{
            self.end -= 1;
            let target = unsafe{ & *self.pointers[self.end]};
            Some(target)
        }

    }
}

pub struct ChunkStridedSliceIter<'a, T, const N: usize>{
    starts: [*const T; N],
    ends: [*const T; N],
    stride: usize,
    _member: PhantomData<&'a T>,
}

impl<'a, T, const N: usize> Iterator for ChunkStridedSliceIter<'a, T, N>{
    type Item = AlongChunkIter<'a, T, N>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item>{
        if self.starts[0] == self.ends[0] {None}
        else{
            let items = AlongChunkIter{
                pointers:self.starts,
                start: 0,
                end: N,
                _member: PhantomData
            };
            self.starts = self.starts.map(|v| unsafe{v.add(self.stride)});
            Some(items)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = unsafe{self.ends[0].offset_from_unsigned(self.starts[0])};
        (len, Some(len))
    }
}
impl<'a, T, const N: usize> ExactSizeIterator for ChunkStridedSliceIter<'a, T, N>{}
impl<'a, T, const N: usize> DoubleEndedIterator for ChunkStridedSliceIter<'a, T, N>{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.starts[0] == self.ends[0] {None}
        else{
            self.ends = self.ends.map(|v| unsafe{v.sub(self.stride)});
            let items = AlongChunkIter{
                pointers:self.ends,
                start: 0,
                end: N,
                _member: PhantomData
            };
            Some(items)
        }
    }
}

pub struct LaneSliceIter<'a, T>{
    base: *const T,
    n_along: usize,
    n_back: usize,
    bounds: [usize; 2],
    _member: PhantomData<&'a T>
}

impl<'a, T> LaneSliceIter<'a, T>{

    pub fn from_slice(arr: &'a [T], shape: &[usize], axis: usize) -> Self{
        assert_eq!(arr.len(), shape.iter().product());
        assert!(axis < shape.len());

        let n_front: usize = shape.iter().take(axis).product();
        let n_along = shape[axis];
        let n_back = shape.iter().skip(axis + 1).product();

        LaneSliceIter {
            base: arr.as_ptr(),
            n_along: n_along,
            n_back: n_back,
            bounds: [0, n_front * n_back],
            _member: PhantomData
        }
    }

    #[inline]
    fn get_front_slice_start(&self) -> usize{
        let ind = self.bounds[0];
        let i_back = ind % self.n_back;
        let i_front = ind / self.n_back;

        i_front * self.n_back * self.n_along + i_back
    }

    #[inline]
    fn get_back_slice_start(&self) -> usize{
        let ind = self.bounds[1];
        let i_back = ind % self.n_back;
        let i_front = ind / self.n_back;

        i_front * self.n_back * self.n_along + i_back
    }
}

impl<'a, T> Iterator for LaneSliceIter<'a, T>{
    type Item = StridedSlice<'a, T>;
    fn next(&mut self) -> Option<Self::Item>{
        if self.bounds[0] >= self.bounds[1] { None}
        else{
            // get the starting index of the slice
            let slice_ptr = unsafe{self.base.add(self.get_front_slice_start())};

            self.bounds[0] += 1;

            Some(
                StridedSlice{
                    base: slice_ptr,
                    length: self.n_along,
                    stride: self.n_back,
                    _member: PhantomData,
            })
        }
    }

    fn nth(&mut self, ind: usize) -> Option<Self::Item>{
        let index = self.bounds[0] + ind;
        if index >= self.bounds[1] { None}
        else{
            self.bounds[0] = index;
            self.next()
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.bounds[1] - self.bounds[0];
        (len, Some(len))
    }
}

impl<'a, T> ExactSizeIterator for LaneSliceIter<'a, T>{}
impl<'a, T> DoubleEndedIterator for LaneSliceIter<'a, T>{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.bounds[0] == self.bounds[1] {None}
        else{

            self.bounds[1] -= 1;

            let slice_ptr = unsafe{self.base.add(self.get_back_slice_start())};
            Some(StridedSlice{
                base: slice_ptr,
                length: self.n_along,
                stride: self.n_back,
                _member: PhantomData
            })
        }
    }
}

pub struct LaneSliceChunkIter<'a, T, const N: usize>{
    base: *const T,
    n_along: usize,
    n_back: usize,
    bounds: [usize; 2],
    _member: PhantomData<&'a T>
}

impl<'a, T, const N:usize> LaneSliceChunkIter<'a, T, N>{

    pub fn from_slice(arr: &'a [T], shape: &[usize], axis: usize) -> (Self, LaneSliceIter<'a, T>){
        assert_eq!(arr.len(), shape.iter().product());
        assert!(axis < shape.len());

        let n_front: usize = shape.iter().take(axis).product();
        let n_along = shape[axis];
        let n_back = shape.iter().skip(axis + 1).product();

        let len = n_front * n_back;
        let rem = len % N;
        let chunk_end = len - rem;

        (
            Self {
                base: arr.as_ptr(),
                n_along: n_along,
                n_back: n_back,
                bounds: [0, chunk_end],
                _member: PhantomData
            },
            LaneSliceIter{
                base: arr.as_ptr(),
                n_along: n_along,
                n_back: n_back,
                bounds:[chunk_end, len],
                _member: PhantomData
            }
        )
    }

    #[inline]
    fn get_front_slice_start(&self) -> usize{
        let ind = self.bounds[0];
        let i_back = ind % self.n_back;
        let i_front = ind / self.n_back;

        i_front * self.n_back * self.n_along + i_back
    }

    #[inline]
    fn get_back_slice_start(&self) -> usize{
        let ind = self.bounds[1];
        let i_back = ind % self.n_back;
        let i_front = ind / self.n_back;

        i_front * self.n_back * self.n_along + i_back
    }
}

impl<'a, T, const N:usize> Iterator for LaneSliceChunkIter<'a, T, N>{
    type Item = ChunkStridedSlice<'a, T, N>;
    fn next(&mut self) -> Option<Self::Item>{
        if self.bounds[0] >= self.bounds[1] { None}
        else{
            // get the starting index of the slice
            let slice_ptrs = std::array::from_fn(|i|{
                let slice_ptr = unsafe{self.base.add(self.get_front_slice_start())};
                self.bounds[0] += 1;
                slice_ptr
            });

            Some(
                ChunkStridedSlice{
                    base: slice_ptrs,
                    length: self.n_along,
                    stride: self.n_back,
                    _member: PhantomData,
            })
        }
    }

    fn nth(&mut self, ind: usize) -> Option<Self::Item>{
        let index = self.bounds[0] + ind;
        if index >= self.bounds[1] { None}
        else{
            self.bounds[0] = index;
            self.next()
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = (self.bounds[1] - self.bounds[0])/N;
        (len, Some(len))
    }
}

impl<'a, T, const N:usize> ExactSizeIterator for LaneSliceChunkIter<'a, T, N>{}
impl<'a, T, const N:usize> DoubleEndedIterator for LaneSliceChunkIter<'a, T, N>{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.bounds[0] == self.bounds[1] {None}
        else{
            let slice_ptrs = std::array::from_fn(|_|{
                self.bounds[1] -= 1;
                unsafe{self.base.add(self.get_back_slice_start())}
            });

            Some(ChunkStridedSlice{
                base: slice_ptrs,
                length: self.n_along,
                stride: self.n_back,
                _member: PhantomData
            })
        }
    }
}


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
    base: [*mut T; N],
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
        let starts = std::array::from_fn(|i|{
            self.base[i] as *const T
        });
        let ends = std::array::from_fn(|i|{
            unsafe{self.base[i].add(end_offset) as *const T}
        });
        ChunkStridedSliceIter {
            starts: starts,
            ends: ends,
            stride:self.stride,
            _member: PhantomData
        }
    }

    #[inline]
    pub fn iter_mut(&'a mut self) -> MutChunkStridedSliceIter<'a, T, N>{
        let end_offset = self.length * self.stride;
        let ends = std::array::from_fn(|i|{
            unsafe{self.base[i].add(end_offset)}
        });
        MutChunkStridedSliceIter {
            starts: self.base,
            ends: ends,
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
            let slice_target = *self.base.get_unchecked(index.1);
            let target = slice_target.add(self.stride * index.0);
            & *target
        }
    }
}

pub struct MutAlongChunkIter<'a, T, const N: usize>{
    pointers: [*mut T; N],
    start: usize,
    end: usize,
    _member: PhantomData<&'a T>
}

impl<'a, T, const N: usize> Iterator for MutAlongChunkIter<'a, T, N>{
    type Item = &'a mut T;
    fn next(&mut self) -> Option<Self::Item>{
        if self.start == self.end {None}
        else{
            let target = unsafe{ &mut *self.pointers[self.start]};
            self.start += 1;
            Some(target)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.end - self.start;
        (len, Some(len))
    }
}
impl<'a, T, const N: usize> ExactSizeIterator for MutAlongChunkIter<'a, T, N>{}
impl<'a, T, const N: usize> DoubleEndedIterator for MutAlongChunkIter<'a, T, N>{

    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start == self.end {None}
        else{
            self.end -= 1;
            let target = unsafe{ &mut *self.pointers[self.end]};
            Some(target)
        }

    }
}

pub struct MutChunkStridedSliceIter<'a, T, const N: usize>{
    starts: [*mut T; N],
    ends: [*mut T; N],
    stride: usize,
    _member: PhantomData<&'a T>,
}

impl<'a, T, const N: usize> Iterator for MutChunkStridedSliceIter<'a, T, N>{
    type Item = MutAlongChunkIter<'a, T, N>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item>{
        if self.starts[0] == self.ends[0] {None}
        else{
            let items = MutAlongChunkIter{
                pointers:self.starts,
                start: 0,
                end: N,
                _member: PhantomData
            };
            self.starts = self.starts.map(|v| unsafe{v.add(self.stride)});
            Some(items)
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = unsafe{self.ends[0].offset_from_unsigned(self.starts[0])};
        (len, Some(len))
    }
}
impl<'a, T, const N: usize> ExactSizeIterator for MutChunkStridedSliceIter<'a, T, N>{}
impl<'a, T, const N: usize> DoubleEndedIterator for MutChunkStridedSliceIter<'a, T, N>{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.starts[0] == self.ends[0] {None}
        else{
            self.ends = self.ends.map(|v| unsafe{v.sub(self.stride)});
            let items = MutAlongChunkIter{
                pointers:self.ends,
                start: 0,
                end: N,
                _member: PhantomData
            };
            Some(items)
        }
    }
}

pub struct MutLaneSliceIter<'a, T>{
    base: *mut T,
    n_along: usize,
    n_back: usize,
    bounds: [usize; 2],
    _member: PhantomData<&'a T>
}

impl<'a, T> MutLaneSliceIter<'a, T>{

    pub fn from_slice_mut(arr: &'a mut [T], shape: &[usize], axis: usize) -> Self{
        assert_eq!(arr.len(), shape.iter().product());
        assert!(axis < shape.len());

        let n_front: usize = shape.iter().take(axis).product();
        let n_along = shape[axis];
        let n_back = shape.iter().skip(axis + 1).product();

        MutLaneSliceIter {
            base: arr.as_mut_ptr(),
            n_along: n_along,
            n_back: n_back,
            bounds: [0, n_front * n_back],
            _member: PhantomData
        }
    }

    #[inline]
    fn get_front_slice_start(&self) -> usize{
        let ind = self.bounds[0];
        let i_back = ind % self.n_back;
        let i_front = ind / self.n_back;

        i_front * self.n_back * self.n_along + i_back
    }

    #[inline]
    fn get_back_slice_start(&self) -> usize{
        let ind = self.bounds[1];
        let i_back = ind % self.n_back;
        let i_front = ind / self.n_back;

        i_front * self.n_back * self.n_along + i_back
    }
}

impl<'a, T> Iterator for MutLaneSliceIter<'a, T>{
    type Item = MutStridedSlice<'a, T>;
    fn next(&mut self) -> Option<Self::Item>{
        if self.bounds[0] >= self.bounds[1] { None}
        else{
            // get the starting index of the slice
            let slice_ptr = unsafe{self.base.add(self.get_front_slice_start())};

            self.bounds[0] += 1;

            Some(
                MutStridedSlice{
                    base: slice_ptr,
                    length: self.n_along,
                    stride: self.n_back,
                    _member: PhantomData,
            })
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.bounds[1] - self.bounds[0];
        (len, Some(len))
    }
}

impl<'a, T> ExactSizeIterator for MutLaneSliceIter<'a, T>{}
impl<'a, T> DoubleEndedIterator for MutLaneSliceIter<'a, T>{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.bounds[0] == self.bounds[1] {None}
        else{

            self.bounds[1] -= 1;

            let slice_ptr = unsafe{self.base.add(self.get_back_slice_start())};
            Some(MutStridedSlice{
                base: slice_ptr,
                length: self.n_along,
                stride: self.n_back,
                _member: PhantomData
            })
        }
    }
}

pub struct MutLaneSliceChunkIter<'a, T, const N: usize>{
    base: *mut T,
    n_along: usize,
    n_back: usize,
    bounds: [usize; 2],
    _member: PhantomData<&'a T>
}

impl<'a, T, const N:usize> MutLaneSliceChunkIter<'a, T, N>{

    pub fn from_slice_mut(arr: &'a mut [T], shape: &[usize], axis: usize) -> (Self, MutLaneSliceIter<'a, T>){
        assert_eq!(arr.len(), shape.iter().product());
        assert!(axis < shape.len());

        let n_front: usize = shape.iter().take(axis).product();
        let n_along = shape[axis];
        let n_back = shape.iter().skip(axis + 1).product();

        let len = n_front * n_back;
        let rem = len % N;
        let chunk_end = len - rem;

        (
            Self {
                base: arr.as_mut_ptr(),
                n_along: n_along,
                n_back: n_back,
                bounds: [0, chunk_end],
                _member: PhantomData
            },
            MutLaneSliceIter{
                base: arr.as_mut_ptr(),
                n_along: n_along,
                n_back: n_back,
                bounds:[chunk_end, len],
                _member: PhantomData
            }
        )
    }

    #[inline]
    fn get_front_slice_start(&self) -> usize{
        let ind = self.bounds[0];
        let i_back = ind % self.n_back;
        let i_front = ind / self.n_back;

        i_front * self.n_back * self.n_along + i_back
    }

    #[inline]
    fn get_back_slice_start(&self) -> usize{
        let ind = self.bounds[1];
        let i_back = ind % self.n_back;
        let i_front = ind / self.n_back;

        i_front * self.n_back * self.n_along + i_back
    }
}

impl<'a, T, const N:usize> Iterator for MutLaneSliceChunkIter<'a, T, N>{
    type Item = MutChunkStridedSlice<'a, T, N>;
    fn next(&mut self) -> Option<Self::Item>{
        if self.bounds[0] >= self.bounds[1] { None}
        else{
            // get the starting index of the slice
            let slice_ptrs = std::array::from_fn(|_|{
                let slice_ptr = unsafe{self.base.add(self.get_front_slice_start())};
                self.bounds[0] += 1;
                slice_ptr
            });

            Some(
                MutChunkStridedSlice{
                    base: slice_ptrs,
                    length: self.n_along,
                    stride: self.n_back,
                    _member: PhantomData,
            })
        }
    }

    fn nth(&mut self, ind: usize) -> Option<Self::Item>{
        let index = self.bounds[0] + ind;
        if index >= self.bounds[1] { None}
        else{
            self.bounds[0] = index;
            self.next()
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = (self.bounds[1] - self.bounds[0])/N;
        (len, Some(len))
    }
}

impl<'a, T, const N:usize> ExactSizeIterator for MutLaneSliceChunkIter<'a, T, N>{}
impl<'a, T, const N:usize> DoubleEndedIterator for MutLaneSliceChunkIter<'a, T, N>{
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.bounds[0] == self.bounds[1] {None}
        else{
            let slice_ptrs = std::array::from_fn(|_|{
                self.bounds[1] -= 1;
                unsafe{self.base.add(self.get_back_slice_start())}
            });

            Some(MutChunkStridedSlice{
                base: slice_ptrs,
                length: self.n_along,
                stride: self.n_back,
                _member: PhantomData
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

// pub mod parallel{
//     use core::slice;

//     pub use rayon::iter::{ParallelIterator, IndexedParallelIterator};
//     use rayon::iter::plumbing::{UnindexedConsumer, Consumer, bridge, ProducerCallback, Producer};

//     use super::*;

//     pub struct LaneSliceParIter<'a, T>{
//         base: &'a [T],
//         n_along: usize,
//         n_back: usize,
//         bounds: [usize; 2],
//     }

//     impl<'a, T> LaneSliceParIter<'a, T>{
//         pub fn from_slice(arr: &'a [T], shape: &[usize], axis: usize) -> Self{
//             assert_eq!(arr.len(), shape.iter().product());
//             assert!(axis < shape.len());
//             let n_front: usize = shape.iter().take(axis).product();
//             let n_along = shape[axis];
//             let n_back = shape.iter().skip(axis + 1).product();

//             LaneSliceParIter {
//                 base: arr,
//                 n_along: n_along,
//                 n_back: n_back,
//                 bounds: [0, n_front * n_back] }
//         }
//     }

//     impl<'a, T: Sync + Send> ParallelIterator for LaneSliceParIter<'a, T>{
//         type Item = StridedSlice<'a, T>;
//         fn drive_unindexed<C>(self, consumer: C) -> C::Result
//         where
//             C: UnindexedConsumer<Self::Item>
//         {
//             bridge(self, consumer)
//         }
//     }

//     impl<'a, T: Sync> IndexedParallelIterator for LaneSliceParIter<'a, T>{
//         fn drive<C>(self, consumer: C) -> C::Result
//         where
//             C: Consumer<Self::Item>,
//         {
//             bridge(self, consumer)
//         }

//         fn len(&self) -> usize{
//             self.bounds[1] - self.bounds[0]
//         }

//         fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
//             callback.callback(
//                 LaneSliceParIter{..self}
//             )
//         }
//     }

//     impl<'a, T: Sync> Producer for LaneSliceParIter<'a, T>{
//         type Item = StridedSlice<'a, T>;
//         type IntoIter = LaneSliceIter<'a, T>;

//         fn into_iter(self) -> Self::IntoIter{
//             LaneSliceIter{
//                 base: self.base,
//                 n_along: self.n_along,
//                 n_back: self.n_back,
//                 bounds: self.bounds
//             }
//         }

//         fn split_at(self, index: usize) -> (Self, Self) {
//             let split = self.bounds[0] + index;
//             let bounds_left = [self.bounds[0], split];
//             let bounds_right = [split, self.bounds[1]];
//             (
//                 Self{
//                     bounds: bounds_left,
//                     ..self
//                 },
//                 Self{
//                     bounds: bounds_right,
//                     ..self
//                 }
//             )
//         }
//     }

//     pub struct MutLaneSliceParIter<'a, T>{
//         base: &'a mut [T],
//         n_along: usize,
//         n_back: usize,
//         bounds: [usize; 2],
//     }

//     impl<'a, T> MutLaneSliceParIter<'a, T>{
//         pub fn from_slice_mut(arr: &'a mut [T], shape: &[usize], axis: usize) -> Self{
//             assert_eq!(arr.len(), shape.iter().product());
//             assert!(axis < shape.len());
//             let n_front: usize = shape.iter().take(axis).product();
//             let n_along = shape[axis];
//             let n_back = shape.iter().skip(axis + 1).product();

//             MutLaneSliceParIter {
//                 base: arr,
//                 n_along: n_along,
//                 n_back: n_back,
//                 bounds: [0, n_front * n_back] }
//         }
//     }

//     impl<'a, T: Send> ParallelIterator for MutLaneSliceParIter<'a, T>{
//         type Item = MutStridedSlice<'a, T>;
//         fn drive_unindexed<C>(self, consumer: C) -> C::Result
//         where
//             C: UnindexedConsumer<Self::Item>
//         {
//             bridge(self, consumer)
//         }
//     }

//     impl<'a, T: Send> IndexedParallelIterator for MutLaneSliceParIter<'a, T>{
//         fn drive<C>(self, consumer: C) -> C::Result
//         where
//             C: Consumer<Self::Item>,
//         {
//             bridge(self, consumer)
//         }

//         fn len(&self) -> usize{
//             self.bounds[1] - self.bounds[0]
//         }

//         fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
//             callback.callback(
//                 MutLaneSliceParIter{..self}
//             )
//         }
//     }

//     impl<'a, T: Send> Producer for MutLaneSliceParIter<'a, T>{
//         type Item = MutStridedSlice<'a, T>;
//         type IntoIter = MutLaneSliceIter<'a, T>;

//         fn into_iter(self) -> Self::IntoIter{
//             MutLaneSliceIter{
//                 base: self.base,
//                 n_along: self.n_along,
//                 n_back: self.n_back,
//                 bounds: self.bounds
//             }
//         }

//         fn split_at(self, index: usize) -> (Self, Self) {
//             let split = self.bounds[0] + index;
//             let bounds_left = [self.bounds[0], split];
//             let bounds_right = [split, self.bounds[1]];

//             let base_cloned = unsafe{slice::from_raw_parts_mut(self.base.as_mut_ptr(), self.base.len())};
//             (
//                 Self{
//                     bounds: bounds_left,
//                     ..self
//                 },
//                 Self{
//                     base: base_cloned,
//                     bounds: bounds_right,
//                     ..self
//                 }
//             )
//         }
//     }

//     pub trait ParallelLanesIterator<T>{
//         fn par_iter_lanes(&self, shape: &[usize], axis: usize) -> LaneSliceParIter<'_, T>;
//         fn par_iter_lanes_mut(&mut self, shape: &[usize], axis: usize) -> MutLaneSliceParIter<'_, T>;
//     }

//     impl<T> ParallelLanesIterator<T> for [T]{
//         fn par_iter_lanes(&self, shape: &[usize], axis: usize) -> LaneSliceParIter<'_, T>{
//             LaneSliceParIter::from_slice(self, shape, axis)
//         }
//         fn par_iter_lanes_mut(&mut self, shape: &[usize], axis: usize) -> MutLaneSliceParIter<'_, T>{
//             MutLaneSliceParIter::from_slice_mut(self, shape, axis)
//         }
//     }
// }

#[cfg(test)]
mod tests{

    use std::iter;

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
                assert_eq!(row.len(), 4);
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

    // mod par{
    //     use super::*;
    //     use super::super::parallel::*;

    //     #[test]
    //     fn test_strided_lane_iter_4d(){
    //         let shape = [3, 4, 5, 6];
    //         let n_t = shape.iter().product();
    //         let arr = (0..n_t).collect::<Vec<_>>();
    //         let strides = (0..4).map(|i| shape.iter().skip(i + 1).product()).collect::<Vec<usize>>();

    //         for axis in 0..shape.len(){
    //             let shape_sub: [usize; 3] = (0..4).filter(|i| * i!= axis).map(|i| shape[i]).collect::<Vec<_>>().try_into().unwrap();
    //             arr.par_iter_lanes(&shape, axis).enumerate().for_each( |(i_lane, lane)| {
    //                 assert_eq!(lane.len(), shape[axis]);

    //                 let inds_sub = unravel(i_lane, &shape_sub);
    //                 let offset: usize = strides.iter().
    //                     enumerate().filter(|(i, _)| *i != axis)
    //                     .zip(inds_sub)
    //                     .map(|((_, off), i_ax)| i_ax * off)
    //                     .sum();

    //                 let collected = lane.iter().map(|ind| *ind).collect::<Vec<_>>();

    //                 let base = (0..shape[axis]).map(|i| strides[axis] * i + offset).collect::<Vec<_>>();
    //                 assert_eq!(collected, base);
    //             });
    //         }
    //     }

    //     #[test]
    //     fn test_strided_lane_iter_mut_4d(){
    //         let shape = [3, 4, 5, 6];
    //         let n_t = shape.iter().product();
    //         let mut arr = vec![0; n_t];

    //         for axis in 0..shape.len(){
    //             arr.par_iter_lanes_mut(&shape, axis).enumerate().for_each( | (lane_ind, mut lane)| {
    //                 assert_eq!(lane.len(), shape[axis]);
    //                 let mut index = lane_ind * shape[axis];
    //                 lane.iter_mut().enumerate().for_each(|(ii, v)| {
    //                     *v = index + ii;
    //                 });
    //             });

    //             let collected = arr.iter_lanes(&shape, axis).map(|lane|{
    //                 lane.iter().map(|v| *v).collect::<Vec<_>>()
    //             }).collect::<Vec<_>>().concat();

    //             assert_eq!(collected, (0..n_t).collect::<Vec<_>>());
    //         }
    //     }
    // }
}