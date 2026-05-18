#[cfg(feature = "ndarray")]
use ndarray::{ArrayRef, Dimension};
use num_traits::Zero;
use std::collections::HashSet;

use aligned_vec::{AVec, avec};

use crate::Wavelets;
use crate::boundarys::BoundaryExtension;
use crate::dwt::{DiscreteTransform, get_outlen};
use crate::iter::LanesIterator;
use crate::utils::{
    clone_avecs_to_strided_chunk, clone_slice_to_strided, clone_strided_chunk_to_avecs,
    clone_strided_to_slice, split_strided, split_strided_chunk, stack_to_strided,
    stack_to_strided_chunk,
};

use crate::{ChunkWidth, Transformable};

use wavelets_macros::generate_wavelet_match_arms;
pub struct WaveletTransform<T, BC, const N: usize>
where
    T: Transformable + Zero + ChunkWidth<T, N>,
    BC: BoundaryExtension,
{
    dwt_forward: fn(&[T], &mut [T], &mut [T], &BC),
    dwt_inverse: fn(&[T], &[T], &mut [T]),
    dwt_adj_inverse: fn(&[T], &mut [T], &mut [T]),
    dwt_per_forward: fn(&[T], &mut [T], &mut [T]),
    dwt_per_inverse: fn(&[T], &[T], &mut [T]),
    dwt_per_adj_forward: fn(&[T], &[T], &mut [T]),
    dwt_per_adj_inverse: fn(&[T], &mut [T], &mut [T]),
    bc: BC,
    width: usize,
}

impl<T, BC, const N: usize> WaveletTransform<T, BC, N>
where
    T: Transformable + Zero + ChunkWidth<T, N>,
    BC: BoundaryExtension,
{
    pub fn new(wvlt: Wavelets, bc: BC) -> Self {
        use crate::dwt::bior::*;
        use crate::dwt::coiflet::*;
        use crate::dwt::daubechies::*;
        use crate::dwt::symlet::*;
        let dwt_forward: fn(&[T], &mut [T], &mut [T], &BC) = generate_wavelet_match_arms! {
            Wavelets,
            wvlt,
            {#wvlt::forward,}
        };
        let dwt_inverse: fn(&[T], &[T], &mut [T]) = generate_wavelet_match_arms! {
            Wavelets,
            wvlt,
            {#wvlt::inverse,}
        };
        let dwt_adj_inverse: fn(&[T], &mut [T], &mut [T]) = generate_wavelet_match_arms! {
            Wavelets,
            wvlt,
            {#wvlt::adjoint_inverse,}
        };
        let dwt_per_forward: fn(&[T], &mut [T], &mut [T]) = generate_wavelet_match_arms! {
            Wavelets,
            wvlt,
            {#wvlt::forward_per,}
        };
        let dwt_per_inverse: fn(&[T], &[T], &mut [T]) = generate_wavelet_match_arms! {
            Wavelets,
            wvlt,
            {#wvlt::inverse_per,}
        };
        let dwt_per_adj_forward: fn(&[T], &[T], &mut [T]) = generate_wavelet_match_arms! {
            Wavelets,
            wvlt,
            {#wvlt::adjoint_forward_per,}
        };
        let dwt_per_adj_inverse: fn(&[T], &mut [T], &mut [T]) = generate_wavelet_match_arms! {
            Wavelets,
            wvlt,
            {#wvlt::adjoint_inverse_per,}
        };
        let width = generate_wavelet_match_arms! {
            Wavelets,
            wvlt,
            {#wvlt::WIDTH,}
        };
        Self {
            dwt_forward,
            dwt_inverse,
            dwt_adj_inverse,
            dwt_per_forward,
            dwt_per_inverse,
            dwt_per_adj_forward,
            dwt_per_adj_inverse,
            bc,
            width,
        }
    }

    pub fn forward_1d(&self, input: &[T], s: &mut [T], d: &mut [T]) {
        (self.dwt_forward)(input, s, d, &self.bc);
    }

    pub fn inverse_1d(&self, s: &[T], d: &[T], output: &mut [T]) {
        (self.dwt_inverse)(&s, &d, output);
    }

    // pub fn adj_forward_1d(&self, s: &[T], d: &[T], output: &mut [T]) {
    //     let (mut s, mut d) = (s.to_owned(), d.to_owned());
    //     (self.lwt_adj_forward)(&mut s, &mut d, &self.bc);
    //     interleave(&s, &d, output);
    // }

    pub fn adj_inverse_1d(&self, input: &[T], s: &mut [T], d: &mut [T]) {
        (self.dwt_adj_inverse)(input, s, d);
    }

    pub fn per_forward_1d(&self, input: &[T], s: &mut [T], d: &mut [T]) {
        (self.dwt_per_forward)(input, s, d);
    }

    pub fn per_inverse_1d(&self, s: &[T], d: &[T], output: &mut [T]) {
        (self.dwt_per_inverse)(&s, &d, output);
    }

    pub fn per_adj_forward_1d(&self, s: &[T], d: &[T], output: &mut [T]) {
        (self.dwt_per_adj_forward)(s, d, output);
    }

    pub fn per_adj_inverse_1d(&self, input: &[T], s: &mut [T], d: &mut [T]) {
        (self.dwt_per_adj_inverse)(input, s, d);
    }

    pub fn forward_nd(&self, input: &[T], output: &mut [T], shape: &[usize], axes: &[usize]) {
        self.forward_multilevel_nd(input, output, shape, &axes, 1);
    }

    pub fn inverse_nd(&self, input: &mut [T], output: &mut [T], shape: &[usize], axes: &[usize]) {
        self.inverse_multilevel_nd(input, output, shape, &axes, 1);
    }

    // pub fn adj_forward_nd(&self, input: &[T], output: &mut [T], shape: &[usize], axes: &[usize]) {
    //     self.adj_forward_multilevel_nd(input, output, shape, &axes, 1);
    // }

    pub fn adj_inverse_nd(&self, input: &[T], output: &mut [T], shape: &[usize], axes: &[usize]) {
        self.adj_inverse_multilevel_nd(input, output, shape, &axes, 1);
    }

    pub fn per_forward_nd(&self, input: &[T], output: &mut [T], shape: &[usize], axes: &[usize]) {
        self.per_forward_multilevel_nd(input, output, shape, &axes, 1);
    }

    pub fn per_inverse_nd(
        &self,
        input: &mut [T],
        output: &mut [T],
        shape: &[usize],
        axes: &[usize],
    ) {
        self.per_inverse_multilevel_nd(input, output, shape, &axes, 1);
    }

    pub fn per_adj_forward_nd(
        &self,
        input: &mut [T],
        output: &mut [T],
        shape: &[usize],
        axes: &[usize],
    ) {
        self.per_adj_forward_multilevel_nd(input, output, shape, &axes, 1);
    }

    pub fn per_adj_inverse_nd(
        &self,
        input: &[T],
        output: &mut [T],
        shape: &[usize],
        axes: &[usize],
    ) {
        self.per_adj_inverse_multilevel_nd(input, output, shape, &axes, 1);
    }

    pub fn forward_multilevel_nd(
        &self,
        input: &[T],
        output: &mut [T],
        in_shape: &[usize],
        axes: &[usize],
        level: usize,
    ) {
        let axes = HashSet::from_iter(axes.iter().cloned());
        let out_shape = get_outshape(in_shape, &axes, level, self.width, false);
        general_nd_forward_multilevel(
            |x, s, d| (self.dwt_forward)(x, s, d, &self.bc),
            input,
            output,
            in_shape,
            &out_shape,
            &axes,
            level,
            self.width,
        );
    }

    pub fn inverse_multilevel_nd(
        &self,
        input: &mut [T],
        output: &mut [T],
        out_shape: &[usize],
        axes: &[usize],
        level: usize,
    ) {
        let axes = HashSet::from_iter(axes.iter().cloned());
        let in_shape = get_outshape(out_shape, &axes, level, self.width, false);
        general_nd_inverse_multilevel(
            |s, d, x| (self.dwt_inverse)(s, d, x),
            input,
            output,
            &in_shape,
            out_shape,
            &axes,
            level,
            self.width,
        );
    }

    // pub fn adj_forward_multilevel_nd(
    //     &self,
    //     input: &[T],
    //     output: &mut [T],
    //     shape: &[usize],
    //     axes: &[usize],
    //     level: usize,
    // ) {
    //     let axes = HashSet::from_iter(axes.iter().cloned());
    //     general_nd_inverse_multilevel(
    //         |s, d| (self.lwt_adj_forward)(s, d, &self.bc),
    //         input,
    //         output,
    //         shape,
    //         &axes,
    //         level,
    //     );
    // }

    pub fn adj_inverse_multilevel_nd(
        &self,
        input: &[T],
        output: &mut [T],
        in_shape: &[usize],
        axes: &[usize],
        level: usize,
    ) {
        let axes = HashSet::from_iter(axes.iter().cloned());
        let out_shape = get_outshape(in_shape, &axes, level, self.width, false);
        general_nd_forward_multilevel(
            |x, s, d| (self.dwt_adj_inverse)(x, s, d),
            input,
            output,
            in_shape,
            &out_shape,
            &axes,
            level,
            self.width,
        );
    }

    pub fn per_forward_multilevel_nd(
        &self,
        input: &[T],
        output: &mut [T],
        shape: &[usize],
        axes: &[usize],
        level: usize,
    ) {
        let axes = HashSet::from_iter(axes.iter().cloned());
        general_nd_forward_multilevel(
            |x, s, d| (self.dwt_per_forward)(x, s, d),
            input,
            output,
            shape,
            shape,
            &axes,
            level,
            self.width,
        );
    }

    pub fn per_inverse_multilevel_nd(
        &self,
        input: &mut [T],
        output: &mut [T],
        shape: &[usize],
        axes: &[usize],
        level: usize,
    ) {
        let axes = HashSet::from_iter(axes.iter().cloned());
        general_nd_inverse_multilevel(
            |s, d, x| (self.dwt_per_inverse)(s, d, x),
            input,
            output,
            shape,
            shape,
            &axes,
            level,
            self.width,
        );
    }

    pub fn per_adj_forward_multilevel_nd(
        &self,
        input: &mut [T],
        output: &mut [T],
        shape: &[usize],
        axes: &[usize],
        level: usize,
    ) {
        let axes = HashSet::from_iter(axes.iter().cloned());
        general_nd_inverse_multilevel(
            |s, d, x| (self.dwt_per_adj_forward)(s, d, x),
            input,
            output,
            shape,
            shape,
            &axes,
            level,
            self.width,
        );
    }

    pub fn per_adj_inverse_multilevel_nd(
        &self,
        input: &[T],
        output: &mut [T],
        shape: &[usize],
        axes: &[usize],
        level: usize,
    ) {
        let axes = HashSet::from_iter(axes.iter().cloned());
        general_nd_forward_multilevel(
            |x, s, d| (self.dwt_per_adj_inverse)(x, s, d),
            input,
            output,
            shape,
            shape,
            &axes,
            level,
            self.width,
        );
    }
}

pub fn get_outshape<'a, IT: IntoIterator<Item = &'a usize>>(
    in_shape: &[usize],
    axes: IT,
    level: usize,
    width: usize,
    per_mode: bool,
) -> Vec<usize> {
    let ndim = in_shape.len();
    let axes: HashSet<usize> = HashSet::from_iter(axes.into_iter().cloned());
    assert!(axes.iter().all(|i| *i < ndim));
    let mut lvl_shape = in_shape.to_owned();
    if per_mode {
        // In per mode, the output shape is the same as the input shape, since odd length transforms are
        // handled by copying the last element to the end of the approximation coefficients.
        return lvl_shape;
    }
    // initialize as shape of input array to copy un-transformed axes
    let mut out_shape = in_shape.to_owned();
    // transformed axes will be replaced by approximation and detail coefficients, so we initialize them to 0 and add the lengths of the coefficients in the loop below.
    for &ax in axes.iter() {
        out_shape[ax] = 0;
    }

    for lvl in 0..level {
        for &ax in axes.iter() {
            let n_ax = lvl_shape[ax];
            let nds = get_outlen(width, n_ax);
            if lvl + 1 < level {
                out_shape[ax] += nds;
            } else {
                out_shape[ax] += 2 * nds;
            }
            lvl_shape[ax] = nds;
        }
    }
    out_shape
}

fn general_nd_forward_multilevel<F, T, L, const N: usize>(
    func: F,
    input: &L,
    output: &mut L,
    in_shape: &[usize],
    out_shape: &[usize],
    axes: &HashSet<usize>,
    level: usize,
    width: usize,
) where
    F: Fn(&[T], &mut [T], &mut [T]),
    L: LanesIterator<Item = T> + ?Sized,
    T: Clone + Zero + ChunkWidth<T, N>,
{
    let ndim = in_shape.len();
    assert_eq!(
        in_shape.len(),
        out_shape.len(),
        "input and output shapes must have the same number of dimensions"
    );
    assert!(axes.iter().all(|i| *i < ndim));
    // note that axes is a HashSet, so they are gauranteed to be different axes.

    let mut first = true;

    let mut in_sub_shape = in_shape.to_owned();
    let mut out_sub_shape = out_shape.to_owned();

    // If the input shape and the output shape are the same, then we are in per mode
    let per_mode = in_shape
        .iter()
        .zip(out_shape.iter())
        .all(|(in_ax, out_ax)| in_ax == out_ax);

    for _level in 0..level {
        let mut sub_shape = in_sub_shape.clone();
        for &ax in axes {
            let n_ax = sub_shape[ax];

            let (n_s, n_d) = if per_mode {
                let n_d = n_ax / 2;
                let n_s = n_ax - n_d;
                (n_s, n_d)
            } else {
                let n_sd = get_outlen(width, n_ax);
                (n_sd, n_sd)
            };
            sub_shape[ax] = out_sub_shape[ax];

            // Note that everything does work for n_s == 1 (or 0 for that matter),
            // just that there really isn't anything useful to do.
            if n_s > 1 {
                match first {
                    true => {
                        let in_chunks = input.iter_lane_chunks::<N>(in_shape, ax);
                        let in_rem = in_chunks.remainder();
                        let out_chunks =
                            output.iter_lane_chunks_sub_mut::<N>(out_shape, &sub_shape, ax);
                        let out_rem = out_chunks.remainder();

                        if in_chunks.len() > 0 {
                            let mut x: [AVec<T>; N] =
                                core::array::from_fn(|_| avec![T::zero(); n_ax]);
                            let mut s: [AVec<T>; N] =
                                core::array::from_fn(|_| avec![T::zero(); n_s]);
                            let mut d: [AVec<T>; N] =
                                core::array::from_fn(|_| avec![T::zero(); n_d]);
                            in_chunks
                                .zip(out_chunks)
                                .for_each(|(in_chunk, mut out_chunk)| {
                                    // copy strided chunks into the local storage
                                    clone_strided_chunk_to_avecs(&in_chunk, &mut x);
                                    x.iter().zip(s.iter_mut().zip(d.iter_mut())).for_each(
                                        |(x, (s, d))| {
                                            func(x, s, d);
                                        },
                                    );
                                    // clone local storage to the output
                                    stack_to_strided_chunk(&s, &d, &mut out_chunk);
                                });
                        }
                        if in_rem.len() > 0 {
                            let mut x = avec![T::zero(); n_ax];
                            let mut s = avec![T::zero(); n_s];
                            let mut d = avec![T::zero(); n_d];
                            in_rem.zip(out_rem).for_each(|(in_slice, mut out_slice)| {
                                // copy strided slice into local dimension storage
                                clone_strided_to_slice(&in_slice, &mut x);
                                func(&x, &mut s, &mut d);
                                // copy local back to output strided slice
                                stack_to_strided(&s, &d, &mut out_slice);
                            });
                        }

                        first = false;
                    }
                    false => {
                        let chunks =
                            output.iter_lane_chunks_sub_mut::<N>(out_shape, &sub_shape, ax);
                        let rem = chunks.remainder();

                        if chunks.len() > 0 {
                            let mut x: [AVec<T>; N] =
                                core::array::from_fn(|_| avec![T::zero(); n_ax]);
                            let mut s: [AVec<T>; N] =
                                core::array::from_fn(|_| avec![T::zero(); n_s]);
                            let mut d: [AVec<T>; N] =
                                core::array::from_fn(|_| avec![T::zero(); n_d]);
                            chunks.for_each(|mut chunk| {
                                // copy (and deinterleave) strided chunks into the local storage
                                clone_strided_chunk_to_avecs(&chunk, &mut x);
                                x.iter().zip(s.iter_mut().zip(d.iter_mut())).for_each(
                                    |(x, (s, d))| {
                                        func(x, s, d);
                                    },
                                );
                                // clone local storage to the output
                                stack_to_strided_chunk(&s, &d, &mut chunk);
                            });
                        }
                        if rem.len() > 0 {
                            let mut x = avec![T::zero(); n_ax];
                            let mut s = avec![T::zero(); n_s];
                            let mut d = avec![T::zero(); n_d];
                            rem.for_each(|mut slc| {
                                // copy strided slice into local dimension storage
                                clone_strided_to_slice(&slc, &mut x);
                                func(&x, &mut s, &mut d);
                                // copy local back to output strided slice
                                stack_to_strided(&s, &d, &mut slc);
                            });
                        }
                    }
                }
            }
        }

        // shrink shape for each axis we used.
        for &ax in axes {
            let n_ax = in_sub_shape[ax];

            let (n_s, n_d) = if per_mode {
                let n_d = n_ax / 2;
                let n_s = n_ax - n_d;
                (n_s, n_d)
            } else {
                let n_sd = get_outlen(width, n_ax);
                (n_sd, n_sd)
            };
            if n_s > 1 {
                out_sub_shape[ax] = out_sub_shape[ax] - n_d;
                in_sub_shape[ax] = n_s;
            }
        }
    }
}

fn general_nd_inverse_multilevel<F, T, L, const N: usize>(
    func: F,
    inwork: &mut L,
    output: &mut L,
    in_shape: &[usize],
    out_shape: &[usize],
    axes: &HashSet<usize>,
    level: usize,
    width: usize,
) where
    F: Fn(&[T], &[T], &mut [T]),
    L: LanesIterator<Item = T> + ?Sized,
    T: Clone + Zero + ChunkWidth<T, N>,
{
    let ndim = in_shape.len();
    assert_eq!(
        in_shape.len(),
        out_shape.len(),
        "input and output shapes must have the same number of dimensions"
    );
    assert!(axes.iter().all(|i| *i < ndim));
    // note that axes is a HashSet, so they are gauranteed to be different axes.

    // If the input shape and the output shape are the same, then we are in per mode
    let per_mode = in_shape
        .iter()
        .zip(out_shape.iter())
        .all(|(in_ax, out_ax)| in_ax == out_ax);

    // make some lists to keep track of the shapes at each level, as we need to iterate in reverse order later.
    let mut ax_shapes = Vec::with_capacity(level);
    let mut out_shapes = Vec::with_capacity(level);
    let mut approx_shapes = Vec::with_capacity(level);
    let mut detail_shapes = Vec::with_capacity(level);

    out_shapes.push(in_shape.to_owned());
    ax_shapes.push(out_shape.to_owned());
    for _level in 0..level {
        // shrink shape for each axis that is used.
        let mut approx_shape = ax_shapes.last().unwrap().clone();
        let mut detail_shape = approx_shape.clone();
        let mut next_out_shape = out_shapes.last().unwrap().clone();
        for &ax in axes {
            let n_ax = approx_shape[ax];

            let (n_s, n_d) = if per_mode {
                let n_d = n_ax / 2;
                let n_s = n_ax - n_d;
                (n_s, n_d)
            } else {
                let n_sd = get_outlen(width, n_ax);
                (n_sd, n_sd)
            };
            if n_s > 1 {
                approx_shape[ax] = n_s;
                detail_shape[ax] = n_d;
                next_out_shape[ax] -= n_d;
            }
        }
        if _level + 1 < level {
            ax_shapes.push(approx_shape.clone());
            out_shapes.push(next_out_shape);
        }
        approx_shapes.push(approx_shape);
        detail_shapes.push(detail_shape);
    }

    if per_mode {
        // In per mode we can copy the input to the output right away and not modify the input array.
        let min_axis = output.min_stride_axis(out_shape);
        let in_chunks = inwork.iter_lane_chunks_sub::<N>(in_shape, out_shape, min_axis);
        let in_rem = in_chunks.remainder();
        let out_chunks = output.iter_lane_chunks_mut::<N>(out_shape, min_axis);
        let out_rem = out_chunks.remainder();

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

    for level in (0..level).rev() {
        let mut sub_shape = out_shapes[level].clone();
        for &ax in axes {
            let n_ax = ax_shapes[level][ax];
            let n_s = approx_shapes[level][ax];
            let n_d = detail_shapes[level][ax];

            // Note that everything does work for n_s == 1 (or 0 for that matter),
            // just that there really isn't anything useful to do.
            if n_s > 1 {
                let chunks = if per_mode {
                    output.iter_lane_chunks_sub_mut(out_shape, &sub_shape, ax)
                } else {
                    inwork.iter_lane_chunks_sub_mut(in_shape, &sub_shape, ax)
                };
                let rem = chunks.remainder();

                if chunks.len() > 0 {
                    let mut x: [AVec<T>; N] = core::array::from_fn(|_| avec![T::zero(); n_ax]);
                    let mut s: [AVec<T>; N] = core::array::from_fn(|_| avec![T::zero(); n_s]);
                    let mut d: [AVec<T>; N] = core::array::from_fn(|_| avec![T::zero(); n_d]);
                    chunks.for_each(|mut chunk| {
                        // split the chunk into the approximation and detail coefficients.
                        split_strided_chunk(&chunk, &mut s, &mut d);
                        x.iter_mut()
                            .zip(s.iter().zip(d.iter()))
                            .for_each(|(x, (s, d))| {
                                func(s, d, x);
                            });
                        // clone local storage to the output
                        clone_avecs_to_strided_chunk(&x, &mut chunk);
                    });
                }
                if rem.len() > 0 {
                    let mut x = avec![T::zero(); n_ax];
                    let mut s = avec![T::zero(); n_s];
                    let mut d = avec![T::zero(); n_d];
                    rem.for_each(|mut slc| {
                        // split the slice into the approximation and detail coefficients.
                        split_strided(&slc, &mut s, &mut d);
                        func(&s, &d, &mut x);
                        // copy local back to output strided slice
                        clone_slice_to_strided(&x, &mut slc);
                    });
                }
                // the next passes sub shape along this dimension will have the size of n_ax
                sub_shape[ax] = n_ax;
            }
        }
    }

    if !per_mode {
        // copy input into output
        let min_axis = output.min_stride_axis(out_shape);
        let in_chunks = inwork.iter_lane_chunks_sub::<N>(in_shape, out_shape, min_axis);
        let in_rem = in_chunks.remainder();
        let out_chunks = output.iter_lane_chunks_mut::<N>(out_shape, min_axis);
        let out_rem = out_chunks.remainder();

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
