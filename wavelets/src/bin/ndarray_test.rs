use ndarray::{ArrayView, Axis, ArrayViewMut};
use rayon::prelude::*;
use wavelets::iter::ndarray::par::{
    ParLaneChunksExactIterator,
    ParLaneChunksExactIteratorMut,
};
use wavelets::iter::ndarray::{
    LaneChunksExactIterator,
    LaneChunksExactIteratorMut
};


fn main(){
    const N: usize = 2;

    let shape = [2, 3, 5];
    let n_total = shape.iter().product();
    let data = (0..n_total).collect::<Vec<_>>();

    let axis = Axis(0);

    let arr_view = ArrayView::from_shape(shape, &data).unwrap();
    let (chunks, rem) = arr_view.lane_chunks_par_iter::<N>(axis);
    chunks.enumerate().for_each(|(i, chunk)|{
        for slice in chunk{
            println!("Chunk {i}: {slice}");
        }
    });
    rem.enumerate().for_each(|(i, chunk)|{
        for slice in chunk{
            println!("Remainder {i}: {slice}");
        }
    });

    for slice in arr_view.lanes(axis){
        println!("{slice}");
    }

    let mut out = vec![0; n_total];

    let mut out_view = ArrayViewMut::from_shape(shape, &mut out).unwrap();

    let (in_chunks, in_rem) = arr_view.lane_chunks_par_iter::<N>(axis);
    let (out_chunks, out_rem) = out_view.lane_chunks_par_iter_mut::<N>(axis);

    in_chunks.zip(out_chunks).for_each(|(in_chunk, mut out_chunk)|{
        in_chunk.iter()
            .zip(out_chunk.iter_mut())
            .for_each(|(in_row, out_row)|{
                out_row.zip_mut_with(in_row, |out, inp|{
                    *out = *inp;
                }
            );
        })
    });

    in_rem.zip(out_rem).for_each(|(in_chunk, mut out_chunk)|{
        in_chunk.iter()
            .zip(out_chunk.iter_mut())
            .for_each(|(in_row, out_row)|{
                out_row.zip_mut_with(in_row, |out, inp|{
                    *out = *inp;
                }
            );
        })
    });

    assert_eq!(&data, &out);
    for (i, (inp, outp)) in data.iter().zip(out.iter()).enumerate(){
        println!("data[{i}]:{inp}, out[{i}]:{outp}");
    }


}