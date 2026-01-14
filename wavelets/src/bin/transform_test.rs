use num_traits::{Num, NumAssignOps};
use wavelets::boundarys::ZeroBoundary;
use wavelets::lwt;
use wavelets::lwt::{LiftingTransform, daubechies};

use chrono::prelude::*;

const BC: ZeroBoundary = ZeroBoundary {};
fn fwd_func<T>(s: &mut [T], d: &mut [T])
where
    T: Num + NumAssignOps + Clone + From<f64>,
{
    daubechies::Daubechies6::forward(s, d, &BC);
}

fn main() {
    const N_REPEAT: usize = 50;

    let shape = [1025, 1020];
    type NDARRAY = ndarray::Array2<f64>;
    let mut arr_in = NDARRAY::zeros(shape);
    arr_in[(3, 3)] = 1.0;

    let mut arr_out = NDARRAY::zeros(shape);

    let mut arr_out2 = NDARRAY::zeros(shape);

    let axes = [1];

    let time1 = Utc::now();
    for _ in 0..N_REPEAT {
        lwt::general_nd_forward(
            fwd_func,
            arr_in.as_slice().unwrap(),
            arr_out.as_slice_mut().unwrap(),
            &shape,
            &axes,
        );
    }
    let time2 = Utc::now();
    let dt = time2 - time1;

    let average_serial_time = dt.as_seconds_f64() / N_REPEAT as f64;

    println!("time serial: {average_serial_time}");

    let time1 = Utc::now();
    for _ in 0..N_REPEAT {
        lwt::parallel::general_nd_forward(
            fwd_func,
            arr_in.as_slice().unwrap(),
            arr_out2.as_slice_mut().unwrap(),
            &shape,
            &axes,
        );
    }
    let time2 = Utc::now();
    let dt = time2 - time1;

    let average_parallel_time = dt.as_seconds_f64() / N_REPEAT as f64;

    println!("time parallel: {average_parallel_time}");

    println!(
        "Parallel speedup: {}",
        average_serial_time / average_parallel_time
    );

    // let time1 = Utc::now();
    // lwt::general_nd_forward(fwd_func, arr_in.as_slice().unwrap(), arr_out.as_slice_mut().unwrap(), &shape, &axes);
    // let time2 = Utc::now();
    // let dt = time2 - time1;
    // println!("time v1, 2: {}", dt.as_seconds_f64());

    // let time1 = Utc::now();
    // lwt::general_nd_forward(fwd_func, arr_in.as_slice().unwrap(), arr_out2.as_slice_mut().unwrap(), &shape, &axes);
    // let time2 = Utc::now();
    // let dt = time2 - time1;
    // println!("time v1, 3: {}", dt.as_seconds_f64());

    // let time1 = Utc::now();
    // lwt::general_nd_forward(fwd_func, arr_in.as_slice().unwrap(), arr_out2.as_slice_mut().unwrap(), &shape, &axes);
    // let time2 = Utc::now();
    // let dt = time2 - time1;
    // println!("time v1, 4: {}", dt.as_seconds_f64());

    // let time1 = Utc::now();
    // lwt::general_nd_forward(fwd_func, arr_in.as_slice().unwrap(), arr_out2.as_slice_mut().unwrap(), &shape, &axes);
    // let time2 = Utc::now();
    // let dt = time2 - time1;

    // println!("time v1, 5: {}", dt.as_seconds_f64());

    // let time1 = Utc::now();
    // lwt::parallel::forward_transform(&wvlt, arr_in.as_slice().unwrap(), arr_out.as_slice_mut().unwrap(), &shape, &axes, &bc);
    // let time2 = Utc::now();
    // let dt = time2 - time1;

    // println!("input:");
    // println!("{arr_in}");

    // println!("output:");
    // println!("{arr_out}");
    assert_eq!(arr_out, arr_out2);
}
