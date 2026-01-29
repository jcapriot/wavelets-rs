use itertools::Itertools;
use num_traits::{NumAssignOps, NumOps};
use std::ops::Neg;

use wavelets::boundarys::ZeroBoundary;
use wavelets::driver::{general_nd_forward, general_nd_inverse};
use wavelets::lwt::LiftingTransform;
use wavelets::wavelets::daubechies;

use chrono::prelude::*;

const BC: ZeroBoundary = ZeroBoundary {};
fn fwd_func<T>(s: &mut [T], d: &mut [T])
where
    T: NumOps + NumAssignOps + Clone + From<f64> + Neg<Output = T>,
{
    daubechies::Daubechies6::forward(s, d, &BC);
}

fn inv_func<T>(s: &mut [T], d: &mut [T])
where
    T: NumOps + NumAssignOps + Clone + From<f64> + Neg<Output = T>,
{
    daubechies::Daubechies6::inverse(s, d, &BC);
}

fn main() {
    const N_REPEAT: usize = 50;

    let shape = [1002, 522];
    let n = shape.iter().product();
    let arr_in = (0..n).map(|v| v as f64 + 1.0).collect_vec();

    let mut arr_out = vec![0.0; n];

    let mut arr_out2 = vec![0.0; n];

    let axes = [0, 1];

    general_nd_forward(fwd_func, &arr_in, &mut arr_out, &shape, &axes);
    general_nd_inverse(inv_func, &arr_out, &mut arr_out2, &shape, &axes);
    wavelets::tests::test_approx_equal(&arr_out2, &arr_in, 1E-12, 0.0);

    let time1 = Utc::now();
    for _ in 0..N_REPEAT {
        general_nd_forward(fwd_func, &arr_in, &mut arr_out, &shape, &axes);
    }
    let time2 = Utc::now();
    let dt = time2 - time1;

    let average_serial_time = dt.as_seconds_f64() / N_REPEAT as f64;

    println!("time serial: {average_serial_time}");

    let time1 = Utc::now();
    for _ in 0..N_REPEAT {
        general_nd_forward(fwd_func, &arr_in, &mut arr_out2, &shape, &axes);
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
