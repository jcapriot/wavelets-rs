use wavelets::lwt;
use wavelets::boundarys::ZeroBoundary;

use ndarray::{Array2, Array3};
use chrono::prelude::*;

fn main(){

    let wvlt = lwt::daubechies::Daubechies6::new();

    let shape = [1025, 1020];
    let mut arr_in = Array2::<f64>::zeros(shape);
    arr_in[(3, 3)] = 1.0;

    let mut arr_out = Array2::<f64>::zeros(shape);

    let mut arr_out2 = Array2::<f64>::zeros(shape);

    let axes = [1];
    let bc = ZeroBoundary{};

    let time1 = Utc::now();
    lwt::forward_transform(&wvlt, arr_in.as_slice().unwrap(), arr_out.as_slice_mut().unwrap(), &shape, &axes, &bc);
    let time2 = Utc::now();
    let dt = time2 - time1;

    println!("time v1, 1: {}", dt.as_seconds_f64());

    let time1 = Utc::now();
    lwt::forward_transform(&wvlt, arr_in.as_slice().unwrap(), arr_out.as_slice_mut().unwrap(), &shape, &axes, &bc);
    let time2 = Utc::now();
    let dt = time2 - time1;
    println!("time v1, 2: {}", dt.as_seconds_f64());



    let time1 = Utc::now();
    lwt::alt::forward_transform(&wvlt, arr_in.as_slice().unwrap(), arr_out2.as_slice_mut().unwrap(), &shape, &axes, &bc);
    let time2 = Utc::now();
    let dt = time2 - time1;
    println!("time v2, 1: {}", dt.as_seconds_f64());



    let time1 = Utc::now();
    lwt::alt::forward_transform(&wvlt, arr_in.as_slice().unwrap(), arr_out2.as_slice_mut().unwrap(), &shape, &axes, &bc);
    let time2 = Utc::now();
    let dt = time2 - time1;
    println!("time v2, 2: {}", dt.as_seconds_f64());


    // let time1 = Utc::now();
    // lwt::parallel::forward_transform(&wvlt, arr_in.as_slice().unwrap(), arr_out.as_slice_mut().unwrap(), &shape, &axes, &bc);
    // let time2 = Utc::now();
    // let dt = time2 - time1;

    // println!("input:");
    // println!("{arr_in}");

    // println!("output:");
    // println!("{arr_out}");

    println!("output v1: {arr_out}");
    println!("output v2: {arr_out2}");
}