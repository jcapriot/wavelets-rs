use wavelets::boundarys::BoundaryCondition;

fn main() {
    type Wvlt = wavelets::coiflet::Coiflet1;

    let n = 50;
    let ns = (n + 1) / 2;
    let nd = (n) / 2;

    let mut e = vec![0.0; n];

    let mut s = vec![0.0; ns];
    let mut d = vec![0.0; nd];
    {
        use wavelets::lwt::LiftingTransform;
        use wavelets::utils::deinterleave;

        let bc = BoundaryCondition::Periodic;

        e[20] = 1.0;
        e[21] = 0.0;
        deinterleave(&e, &mut s, &mut d);
        Wvlt::forward(&mut s, &mut d, &bc);

        println!("LWT:");
        println!("s_10 = {:?}", s);
        println!("d_10 = {:?}", d);

        e[20] = 0.0;
        e[21] = 1.0;
        deinterleave(&e, &mut s, &mut d);
        Wvlt::forward(&mut s, &mut d, &bc);

        println!("s_11 = {:?}", s);
        println!("d_11 = {:?}", d);
    }

    {
        use wavelets::dwt::DiscreteTransform;

        e[20] = 1.0;
        e[21] = 0.0;
        Wvlt::forward_per(&e, &mut s, &mut d);

        println!("DWT:");
        println!("s_10 = {:?}", s);
        println!("d_10 = {:?}", d);

        e[20] = 0.0;
        e[21] = 1.0;
        Wvlt::forward_per(&e, &mut s, &mut d);

        println!("s_11 = {:?}", s);
        println!("d_11 = {:?}", d);
    }
}
