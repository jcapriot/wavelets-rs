pub mod boundarys;
pub mod lwt;

pub enum TransformDirection{
    Forward,
    Inverse,
}

pub trait WaveletLength{
    const WIDTH: usize;
}


#[cfg(test)]
mod tests {
    #[macro_export]
    macro_rules! test_approx_equal {
        ($actual:expr, $desired:expr, $rtol:expr, $atol:expr) => {{
            let actual = $actual;
            let desired = $desired;
            let rtol = $rtol;
            let atol = $atol;

            let mut mismatch = None;
            actual.iter().zip(desired.iter()).for_each(|(a, d)| {
                let abs_diff = (*a - *d).abs();
                if abs_diff > rtol * d.abs() + atol {
                    mismatch = Some(mismatch.unwrap_or(0) + 1);
                }
            });

            if let Some(mismatch) = mismatch {
                panic!(
                    "{} mismatched elements: \n  actual: {:?}\n desired: {:?}",
                    mismatch, actual, desired
                );
            }
        }};
    }
}
