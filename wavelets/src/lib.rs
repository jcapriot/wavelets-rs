pub mod wavelets;
pub mod boundarys;
pub mod lwt;
pub mod dwt;
//pub mod vector;
pub mod iter;

pub enum TransformDirection{
    Forward,
    Inverse,
}


#[cfg(test)]
mod tests {
    pub fn test_approx_equal<T>(actual: &[T], desired: &[T], rtol: T, atol: T)
    where 
        T: num_traits::Float + std::fmt::Debug
    {
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

    }
}