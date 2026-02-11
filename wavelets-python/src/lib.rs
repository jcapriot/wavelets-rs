//generate_wavelet_enum!(Wavelets);
use pyo3::prelude::*;

#[pymodule]
mod wavelets_ext {
    use super::*;
    use wavelets::Wavelets as WaveletsEnum;

    #[pyclass]
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum Wavelets {
        Daubechies1,
        Daubechies2,
        Daubechies3,
        Daubechies4,
    }

    impl Wavelets {
        fn to_enum(&self) -> WaveletsEnum {
            match self {
                Wavelets::Daubechies1 => WaveletsEnum::Daubechies1,
                Wavelets::Daubechies2 => WaveletsEnum::Daubechies2,
                Wavelets::Daubechies3 => WaveletsEnum::Daubechies3,
                Wavelets::Daubechies4 => WaveletsEnum::Daubechies4,
            }
        }
    }

    use numpy::{PyReadonlyArrayDyn, PyReadwriteArrayDyn};

    #[pyfunction]
    fn forward_transform<'py>(
        py: Python<'py>,
        wavelet: Wavelets,
        x: PyReadonlyArrayDyn<f64>,
        mut y: PyReadwriteArrayDyn<f64>,
    ) -> PyResult<()> {
        let x = x.as_array();
        if x.strides().iter().any(|s| *s < 0) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Negative strides are not supported",
            ));
        }
        let ptr = x.as_ptr();
        let shape = x.shape();
        let stride = x.strides().iter().map(|s| *s as usize).collect::<Vec<_>>();
        let max_offset = shape
            .iter()
            .zip(stride.iter())
            .map(|(n, step)| n * step)
            .max()
            .unwrap_or(0);

        let first_element_ptr = x.as_ptr();
        let y = y.as_array_mut();
        let wvlt = wavelet.to_enum();

        Ok(())
    }
}

// #[pyfunction]
// fn guess_the_number() {
//     println!("Guess the number!");

//     let secret_number = rand::rng().random_range(1..101);

//     loop {
//         println!("Please input your guess.");

//         let mut guess = String::new();

//         io::stdin()
//             .read_line(&mut guess)
//             .expect("Failed to read line");

//         let guess: u32 = match guess.trim().parse() {
//             Ok(num) => num,
//             Err(_) => continue,
//         };

//         println!("You guessed: {}", guess);

//         match guess.cmp(&secret_number) {
//             Ordering::Less => println!("Too small!"),
//             Ordering::Greater => println!("Too big!"),
//             Ordering::Equal => {
//                 println!("You win!");
//                 break;
//             }
//         }
//     }
// }

// /// A Python module implemented in Rust. The name of this function must match
// /// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
// /// import the module.
// #[pymodule]
// fn guessing_game(m: &Bound<'_, PyModule>) -> PyResult<()> {
//     m.add_function(wrap_pyfunction!(guess_the_number, m)?)?;

//     Ok(())
// }
