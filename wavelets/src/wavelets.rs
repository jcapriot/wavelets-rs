

macro_rules! gen_wavelet_struct {
    (
        $( ($name:ident, $width:expr) ),* $(,)?
    ) => {
        $(
            pub struct $name;
            impl $name{
                pub const WIDTH: usize = $width;

                pub fn new() -> Self{ Self{}}
            }
        )*
    };
}
pub mod daubechies{
    gen_wavelet_struct!(
        (Daubechies1, 2),
        (Daubechies2, 4),
        (Daubechies3, 6),
        (Daubechies4, 8),
        (Daubechies5, 10),
        (Daubechies6, 12)
    );
}