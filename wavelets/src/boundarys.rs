use num_traits::Num;


pub trait BoundaryExtension{
    fn extend_front<T: Num>(data: &[T], i: isize) -> T;
    fn extend_back<T: Num>(data: &[T], i: usize) -> T;
}

pub struct ZeroBoundary;
impl BoundaryExtension for ZeroBoundary{
    fn extend_front<T: Num>(_: &[T], _: isize) -> T{
        T::zero()
    }

    fn extend_back<T: Num>(_: &[T], _: usize) -> T{
        T::zero()
    }
}

pub enum BoundaryCondition{
    ZeroBoundary,
}

#[cfg(test)]
mod tests{
    use super::*;

    #[test]
    fn test_zero_boundary(){
        let data = [1,2,3,4,5];

        assert_eq!(ZeroBoundary::extend_front(&data, -1), 0);
        assert_eq!(ZeroBoundary::extend_front(&data, -10), 0);
        assert_eq!(ZeroBoundary::extend_back(&data, 5), 0);
        assert_eq!(ZeroBoundary::extend_back(&data, 10), 0);
    }
}