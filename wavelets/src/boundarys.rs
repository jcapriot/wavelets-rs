use num_traits::Num;

pub trait BoundaryExtension{
    fn get_bc<T: Num + Clone>(data: &[T], i: isize) -> T;
    #[inline(always)]
    fn extend_front<T: Num + Clone>(data: &[T], i: isize) -> T{
        Self::get_bc(data, i)
    }
    #[inline(always)]
    fn extend_back<T: Num + Clone>(data: &[T], i: usize) -> T{
        Self::get_bc(data, i as isize)
    }
}

pub struct ZeroBoundary;
impl BoundaryExtension for ZeroBoundary{
    #[inline(always)]
    fn get_bc<T: Num + Clone>(data: &[T], i: isize) -> T{
        if i < 0 {
            T::zero()
        }else{
            data.get(i as usize).cloned().unwrap_or(T::zero())
        }
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