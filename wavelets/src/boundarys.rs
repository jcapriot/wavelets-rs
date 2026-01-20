use num_traits::Num;

pub trait BoundaryExtension {
    fn get_bc<T: Num + Clone>(&self, data: &[T], i: isize) -> T;
    #[inline(always)]
    fn extend_front<T: Num + Clone>(&self, data: &[T], i: isize) -> T {
        self.get_bc(data, i)
    }
    #[inline(always)]
    fn extend_back<T: Num + Clone>(&self, data: &[T], i: usize) -> T {
        self.get_bc(data, i as isize)
    }
}

pub enum BoundaryCondition {
    ZeroBoundary,
    PeriodicBoundary,
}
impl BoundaryExtension for BoundaryCondition {
    #[inline(always)]
    fn get_bc<T: Num + Clone>(&self, data: &[T], i: isize) -> T {
        match self {
            Self::ZeroBoundary => {
                if i < 0 {
                    T::zero()
                } else {
                    data.get(i as usize).cloned().unwrap_or(T::zero())
                }
            }
            Self::PeriodicBoundary => {
                let i = i.rem_euclid(data.len() as isize) as usize;
                data.get(i).cloned().unwrap()
            }
        }
    }
}

pub struct ZeroBoundary;
impl BoundaryExtension for ZeroBoundary {
    #[inline(always)]
    fn get_bc<T: Num + Clone>(&self, data: &[T], i: isize) -> T {
        if i < 0 {
            T::zero()
        } else {
            data.get(i as usize).cloned().unwrap_or(T::zero())
        }
    }
}
pub struct PeriodicBoundary;
impl BoundaryExtension for PeriodicBoundary {
    #[inline(always)]
    fn get_bc<T: Num + Clone>(&self, data: &[T], i: isize) -> T {
        let i = i.rem_euclid(data.len() as isize) as usize;
        data.get(i).cloned().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_boundary() {
        let data = [1, 2, 3, 4, 5];
        let bc = ZeroBoundary {};

        assert_eq!(bc.extend_front(&data, -1), 0);
        assert_eq!(bc.extend_front(&data, -10), 0);
        assert_eq!(bc.extend_back(&data, 5), 0);
        assert_eq!(bc.extend_back(&data, 10), 0);
    }
}
