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
    Zero,
    Periodic,
    Constant,
    Symmetric,
    Reflect,
}
impl BoundaryExtension for BoundaryCondition {
    #[inline(always)]
    fn get_bc<T: Num + Clone>(&self, data: &[T], i: isize) -> T {
        match self {
            Self::Zero => {
                if i < 0 {
                    T::zero()
                } else {
                    data.get(i as usize).cloned().unwrap_or(T::zero())
                }
            }
            Self::Periodic => {
                let i = i.rem_euclid(data.len() as isize) as usize;
                data.get(i).cloned().unwrap()
            }
            Self::Constant => {
                if i < 0 {
                    data.first().cloned().unwrap_or(T::zero())
                } else {
                    data.get(i as usize)
                        .or(data.last())
                        .cloned()
                        .unwrap_or(T::zero())
                }
            }
            Self::Symmetric => {
                let mut io = i;
                let n = data.len() as isize;
                while io >= n || io < 0 {
                    if io < 0 {
                        io = -(io + 1);
                    } else {
                        io = 2 * (n - 1) - (io - 1);
                    }
                }
                data.get(io as usize).cloned().unwrap_or(T::zero())
            }
            Self::Reflect => {
                let mut io = i;
                let n = data.len() as isize;
                while io >= n || io < 0 {
                    if io < 0 {
                        io = -io;
                    } else {
                        io = 2 * (n - 1) - io;
                    }
                }
                data.get(io as usize).cloned().unwrap_or(T::zero())
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
