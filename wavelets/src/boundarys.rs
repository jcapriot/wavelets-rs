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
            Self::Zero => data.get(i as usize).cloned().unwrap_or(T::zero()),
            Self::Periodic => {
                let i = i.rem_euclid(data.len() as isize) as usize;
                data.get(i).cloned().unwrap_or(T::zero())
            }
            Self::Constant => {
                let i = i.clamp(0, data.len() as isize - 1) as usize;
                data.get(i).cloned().unwrap_or(T::zero())
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

pub trait UpdateOperation {
    fn update_assign<T: Num + Clone>(left: T, right: T) -> T;
}
pub struct ForwardUpdate {}
pub struct InverseUpdate {}

impl UpdateOperation for ForwardUpdate {
    #[inline(always)]
    fn update_assign<T: Num + Clone>(left: T, right: T) -> T {
        left + right
    }
}
impl UpdateOperation for InverseUpdate {
    #[inline(always)]
    fn update_assign<T: Num>(left: T, right: T) -> T {
        left - right
    }
}

pub trait LiftedAdjointBoundary {
    fn adjoint_op<T: Num + Clone, const N: usize, OP: UpdateOperation>(
        &self,
        op: &OP,
        left: &mut [T],
        right: &mut [T],
        rev_offset: isize,
        rev_c: &[T; N],
        i_left: isize,
    );
}
impl LiftedAdjointBoundary for BoundaryCondition {
    #[inline(always)]
    fn adjoint_op<T: Num + Clone, const N: usize, OP: UpdateOperation>(
        &self,
        _op: &OP,
        left: &mut [T],
        right: &mut [T],
        rev_offset: isize,
        rev_c: &[T; N],
        i_left: isize,
    ) {
        let i_right = i_left + rev_offset;
        let io = match self {
            Self::Zero => return, // do nothing
            Self::Periodic => i_left.rem_euclid(left.len() as isize) as usize,
            Self::Constant => i_left.clamp(0, left.len() as isize - 1) as usize,
            Self::Symmetric => {
                let mut io = i_left;
                let n = left.len() as isize;
                while io >= n || io < 0 {
                    if io < 0 {
                        io = -(io + 1);
                    } else {
                        io = 2 * (n - 1) - (io - 1);
                    }
                }
                io as usize
            }
            Self::Reflect => {
                let mut io = i_left;
                let n = left.len() as isize;
                while io >= n || io < 0 {
                    if io < 0 {
                        io = -io;
                    } else {
                        io = 2 * (n - 1) - io;
                    }
                }
                io as usize
            }
        };

        if let Some(yi) = left.get_mut(io) {
            *yi = OP::update_assign(
                yi.clone(),
                (i_right..i_right + N as isize)
                    .zip(rev_c.iter())
                    .filter_map(|(j, c)| {
                        right
                            .get(j as usize)
                            .and_then(|v| Some(c.clone() * v.clone()))
                    })
                    .fold(T::zero(), |acc, v| acc + v),
            );
        }
    }
}

pub struct ZeroBoundary;
impl BoundaryExtension for ZeroBoundary {
    #[inline(always)]
    fn get_bc<T: Num + Clone>(&self, data: &[T], i: isize) -> T {
        data.get(i as usize).cloned().unwrap_or(T::zero())
    }
}
impl LiftedAdjointBoundary for ZeroBoundary {
    #[inline(always)]
    fn adjoint_op<T: Num + Clone, const N: usize, OP: UpdateOperation>(
        &self,
        _op: &OP,
        _left: &mut [T],
        _right: &mut [T],
        _rev_offset: isize,
        _rev_c: &[T; N],
        _i_left: isize,
    ) {
        // Do nothing
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

impl LiftedAdjointBoundary for PeriodicBoundary {
    #[inline(always)]
    fn adjoint_op<T: Num + Clone, const N: usize, OP: UpdateOperation>(
        &self,
        _op: &OP,
        left: &mut [T],
        right: &mut [T],
        rev_offset: isize,
        rev_c: &[T; N],
        i_left: isize,
    ) {
        let i_right = i_left + rev_offset;
        let io = i_left.rem_euclid(left.len() as isize) as usize;

        if let Some(yi) = left.get_mut(io) {
            *yi = OP::update_assign(
                yi.clone(),
                (i_right..i_right + N as isize)
                    .zip(rev_c.iter())
                    .filter_map(|(j, c)| {
                        right
                            .get(j as usize)
                            .and_then(|v| Some(c.clone() * v.clone()))
                    })
                    .fold(T::zero(), |acc, v| acc + v),
            )
        }
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
