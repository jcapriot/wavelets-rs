use crate::Transformable;

pub trait BoundaryExtension {
    fn get_bc<T: Transformable>(&self, data: &[T], i: isize) -> Option<T>;
    fn get_parts<T: Transformable>(
        &self,
        n: usize,
        i: isize,
    ) -> Vec<(Option<T::ScalarType>, usize)>
    where
        T::ScalarType: From<f64>;
    #[inline(always)]
    fn extend_front<T: Transformable>(&self, data: &[T], i: isize) -> Option<T> {
        self.get_bc(data, i)
    }
    #[inline(always)]
    fn extend_back<T: Transformable>(&self, data: &[T], i: usize) -> Option<T> {
        self.get_bc(data, i as isize)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum BoundaryCondition {
    Zero,
    Periodic,
    Constant,
    Symmetric,
    Reflect,
    Antisymmetric,
    //Smooth,
}

unsafe impl Sync for BoundaryCondition {}
unsafe impl Send for BoundaryCondition {}

impl BoundaryExtension for BoundaryCondition {
    #[inline(always)]
    fn get_bc<T: Transformable>(&self, data: &[T], i: isize) -> Option<T> {
        match self {
            Self::Zero => data.get(i as usize).cloned(),
            Self::Periodic => {
                let i = i.rem_euclid(data.len() as isize) as usize;
                data.get(i).cloned()
            }
            Self::Constant => {
                let i = i.clamp(0, data.len() as isize - 1) as usize;
                data.get(i).cloned()
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
                data.get(io as usize).cloned()
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
                data.get(io as usize).cloned()
            }
            Self::Antisymmetric => {
                let mut io = i;
                let mut neg = false;
                let n = data.len() as isize;
                while io >= n || io < 0 {
                    if io < 0 {
                        io = -(io + 1);
                    } else {
                        io = 2 * (n - 1) - (io - 1);
                    }
                    neg = !neg;
                }
                let v = data.get(io as usize).cloned();
                match neg {
                    true => v.and_then(|v| Some(-v)),
                    false => v,
                }
            } // Self::Smooth => {
              //     if i < 0 {
              //         let vs = data.first_chunk::<2>().cloned();
              //         if let Some(vs) = vs {
              //             Some(vs[0].clone() + (vs[1].clone() - vs[0].clone()))
              //         } else {
              //             None
              //         }
              //     } else if i >= data.len() as isize {
              //         let vs = data.last_chunk::<2>().cloned();
              //         if let Some(vs) = vs {
              //             Some(vs[1].clone() + (vs[1].clone() - vs[0].clone()))
              //         } else {
              //             None
              //         }
              //     } else {
              //         return data.get(i as usize).cloned();
              //     }
              // }
        }
    }

    fn get_parts<T: Transformable>(&self, n: usize, i: isize) -> Vec<(Option<T::ScalarType>, usize)>
    where
        T::ScalarType: From<f64>,
    {
        match self {
            Self::Zero => {
                if i >= 0 && i < n as isize {
                    vec![(None, i as usize)]
                } else {
                    vec![]
                }
            }
            Self::Periodic => {
                let i = i.rem_euclid(n as isize) as usize;
                vec![(None, i)]
            }
            Self::Constant => {
                let i = i.clamp(0, n as isize - 1) as usize;
                vec![(None, i)]
            }
            Self::Symmetric => {
                let mut io = i;
                let n = n as isize;
                while io >= n || io < 0 {
                    if io < 0 {
                        io = -(io + 1);
                    } else {
                        io = 2 * (n - 1) - (io - 1);
                    }
                }
                vec![(None, io as usize)]
            }
            Self::Reflect => {
                let mut io = i;
                let n = n as isize;
                while io >= n || io < 0 {
                    if io < 0 {
                        io = -io;
                    } else {
                        io = 2 * (n - 1) - io;
                    }
                }
                vec![(None, io as usize)]
            }
            Self::Antisymmetric => {
                let mut io = i;
                let mut neg = false;
                let n = n as isize;
                while io >= n || io < 0 {
                    if io < 0 {
                        io = -(io + 1);
                    } else {
                        io = 2 * (n - 1) - (io - 1);
                    }
                    neg = !neg;
                }
                match neg {
                    true => vec![(Some(T::ScalarType::from(-1.0)), io as usize)],
                    false => vec![(None, io as usize)],
                }
            }
        }
    }
}

pub trait LiftedAdjointBoundary {
    fn adjoint_op<F: Fn(&mut T, T), T: Transformable, const N: usize>(
        &self,
        op: F,
        left: &mut [T],
        right: &mut [T],
        rev_offset: isize,
        rev_c: &[T::ScalarType; N],
        i_left: isize,
    );
}
impl LiftedAdjointBoundary for BoundaryCondition {
    #[inline(always)]
    fn adjoint_op<F: Fn(&mut T, T), T: Transformable, const N: usize>(
        &self,
        op: F,
        left: &mut [T],
        right: &mut [T],
        rev_offset: isize,
        rev_c: &[T::ScalarType; N],
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
            Self::Antisymmetric => {
                let mut io = i_left;
                let n = left.len() as isize;
                let mut neg = false;
                while io >= n || io < 0 {
                    if io < 0 {
                        io = -(io + 1);
                    } else {
                        io = 2 * (n - 1) - (io - 1);
                    }
                    neg = !neg;
                }
                let io = io as usize;

                if let Some(yi) = left.get_mut(io) {
                    let right = (i_right..i_right + N as isize)
                        .zip(rev_c.iter())
                        .filter_map(|(j, c)| {
                            right
                                .get(j as usize)
                                .and_then(|v| Some(v.clone() * c.clone()))
                        })
                        .reduce(|acc, v| acc + v);

                    if let Some(right) = right {
                        match neg {
                            true => op(yi, -right),
                            false => op(yi, right),
                        };
                    }
                }
                return;
            }
        };

        if let Some(yi) = left.get_mut(io) {
            let right = (i_right..i_right + N as isize)
                .zip(rev_c.iter())
                .filter_map(|(j, c)| {
                    right
                        .get(j as usize)
                        .and_then(|v| Some(v.clone() * c.clone()))
                })
                .reduce(|acc, v| acc + v);

            if let Some(right) = right {
                op(yi, right);
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ZeroBoundary;

unsafe impl Sync for ZeroBoundary {}
unsafe impl Send for ZeroBoundary {}

impl BoundaryExtension for ZeroBoundary {
    #[inline(always)]
    fn get_bc<T: Transformable>(&self, data: &[T], i: isize) -> Option<T> {
        data.get(i as usize).cloned()
    }

    fn get_parts<T: Transformable>(
        &self,
        n: usize,
        i: isize,
    ) -> Vec<(Option<T::ScalarType>, usize)> {
        if i >= 0 && i < n as isize {
            vec![(None, i as usize)]
        } else {
            vec![]
        }
    }
}
impl LiftedAdjointBoundary for ZeroBoundary {
    #[inline(always)]
    fn adjoint_op<F: Fn(&mut T, T), T: Transformable, const N: usize>(
        &self,
        _op: F,
        _left: &mut [T],
        _right: &mut [T],
        _rev_offset: isize,
        _rev_c: &[T::ScalarType; N],
        _i_left: isize,
    ) {
        // Do nothing
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PeriodicBoundary;
impl BoundaryExtension for PeriodicBoundary {
    #[inline(always)]
    fn get_bc<T: Transformable>(&self, data: &[T], i: isize) -> Option<T> {
        let i = i.rem_euclid(data.len() as isize) as usize;
        data.get(i).cloned()
    }

    fn get_parts<T: Transformable>(
        &self,
        n: usize,
        i: isize,
    ) -> Vec<(Option<T::ScalarType>, usize)> {
        let i = i.rem_euclid(n as isize) as usize;
        vec![(None, i)]
    }
}

unsafe impl Sync for PeriodicBoundary {}
unsafe impl Send for PeriodicBoundary {}

impl LiftedAdjointBoundary for PeriodicBoundary {
    #[inline(always)]
    fn adjoint_op<F: Fn(&mut T, T), T: Transformable, const N: usize>(
        &self,
        op: F,
        left: &mut [T],
        right: &mut [T],
        rev_offset: isize,
        rev_c: &[T::ScalarType; N],
        i_left: isize,
    ) {
        let i_right = i_left + rev_offset;
        let io = i_left.rem_euclid(left.len() as isize) as usize;

        if let Some(yi) = left.get_mut(io) {
            if let Some(right) = (i_right..i_right + N as isize)
                .zip(rev_c.iter())
                .filter_map(|(j, c)| {
                    right
                        .get(j as usize)
                        .and_then(|v| Some(v.clone() * c.clone()))
                })
                .reduce(|acc, v| acc + v)
            {
                op(yi, right);
            }
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

        assert_eq!(bc.extend_front(&data, -1), None);
        assert_eq!(bc.extend_front(&data, -10), None);
        assert_eq!(bc.extend_back(&data, 5), None);
        assert_eq!(bc.extend_back(&data, 10), None);
    }
}
