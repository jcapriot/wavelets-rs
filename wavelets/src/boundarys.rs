//! Boundary extension strategies for wavelet transforms.
//!
//! When a filter kernel reaches the edge of a signal it needs values outside the
//! array bounds.  Each type in this module implements a different convention for
//! providing those virtual out-of-bounds values.

use crate::Transformable;

/// Strategy for extending a signal beyond its boundaries.
///
/// Implementors define how out-of-bounds index `i` maps to an in-bounds value (or
/// a linear combination of in-bounds values).  Two complementary methods are required:
///
/// - `get_bc` — returns the actual extended value, used during the forward transform.
/// - `get_parts` — returns a sparse weighted sum of in-bounds indices, used to build
///   the adjoint (transpose) of the forward transform.
pub trait BoundaryExtension: Sync {
    /// Return the signal value at virtual index `i` under this boundary condition.
    ///
    /// Returns `None` when the boundary condition maps `i` to zero (e.g. zero-padding
    /// outside bounds), so callers can skip the contribution entirely.
    fn get_bc<T: Transformable>(&self, data: &[T], i: isize) -> Option<T>;

    /// Decompose virtual index `i` into a weighted sum of in-bounds indices.
    ///
    /// Each entry is `(scale, j)` meaning the contribution is `scale * data[j]` (or
    /// just `data[j]` when `scale` is `None`, implying a weight of 1).  Used to form
    /// the adjoint operator without materialising extra memory.
    fn get_parts<T: Transformable>(&self, n: usize, i: isize) -> Vec<(Option<T::Scalar>, usize)>;
}

/// Runtime-selectable boundary extension condition.
///
/// Choose the variant that best matches the physical interpretation of your signal:
///
/// | Variant | Extension rule |
/// |---------|---------------|
/// | `Zero` | Out-of-bounds values are zero (zero-padding). |
/// | `Periodic` | Signal repeats periodically. |
/// | `Constant` | Nearest edge value is repeated. |
/// | `Symmetric` | Signal is reflected about the edge sample (even symmetry). |
/// | `Reflect` | Signal is reflected about the point just outside the edge (point reflection). |
/// | `Antisymmetric` | Like `Symmetric` but with sign flip on each reflection. |
/// | `Smooth` | Linear extrapolation using the two nearest edge samples. |
/// | `Antireflect` | Odd-symmetric extension preserving derivatives at boundaries. |
#[derive(Clone, Copy, Debug)]
pub enum BoundaryCondition {
    /// Out-of-bounds indices contribute zero (zero-padding).
    Zero,
    /// Signal wraps around periodically.
    Periodic,
    /// Nearest edge sample is repeated for all out-of-bounds indices.
    Constant,
    /// Even (mirror) reflection about the edge sample.
    Symmetric,
    /// Odd reflection about the point just outside the edge (no edge repetition).
    Reflect,
    /// Even reflection with sign flip on each reflection.
    Antisymmetric,
    /// Linear extrapolation from the two nearest edge samples.
    Smooth,
    /// Odd-symmetric extension that preserves the first derivative at boundaries.
    Antireflect,
}

unsafe impl Sync for BoundaryCondition {}
unsafe impl Send for BoundaryCondition {}

impl BoundaryExtension for BoundaryCondition {
    #[inline(always)]
    fn get_bc<T: Transformable>(&self, data: &[T], i: isize) -> Option<T> {
        if data.is_empty() {
            return None;
        }
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
                let n = data.len() as isize;
                let mut io = i;
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
                if data.len() == 1 {
                    return data.first().cloned();
                }
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
                    true => v.map(|v| -v),
                    false => v,
                }
            }
            Self::Smooth => {
                if data.len() == 1 {
                    return data.first().cloned();
                }
                // unwrap first_chunk/last_chunk because data.len() >= 2 at this point.
                if i < 0 {
                    let vs = data.first_chunk::<2>().cloned().unwrap();

                    let scale = T::scalar_type_from_isize(i);
                    Some(vs[0].clone() + (vs[1].clone() - vs[0].clone()) * scale)
                } else if i >= data.len() as isize {
                    let vs = data.last_chunk::<2>().cloned().unwrap();
                    let scale = T::scalar_type_from_isize(i - (data.len() as isize - 1));
                    Some(vs[1].clone() + (vs[1].clone() - vs[0].clone()) * scale)
                } else {
                    return data.get(i as usize).cloned();
                }
            }
            Self::Antireflect => {
                if data.len() == 1 {
                    return data.first().cloned();
                }
                let mut io = i;
                let n = data.len() as isize;
                let mut neg = false;
                let mut left_c = 0;
                let mut right_c = 0;
                while io >= n || io < 0 {
                    if io < 0 {
                        io = -io;
                        left_c += if neg { -2 } else { 2 };
                    } else {
                        io = 2 * (n - 1) - io;
                        right_c += if neg { -2 } else { 2 };
                    }
                    neg = !neg;
                }
                let mut v = None;
                if left_c != 0 {
                    v = data
                        .first()
                        .cloned()
                        .map(|u| u * T::scalar_type_from_isize(left_c))
                };
                if right_c != 0 {
                    let u = data
                        .last()
                        .cloned()
                        .map(|u| u * T::scalar_type_from_isize(right_c));

                    v = match (v, u) {
                        (Some(v), Some(u)) => Some(v + u),
                        (Some(v), None) => Some(v),
                        (None, Some(u)) => Some(u),
                        (None, None) => None,
                    };
                }

                let d = data
                    .get(io as usize)
                    .cloned()
                    .map(|v| if neg { -v } else { v });

                match (v, d) {
                    (Some(v), Some(d)) => Some(d + v),
                    (Some(v), None) => Some(v),
                    (None, Some(d)) => Some(d),
                    (None, None) => None,
                }
            }
        }
    }

    fn get_parts<T: Transformable>(&self, n: usize, i: isize) -> Vec<(Option<T::Scalar>, usize)> {
        if n == 0 {
            return vec![];
        }
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
                if n == 1 {
                    return vec![(None, 0)];
                }
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
                    true => vec![(Some(T::scalar_type_from_f64(-1.0)), io as usize)],
                    false => vec![(None, io as usize)],
                }
            }
            Self::Smooth => {
                if n == 1 {
                    return vec![(None, 0)];
                }
                if i < 0 {
                    // Some(vs[0].clone() + (vs[1].clone() - vs[0].clone()) * scale)
                    // Some(vs[0].clone() * (1 - scale) + vs[1].clone() * scale
                    let scale_0 = T::scalar_type_from_isize(1 - i);
                    let scale_1 = T::scalar_type_from_isize(i);
                    vec![(Some(scale_0), 0), (Some(scale_1), 1)]
                } else if i >= n as isize {
                    let d_i = i - (n as isize - 1);
                    let scale_m2 = T::scalar_type_from_isize(-d_i);
                    let scale_m1 = T::scalar_type_from_isize(d_i + 1);
                    // Some(vs[1].clone() + (vs[1].clone() - vs[0].clone()) * scale)
                    // Some(vs[1].clone() * (scale + 1) + vs[0].clone() * (-scale);
                    vec![(Some(scale_m2), n - 2), (Some(scale_m1), n - 1)]
                } else {
                    vec![(None, i as usize)]
                }
            }
            Self::Antireflect => {
                if n == 1 {
                    return vec![(None, 0)];
                }
                let mut io = i;
                let n = n as isize;
                let mut neg = false;
                let mut left_c = 0;
                let mut right_c = 0;
                while io >= n || io < 0 {
                    if io < 0 {
                        io = -io;
                        left_c += if neg { -2 } else { 2 };
                    } else {
                        io = 2 * (n - 1) - io;
                        right_c += if neg { -2 } else { 2 };
                    }
                    neg = !neg;
                }
                let mut items = Vec::with_capacity(3);
                if left_c != 0 {
                    items.push((Some(T::scalar_type_from_isize(left_c)), 0));
                }
                if right_c != 0 {
                    items.push((Some(T::scalar_type_from_isize(right_c)), n as usize - 1));
                }
                items.push((
                    if neg {
                        Some(T::scalar_type_from_isize(-1))
                    } else {
                        None
                    },
                    io as usize,
                ));
                items
            }
        }
    }
}

/// Zero-padding boundary: out-of-bounds indices return `None` (treated as zero).
///
/// This is a zero-cost alternative to `BoundaryCondition::Zero` for contexts where
/// the boundary type is fixed at compile time.
#[derive(Clone, Copy, Debug)]
pub struct ZeroBoundary;

unsafe impl Sync for ZeroBoundary {}
unsafe impl Send for ZeroBoundary {}

impl BoundaryExtension for ZeroBoundary {
    #[inline(always)]
    fn get_bc<T: Transformable>(&self, data: &[T], i: isize) -> Option<T> {
        data.get(i as usize).cloned()
    }

    fn get_parts<T: Transformable>(&self, n: usize, i: isize) -> Vec<(Option<T::Scalar>, usize)> {
        if i >= 0 && i < n as isize {
            vec![(None, i as usize)]
        } else {
            vec![]
        }
    }
}

/// Periodic boundary: indices wrap around modulo the signal length.
///
/// This is a zero-cost alternative to `BoundaryCondition::Periodic`.
#[derive(Clone, Copy, Debug)]
pub struct PeriodicBoundary;
impl BoundaryExtension for PeriodicBoundary {
    #[inline(always)]
    fn get_bc<T: Transformable>(&self, data: &[T], i: isize) -> Option<T> {
        let i = i.rem_euclid(data.len() as isize) as usize;
        data.get(i).cloned()
    }

    fn get_parts<T: Transformable>(&self, n: usize, i: isize) -> Vec<(Option<T::Scalar>, usize)> {
        let i = i.rem_euclid(n as isize) as usize;
        vec![(None, i)]
    }
}

unsafe impl Sync for PeriodicBoundary {}
unsafe impl Send for PeriodicBoundary {}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_value<T: Transformable + num_traits::Zero>(
        bc: impl BoundaryExtension,
        io: isize,
        data: &[T],
    ) -> T {
        let vs = bc.get_parts::<T>(data.len(), io);
        let left: T = vs
            .iter()
            .map(|(scale, i)| match scale {
                Some(v) => data[*i].clone() * v.clone(),
                None => data[*i].clone(),
            })
            .fold(T::zero(), |acc, v| acc + v);
        left
    }

    macro_rules! test_boundary {
        ($get_bc_test_name:ident, $get_parts_test_name:ident, $bc:ident, [$($val:expr),*], [ $( ($input:expr, $expected:expr) ),* $(,)? ]) => {

        #[test]
        fn $get_bc_test_name(){
            let data = [$($val),*];
            let bc = BoundaryCondition::$bc;

            $(
                assert_eq!(bc.get_bc(&data, $input), Some($expected));
            )*
        }
        #[test]
        fn $get_parts_test_name(){
            let data  = [$($val),*];
            let bc = BoundaryCondition::$bc;

            $(
                assert_eq!(get_value(bc, $input, &data), $expected);
            )*
        }
        };
    }

    test_boundary!(
        test_antireflect_boundary,
        test_antireflect_boundary_parts,
        Antireflect,
        [4, 4, 9, 16, 25, 16, 9],
        [
            (-40, -47),
            (-10, -1),
            (-3, -8),
            (-2, -1),
            (-1, 4),
            (0, 4),
            (1, 4),
            (2, 9),
            (7, 2),
            (8, -7),
            (9, 2),
            (15, 26),
            (45, 32)
        ]
    );

    test_boundary!(
        test_antisymmetric_boundary,
        test_antisymmetric_boundary_parts,
        Antisymmetric,
        [1, 2, 3, 4, 5],
        [
            (-7, 4),
            (-6, 5),
            (-2, -2),
            (-1, -1),
            (0, 1),
            (4, 5),
            (5, -5),
            (6, -4),
            (10, 1),
            (11, 2)
        ]
    );

    test_boundary!(
        test_symmetric_boundary,
        test_symmetric_boundary_parts,
        Symmetric,
        [1, 2, 3, 4, 5],
        [
            (-7, 4),
            (-6, 5),
            (-2, 2),
            (-1, 1),
            (0, 1),
            (4, 5),
            (5, 5),
            (6, 4),
            (10, 1),
            (11, 2)
        ]
    );

    test_boundary!(
        test_reflect_boundary,
        test_reflect_boundary_parts,
        Reflect,
        [1, 2, 3, 4, 5],
        [
            (-6, 3),
            (-2, 3),
            (-1, 2),
            (0, 1),
            (4, 5),
            (5, 4),
            (6, 3),
            (10, 3)
        ]
    );

    test_boundary!(
        test_periodic_boundary,
        test_periodic_boundary_parts,
        Periodic,
        [1, 2, 3, 4, 5],
        [
            (-1, 5),
            (-2, 4),
            (-6, 5),
            (0, 1),
            (4, 5),
            (5, 1),
            (6, 2),
            (10, 1)
        ]
    );

    test_boundary!(
        test_constant_boundary,
        test_constant_boundary_parts,
        Constant,
        [1, 2, 3, 4, 5],
        [
            (-1, 1),
            (-2, 1),
            (-6, 1),
            (0, 1),
            (4, 5),
            (5, 5),
            (6, 5),
            (10, 5)
        ]
    );

    test_boundary!(
        test_smooth_boundary,
        test_smooth_boundary_parts,
        Smooth,
        [1, 2, 3, 4, 5],
        [(-2, -1), (-1, 0), (0, 1), (4, 5), (5, 6), (6, 7)]
    );

    #[test]
    fn test_zero_boundary() {
        let data = [1, 2, 3, 4, 5];
        let bc = ZeroBoundary {};

        assert_eq!(bc.get_bc(&data, -1), None);
        assert_eq!(bc.get_bc(&data, -10), None);
        assert_eq!(bc.get_bc(&data, 0), Some(1));
        assert_eq!(bc.get_bc(&data, 4), Some(5));
        assert_eq!(bc.get_bc(&data, 5), None);
        assert_eq!(bc.get_bc(&data, 10), None);
    }

    #[test]
    fn test_zero_boundary_parts() {
        let data = [1, 2, 3, 4, 5];
        let bc = BoundaryCondition::Zero;
        assert_eq!(get_value(bc, -10, &data), 0);
        assert_eq!(get_value(bc, -1, &data), 0);
        assert_eq!(get_value(bc, 0, &data), 1);
        assert_eq!(get_value(bc, 4, &data), 5);
        assert_eq!(get_value(bc, 5, &data), 0);
        assert_eq!(get_value(bc, 10, &data), 0);
    }
}
