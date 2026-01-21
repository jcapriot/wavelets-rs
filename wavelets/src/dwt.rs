use itertools::Itertools;
use num_traits::Num;

use crate::boundarys::BoundaryExtension;

pub mod bior;
pub mod daubechies;

pub trait DiscreteTransform<U: Clone, const N: usize> {
    const G: [U; N];
    const H: [U; N];
    const GI: [U; N];
    const HI: [U; N];

    #[inline]
    fn forward<T: Num + Clone + From<U>, BC: BoundaryExtension>(
        x: &[T],
        s: &mut [T],
        d: &mut [T],
        bc: &BC,
    ) {
        dwt_forward(&Self::G, &Self::H, x, s, d, bc);
    }

    #[inline]
    fn inverse<T: Num + Clone + From<U>>(s: &[T], d: &[T], x: &mut [T]) {
        dwt_inverse(&Self::GI, &Self::HI, s, d, x);
    }

    #[inline]
    fn get_outlen(n: usize) -> usize {
        get_outlen::<N>(n)
    }
}

#[inline]
pub fn get_outlen<const N: usize>(n: usize) -> usize {
    let offset = (N - 2) / 2;
    let n_ds = (n + 1) / 2 + 2 * (N / 4);
    if (offset % 2 == 1) && (n % 2 == 1) {
        n_ds - 1
    } else {
        n_ds
    }
}
pub fn dwt_forward<T: Num + Clone + From<U>, U: Clone, const N: usize, BC: BoundaryExtension>(
    g: &[U; N],
    h: &[U; N],
    x: &[T],
    s: &mut [T],
    d: &mut [T],
    bc: &BC,
) {
    let (nx, ns, nd) = (x.len(), s.len(), d.len());

    assert_eq!(ns, nd, "'d.len()' must be equal to 's.len()'");

    assert_eq!(
        get_outlen::<N>(nx),
        ns,
        "'s.len()` and `d.len()' are inconsistent with 'x.len()'"
    );

    let offset = (N as isize - 2) / 2;
    let g: [T; N] = g
        .iter()
        .rev()
        .map(|v| v.clone().into())
        .collect_array()
        .expect("N=N");
    let h: [T; N] = h
        .iter()
        .rev()
        .map(|v| v.clone().into())
        .collect_array()
        .expect("N=N");

    let gh_iter = g.iter().zip(h.iter());

    // front boundary:
    let n_bcs = N as isize / 4;
    let mut sd_iter = (-n_bcs..(ns as isize - n_bcs)).zip(s.iter_mut().zip(d.iter_mut()));

    sd_iter
        .by_ref()
        .take(2 * n_bcs as usize)
        .for_each(|(i, (s, d))| {
            let ix = 2 * i - offset;
            (*s, *d) = (0..N as isize)
                .zip(gh_iter.clone())
                .map(|(j, (g, h))| {
                    let xo = bc.get_bc(x, ix + j);
                    (g.clone() * xo.clone(), h.clone() * xo)
                })
                .fold((T::zero(), T::zero()), |(s_s, d_s), (s, d)| {
                    (s + s_s, d + d_s)
                });
        });

    let first_x = offset as usize % 2;

    sd_iter
        .by_ref()
        .zip(x[first_x..].windows(N).step_by(2))
        .for_each(|((_i, (s, d)), x)| {
            (*s, *d) = gh_iter
                .clone()
                .zip(x)
                .map(|((g, h), xi)| (g.clone() * xi.clone(), h.clone() * xi.clone()))
                .fold((T::zero(), T::zero()), |(s_s, d_s), (s, d)| {
                    (s + s_s, d + d_s)
                });
        });

    sd_iter.for_each(|(i, (s, d))| {
        let ix = 2 * i - offset;
        (*s, *d) = (0..N as isize)
            .zip(gh_iter.clone())
            .map(|(j, (g, h))| {
                let xo = bc.get_bc(x, ix + j);
                (g.clone() * xo.clone(), h.clone() * xo)
            })
            .fold((T::zero(), T::zero()), |(s_s, d_s), (s, d)| {
                (s + s_s, d + d_s)
            });
    });
}

pub fn dwt_inverse<T: Num + Clone + From<U>, U: Clone, const N: usize>(
    gi: &[U; N],
    hi: &[U; N],
    s: &[T],
    d: &[T],
    x: &mut [T],
) {
    let (nx, ns, nd) = (x.len(), s.len(), d.len());

    assert_eq!(ns, nd, "'d.len()' must be equal to 's.len()'");

    assert_eq!(
        get_outlen::<N>(nx),
        ns,
        "'s.len()` and `d.len()' are inconsistent with 'x.len()'"
    );

    let offset = (N as isize - 2) / 2;
    let n_bcs = N as isize / 4;
    // TODO: Remove enumeratiion part of the sd_iter after more testing.
    let mut sd_iter = (-n_bcs..(ns as isize - n_bcs)).zip(s.windows(N / 2).zip(d.windows(N / 2)));
    let g: [T; N] = gi
        .iter()
        .rev()
        .map(|v| v.clone().into())
        .collect_array()
        .expect("N=N");
    let h: [T; N] = hi
        .iter()
        .rev()
        .map(|v| v.clone().into())
        .collect_array()
        .expect("N=N");

    let gh_iter = g.chunks_exact(2).zip(h.chunks_exact(2));
    let pair_shift = offset as usize % 2;

    if pair_shift > 0
        && let (Some(x), Some((_i_s, (s, d)))) = (x.first_mut(), sd_iter.next())
    {
        *x = gh_iter
            .clone()
            .zip(s.iter().zip(d.iter()))
            .map(|((g, h), (s, d))| g[0].clone() * s.clone() + h[0].clone() * d.clone())
            .fold(T::zero(), |acc, v| acc + v);
    }

    sd_iter
        .by_ref()
        .zip(x[pair_shift..].chunks_exact_mut(2))
        .for_each(|((_i_s, (s, d)), x)| {
            // s and d have lengths equal to N / 2
            // gh_iter is an N/2 length iterator that produces items of length 2
            // need to do for each x0 =
            (x[0], x[1]) = gh_iter
                .clone()
                .zip(s.iter().zip(d.iter()))
                .map(|((g, h), (s, d))| {
                    (
                        g[1].clone() * s.clone() + h[1].clone() * d.clone(),
                        g[0].clone() * s.clone() + h[0].clone() * d.clone(),
                    )
                })
                .fold((T::zero(), T::zero()), |(x0_acc, x1_acc), (x0, x1)| {
                    (x0_acc + x0, x1_acc + x1)
                });
        });

    if let Some(x) = x.last_mut() {
        sd_iter.for_each(|(_i_s, (s, d))| {
            *x = gh_iter
                .clone()
                .zip(s.iter().zip(d.iter()))
                .map(|((g, h), (s, d))| g[1].clone() * s.clone() + h[1].clone() * d.clone())
                .fold(T::zero(), |acc, v| acc + v);
        });
    }
}

#[cfg(test)]
mod test {
    use crate::boundarys::ZeroBoundary;

    use super::*;

    #[test]
    fn test_simple() {
        const N: usize = 4;
        let g = [1.0; N];
        let h = std::array::from_fn(|i| (-1 * (i as isize % 2)) as f64 * 1.0);

        let bc = ZeroBoundary {};

        let nx = 33;
        let x = (0..nx).map(|i| (i + 1) as f64).collect::<Vec<_>>();
        let nsd = dbg!(get_outlen::<N>(nx));

        // let ns = (nx + 1) / 2;
        // let nd = nx / 2;

        let mut s = vec![0.0; nsd];
        let mut d = vec![0.0; nsd];

        dwt_forward(&g, &h, &x, &mut s, &mut d, &bc);

        let mut x = vec![0.0; nx];
        dwt_inverse(&g, &h, &s, &d, &mut x);

        // dbg!(&x);
        // dbg!(&s);
        // dbg!(&d);
        // panic!();
    }
}
