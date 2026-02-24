use itertools::Itertools;

use crate::Transformable;
use crate::boundarys::{BoundaryExtension, PeriodicBoundary, ZeroBoundary};

pub mod bior;
pub mod coiflet;
pub mod daubechies;
//pub mod driver;
pub mod symlet;

pub trait DiscreteTransform<const N: usize> {
    const G: [f64; N];
    const H: [f64; N];
    const GI: [f64; N];
    const HI: [f64; N];

    #[inline]
    fn forward<T: Transformable, BC: BoundaryExtension>(
        x: &[T],
        s: &mut [T],
        d: &mut [T],
        bc: &BC,
    ) {
        dwt_forward(&Self::G, &Self::H, x, s, d, bc);
    }

    #[inline]
    fn inverse<T: Transformable>(s: &[T], d: &[T], x: &mut [T]) {
        dwt_inverse(&Self::GI, &Self::HI, s, d, x);
    }

    // #[inline]
    // fn adjoint_forward<T: Transformable>(s: &[T], d: &[T], x: &mut [T], &BC: BoundaryExtension)
    // where
    //     T::ScalarType: From<U>,
    // {
    //     let ga: [_; N] = Self::G.clone().into_iter().rev().collect_array().unwrap();
    //     let ha: [_; N] = Self::H.clone().into_iter().rev().collect_array().unwrap();
    //     dwt_inverse(&ga, &ha, s, d, x);
    // }

    #[inline]
    fn adjoint_inverse<T: Transformable>(x: &[T], s: &mut [T], d: &mut [T]) {
        let ga: [_; N] = Self::GI.clone().into_iter().rev().collect_array().unwrap();
        let ha: [_; N] = Self::HI.clone().into_iter().rev().collect_array().unwrap();
        dwt_forward(&ga, &ha, x, s, d, &ZeroBoundary {});
    }

    #[inline]
    fn forward_per<T: Transformable>(x: &[T], s: &mut [T], d: &mut [T]) {
        dwt_per_forward(&Self::G, &Self::H, x, s, d);
    }

    #[inline]
    fn adjoint_forward_per<T: Transformable>(s: &[T], d: &[T], x: &mut [T]) {
        let ga: [_; N] = Self::G.clone().into_iter().rev().collect_array().unwrap();
        let ha: [_; N] = Self::H.clone().into_iter().rev().collect_array().unwrap();
        dwt_per_inverse(&ga, &ha, s, d, x);
    }

    #[inline]
    fn inverse_per<T: Transformable>(s: &[T], d: &[T], x: &mut [T]) {
        dwt_per_inverse(&Self::GI, &Self::HI, s, d, x);
    }

    #[inline]
    fn adjoint_inverse_per<T: Transformable>(x: &[T], s: &mut [T], d: &mut [T]) {
        let gia: [_; N] = Self::GI.clone().into_iter().rev().collect_array().unwrap();
        let hia: [_; N] = Self::HI.clone().into_iter().rev().collect_array().unwrap();
        dwt_per_forward(&gia, &hia, x, s, d);
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

pub fn dwt_forward<T: Transformable, const N: usize, BC: BoundaryExtension>(
    g: &[f64; N],
    h: &[f64; N],
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
    let g: [T::Scalar; N] = g
        .iter()
        .rev()
        .map(|v| T::scalar_type_from_f64(*v))
        .collect_array()
        .expect("N=N");
    let h: [T::Scalar; N] = h
        .iter()
        .rev()
        .map(|v| T::scalar_type_from_f64(*v))
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
            if let Some((v1, v2)) = (0..N as isize)
                .zip(gh_iter.clone())
                .filter_map(|(j, (g, h))| {
                    let xo = bc.get_bc(x, ix + j)?;
                    Some((xo.clone() * g.clone(), xo * h.clone()))
                })
                .reduce(|(s_s, d_s), (s, d)| (s + s_s, d + d_s))
            {
                (*s, *d) = (v1, v2);
            }
        });

    let first_x = offset as usize % 2;

    sd_iter
        .by_ref()
        .zip(x[first_x..].windows(N).step_by(2))
        .for_each(|((_i, (s, d)), x)| {
            let v = gh_iter
                .clone()
                .zip(x)
                .map(|((g, h), xi)| (xi.clone() * g.clone(), xi.clone() * h.clone()))
                .reduce(|(s_s, d_s), (s, d)| (s + s_s, d + d_s));
            // only undefined if N==0, but windows(N) would've already paniced if that was the case.
            (*s, *d) = unsafe { v.unwrap_unchecked() };
        });

    sd_iter.for_each(|(i, (s, d))| {
        let ix = 2 * i - offset;
        if let Some((v1, v2)) = (0..N as isize)
            .zip(gh_iter.clone())
            .filter_map(|(j, (g, h))| {
                let xo = bc.get_bc(x, ix + j)?;
                Some((xo.clone() * g.clone(), xo * h.clone()))
            })
            .reduce(|(s_s, d_s), (s, d)| (s + s_s, d + d_s))
        {
            (*s, *d) = (v1, v2);
        }
    });
}

pub fn dwt_inverse<T: Transformable, const N: usize>(
    gi: &[f64; N],
    hi: &[f64; N],
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
    let g: [T::Scalar; N] = gi
        .iter()
        .rev()
        .map(|v| T::scalar_type_from_f64(*v))
        .collect_array()
        .expect("N=N");
    let h: [T::Scalar; N] = hi
        .iter()
        .rev()
        .map(|v| T::scalar_type_from_f64(*v))
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
            .map(|((g, h), (s, d))| s.clone() * g[0].clone() + d.clone() * h[0].clone())
            .reduce(|acc, v| acc + v)
            .unwrap();
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
                        s.clone() * g[1].clone() + d.clone() * h[1].clone(),
                        s.clone() * g[0].clone() + d.clone() * h[0].clone(),
                    )
                })
                .reduce(|(x0_acc, x1_acc), (x0, x1)| (x0_acc + x0, x1_acc + x1))
                .unwrap();
        });

    if let Some(x) = x.last_mut() {
        sd_iter.for_each(|(_i_s, (s, d))| {
            *x = gh_iter
                .clone()
                .zip(s.iter().zip(d.iter()))
                .map(|((g, h), (s, d))| s.clone() * g[1].clone() + d.clone() * h[1].clone())
                .reduce(|acc, v| acc + v)
                .unwrap();
        });
    }
}

pub fn dwt_per_forward<T: Transformable, const N: usize>(
    g: &[f64; N],
    h: &[f64; N],
    x: &[T],
    s: &mut [T],
    d: &mut [T],
) {
    let (nx, ns, nd) = (x.len(), s.len(), d.len());

    assert!(
        (ns == nd) || (ns == nd + 1),
        "'d.len()' must be equal to or 1 less than 's.len()'"
    );

    assert_eq!(
        nx,
        ns + nd,
        "'s.len()` + `d.len()' must be equal to `x.len()'"
    );
    let (x, s) = if ns > nd {
        // for the odd length transform, the last x value just becomes the last approximation coefficient.
        // Then shorten x and s by one element.
        if let Some(sl) = s.last_mut()
            && let Some(xl) = x.last()
        {
            *sl = xl.clone();
        }
        (&x[0..nx - 1], &mut s[0..nd])
    } else {
        (x, s)
    };

    let offset = (N as isize - 2) / 2;
    let g: [T::Scalar; N] = g
        .iter()
        .rev()
        .map(|v| T::scalar_type_from_f64(*v))
        .collect_array()
        .expect("N=N");
    let h: [T::Scalar; N] = h
        .iter()
        .rev()
        .map(|v| T::scalar_type_from_f64(*v))
        .collect_array()
        .expect("N=N");
    let gh_iter = g.iter().zip(h.iter());

    // front boundary:
    let n_bcs = N / 4;
    let mut sd_iter = (0..nd as isize).zip(s.iter_mut().zip(d.iter_mut()));

    let per_bc = PeriodicBoundary {};

    sd_iter.by_ref().take(n_bcs).for_each(|(i, (s, d))| {
        let ix = 2 * i - offset;
        if let Some((v1, v2)) = (0..N as isize)
            .zip(gh_iter.clone())
            .filter_map(|(j, (g, h))| {
                let xo = per_bc.get_bc(x, ix + j)?;
                Some((xo.clone() * g.clone(), xo * h.clone()))
            })
            .reduce(|(s_s, d_s), (s, d)| (s + s_s, d + d_s))
        {
            (*s, *d) = (v1, v2)
        }
    });

    let first_x = offset as usize % 2;

    sd_iter
        .by_ref()
        .zip(x[first_x..].windows(N).step_by(2))
        .for_each(|((_i, (s, d)), x)| {
            let v = gh_iter
                .clone()
                .zip(x)
                .map(|((g, h), xi)| (xi.clone() * g.clone(), xi.clone() * h.clone()))
                .reduce(|(s_s, d_s), (s, d)| (s + s_s, d + d_s));
            // only undefined if N == 0, but would've paniced at windows() before this.
            (*s, *d) = unsafe { v.unwrap_unchecked() };
        });

    sd_iter.for_each(|(i, (s, d))| {
        let ix = 2 * i - offset;
        if let Some((v1, v2)) = (0..N as isize)
            .zip(gh_iter.clone())
            .filter_map(|(j, (g, h))| {
                let xo = per_bc.get_bc(x, ix + j)?;
                Some((xo.clone() * g.clone(), xo * h.clone()))
            })
            .reduce(|(s_s, d_s), (s, d)| (s + s_s, d + d_s))
        {
            (*s, *d) = (v1, v2)
        }
    });
}

pub fn dwt_per_inverse<T: Transformable, const N: usize>(
    gi: &[f64; N],
    hi: &[f64; N],
    s: &[T],
    d: &[T],
    x: &mut [T],
) {
    let (nx, ns, nd) = (x.len(), s.len(), d.len());

    assert!(
        (ns == nd) || (ns == nd + 1),
        "'d.len()' must be equal to or 1 less than 's.len()'"
    );

    assert_eq!(
        nx,
        ns + nd,
        "'s.len()` + `d.len()' must be equal to `x.len()'"
    );
    let (x, s) = if ns > nd {
        // for the odd length inverse transform, the last smooth coefficient just becomes the last x coefficient.
        // Then shorten x and s by one element.
        if let Some(sl) = s.last()
            && let Some(xl) = x.last_mut()
        {
            *xl = sl.clone();
        }
        (&mut x[0..nx - 1], &s[0..nd])
    } else {
        (x, s)
    };

    let offset = (N as isize - 2) / 2;
    let n_bcs = N as isize / 4;
    // TODO: Remove enumeratiion part of the sd_iter after more testing.
    let g: [T::Scalar; N] = gi
        .iter()
        .rev()
        .map(|v| T::scalar_type_from_f64(*v))
        .collect_array()
        .expect("N=N");
    let h: [T::Scalar; N] = hi
        .iter()
        .rev()
        .map(|v| T::scalar_type_from_f64(*v))
        .collect_array()
        .expect("N=N");

    let gh_iter = g.chunks_exact(2).zip(h.chunks_exact(2));
    let pair_shift = offset as usize % 2;

    // s and d have lengths equal to N / 2
    // gh_iter is an N/2 length iterator that produces items of length 2
    // need to do for each x0 =

    let per_bc = PeriodicBoundary {};

    if pair_shift > 0
        && let Some(x) = x.first_mut()
    {
        let i_sd = -n_bcs;
        if let Some(v) = (0..N as isize / 2)
            .zip(gh_iter.clone())
            .filter_map(|(j, (g, h))| {
                let sg = per_bc
                    .get_bc(s, i_sd + j)
                    .and_then(|s| Some(s * g[0].clone()));
                let dh = per_bc
                    .get_bc(d, i_sd + j)
                    .and_then(|d| Some(d * h[0].clone()));
                if let Some(sg) = sg {
                    if let Some(dh) = dh {
                        Some(sg + dh)
                    } else {
                        Some(sg)
                    }
                } else {
                    dh
                }
            })
            .reduce(|acc, v| acc + v)
        {
            *x = v
        }
    }
    let mut x_iter =
        (pair_shift as isize - n_bcs..nd as isize - n_bcs).zip(x[pair_shift..].chunks_exact_mut(2));

    // front boundarys
    x_iter
        .by_ref()
        .take(n_bcs as usize - pair_shift)
        .for_each(|(i_sd, x)| {
            if let Some((v1, v2)) = (0..N as isize / 2)
                .zip(gh_iter.clone())
                .filter_map(|(j, (g, h))| {
                    let sg = per_bc
                        .get_bc(s, i_sd + j)
                        .and_then(|s| Some((s.clone() * g[1].clone(), s * g[0].clone())));
                    let dh = per_bc
                        .get_bc(d, i_sd + j)
                        .and_then(|d| Some((d.clone() * h[1].clone(), d * h[0].clone())));
                    if let Some(sg) = sg {
                        if let Some(dh) = dh {
                            Some((sg.0 + dh.0, sg.1 + dh.1))
                        } else {
                            Some(sg)
                        }
                    } else {
                        dh
                    }
                })
                .reduce(|(x0_acc, x1_acc), (x0, x1)| (x0_acc + x0, x1_acc + x1))
            {
                (x[0], x[1]) = (v1, v2)
            }
        });

    // main loop
    x_iter
        .by_ref()
        .zip(s.windows(N / 2).zip(d.windows(N / 2)))
        .for_each(|((_i_sd, x), (s, d))| {
            (x[0], x[1]) = gh_iter
                .clone()
                .zip(s.iter().zip(d.iter()))
                .map(|((g, h), (s, d))| {
                    (
                        s.clone() * g[1].clone() + d.clone() * h[1].clone(),
                        s.clone() * g[0].clone() + d.clone() * h[0].clone(),
                    )
                })
                .reduce(|(x0_acc, x1_acc), (x0, x1)| (x0_acc + x0, x1_acc + x1))
                .unwrap();
        });

    // back bc loop until the x chunks run out
    x_iter
        .by_ref()
        .take(n_bcs as usize - pair_shift)
        .for_each(|(i_sd, x)| {
            if let Some((v1, v2)) = (0..N as isize / 2)
                .zip(gh_iter.clone())
                .filter_map(|(j, (g, h))| {
                    let sg = per_bc
                        .get_bc(s, i_sd + j)
                        .and_then(|s| Some((s.clone() * g[1].clone(), s * g[0].clone())));
                    let dh = per_bc
                        .get_bc(d, i_sd + j)
                        .and_then(|d| Some((d.clone() * h[1].clone(), d * h[0].clone())));
                    if let Some(sg) = sg {
                        if let Some(dh) = dh {
                            Some((sg.0 + dh.0, sg.1 + dh.1))
                        } else {
                            Some(sg)
                        }
                    } else {
                        dh
                    }
                })
                .reduce(|(x0_acc, x1_acc), (x0, x1)| (x0_acc + x0, x1_acc + x1))
            {
                (x[0], x[1]) = (v1, v2)
            }
        });

    if pair_shift > 0
        && let Some(x) = x.last_mut()
    {
        let i_sd = nd as isize - n_bcs;

        if let Some(v) = (0..N as isize / 2)
            .zip(gh_iter.clone())
            .filter_map(|(j, (g, h))| {
                let sg = per_bc
                    .get_bc(s, i_sd + j)
                    .and_then(|s| Some(s * g[1].clone()));
                let dh = per_bc
                    .get_bc(d, i_sd + j)
                    .and_then(|d| Some(d * h[1].clone()));
                if let Some(sg) = sg {
                    if let Some(dh) = dh {
                        Some(sg + dh)
                    } else {
                        Some(sg)
                    }
                } else {
                    dh
                }
            })
            .reduce(|acc, v| acc + v)
        {
            *x = v
        }
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
    }
}
