use num_traits::Num;

use crate::boundarys::BoundaryExtension;

pub mod daubechies;

pub trait DiscreteTransform {
    type FilterType;
    const G: Self::FilterType;
    const H: Self::FilterType;

    fn forward<T: From<f64>, BC: BoundaryExtension>(x: &[T], s: &mut [T], d: &mut [T], bc: &BC);
    fn inverse<T: From<f64>, BC: BoundaryExtension>(s: &[T], d: &[T], x: &mut [T], bc: &BC);
}

pub fn dwt_forward<T: Num + Clone + std::fmt::Debug, const N: usize, BC: BoundaryExtension>(
    g: &[T; N],
    h: &[T; N],
    x: &[T],
    s: &mut [T],
    d: &mut [T],
    _bc: &BC,
) {
    let (nx, ns, nd) = (x.len(), s.len(), d.len());
    assert!(
        ns == nd || ns == nd + 1,
        "'d.len()' must be equal to or 1 less than 's.len()'"
    );
    assert_eq!(
        nx,
        ns + nd,
        "'s.len() + d.len()' must be equal to 'x.len()'"
    );

    let offset = (N as isize - 2) / 2;
    let mut sd_iter = (0..nd as isize).zip(s.iter_mut().zip(d.iter_mut()));

    let gh_iter = g.iter().zip(h.iter()).rev();

    // front boundary:
    let n_front = (offset as usize + 1) / 2;
    sd_iter.by_ref().take(n_front).for_each(|(i, (s, d))| {
        let ix = 2 * i - offset;
        (*s, *d) = (0..N as isize)
            .zip(gh_iter.clone())
            .map(|(j, (g, h))| {
                let xo = BC::get_bc(x, ix + j);
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
                let xo = BC::get_bc(x, ix + j);
                (g.clone() * xo.clone(), h.clone() * xo)
            })
            .fold((T::zero(), T::zero()), |(s_s, d_s), (s, d)| {
                (s + s_s, d + d_s)
            });
    });

    // also deal with x as odd length
    if nd < ns {
        let ix = 2 * nd as isize - offset;
        s[nd] = (0..N as isize)
            .zip(gh_iter.clone())
            .map(|(j, (g, _))| {
                let xo = BC::get_bc(x, ix + j);
                g.clone() * xo.clone()
            })
            .reduce(|acc, v| acc + v)
            .unwrap_or(T::zero());
    }
}

#[cfg(test)]
mod test {
    use crate::boundarys::ZeroBoundary;

    use super::*;

    #[test]
    fn test_simple() {
        const N: usize = 10;
        let g = [1.0; N];
        let h = std::array::from_fn(|i| (-1 * (i as isize % 2)) as f64 * 1.0);

        let bc = ZeroBoundary {};

        let nx = 20;
        let x = (0..nx).map(|i| (i + 1) as f64).collect::<Vec<_>>();

        let ns = (nx + 1) / 2;
        let nd = nx / 2;

        let mut s = vec![0.0; ns];
        let mut d = vec![0.0; nd];

        dwt_forward(&g, &h, &x, &mut s, &mut d, &bc);

        dbg!(&x);
        dbg!(&s);
        dbg!(&d);

        assert_eq!(s, d);

        panic!();
    }
}
