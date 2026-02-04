pub use crate::daubechies::*;
pub use crate::dwt::DiscreteTransform;
use wavelets_macros::implement_dwt_orthogonal;

implement_dwt_orthogonal! {
    Daubechies1,
    [
        7.071067811865475244008443621048490392848359376884740365883398e-01,
        7.071067811865475244008443621048490392848359376884740365883398e-01,
    ]
}

implement_dwt_orthogonal! {
    Daubechies2,
    [
        -1.294095225512603811744494188120241641745344506599652569070016e-01,
        2.241438680420133810259727622404003554678835181842717613871683e-01,
        8.365163037378079055752937809168732034593703883484392934953414e-01,
        4.829629131445341433748715998644486838169524195042022752011715e-01,
    ]
}

implement_dwt_orthogonal! {
    Daubechies3,
    [
        3.522629188570953660274066471551002932775838791743161039893406e-02,
        -8.544127388202666169281916918177331153619763898808662976351748e-02,
        -1.350110200102545886963899066993744805622198452237811919756862e-01,
        4.598775021184915700951519421476167208081101774314923066433867e-01,
        8.068915093110925764944936040887134905192973949948236181650920e-01,
        3.326705529500826159985115891390056300129233992450683597084705e-01,
    ]
}

implement_dwt_orthogonal! {
    Daubechies4,
    [
        -1.059740178506903210488320852402722918109996490637641983484974e-02,
        3.288301166688519973540751354924438866454194113754971259727278e-02,
        3.084138183556076362721936253495905017031482172003403341821219e-02,
        -1.870348117190930840795706727890814195845441743745800912057770e-01,
        -2.798376941685985421141374718007538541198732022449175284003358e-02,
        6.308807679298589078817163383006152202032229226771951174057473e-01,
        7.148465705529156470899219552739926037076084010993081758450110e-01,
        2.303778133088965008632911830440708500016152482483092977910968e-01,
    ]
}

implement_dwt_orthogonal! {
    Daubechies5,
    [
        3.335725285473771277998183415817355747636524742305315099706428e-03,
        -1.258075199908199946850973993177579294920459162609785020169232e-02,
        -6.241490212798274274190519112920192970763557165687607323417435e-03,
        7.757149384004571352313048938860181980623099452012527983210146e-02,
        -3.224486958463837464847975506213492831356498416379847225434268e-02,
        -2.422948870663820318625713794746163619914908080626185983913726e-01,
        1.384281459013207315053971463390246973141057911739561022694652e-01,
        7.243085284377729277280712441022186407687562182320073725767335e-01,
        6.038292697971896705401193065250621075074221631016986987969283e-01,
        1.601023979741929144807237480204207336505441246250578327725699e-01
    ]
}

implement_dwt_orthogonal! {
    Daubechies6,
    [
        -1.077301085308479564852621609587200035235233609334419689818580e-03,
        4.777257510945510639635975246820707050230501216581434297593254e-03,
        5.538422011614961392519183980465012206110262773864964295476524e-04,
        -3.158203931748602956507908069984866905747953237314842337511464e-02,
        2.752286553030572862554083950419321365738758783043454321494202e-02,
        9.750160558732304910234355253812534233983074749525514279893193e-02,
        -1.297668675672619355622896058765854608452337492235814701599310e-01,
        -2.262646939654398200763145006609034656705401539728969940143487e-01,
        3.152503517091976290859896548109263966495199235172945244404163e-01,
        7.511339080210953506789344984397316855802547833382612009730420e-01,
        4.946238903984530856772041768778555886377863828962743623531834e-01,
        1.115407433501094636213239172409234390425395919844216759082360e-01
    ]
}

#[cfg(test)]
mod test {
    use crate::{boundarys::BoundaryCondition, tests::test_approx_equal};

    use super::*;
    use crate::dwt::DiscreteTransform;

    #[test]
    fn test_db1() {
        let n = 32;
        type WVLT = Daubechies1;
        let x: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let mut x2 = vec![0.0; n];

        let nsd = WVLT::get_outlen(n);

        let mut s = vec![0.0; nsd];
        let mut d = vec![0.0; nsd];

        let bcs = [
            BoundaryCondition::Periodic,
            BoundaryCondition::Symmetric,
            BoundaryCondition::Constant,
            BoundaryCondition::Reflect,
            BoundaryCondition::Zero,
        ];

        for bc in bcs {
            WVLT::forward(&x, &mut s, &mut d, &bc);
            WVLT::inverse(&s, &d, &mut x2);

            test_approx_equal(&x2, &x, 1E-14, 0.0);
        }
    }

    #[test]
    fn test_db1_per() {
        let n = 31;
        type WVLT = Daubechies1;
        let x: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let mut x2 = vec![0.0; n];
        let nd = n / 2;
        let ns = (n + 1) / 2;

        let mut s = vec![0.0; ns];
        let mut d = vec![0.0; nd];

        WVLT::forward_per(&x, &mut s, &mut d);
        WVLT::inverse_per(&s, &d, &mut x2);

        test_approx_equal(&x2, &x, 1E-14, 0.0);
    }

    #[test]
    fn test_db2() {
        let n = 32;
        type WVLT = Daubechies2;
        let x: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let mut x2 = vec![0.0; n];

        let nsd = WVLT::get_outlen(n);

        let mut s = vec![0.0; nsd];
        let mut d = vec![0.0; nsd];

        let bcs = [
            BoundaryCondition::Periodic,
            BoundaryCondition::Symmetric,
            BoundaryCondition::Constant,
            BoundaryCondition::Reflect,
            BoundaryCondition::Zero,
        ];

        for bc in bcs {
            WVLT::forward(&x, &mut s, &mut d, &bc);
            WVLT::inverse(&s, &d, &mut x2);

            test_approx_equal(&x2, &x, 1E-14, 0.0);
        }
    }

    #[test]
    fn test_db2_per() {
        let n = 32;
        type WVLT = Daubechies2;
        let x: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let mut x2 = vec![0.0; n];
        let nd = n / 2;
        let ns = (n + 1) / 2;

        let mut s = vec![0.0; ns];
        let mut d = vec![0.0; nd];

        WVLT::forward_per(&x, &mut s, &mut d);
        WVLT::inverse_per(&s, &d, &mut x2);

        test_approx_equal(&x2, &x, 1E-14, 0.0);
    }

    #[test]
    fn test_db3() {
        let n = 32;
        type WVLT = Daubechies3;
        let x: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let mut x2 = vec![0.0; n];

        let nsd = WVLT::get_outlen(n);

        let mut s = vec![0.0; nsd];
        let mut d = vec![0.0; nsd];

        let bcs = [
            BoundaryCondition::Periodic,
            BoundaryCondition::Symmetric,
            BoundaryCondition::Constant,
            BoundaryCondition::Reflect,
            BoundaryCondition::Zero,
        ];

        for bc in bcs {
            WVLT::forward(&x, &mut s, &mut d, &bc);
            WVLT::inverse(&s, &d, &mut x2);

            test_approx_equal(&x2, &x, 1E-14, 0.0);
        }
    }

    #[test]
    fn test_db3_per() {
        let n = 32;
        type WVLT = Daubechies3;
        let x: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let mut x2 = vec![0.0; n];
        let nd = n / 2;
        let ns = (n + 1) / 2;

        let mut s = vec![0.0; ns];
        let mut d = vec![0.0; nd];

        WVLT::forward_per(&x, &mut s, &mut d);

        dbg!(&s);
        dbg!(&d);

        WVLT::inverse_per(&s, &d, &mut x2);

        test_approx_equal(&x2, &x, 1E-14, 0.0);
    }

    #[test]
    fn test_db4() {
        let n = 32;
        type WVLT = Daubechies4;
        let x: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let mut x2 = vec![0.0; n];

        let nsd = WVLT::get_outlen(n);

        let mut s = vec![0.0; nsd];
        let mut d = vec![0.0; nsd];

        let bcs = [
            BoundaryCondition::Periodic,
            BoundaryCondition::Symmetric,
            BoundaryCondition::Constant,
            BoundaryCondition::Reflect,
            BoundaryCondition::Zero,
        ];

        for bc in bcs {
            WVLT::forward(&x, &mut s, &mut d, &bc);
            WVLT::inverse(&s, &d, &mut x2);

            test_approx_equal(&x2, &x, 1E-14, 0.0);
        }
    }

    #[test]
    fn test_db5() {
        let n = 32;
        type WVLT = Daubechies5;
        let x: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let mut x2 = vec![0.0; n];

        let nsd = WVLT::get_outlen(n);

        let mut s = vec![0.0; nsd];
        let mut d = vec![0.0; nsd];

        let bcs = [
            BoundaryCondition::Periodic,
            BoundaryCondition::Symmetric,
            BoundaryCondition::Constant,
            BoundaryCondition::Reflect,
            BoundaryCondition::Zero,
        ];

        for bc in bcs {
            WVLT::forward(&x, &mut s, &mut d, &bc);
            WVLT::inverse(&s, &d, &mut x2);

            test_approx_equal(&x2, &x, 1E-14, 0.0);
        }
    }

    #[test]
    fn test_db6() {
        let n = 32;
        type WVLT = Daubechies6;
        let x: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let mut x2 = vec![0.0; n];

        let nsd = WVLT::get_outlen(n);

        let mut s = vec![0.0; nsd];
        let mut d = vec![0.0; nsd];

        let bcs = [
            BoundaryCondition::Periodic,
            BoundaryCondition::Symmetric,
            BoundaryCondition::Constant,
            BoundaryCondition::Reflect,
            BoundaryCondition::Zero,
        ];

        for bc in bcs {
            WVLT::forward(&x, &mut s, &mut d, &bc);
            WVLT::inverse(&s, &d, &mut x2);

            test_approx_equal(&x2, &x, 1E-14, 0.0);
        }
    }
}
