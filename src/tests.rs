#![doc(hidden)]
use crate::*;

/// Epsilon for curve difference.
pub const EPS: f64 = 2.2e-14;
pub const RES: usize = 1000;

/// Error between two curves, the length of the curves must be the same.
pub fn curve_diff<const D: usize>(a: &[Coord<D>], b: &[Coord<D>]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a.l2_norm(b)).sum::<f64>() / a.len() as f64
}

#[test]
fn error() {
    let coeff = Coeffs2::from_column_slice(&[10., 20., 20., 10., 3., 4., 4., 3.]);
    let a = Efd2::try_from_coeffs_unnorm(coeff).unwrap();
    let coeff = Coeffs2::from_column_slice(&[10., 20., 20., 10.]);
    let b = Efd2::try_from_coeffs_unnorm(coeff).unwrap();
    assert_eq!(a.square_err(&b), 50.);
    assert_eq!(b.square_err(&a), 50.);
}

#[test]
fn efd2d_open() {
    use approx::assert_abs_diff_eq;
    let efd = Efd2::from_curve(PATH_OPEN, true);
    assert_eq!(efd.harmonic(), 14);
    // Test transformation
    let geo = efd.as_geo();
    assert_abs_diff_eq!(geo.trans()[0], 43.03456791427352);
    assert_abs_diff_eq!(geo.trans()[1], 48.107208358019015);
    assert_abs_diff_eq!(geo.rot().angle(), 2.7330524299596815);
    assert_abs_diff_eq!(geo.scale(), 33.930916934329495);
    // Test normalized
    let norm = efd.generate_norm_half(NORM_OPEN.len());
    assert_abs_diff_eq!(curve_diff(&norm, NORM_OPEN), 0.);
    // Test reconstruction
    let target = efd.generate_half(TARGET_OPEN.len());
    assert_abs_diff_eq!(curve_diff(&target, TARGET_OPEN), 0.);
}

#[test]
fn efd2d() {
    use approx::assert_abs_diff_eq;
    let efd = Efd2::from_curve(PATH, false);
    // Test starting point
    let path = PATH
        .iter()
        .cycle()
        .skip(PATH.len() / 2)
        .take(PATH.len())
        .copied()
        .collect::<alloc::vec::Vec<_>>();
    let efd_half = Efd2::from_curve(path, false);
    assert_abs_diff_eq!(efd.l1_norm(&efd_half), 0., epsilon = 1e-12);
    assert_eq!(efd.harmonic(), 8);
    // Test transformation
    let geo = efd.as_geo();
    assert_abs_diff_eq!(geo.trans()[0], -1.248409055632358);
    assert_abs_diff_eq!(geo.trans()[1], 55.26080122817753);
    assert_abs_diff_eq!(geo.rot().angle(), 0.6423416350347734);
    assert_abs_diff_eq!(geo.scale(), 48.16765830752243);
    // Test normalized
    let norm = efd.generate_norm(NORM.len());
    assert_abs_diff_eq!(curve_diff(&norm, NORM), 0.);
    // Test reconstruction
    let target = efd.generate(TARGET.len());
    assert_abs_diff_eq!(curve_diff(&target, TARGET), 0.);
}

#[test]
fn efd3d() {
    use approx::assert_abs_diff_eq;
    let efd = Efd3::from_curve(PATH3D, false);
    // Test starting point
    let path = PATH3D
        .iter()
        .cycle()
        .skip(PATH3D.len() / 2)
        .take(PATH3D.len())
        .copied()
        .collect::<alloc::vec::Vec<_>>();
    let efd_half = Efd3::from_curve_nyquist(path, false);
    assert_abs_diff_eq!(efd.l1_norm(&efd_half), 0., epsilon = 1e-12);
    assert_eq!(efd.harmonic(), 5);
    // Test transformation
    let geo = efd.as_geo();
    assert_abs_diff_eq!(geo.trans()[0], 0.7239345388499508);
    assert_abs_diff_eq!(geo.trans()[1], 0.09100107896533066);
    assert_abs_diff_eq!(geo.trans()[2], 0.49979194975846675);
    assert_abs_diff_eq!(geo.rot().angle(), 2.9160714030359416);
    assert_abs_diff_eq!(geo.scale(), 0.5629099155595344);
    // Test normalized
    let norm = efd.generate_norm(NORM3D.len());
    assert_abs_diff_eq!(curve_diff(&norm, NORM3D), 0.);
    // Test reconstruction
    let target = efd.generate(NORM3D.len());
    assert_abs_diff_eq!(curve_diff(&target, TARGET3D), 0.);
}

#[test]
#[cfg(feature = "std")]
fn plot2d_closed() -> Result<(), Box<dyn std::error::Error>> {
    #[rustfmt::skip]
    let coeff = crate::Coeffs2::from_column_slice(&[
        12., 35., 35., 13.,
        5., 21., 21., 5.,
        1., 12., 12., 1.,
    ]);
    plot2d(coeff, "img/2d.svg")
}

#[test]
#[cfg(feature = "std")]
fn plot2d_open() -> Result<(), Box<dyn std::error::Error>> {
    #[rustfmt::skip]
    let coeff = crate::Coeffs2::from_column_slice(&[
        35., 8., 0., 0.,
        10., 24., 0., 0.,
        5., -8., 0., 0.,
    ]);
    plot2d(coeff, "img/2d_open.svg")
}

#[test]
#[cfg(feature = "std")]
fn plot3d_closed() -> Result<(), Box<dyn std::error::Error>> {
    #[rustfmt::skip]
    let coeff = crate::Coeffs3::from_column_slice(&[
        12., 35., 20., 22., 5., 21.,
        21., 5., 1., 12., 12., 1.,
        3., 7., 12., 3., 5., 21.,
    ]);
    plot3d(coeff, "img/3d.svg")
}

#[test]
#[cfg(feature = "std")]
fn plot3d_open() -> Result<(), Box<dyn std::error::Error>> {
    #[rustfmt::skip]
    let coeff = crate::Coeffs3::from_column_slice(&[
        16., 35., 27., 0., 0., 0.,
        21., 8., 16., 0., 0., 0.,
        3., 7., 12., 0., 0., 0.,
    ]);
    plot3d(coeff, "img/3d_open.svg")
}

#[cfg(all(test, feature = "std"))]
fn get_area<const D: usize>(pts: &[[f64; D]]) -> [[f64; 2]; D] {
    pts.iter()
        .fold([[f64::INFINITY, f64::NEG_INFINITY]; D], |mut b, c| {
            b.iter_mut()
                .zip(c.iter())
                .for_each(|(b, c)| *b = [b[0].min(*c), b[1].max(*c)]);
            b
        })
}

#[cfg(all(test, feature = "std"))]
fn plot2d(coeff: crate::Coeffs2, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    use plotters::prelude::*;

    fn bounding_box(pts: &[[f64; 2]]) -> [f64; 4] {
        let [[x_min, x_max], [y_min, y_max]] = get_area(pts);
        let dx = (x_max - x_min).abs();
        let dy = (y_max - y_min).abs();
        if dx > dy {
            let cen = (y_min + y_max) * 0.5;
            let r = dx * 0.5;
            let mg = dx * 0.1;
            [x_min - mg, x_max + mg, cen - r - mg, cen + r + mg]
        } else {
            let cen = (x_min + x_max) * 0.5;
            let r = dy * 0.5;
            let mg = dy * 0.1;
            [cen - r - mg, cen + r + mg, y_min - mg, y_max + mg]
        }
    }

    let efd = Efd2::try_from_coeffs_unnorm(coeff).unwrap();
    let curve = efd.generate(360);
    let [x_min, x_max, y_min, y_max] = bounding_box(&curve);
    let b = SVGBackend::new(path, (1200, 1200));
    let root = b.into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root).build_cartesian_2d(x_min..x_max, y_min..y_max)?;
    let p0 = curve[0];
    chart.draw_series([Circle::new((p0[0], p0[1]), 3, BLACK.filled())])?;
    for (p, color) in [((10., 0.), RED), ((0., 10.), BLUE)] {
        chart.draw_series(LineSeries::new([(0., 0.), p], color.stroke_width(10)))?;
    }
    let geo0 = efd.as_geo();
    let mut c0 = [0.; 2];
    for m in efd.coeffs_iter() {
        let geo = geo0 * GeoVar2::new(c0, na::UnitComplex::new(0.), 1.);
        const N: usize = 100;
        const N_F: f64 = N as f64;
        let ellipse = (0..N)
            .map(|i| {
                let t = i as f64 * std::f64::consts::TAU / N_F;
                m * na::matrix![t.cos(); t.sin()]
            })
            .map(|c| geo.transform_pt(c.data.0[0]).into());
        let p1 = c0;
        c0.iter_mut()
            .zip(m.column(0).iter())
            .for_each(|(c, u)| *c += u);
        let p1 = geo0.transform_pt(p1).into();
        let p2 = geo0.transform_pt(c0).into();
        chart.draw_series([Circle::new(p2, 5, RED.filled())])?;
        chart.draw_series(LineSeries::new([p1, p2], RED.stroke_width(5)))?;
        chart.draw_series(LineSeries::new(ellipse, RED.stroke_width(7)))?;
    }
    chart.draw_series(LineSeries::new(
        curve.into_iter().map(|[x, y]| (x, y)),
        BLACK.stroke_width(10),
    ))?;
    Ok(())
}

#[cfg(all(test, feature = "std"))]
fn plot3d(coeff: crate::Coeffs3, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    use plotters::prelude::*;

    fn bounding_box(pts: &[[f64; 3]]) -> [f64; 6] {
        let mut b = get_area(pts);
        let center = b.map(|[min, max]| (min + max) * 0.5);
        let width = b
            .iter()
            .map(|[min, max]| (max - min).abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        b.iter_mut().zip(center).for_each(|([min, max], c)| {
            *min = c - width * 0.5;
            *max = c + width * 0.5;
        });
        let [[x_min, x_max], [y_min, y_max], [z_min, z_max]] = b;
        [x_min, x_max, y_min, y_max, z_min, z_max]
    }

    let efd = Efd3::try_from_coeffs_unnorm(coeff).unwrap();
    let curve = efd.generate(360);
    let [x_min, x_max, y_min, y_max, z_min, z_max] = bounding_box(&curve);
    let b = SVGBackend::new(path, (1200, 1200));
    let root = b.into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart =
        ChartBuilder::on(&root).build_cartesian_3d(x_min..x_max, y_min..y_max, z_min..z_max)?;
    chart.with_projection(|mut pb| {
        pb.yaw = 0.3;
        pb.scale = 1.4;
        pb.into_matrix()
    });
    let p0 = curve[0];
    chart.draw_series([Circle::new((p0[0], p0[1], p0[2]), 3, BLACK.filled())])?;
    for (p, color) in [
        ((10., 0., 0.), RED),
        ((0., 10., 0.), BLUE),
        ((0., 0., 10.), GREEN),
    ] {
        chart.draw_series(LineSeries::new([(0., 0., 0.), p], color.stroke_width(10)))?;
    }
    let geo0 = efd.as_geo();
    let mut c0 = [0.; 3];
    for m in efd.coeffs_iter() {
        let geo = geo0 * GeoVar3::new(c0, na::UnitQuaternion::identity(), 1.);
        const N: usize = 100;
        const N_F: f64 = N as f64;
        let ellipse = (0..N)
            .map(|i| {
                let t = i as f64 * std::f64::consts::TAU / N_F;
                m * na::matrix![t.cos(); t.sin()]
            })
            .map(|c| geo.transform_pt(c.data.0[0]).into());
        let p1 = c0;
        c0.iter_mut()
            .zip(m.column(0).iter())
            .for_each(|(c, u)| *c += u);
        let p1 = geo0.transform_pt(p1).into();
        let p2 = geo0.transform_pt(c0).into();
        chart.draw_series([Circle::new(p2, 5, RED.filled())])?;
        chart.draw_series(LineSeries::new([p1, p2], RED.stroke_width(5)))?;
        chart.draw_series(LineSeries::new(ellipse, RED.stroke_width(7)))?;
    }
    chart.draw_series(LineSeries::new(
        curve.into_iter().map(|c| c.into()),
        BLACK.stroke_width(10),
    ))?;
    Ok(())
}

pub const PATH: &[[f64; 2]] = &[
    [14.928108089437242, 90.01002059789568],
    [-3.25371009238094, 85.46456605244113],
    [-16.763462931659024, 76.52439024390245],
    [-39.6173464560173, 57.055475143350215],
    [-49.46583130450215, 35.085778173653246],
    [-27.739072687756586, 14.939024390243903],
    [-2.117346456017304, 19.17668726456234],
    [17.958411119740273, 37.7372933251684],
    [26.291744453073605, 57.81305090092597],
    [43.71598687731603, 68.41911150698658],
    [47.12507778640693, 80.5403236281987],
    [38.41295657428572, 90.38880847668355],
    [27.80689596822512, 91.1463842342593],
];
pub const NORM: &[[f64; 2]] = &[
    [1.121867867415495, -0.045464239084766815],
    [0.9872417926127977, 0.18638949876194405],
    [0.7476580005002319, 0.34779177855648996],
    [0.48905080113405003, 0.4672825594971016],
    [0.20973982300907326, 0.540823069913559],
    [-0.07222723523517134, 0.5367575823892701],
    [-0.3691999131281563, 0.5280943367180384],
    [-0.6358172824767391, 0.4819309005033445],
    [-0.91279332818287, 0.3541149750143424],
    [-1.0322980468221987, 0.16159118041569326],
    [-0.986534645084418, -0.13686891144169946],
    [-0.8755517667383229, -0.36697981788913797],
    [-0.6245607058741152, -0.5135694354439416],
    [-0.35544697825948657, -0.5780626722560755],
    [-0.07055780263354745, -0.5506666212441792],
    [0.2043462020076517, -0.47418098033451317],
    [0.44664970830871464, -0.328108758578716],
    [0.7304825634231278, -0.3219001870135995],
    [0.9979509460238818, -0.288974258483154],
    [1.121867867415495, -0.04546423908476701],
];
pub const TARGET: &[[f64; 2]] = &[
    [43.33126493721068, 85.87985697497079],
    [31.448711552462917, 90.93714551548459],
    [17.551129802024384, 90.24864068027458],
    [4.129254558611014, 87.39477202741728],
    [-8.765179714679514, 82.17128515954673],
    [-19.522662025047925, 73.87807684015954],
    [-30.72620279299187, 64.97454337279662],
    [-39.676910244933445, 55.500646807734725],
    [-46.670971329200704, 42.57869997267558],
    [-45.7245360925009, 31.705101541410578],
    [-35.34721760389383, 21.514753026111038],
    [-24.426816766440876, 15.842435703492256],
    [-10.51671656605222, 17.431396485950074],
    [1.7233669357074701, 22.70956427286999],
    [11.920324010596742, 31.986896987031265],
    [20.3156527894926, 42.86936681205345],
    [25.445675097573584, 55.49490816543492],
    [36.21327612361932, 63.92460935569895],
    [45.5787852714277, 72.91252363426],
    [43.33126493721069, 85.87985697497079],
];
pub const PATH_OPEN: &[[f64; 2]] = &[
    [0.028607755880487345, 47.07692307692308],
    [6.182453909726641, 52.76923076923077],
    [14.797838525111256, 57.07692307692308],
    [24.643992371265103, 58.61538461538461],
    [41.10553083280357, 59.07692307692308],
    [50.18245390972664, 56.76923076923077],
    [60.6439923712651, 51.53846153846154],
    [65.41322314049587, 46.0],
    [68.79783852511126, 36.92307692307692],
    [67.41322314049587, 25.384615384615383],
    [60.6439923712651, 18.153846153846153],
];
pub const NORM_OPEN: &[[f64; 2]] = &[
    [1.1359143408478851, 0.4932627535583506],
    [1.0107881344064362, 0.25203693163270735],
    [0.7986359597354069, 0.04362118993327746],
    [0.5352216449239837, -0.10532506462439944],
    [0.2576057614333437, -0.2383842862958325],
    [-0.041034602995239464, -0.3121009834802267],
    [-0.3464267744182665, -0.30630825352973684],
    [-0.6230294787802827, -0.1959887930322815],
    [-0.8262117214963934, 0.02524784944958026],
    [-0.9146986927465437, 0.3083655207679895],
    [-0.8376148009727739, 0.5644090247994938],
];
pub const TARGET_OPEN: &[[f64; 2]] = &[
    [1.0148902072201196, 48.05959437965015],
    [8.162788875132385, 53.88432660760344],
    [17.578270939474194, 57.51431607681633],
    [27.78834398083798, 58.601516958827375],
    [38.22647895537773, 59.00258664455835],
    [48.51936215761635, 57.2724169244058],
    [57.95071989825114, 52.97543340836404],
    [65.07662487492445, 45.81172287359621],
    [68.42119498324773, 36.183910663649094],
    [67.36019072112205, 26.17528079383288],
    [61.50851730628217, 19.241550877423137],
];
#[rustfmt::skip]
pub const PATH3D: &[[f64; 3]] = &[
    [0.6999100262927096,0.43028119112952473,0.5700737247541725],
    [0.6296590344074013,0.4512760872295425,0.6323601770225047],
    [0.5638138739696974,0.46183051089148464,0.6847090584540213],
    [0.5058369206546075,0.4615280764519245,0.7287803814245077],
    [0.45702863707701546,0.4510622388540364,0.7665948614304103],
    [0.41723197726966244,0.4317766784472379,0.799678921250722],
    [0.3856541747672426,0.40523683682222406,0.8288871838597307],
    [0.3614034251971859,0.3729768425814701,0.8545617819407205],
    [0.3437381466626122,0.33639334903869056,0.8767460300745514],
    [0.3321447389761846,0.29671906154191235,0.8953422088163436],
    [0.3263372629749841,0.25502826620566654,0.9102002934684913],
    [0.32622858809088523,0.21225076122841705,0.9211539082423659],
    [0.3318929803861661,0.16918542545203516,0.928021196624841],
    [0.34352480290022414,0.1265118028240236,0.9305834049340109],
    [0.36139084538662236,0.08480105307198857,0.9285501808026447],
    [0.38577056486989364,0.04452862638492877,0.9215195454857344],
    [0.41687739559371884,0.0060909757379047635,0.9089423177834546],
    [0.45475516434546037,-0.030172412132808968,0.8901052556002697],
    [0.49914754370187137,-0.06395289690096129,0.8641537806399759],
    [0.5493470560271839,-0.09494849409160033,0.8301822664355119],
    [0.6040449685807827,-0.1228552958162816,0.7874238072487618],
    [0.6612249903953387,-0.14739768525858138,0.7355647044666407],
    [0.7181712990997067,-0.1684251622294386,0.6751762361616609],
    [0.7716940290779204,-0.18608109658863914,0.6081629312757029],
    [0.8186957920221034,-0.20095221972437277,0.5379176568148215],
    [0.8570911228971989,-0.21393997152327254,0.46864111603170266],
    [0.88661962597743,-0.2256744734813517,0.403703691896119],
    [0.9087058112647335,-0.2359493070484034,0.3443568978212587],
    [0.9253987983439792,-0.24384980367577586,0.29012813939844717],
    [0.9384640530551117,-0.24830445184260166,0.24006274246229356],
    [0.9491216566898215,-0.2484529388271337,0.19349216519158882],
    [0.9581400770939057,-0.2437369857356437,0.1501994489037393],
    [0.9659871776934699,-0.2338686278496317,0.11033692690815355],
    [0.9729414021915203,-0.21877677851516128,0.07430847249221559],
    [0.979158591144934,-0.1985626427303512,0.04267704651223003],
    [0.984708076435491,-0.17346680001041803,0.0161019718314504],
    [0.9895891445905367,-0.1438444290548901,-0.004701610196814504],
    [0.9937341248474196,-0.11014503640716575,-0.01898841932855333],
    [0.997000753888979,-0.07289400668205587,-0.02599924103976281],
    [0.9991540450179144,-0.03267407497300495,-0.024971967263275538],
    [0.9998362404365794,0.009894685600921771,-0.015152145277328999],
    [0.9985225360959517,0.05417737861443478,0.004190054292864898],
    [0.9944609178696164,0.09954962956012486,0.03372468064136638],
    [0.9865989401537969,0.14542428454053347,0.07398857177482451],
    [0.9735136458469561,0.19127348741773503,0.12524230260109778],
    [0.9533914378420906,0.2366305939786505,0.18721839718014333],
    [0.9241589851285424,0.2810406966408692,0.25873982499375203],
    [0.8839235581077262,0.32391579444519214,0.33727985639690555],
    [0.8318395864591331,0.36427712952733915,0.4187422540212782],
    [0.7691918202384619,0.4004903064827955,0.49794724428553494],
];
#[rustfmt::skip]
pub const NORM3D: &[[f64; 3]] = &[
    [1.050227461832115, -0.0005073577983769964, -0.13398960704180338],
    [1.0409654686697103, 0.09178711336572118, -0.13075380791170074],
    [1.0151358207877603, 0.17957003152192338, -0.12164579414882536],
    [0.9743071746687448, 0.25935684076368853, -0.10709090301194589],
    [0.9206966963485949, 0.3287803476339455, -0.08778522995837323],
    [0.8568177382307565, 0.38675251927207704, -0.06468731553153995],
    [0.7851237349711093, 0.43337044680659426, -0.0389815629897],
    [0.7077244002261751, 0.46961236789347416, -0.012005149889308972],
    [0.6262274495758018, 0.49691767664275016, 0.014859439825599967],
    [0.5417236047369597, 0.516766679700503, 0.04031217608400313],
    [0.45489434942195395, 0.5303663107866716, 0.06325974426131489],
    [0.36619123214489024, 0.53851039737721, 0.08290855901195963],
    [0.27602065193916303, 0.5416282350884262, 0.09880259572341114],
    [0.1848727709368072, 0.5399786015444935, 0.11079353595558848],
    [0.09335575109573839, 0.5339037422206833, 0.11894731514179564],
    [0.002130193904912661, 0.5240411621089527, 0.12340870306719053],
    [-0.08822638949837375, 0.5114049500433689, 0.12425915654852249],
    [-0.17736931446205084, 0.49728904303715976, 0.12140864041522946],
    [-0.2652858159600485, 0.4830009999054962, 0.11455709426191873],
    [-0.3522895916048328, 0.46949072730129615, 0.10324607875536446],
    [-0.43888541426938, 0.45697822867515725, 0.08699897901594812],
    [-0.5255170490386677, 0.4446960441414569, 0.06552419306030442],
    [-0.6122498229906431, 0.4308412852616422, 0.03893606093979722],
    [-0.6984659656318329, 0.41278346302623825, 0.007938271365220542],
    [-0.7826581252568905, 0.3875097873345675, -0.026082636760474155],
    [-0.8623911241548298, 0.3522259920018012, -0.061090109450567476],
    [-0.9344668691075982, 0.3049849151931595, -0.09456725232600914],
    [-0.9952805851534361, 0.2451996506542417, -0.12380365721923461],
    [-1.041309728906082, 0.17391799819341314, -0.1462522487187565],
    [-1.0696419801272443, 0.09378601479425208, -0.15989092530823687],
    [-1.0784347992690009, 0.00869843554962985, -0.1635195891421702],
    [-1.0672102114892057, -0.07679464425725145, -0.1569342447387529],
    [-1.0369225035447835, -0.1581999358387836, -0.14094395727640763],
    [-0.9897854074771137, -0.23176143207953973, -0.11722825112520766],
    [-0.9288973440013031, -0.29496898796690496, -0.08806481527363849],
    [-0.8577458800052447, -0.34679812845049457, -0.05598286269131613],
    [-0.7796955924479362, -0.3876492180032384, -0.023410495825878994],
    [-0.6975618570473225, -0.41902421256341826, 0.00761773200324507],
    [-0.6133477491688857, -0.44303680455538963, 0.03564330718689209],
    [-0.528179185281803, -0.46188289574832475, 0.059833056517277446],
    [-0.4424255885847685, -0.47739618687357627, 0.0799021644604981],
    [-0.3559518117051911, -0.4907802568321497, 0.09594453533454746],
    [-0.2684218801853755, -0.5025539177856179, 0.10822144732036201],
    [-0.17957186643771475, -0.512686240169666, 0.11696387505846025],
    [-0.08938751750661104, -0.5208478814548784, 0.12223515082761294],
    [0.0018438463983321532, -0.5266793173670363, 0.12388170172104114],
    [0.09359807481392228, -0.5299805519747676, 0.12157565128588482],
    [0.18525232061302777, -0.5307587706425019, 0.1149302190271396],
    [0.2762338250563933, -0.5291204033300677, 0.10365253843125763],
    [0.3661167852751386, -0.5250474775511504, 0.08769220733248],
    [0.4546239908690802, -0.5181394303395053, 0.0673483084462124],
    [0.5415223714184638, -0.5074186537949871, 0.04331076524364217],
    [0.6264377339938791, -0.49128569824759005, 0.01662964744166247],
    [0.708643293941108, -0.4676712330852152, -0.011376528242115474],
    [0.7868905393204704, -0.4343770077674127, -0.03924978397692278],
    [0.8593452256260729, -0.38954222756530044, -0.06553011711480797],
    [0.9236670412688771, -0.3321305420651166, -0.08887549669281929],
    [0.9772350358236068, -0.2623180783789554, -0.10814778433707047],
    [1.0174820194028202, -0.1816794668562026, -0.12246013098691194],
    [1.0422703670017814, -0.09311304849657658, -0.131192593380888],
    [1.050227461832115, -0.0005073577983772137, -0.13398960704180338],
];
#[rustfmt::skip]
pub const TARGET3D: &[[f64; 3]] = &[
    [0.3260958725587005, 0.2626914373285255, 0.9089784150350606],
    [0.32578144663123737, 0.2120944697131259, 0.9219980701322934],
    [0.33370591212609335, 0.161420097024903, 0.928978947539205],
    [0.3493514104777207, 0.11276885734180067, 0.929857444661391],
    [0.3718101631920913, 0.06774034427097282, 0.924956117867235],
    [0.39989272818276195, 0.027287279438196158, 0.9148627028403553],
    [0.4322636914587475, -0.008292997869433702, 0.900272242990724],
    [0.4675818566515592, -0.03924061637611764, 0.8818356399513159],
    [0.5046220619146035, -0.06612994872822492, 0.8600531121510557],
    [0.5423604777287174, -0.08964059239897768, 0.8352357460837865],
    [0.5800135236278517, -0.11036408928771607, 0.8075371011536572],
    [0.6170304843591229, -0.12869333227350999, 0.777035853915857],
    [0.6530492512622788, -0.14481468265426684, 0.7438357974312979],
    [0.687831292579487, -0.15879115629786344, 0.708145382500883],
    [0.7211945715146565, -0.17069749625584313, 0.6703066119397451],
    [0.7529612825727692, -0.18075226225800145, 0.6307602986825889],
    [0.7829316324698207, -0.18939270537930958, 0.5899564839232165],
    [0.8108869979393252, -0.1972551348933287, 0.5482387547348462],
    [0.8366176732823795, -0.20505217114689472, 0.5057432310505893],
    [0.8599640687891229, -0.21337082777927896, 0.46235312409291035],
    [0.8808571116416533, -0.2224426700589897, 0.41773728432686424],
    [0.8993443439680942, -0.23195148369725715, 0.3714789080813883],
    [0.9155924352968428, -0.24094048514036054, 0.32327430163491433],
    [0.9298633028955574, -0.24786029816941157, 0.2731584558798579],
    [0.9424680252852653, -0.2507655976866893, 0.2217007999315233],
    [0.9537084491736222, -0.24763084111678219, 0.17011517965093764],
    [0.9638194363039794, -0.23672354810738633, 0.12024357043003747],
    [0.9729244187208099, -0.21695561008808684, 0.0744001125248217],
    [0.9810135438902163, -0.18813411787982998, 0.03509440202436903],
    [0.9879481865105852, -0.15105338988136385, 0.0046826248029508966],
    [0.9934894592013293, -0.1074045647679859, -0.014985449231401882],
    [0.9973431296794861, -0.059519868512043855, -0.02285154186080629],
    [0.999210283805767, -0.010005484778035562, -0.018806669388064745],
    [0.9988327719443517, 0.03865914657889336, -0.0036890676835223957],
    [0.9960248012457309, 0.08447250839867437, 0.020875901585369283],
    [0.990686182532988, 0.12614389474817164, 0.05275805449015886],
    [0.9827974905407914, 0.16316920230375498, 0.08966577483931565],
    [0.9724014763633638, 0.19574859718182563, 0.12947864696273398],
    [0.9595775003638263, 0.22458102727431392, 0.17050366370933018],
    [0.9444160952610361, 0.25059588716226644, 0.2116105102349466],
    [0.926999213451773, 0.2746909442880491, 0.2522320007351945],
    [0.9073889797387367, 0.29753648855138504, 0.2922479325268073],
    [0.8856248616097713, 0.3194818829678534, 0.3317964244442485],
    [0.8617270456027556, 0.3405693118917297, 0.371070072071083],
    [0.8357030940885343, 0.3606293539760703, 0.41015220639928174],
    [0.807555750894855, 0.37941222388457463, 0.4489324241818712],
    [0.7772916182043683, 0.3967024798279609, 0.48711520242172035],
    [0.7449324952165203, 0.41237481484988064, 0.524307899314948],
    [0.7105324855268182, 0.4263709200896445, 0.5601522002656646],
    [0.6742037832861655, 0.43860554185288597, 0.5944518853784275],
    [0.6361520526183972, 0.44883541629010093, 0.6272523465314259],
    [0.5967188487230278, 0.456540200010284, 0.6588424757552608],
    [0.5564244889989518, 0.46086514637189036, 0.6896728968891997],
    [0.5160014095998329, 0.46066061244683537, 0.7202094111211075],
    [0.476406577773427, 0.45462736108879853, 0.7507600608276456],
    [0.43880283974270917, 0.4415461790958255, 0.7813230387048884],
    [0.40450336412199756, 0.4205440430163977, 0.8114984239457868],
    [0.3748799882081265, 0.39133442311781086, 0.840490645062135],
    [0.35124396812186137, 0.3543707641829752, 0.8672050893497913],
    [0.3347146215240782, 0.3108698548364523, 0.8904178079577785],
    [0.3260958725587005, 0.2626914373285256, 0.9089784150350605],
];
