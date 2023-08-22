#![doc(hidden)]
#[cfg(test)]
use core::f64::consts::TAU;

/// Epsilon for curve difference.
pub const EPS: f64 = 2.2e-14;
pub const RES: usize = 1000;

#[test]
fn error() {
    use crate::*;
    let coeff = Coeff2::from_column_slice(&[10., 20., 20., 10., 3., 4., 4., 3.]);
    let a = Efd2::try_from_coeffs_unnorm(coeff).unwrap();
    let coeff = Coeff2::from_column_slice(&[10., 20., 20., 10.]);
    let b = Efd2::try_from_coeffs_unnorm(coeff).unwrap();
    assert_eq!(a.square_err(&b), 50.);
    assert_eq!(b.square_err(&a), 50.);
}

#[test]
fn efd2d_open() {
    use crate::*;
    use approx::assert_abs_diff_eq;
    let efd = Efd2::from_curve(PATH_OPEN, true);
    assert_eq!(efd.harmonic(), 14);
    // Test transformation
    let trans = efd.as_trans();
    assert_abs_diff_eq!(trans.trans()[0], 43.03456791427352);
    assert_abs_diff_eq!(trans.trans()[1], 48.107208358019015);
    assert_abs_diff_eq!(trans.rot().angle(), 2.7330524299596815);
    assert_abs_diff_eq!(trans.scale(), 33.930916934329495);
    // Test normalized
    let norm = efd.generate_norm_in(NORM_OPEN.len(), TAU);
    assert!(curve_diff(&norm, NORM_OPEN) < EPS);
    // Test reconstruction
    let target = efd.generate(TARGET_OPEN.len());
    assert!(curve_diff(&target, TARGET_OPEN) < EPS);
}

#[test]
fn efd2d() {
    use crate::*;
    use alloc::vec::Vec;
    use approx::assert_abs_diff_eq;
    let efd = Efd2::from_curve(PATH, false);
    // Test starting point
    let path = PATH
        .iter()
        .cycle()
        .skip(0)
        .take(PATH.len())
        .copied()
        .collect::<Vec<_>>();
    let efd_half = Efd2::from_curve(path, false);
    assert!(efd.l1_norm(&efd_half) < EPS);
    // Test transformation
    let trans = efd.as_trans();
    assert_abs_diff_eq!(trans.trans()[0], -1.248409055632358);
    assert_abs_diff_eq!(trans.trans()[1], 55.26080122817753);
    assert_abs_diff_eq!(trans.rot().angle(), 0.6423416350347734);
    assert_abs_diff_eq!(trans.scale(), 48.16765830752243);
    assert_eq!(efd.harmonic(), 8);
    // Test normalized
    let norm = efd.generate_norm_in(NORM.len(), TAU);
    assert!(curve_diff(&norm, NORM) < EPS);
    // Test reconstruction
    let target = efd.generate(TARGET.len());
    assert!(curve_diff(&target, TARGET) < EPS);
}

#[test]
fn efd3d() {
    use crate::*;
    use alloc::vec::Vec;
    use approx::assert_abs_diff_eq;
    let efd = Efd3::from_curve(PATH3D, false);
    // Test starting point
    let path = PATH3D
        .iter()
        .cycle()
        .skip(PATH3D.len() / 2)
        .take(PATH3D.len())
        .copied()
        .collect::<Vec<_>>();
    let efd_half = Efd3::from_curve_nyquist(path, false);
    assert!(efd.l1_norm(&efd_half) < EPS);
    // Test transformation
    let trans = efd.as_trans();
    assert_abs_diff_eq!(trans.trans()[0], 0.7239345388499508);
    assert_abs_diff_eq!(trans.trans()[1], 0.09100107896533066);
    assert_abs_diff_eq!(trans.trans()[2], 0.49979194975846675);
    assert_abs_diff_eq!(trans.rot().angle(), 2.9160714030359416);
    assert_abs_diff_eq!(trans.scale(), 0.5629099155595344);
    assert_eq!(efd.harmonic(), 5);
    // Test normalized
    let norm = efd.generate_norm_in(NORM3D.len(), TAU);
    assert!(curve_diff(&norm, NORM3D) < EPS);
    // Test reconstruction
    let target = efd.generate(NORM3D.len());
    assert!(curve_diff(&target, TARGET3D) < EPS);
}

#[cfg(test)]
fn get_area<const N: usize>(pts: &[[f64; N]]) -> [[f64; 2]; N] {
    pts.iter()
        .fold([[f64::INFINITY, f64::NEG_INFINITY]; N], |mut b, c| {
            b.iter_mut()
                .zip(c.iter())
                .for_each(|(b, c)| *b = [b[0].min(*c), b[1].max(*c)]);
            b
        })
}

#[test]
#[cfg(feature = "std")]
fn plot2d() -> Result<(), Box<dyn std::error::Error>> {
    use crate::*;
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

    #[rustfmt::skip]
    let coeff = Coeff2::from_column_slice(&[
        12., 35., 35., 13.,
        5., 21., 21., 5.,
        1., 12., 12., 1.,
    ]);
    let efd = Efd2::try_from_coeffs_unnorm(coeff).unwrap();
    let path = efd.generate(360);
    let [x_min, x_max, y_min, y_max] = bounding_box(&path);
    let b = SVGBackend::new("test2d.svg", (1200, 1200));
    let root = b.into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root).build_cartesian_2d(x_min..x_max, y_min..y_max)?;
    let p0 = path[0];
    chart.draw_series([Circle::new((p0[0], p0[1]), 3, BLACK.filled())])?;
    for (p, color) in [((10., 0.), RED), ((0., 10.), BLUE)] {
        chart.draw_series(LineSeries::new([(0., 0.), p], color.stroke_width(10)))?;
    }
    let trans0 = efd.as_trans();
    let mut c0 = [0.; 2];
    for m in efd.coeffs_iter() {
        let trans = trans0 * Transform2::new(c0, na::UnitComplex::new(0.), 1.);
        const N: usize = 100;
        const N_F: f64 = N as f64;
        let ellipse = (0..N)
            .map(|i| {
                let t = i as f64 * std::f64::consts::TAU / N_F;
                m * na::matrix![t.cos(); t.sin()]
            })
            .map(|c| trans.transform_pt(&c.data.0[0]).into());
        let p1 = c0;
        c0.iter_mut()
            .zip(m.column(0).iter())
            .for_each(|(c, u)| *c += u);
        let p1 = trans0.transform_pt(&p1).into();
        let p2 = trans0.transform_pt(&c0).into();
        chart.draw_series([Circle::new(p2, 5, RED.filled())])?;
        chart.draw_series(LineSeries::new([p1, p2], RED.stroke_width(5)))?;
        chart.draw_series(LineSeries::new(ellipse, RED.stroke_width(7)))?;
    }
    chart.draw_series(LineSeries::new(
        path.into_iter().map(|[x, y]| (x, y)),
        BLACK.stroke_width(10),
    ))?;
    Ok(())
}

#[test]
#[cfg(feature = "std")]
fn plot3d() -> Result<(), Box<dyn std::error::Error>> {
    use crate::*;
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

    #[rustfmt::skip]
    let coeff = Coeff3::from_column_slice(&[
        12., 35., 20., 22., 5., 21.,
        21., 5., 1., 12., 12., 1.,
        3., 7., 12., 3., 5., 21.,
    ]);
    let efd = Efd3::try_from_coeffs_unnorm(coeff).unwrap();
    let path = efd.generate(360);
    let [x_min, x_max, y_min, y_max, z_min, z_max] = bounding_box(&path);
    let b = SVGBackend::new("test3d.svg", (1200, 1200));
    let root = b.into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart =
        ChartBuilder::on(&root).build_cartesian_3d(x_min..x_max, y_min..y_max, z_min..z_max)?;
    chart.with_projection(|mut pb| {
        pb.yaw = 0.3;
        pb.scale = 1.4;
        pb.into_matrix()
    });
    let p0 = path[0];
    chart.draw_series([Circle::new((p0[0], p0[1], p0[2]), 3, BLACK.filled())])?;
    for (p, color) in [
        ((10., 0., 0.), RED),
        ((0., 10., 0.), BLUE),
        ((0., 0., 10.), GREEN),
    ] {
        chart.draw_series(LineSeries::new([(0., 0., 0.), p], color.stroke_width(10)))?;
    }
    let trans0 = efd.as_trans();
    let mut c0 = [0.; 3];
    for m in efd.coeffs_iter() {
        let trans = trans0 * Transform3::new(c0, na::UnitQuaternion::identity(), 1.);
        const N: usize = 100;
        const N_F: f64 = N as f64;
        let ellipse = (0..N)
            .map(|i| {
                let t = i as f64 * std::f64::consts::TAU / N_F;
                m * na::matrix![t.cos(); t.sin()]
            })
            .map(|c| trans.transform_pt(&c.data.0[0]).into());
        let p1 = c0;
        c0.iter_mut()
            .zip(m.column(0).iter())
            .for_each(|(c, u)| *c += u);
        let p1 = trans0.transform_pt(&p1).into();
        let p2 = trans0.transform_pt(&c0).into();
        chart.draw_series([Circle::new(p2, 5, RED.filled())])?;
        chart.draw_series(LineSeries::new([p1, p2], RED.stroke_width(5)))?;
        chart.draw_series(LineSeries::new(ellipse, RED.stroke_width(7)))?;
    }
    chart.draw_series(LineSeries::new(
        path.into_iter().map(|c| c.into()),
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
    [0.7986359597354069, 0.04362118993327746],
    [0.2576057614333437, -0.2383842862958325],
    [-0.3464267744182665, -0.30630825352973684],
    [-0.8262117214963934, 0.02524784944958026],
    [-0.8376148009727739, 0.5644090247994938],
    [-0.8262117214963935, 0.025247849449580518],
    [-0.34642677441826664, -0.3063082535297368],
    [0.25760576143334346, -0.23838428629583264],
    [0.7986359597354066, 0.0436211899332773],
    [1.1359143408478851, 0.4932627535583506],
];
pub const TARGET_OPEN: &[[f64; 2]] = &[
    [1.0148902072201196, 48.05959437965015],
    [17.578270939474194, 57.51431607681633],
    [38.22647895537773, 59.00258664455835],
    [57.95071989825114, 52.97543340836404],
    [68.42119498324773, 36.183910663649094],
    [61.50851730628217, 19.241550877423137],
    [68.42119498324773, 36.18391066364909],
    [57.95071989825114, 52.97543340836404],
    [38.22647895537774, 59.00258664455835],
    [17.5782709394742, 57.51431607681633],
    [1.0148902072201196, 48.05959437965015],
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
    [1.050227461832115, -0.0005073577983771577, -0.13398960704180335],
    [1.0409654686697103, 0.09178711336572103, -0.13075380791170071],
    [1.0151358207877603, 0.17957003152192322, -0.12164579414882534],
    [0.9743071746687448, 0.25935684076368837, -0.10709090301194589],
    [0.9206966963485947, 0.3287803476339453, -0.08778522995837323],
    [0.8568177382307564, 0.38675251927207693, -0.06468731553153996],
    [0.7851237349711093, 0.43337044680659403, -0.038981562989700035],
    [0.7077244002261751, 0.46961236789347405, -0.012005149889309008],
    [0.6262274495758018, 0.49691767664275005, 0.014859439825599924],
    [0.5417236047369597, 0.5167666797005029, 0.04031217608400308],
    [0.45489434942195384, 0.5303663107866715, 0.06325974426131484],
    [0.36619123214489024, 0.53851039737721, 0.08290855901195958],
    [0.27602065193916303, 0.5416282350884261, 0.09880259572341107],
    [0.18487277093680712, 0.5399786015444937, 0.1107935359555884],
    [0.09335575109573833, 0.5339037422206833, 0.11894731514179559],
    [0.0021301939049125887, 0.5240411621089528, 0.1234087030671905],
    [-0.08822638949837383, 0.5114049500433689, 0.12425915654852243],
    [-0.1773693144620509, 0.49728904303715976, 0.12140864041522938],
    [-0.2652858159600486, 0.48300099990549633, 0.11455709426191867],
    [-0.35228959160483286, 0.46949072730129626, 0.10324607875536439],
    [-0.43888541426938005, 0.4569782286751573, 0.08699897901594807],
    [-0.5255170490386677, 0.444696044141457, 0.06552419306030437],
    [-0.6122498229906431, 0.43084128526164234, 0.03893606093979718],
    [-0.6984659656318329, 0.41278346302623836, 0.007938271365220503],
    [-0.7826581252568905, 0.38750978733456765, -0.02608263676047418],
    [-0.8623911241548299, 0.3522259920018013, -0.061090109450567476],
    [-0.9344668691075984, 0.3049849151931597, -0.09456725232600914],
    [-0.9952805851534361, 0.24519965065424185, -0.12380365721923461],
    [-1.041309728906082, 0.17391799819341328, -0.14625224871875647],
    [-1.0696419801272443, 0.09378601479425225, -0.15989092530823687],
    [-1.0784347992690009, 0.008698435549630004, -0.16351958914217019],
    [-1.0672102114892057, -0.07679464425725128, -0.1569342447387529],
    [-1.0369225035447835, -0.15819993583878347, -0.14094395727640757],
    [-0.9897854074771137, -0.2317614320795396, -0.11722825112520763],
    [-0.928897344001303, -0.2949689879669048, -0.08806481527363846],
    [-0.8577458800052445, -0.34679812845049446, -0.05598286269131609],
    [-0.7796955924479362, -0.3876492180032383, -0.023410495825878955],
    [-0.6975618570473227, -0.41902421256341815, 0.007617732003245104],
    [-0.6133477491688857, -0.4430368045553895, 0.035643307186892126],
    [-0.528179185281803, -0.46188289574832464, 0.05983305651727745],
    [-0.4424255885847685, -0.47739618687357616, 0.07990216446049812],
    [-0.3559518117051911, -0.4907802568321496, 0.09594453533454747],
    [-0.26842188018537544, -0.5025539177856178, 0.10822144732036205],
    [-0.17957186643771467, -0.512686240169666, 0.11696387505846027],
    [-0.08938751750661099, -0.5208478814548784, 0.12223515082761294],
    [0.001843846398332216, -0.5266793173670364, 0.12388170172104111],
    [0.09359807481392234, -0.5299805519747676, 0.12157565128588482],
    [0.18525232061302785, -0.5307587706425019, 0.11493021902713962],
    [0.2762338250563934, -0.5291204033300678, 0.10365253843125767],
    [0.3661167852751387, -0.5250474775511506, 0.08769220733248002],
    [0.4546239908690802, -0.5181394303395053, 0.06734830844621242],
    [0.5415223714184638, -0.5074186537949872, 0.04331076524364221],
    [0.6264377339938791, -0.49128569824759005, 0.016629647441662513],
    [0.708643293941108, -0.4676712330852153, -0.01137652824211543],
    [0.7868905393204704, -0.43437700776741284, -0.03924978397692273],
    [0.859345225626073, -0.38954222756530044, -0.06553011711480793],
    [0.9236670412688772, -0.3321305420651167, -0.08887549669281923],
    [0.9772350358236068, -0.2623180783789555, -0.10814778433707042],
    [1.0174820194028202, -0.1816794668562027, -0.12246013098691189],
    [1.0422703670017814, -0.09311304849657671, -0.131192593380888],
    [1.050227461832115, -0.000507357798377375, -0.13398960704180335],
];
#[rustfmt::skip]
pub const TARGET3D: &[[f64; 3]] = &[
    [0.32609587255870043, 0.2626914373285256, 0.9089784150350606],
    [0.32578144663123726, 0.212094469713126, 0.9219980701322936],
    [0.3337059121260933, 0.1614200970249031, 0.928978947539205],
    [0.34935141047772067, 0.11276885734180074, 0.929857444661391],
    [0.3718101631920916, 0.06774034427097281, 0.924956117867235],
    [0.39989272818276195, 0.027287279438196185, 0.9148627028403553],
    [0.4322636914587473, -0.008292997869433591, 0.900272242990724],
    [0.46758185665155916, -0.039240616376117615, 0.8818356399513159],
    [0.5046220619146033, -0.06612994872822489, 0.8600531121510557],
    [0.5423604777287174, -0.08964059239897773, 0.8352357460837865],
    [0.5800135236278517, -0.11036408928771599, 0.8075371011536572],
    [0.6170304843591228, -0.12869333227351, 0.777035853915857],
    [0.6530492512622788, -0.14481468265426684, 0.7438357974312979],
    [0.6878312925794869, -0.15879115629786375, 0.708145382500883],
    [0.7211945715146565, -0.17069749625584318, 0.6703066119397451],
    [0.7529612825727692, -0.18075226225800167, 0.6307602986825889],
    [0.7829316324698207, -0.18939270537930963, 0.5899564839232165],
    [0.8108869979393252, -0.19725513489332874, 0.5482387547348461],
    [0.8366176732823796, -0.20505217114689495, 0.5057432310505892],
    [0.859964068789123, -0.21337082777927924, 0.4623531240929103],
    [0.8808571116416533, -0.22244267005898988, 0.4177372843268643],
    [0.8993443439680942, -0.23195148369725727, 0.37147890808138817],
    [0.9155924352968428, -0.24094048514036054, 0.3232743016349142],
    [0.9298633028955575, -0.24786029816941169, 0.27315845587985776],
    [0.9424680252852653, -0.25076559768668943, 0.22170079993152325],
    [0.9537084491736223, -0.24763084111678224, 0.1701151796509376],
    [0.9638194363039794, -0.23672354810738638, 0.1202435704300373],
    [0.9729244187208099, -0.2169556100880869, 0.07440011252482165],
    [0.9810135438902164, -0.1881341178798301, 0.03509440202436909],
    [0.987948186510585, -0.15105338988136405, 0.004682624802950841],
    [0.9934894592013293, -0.10740456476798604, -0.014985449231401993],
    [0.9973431296794862, -0.059519868512043966, -0.0228515418608064],
    [0.9992102838057672, -0.010005484778035645, -0.018806669388064745],
    [0.9988327719443518, 0.038659146578893334, -0.0036890676835223957],
    [0.9960248012457311, 0.08447250839867435, 0.020875901585369228],
    [0.990686182532988, 0.1261438947481716, 0.05275805449015886],
    [0.9827974905407915, 0.16316920230375492, 0.08966577483931559],
    [0.9724014763633642, 0.19574859718182563, 0.12947864696273376],
    [0.9595775003638264, 0.22458102727431387, 0.17050366370933012],
    [0.9444160952610361, 0.25059588716226644, 0.21161051023494654],
    [0.926999213451773, 0.2746909442880491, 0.2522320007351945],
    [0.9073889797387367, 0.29753648855138504, 0.2922479325268073],
    [0.8856248616097713, 0.3194818829678534, 0.3317964244442485],
    [0.8617270456027556, 0.34056931189172973, 0.371070072071083],
    [0.8357030940885343, 0.36062935397607043, 0.41015220639928174],
    [0.807555750894855, 0.37941222388457474, 0.4489324241818712],
    [0.7772916182043683, 0.396702479827961, 0.4871152024217204],
    [0.7449324952165203, 0.4123748148498807, 0.524307899314948],
    [0.7105324855268182, 0.42637092008964467, 0.5601522002656647],
    [0.6742037832861655, 0.4386055418528862, 0.5944518853784275],
    [0.6361520526183972, 0.448835416290101, 0.6272523465314259],
    [0.5967188487230278, 0.4565402000102843, 0.6588424757552608],
    [0.5564244889989517, 0.46086514637189047, 0.6896728968891999],
    [0.5160014095998329, 0.46066061244683565, 0.7202094111211076],
    [0.4764065777734269, 0.45462736108879864, 0.7507600608276457],
    [0.4388028397427089, 0.44154617909582555, 0.7813230387048886],
    [0.40450336412199756, 0.42054404301639786, 0.811498423945787],
    [0.37487998820812657, 0.39133442311781097, 0.840490645062135],
    [0.35124396812186137, 0.3543707641829753, 0.8672050893497913],
    [0.33471462152407827, 0.31086985483645246, 0.8904178079577785],
    [0.32609587255870043, 0.26269143732852573, 0.9089784150350606],
];
