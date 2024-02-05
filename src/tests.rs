#![doc(hidden)]
use crate::*;
#[cfg(test)]
use alloc::{vec, vec::Vec};
use core::iter::zip;

/// Epsilon for curve difference.
pub const EPS: f64 = 2.2e-14;
pub const RES: usize = 1000;

/// Error between two curves, the length of the curves must be the same.
pub fn curve_diff<const D: usize>(a: impl Curve<D>, b: impl Curve<D>) -> f64 {
    zip(a.as_curve(), b.as_curve())
        .map(|(a, b)| a.l2_norm(b))
        .sum::<f64>()
        / a.len() as f64
}

#[test]
fn error() {
    let coeff = vec![
        na::matrix![10., 20.; 20., 10.],
        na::matrix![ 3., 4.; 4., 3.],
    ];
    let a = Efd2::try_from_coeffs_unnorm(coeff).unwrap();
    let coeff = vec![na::matrix![10., 20.; 20., 10.]];
    let b = Efd2::try_from_coeffs_unnorm(coeff).unwrap();
    assert_eq!(a.square_err(&b), 50.);
    assert_eq!(b.square_err(&a), 50.);
}

#[test]
fn efd2d() {
    use approx::assert_abs_diff_eq;
    let efd = Efd2::from_curve(CURVE2D, false);
    assert!(!efd.is_open());
    // Test starting point
    let curve = CURVE2D
        .iter()
        .cycle()
        .skip(CURVE2D.len() / 2)
        .take(CURVE2D.len())
        .copied()
        .collect::<Vec<_>>();
    let efd_half = Efd2::from_curve(curve, false);
    assert_abs_diff_eq!(efd.l1_norm(&efd_half), 0., epsilon = 1e-12);
    assert_eq!(efd.harmonic(), 8);
    // Test rotation
    for ang in 0..6 {
        let ang = core::f64::consts::TAU * ang as f64 / 6.;
        let curve = GeoVar::from_rot(na::UnitComplex::new(ang)).transform(CURVE2D);
        let efd_rot = Efd2::from_curve_harmonic(curve, false, efd.harmonic());
        assert_abs_diff_eq!(efd_rot.l1_norm(&efd), 0., epsilon = 1e-12);
    }
    // Test transformation
    let geo = efd.as_geo();
    assert_abs_diff_eq!(geo.trans()[0], -1.248409055632358);
    assert_abs_diff_eq!(geo.trans()[1], 55.26080122817753);
    assert_abs_diff_eq!(geo.rot().angle(), -2.49925101855502);
    assert_abs_diff_eq!(geo.scale(), 48.16765830752243);
    // Test reconstruction
    let (pos, _) = get_target_pos(CURVE2D, false);
    let target = efd.generate_norm_by(&pos);
    let curve = efd.as_geo().inverse().transform(CURVE2D);
    assert!(curve_diff(target, curve) < 1.6565);
}

#[test]
fn efd2d_open() {
    use approx::assert_abs_diff_eq;
    let efd = Efd2::from_curve(CURVE2D_OPEN, true);
    assert!(efd.is_open());
    assert_eq!(efd.harmonic(), 14);
    // Test transformation
    let geo = efd.as_geo();
    assert_abs_diff_eq!(geo.trans()[0], 43.03456791427352);
    assert_abs_diff_eq!(geo.trans()[1], 48.107208358019015);
    assert_abs_diff_eq!(geo.rot().angle(), 2.7330524299596815);
    assert_abs_diff_eq!(geo.scale(), 33.930916934329495);
    // Test reconstruction
    let (pos, _) = get_target_pos(CURVE2D_OPEN, true);
    let target = efd.generate_norm_by(&pos);
    let curve = efd.as_geo().inverse().transform(CURVE2D_OPEN);
    assert!(curve_diff(target, curve) < 0.0143);
}

#[test]
fn efd3d() {
    use approx::assert_abs_diff_eq;
    let efd = Efd3::from_curve(CURVE3D, false);
    assert!(!efd.is_open());
    // Test starting point
    let curve = CURVE3D
        .iter()
        .cycle()
        .skip(CURVE3D.len() / 2)
        .take(CURVE3D.len())
        .copied()
        .collect::<Vec<_>>();
    let efd_half = Efd3::from_curve(curve, false);
    assert_abs_diff_eq!(efd.l1_norm(&efd_half), 0., epsilon = 1e-12);
    assert_eq!(efd.harmonic(), 5);
    // Test rotation
    for ang in 0..6 {
        let ang = core::f64::consts::TAU * ang as f64 / 6.;
        let curve = GeoVar::from_rot(na::UnitQuaternion::new(na::matrix![1.; 1.; 0.] * ang))
            .transform(CURVE3D);
        let efd_rot = Efd3::from_curve_harmonic(curve, false, efd.harmonic());
        assert_abs_diff_eq!(efd.l1_norm(&efd_rot), 0., epsilon = 1e-12);
    }
    // Test transformation
    let geo = efd.as_geo();
    assert_abs_diff_eq!(geo.trans()[0], 0.7239345388499508);
    assert_abs_diff_eq!(geo.trans()[1], 0.09100107896533066);
    assert_abs_diff_eq!(geo.trans()[2], 0.49979194975846675);
    assert_abs_diff_eq!(geo.rot().angle(), 2.9160714030359416);
    assert_abs_diff_eq!(geo.scale(), 0.5629099155595344);
    // Test reconstruction
    let (pos, _) = get_target_pos(CURVE3D, false);
    let target = efd.generate_norm_by(&pos);
    let curve = efd.as_geo().inverse().transform(CURVE3D);
    assert!(curve_diff(target, curve) < 0.0042);
}

#[test]
fn posed_efd_open() {
    use approx::assert_abs_diff_eq;
    let efd = PosedEfd2::from_angles(CURVE2D_POSE, ANGLE2D_POSE, true);
    assert!(efd.is_open());
    assert_eq!(efd.harmonic(), 16);
    // Test rotation
    for ang in 0..6 {
        let ang = core::f64::consts::TAU * ang as f64 / 6.;
        let curve = GeoVar::from_rot(na::UnitComplex::new(ang)).transform(CURVE2D_POSE);
        let angles = ANGLE2D_POSE.iter().map(|a| a + ang).collect::<Vec<_>>();
        let efd_rot = PosedEfd2::from_angles(curve, &angles, true);
        assert_abs_diff_eq!(efd.l1_norm(&efd_rot), 0., epsilon = 1e-12);
    }
}

#[test]
#[cfg(feature = "std")]
fn plot2d_closed() -> Result<(), Box<dyn std::error::Error>> {
    let coeff = vec![
        na::matrix![12., 35.; 35., 13.],
        na::matrix![5., 21.; 21., 5.],
        na::matrix![1., 12.; 12., 1.],
    ];
    plot2d(coeff, "img/2d.svg")
}

#[test]
#[cfg(feature = "std")]
fn plot2d_open() -> Result<(), Box<dyn std::error::Error>> {
    let coeff = vec![
        na::matrix![35., 0.; 8., 0.],
        na::matrix![10., 0.; 24., 0.],
        na::matrix![5., 0.; -8., 0.],
    ];
    plot2d(coeff, "img/2d_open.svg")
}

#[test]
#[cfg(feature = "std")]
fn plot3d_closed() -> Result<(), Box<dyn std::error::Error>> {
    let coeff = vec![
        na::matrix![12., 22.; 35., 5.; 20., 21.],
        na::matrix![21., 12.; 5., 12.; 1., 1.],
        na::matrix![3., 3.; 7., 5.; 12., 21.],
    ];
    plot3d(coeff, "img/3d.svg")
}

#[test]
#[cfg(feature = "std")]
fn plot3d_open() -> Result<(), Box<dyn std::error::Error>> {
    let coeff = vec![
        na::matrix![16., 0.; 35., 0.; 27., 0.],
        na::matrix![21., 0.; 8., 0.; 16., 0.],
        na::matrix![3., 0.; 7., 0.; 12., 0.],
    ];
    plot3d(coeff, "img/3d_open.svg")
}

#[cfg(all(test, feature = "std"))]
fn get_area<const D: usize>(pts: &[[f64; D]]) -> [[f64; 2]; D] {
    pts.iter()
        .fold([[f64::INFINITY, f64::NEG_INFINITY]; D], |mut b, c| {
            zip(&mut b, c).for_each(|(b, c)| *b = [b[0].min(*c), b[1].max(*c)]);
            b
        })
}

#[cfg(all(test, feature = "std"))]
fn plot2d(coeff: Coeffs2, path: &str) -> Result<(), Box<dyn std::error::Error>> {
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
        zip(&mut c0, &m.column(0)).for_each(|(c, u)| *c += u);
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

#[cfg(all(test, feature = "std"))]
fn plot3d(coeff: Coeffs3, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    use plotters::prelude::*;

    fn bounding_box(pts: &[[f64; 3]]) -> [f64; 6] {
        let mut b = get_area(pts);
        let center = b.map(|[min, max]| (min + max) * 0.5);
        let width = b
            .iter()
            .map(|[min, max]| (max - min).abs())
            .fold(0., f64::max);
        for ([min, max], c) in zip(&mut b, center) {
            *min = c - width * 0.5;
            *max = c + width * 0.5;
        }
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
        zip(&mut c0, &m.column(0)).for_each(|(c, u)| *c += u);
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

pub const CURVE2D: &[[f64; 2]] = &[
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
pub const CURVE2D_OPEN: &[[f64; 2]] = &[
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
#[rustfmt::skip]
pub const CURVE3D: &[[f64; 3]] = &[
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
pub const CURVE2D_POSE: &[[f64; 2]] = &[
    [18.8, 12.1],
    [13.3, 18.1],
    [6.3, 19.8],
    [-0.4, 17.1],
    [-2.7, 10.3],
    [-1.1, 6.0],
    [0.2, 1.7],
    [3.4, -2.2],
    [7.8, -4.9],
];
pub const ANGLE2D_POSE: &[f64] = &[-0.9, 0., 0.7, 1.5, 2.8, -2.3, -2., -1.9, -2.1];
