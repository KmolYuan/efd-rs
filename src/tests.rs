#![doc(hidden)]
#[cfg(test)]
use core::f64::consts::TAU;

/// Epsilon for curve difference.
pub const EPS: f64 = 1.6e-14;
pub const RES: usize = 1000;

#[test]
fn error() {
    use crate::*;
    use ndarray::*;
    let coeff = arr2(&[[10., 20., 20., 10.], [3., 4., 4., 3.]]);
    let a = Efd2::try_from_coeffs(coeff).unwrap();
    let coeff = arr2(&[[10., 20., 20., 10.]]);
    let b = Efd2::try_from_coeffs(coeff).unwrap();
    assert_eq!(a.square_err(&b), 50.);
    assert_eq!(b.square_err(&a), 50.);
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
    assert_abs_diff_eq!(trans.trans()[0], -1.248409055632365);
    assert_abs_diff_eq!(trans.trans()[1], 55.26080122817753);
    assert_abs_diff_eq!(trans.rot().angle(), 0.6423416350347734);
    assert_abs_diff_eq!(trans.scale(), 48.16765830752243);
    assert_eq!(efd.harmonic(), 6);
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
    let efd_half = Efd3::from_curve(path, false);
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

#[test]
#[cfg(feature = "std")]
fn plot() -> Result<(), Box<dyn std::error::Error>> {
    use crate::*;
    use ndarray::*;
    use plotters::prelude::*;

    pub fn bounding_box<'a>(pts: impl IntoIterator<Item = &'a [f64; 2]>) -> [f64; 4] {
        let [mut x_min, mut x_max] = [&f64::INFINITY, &-f64::INFINITY];
        let [mut y_min, mut y_max] = [&f64::INFINITY, &-f64::INFINITY];
        for [x, y] in pts {
            if x < x_min {
                x_min = x;
            }
            if x > x_max {
                x_max = x;
            }
            if y < y_min {
                y_min = y;
            }
            if y > y_max {
                y_max = y;
            }
        }
        let dx = (x_max - x_min).abs();
        let dy = (y_max - y_min).abs();
        if dx > dy {
            let cen = (y_min + y_max) * 0.5;
            let r = dx * 0.5;
            [*x_min, *x_max, cen - r, cen + r]
        } else {
            let cen = (x_min + x_max) * 0.5;
            let r = dy * 0.5;
            [cen - r, cen + r, *y_min, *y_max]
        }
    }

    let coeff = arr2(&[[12., 35., 35., 13.], [5., 21., 21., 5.], [1., 12., 12., 1.]]);
    let efd = Efd2::try_from_coeffs(coeff).unwrap();
    let path = efd.generate(360);
    let [x_min, x_max, y_min, y_max] = bounding_box(&path);
    let b = SVGBackend::new("test.svg", (1200, 1200));
    let root = b.into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .set_label_area_size(LabelAreaPosition::Left, (8).percent())
        .set_label_area_size(LabelAreaPosition::Bottom, (4).percent())
        .margin((8).percent())
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;
    let p0 = path[0];
    chart.draw_series([Circle::new((p0[0], p0[1]), 3, BLACK.filled())])?;
    for (p, color) in [((10., 0.), RED), ((0., 10.), BLUE)] {
        chart.draw_series(LineSeries::new([(0., 0.), p], color.stroke_width(10)))?;
    }
    let trans0 = efd.as_trans();
    let mut c0 = [0.; 2];
    for c in efd.coeffs().axis_iter(Axis(0)) {
        let m = na::matrix![c[0], c[1]; c[2], c[3]];
        let f = |t: f64| {
            let v = m * na::matrix![t.cos(); t.sin()];
            [v[0], v[1]]
        };
        let t = Array1::linspace(0., std::f64::consts::TAU, 100);
        let trans = trans0 * Transform2::new(c0, na::UnitComplex::new(0.), 1.);
        let ellipse = t.into_iter().map(f).map(|[x, y]| {
            let [x, y] = trans.transform_pt(&[x, y]);
            (x, y)
        });
        let p1 = c0;
        c0[0] += c[0];
        c0[1] += c[2];
        let p2 = c0;
        let [x1, y1] = trans0.transform_pt(&p1);
        let [x2, y2] = trans0.transform_pt(&p2);
        chart.draw_series([Circle::new((x2, y2), 5, RED.filled())])?;
        chart.draw_series(LineSeries::new([(x1, y1), (x2, y2)], RED.stroke_width(5)))?;
        chart.draw_series(LineSeries::new(ellipse, RED.stroke_width(7)))?;
    }
    chart.draw_series(LineSeries::new(
        path.into_iter().map(|[x, y]| (x, y)),
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
    [1.119379152671337, -0.05553234647386943],
    [0.9910207666952372, 0.19246492442111932],
    [0.7433455516363754, 0.3466236207064233],
    [0.4912450590109462, 0.4662127118631491],
    [0.21253057838553358, 0.5404258318025645],
    [-0.08088496699201134, 0.5402447537989888],
    [-0.35699414927505113, 0.5234770691714837],
    [-0.6467599401741159, 0.483330568432713],
    [-0.9080511096122624, 0.3595326487685509],
    [-1.0284506874180428, 0.14947774345139578],
    [-0.9974323951564845, -0.12263773651284177],
    [-0.8623052545037516, -0.3764679704172556],
    [-0.6347643368252188, -0.5140999552626939],
    [-0.35164920002427524, -0.5670235417766133],
    [-0.06800914369245911, -0.5674017234826676],
    [0.19828936299781613, -0.45931628321452417],
    [0.4527171824458201, -0.3348867313490356],
    [0.7264030988252986, -0.32487721086570215],
    [1.0003704310053017, -0.27954637306118696],
    [1.1193791526713373, -0.0555323464738718],
];
pub const TARGET: &[[f64; 2]] = &[
    [43.52580541936114, 85.41974003908122],
    [31.41914569934586, 91.2805058483037],
    [17.418517196895017, 90.07914804660189],
    [4.244753253462857, 87.41682783854023],
    [-8.646084287386033, 82.23649438254617],
    [-19.957195505252518, 73.7627420984368],
    [-30.122220256970994, 65.14867465329937],
    [-40.13933107431942, 55.23886845955356],
    [-46.64440712361967, 42.92448766942674],
    [-45.226608554249765, 31.348934230012294],
    [-36.17817108831738, 21.749151088169825],
    [-23.642141300790406, 15.858740317938057],
    [-10.894937602526898, 17.11650068817552],
    [1.5512941624695515, 23.24490544911555],
    [12.501526069295421, 31.415007924947],
    [19.653120825390836, 43.267887138455805],
    [25.87526755420776, 55.40858035781829],
    [36.1418456145505, 63.692076052934084],
    [45.40004894143899, 73.34595105201664],
    [43.52580541936122, 85.41974003908112],
];
#[rustfmt::skip]
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
    [1.050227461832115, -0.0005073577983768646, -0.13398960704180327],
    [1.0409654686697103, 0.09178711336572129, -0.1307538079117006],
    [1.0151358207877603, 0.17957003152192352, -0.12164579414882525],
    [0.9743071746687448, 0.2593568407636886, -0.10709090301194578],
    [0.9206966963485949, 0.32878034763394554, -0.08778522995837311],
    [0.8568177382307565, 0.3867525192720771, -0.06468731553153984],
    [0.7851237349711094, 0.4333704468065942, -0.038981562989699896],
    [0.7077244002261753, 0.46961236789347405, -0.012005149889308918],
    [0.6262274495758019, 0.49691767664275016, 0.014859439825600035],
    [0.5417236047369598, 0.516766679700503, 0.04031217608400317],
    [0.45489434942195395, 0.5303663107866715, 0.06325974426131492],
    [0.3661912321448904, 0.53851039737721, 0.08290855901195963],
    [0.27602065193916314, 0.541628235088426, 0.09880259572341112],
    [0.1848727709368075, 0.5399786015444934, 0.11079353595558838],
    [0.09335575109573871, 0.533903742220683, 0.11894731514179559],
    [0.0021301939049129465, 0.5240411621089526, 0.12340870306719041],
    [-0.08822638949837364, 0.5114049500433687, 0.12425915654852235],
    [-0.1773693144620507, 0.49728904303715965, 0.1214086404152293],
    [-0.2652858159600484, 0.4830009999054961, 0.11455709426191857],
    [-0.3522895916048327, 0.4694907273012961, 0.1032460787553643],
    [-0.4388854142693799, 0.4569782286751572, 0.08699897901594796],
    [-0.5255170490386676, 0.4446960441414569, 0.06552419306030427],
    [-0.612249822990643, 0.4308412852616422, 0.03893606093979707],
    [-0.6984659656318325, 0.4127834630262383, 0.007938271365220547],
    [-0.7826581252568904, 0.38750978733456737, -0.026082636760474263],
    [-0.8623911241548294, 0.3522259920018013, -0.06109010945056742],
    [-0.934466869107598, 0.3049849151931597, -0.09456725232600907],
    [-0.9952805851534358, 0.24519965065424193, -0.12380365721923452],
    [-1.0413097289060818, 0.17391799819341328, -0.14625224871875642],
    [-1.0696419801272443, 0.09378601479425197, -0.1598909253082369],
    [-1.0784347992690009, 0.00869843554963011, -0.16351958914217016],
    [-1.0672102114892057, -0.07679464425725122, -0.15693424473875286],
    [-1.0369225035447835, -0.15819993583878375, -0.14094395727640754],
    [-0.9897854074771137, -0.2317614320795398, -0.11722825112520756],
    [-0.9288973440013026, -0.2949689879669055, -0.08806481527363814],
    [-0.8577458800052444, -0.3467981284504949, -0.05598286269131593],
    [-0.7796955924479353, -0.38764921800323904, -0.02341049582587853],
    [-0.6975618570473215, -0.4190242125634186, 0.007617732003245507],
    [-0.6133477491688839, -0.44303680455539, 0.03564330718689269],
    [-0.528179185281801, -0.4618828957483251, 0.05983305651727803],
    [-0.4424255885847663, -0.47739618687357654, 0.07990216446049853],
    [-0.355951811705189, -0.49078025683215, 0.09594453533454778],
    [-0.26842188018537266, -0.5025539177856181, 0.10822144732036237],
    [-0.17957186643771178, -0.512686240169666, 0.11696387505846045],
    [-0.08938751750660723, -0.5208478814548785, 0.12223515082761306],
    [0.0018438463983360028, -0.5266793173670364, 0.12388170172104106],
    [0.09359807481392532, -0.5299805519747676, 0.12157565128588461],
    [0.1852523206130316, -0.5307587706425018, 0.11493021902713921],
    [0.2762338250563979, -0.5291204033300674, 0.10365253843125691],
    [0.36611678527514374, -0.5250474775511501, 0.08769220733247889],
    [0.45462399086908445, -0.5181394303395048, 0.06734830844621126],
    [0.5415223714184689, -0.5074186537949862, 0.04331076524364066],
    [0.6264377339938841, -0.49128569824758883, 0.01662964744166088],
    [0.7086432939411135, -0.46767123308521324, -0.011376528242117365],
    [0.7868905393204754, -0.43437700776740995, -0.03924978397692458],
    [0.8593452256260783, -0.38954222756529633, -0.06553011711480981],
    [0.923667041268881, -0.3321305420651121, -0.08887549669282066],
    [0.9772350358236104, -0.26231807837894955, -0.10814778433707171],
    [1.0174820194028231, -0.18167946685619532, -0.12246013098691282],
    [1.042270367001783, -0.09311304849656796, -0.13119259338088846],
    [1.050227461832115, -0.0005073577983699174, -0.13398960704180324],
];
#[rustfmt::skip]
pub const TARGET3D: &[[f64; 3]] = &[
    [0.3260958725587006, 0.26269143732852546, 0.9089784150350606],
    [0.3257814466312374, 0.21209446971312584, 0.9219980701322935],
    [0.3337059121260934, 0.16142009702490295, 0.928978947539205],
    [0.34935141047772084, 0.11276885734180062, 0.929857444661391],
    [0.37181016319209154, 0.06774034427097272, 0.924956117867235],
    [0.39989272818276217, 0.027287279438196144, 0.9148627028403553],
    [0.4322636914587476, -0.008292997869433702, 0.900272242990724],
    [0.46758185665155927, -0.03924061637611756, 0.881835639951316],
    [0.5046220619146036, -0.06612994872822495, 0.8600531121510557],
    [0.5423604777287174, -0.08964059239897765, 0.8352357460837865],
    [0.5800135236278519, -0.1103640892877159, 0.8075371011536572],
    [0.6170304843591228, -0.12869333227350993, 0.7770358539158572],
    [0.6530492512622788, -0.14481468265426667, 0.7438357974312979],
    [0.6878312925794869, -0.15879115629786347, 0.7081453825008831],
    [0.7211945715146565, -0.17069749625584285, 0.6703066119397452],
    [0.7529612825727691, -0.18075226225800134, 0.630760298682589],
    [0.7829316324698207, -0.18939270537930952, 0.5899564839232165],
    [0.8108869979393252, -0.19725513489332852, 0.5482387547348462],
    [0.8366176732823796, -0.20505217114689467, 0.5057432310505892],
    [0.8599640687891228, -0.21337082777927896, 0.46235312409291035],
    [0.8808571116416533, -0.2224426700589896, 0.4177372843268643],
    [0.8993443439680942, -0.2319514836972571, 0.3714789080813882],
    [0.9155924352968426, -0.24094048514036037, 0.32327430163491433],
    [0.9298633028955572, -0.24786029816941146, 0.27315845587985815],
    [0.9424680252852651, -0.2507655976866892, 0.22170079993152336],
    [0.9537084491736221, -0.24763084111678213, 0.17011517965093786],
    [0.9638194363039793, -0.23672354810738627, 0.12024357043003764],
    [0.9729244187208097, -0.21695561008808678, 0.07440011252482198],
    [0.9810135438902163, -0.18813411787982998, 0.03509440202436931],
    [0.9879481865105851, -0.1510533898813638, 0.004682624802950952],
    [0.9934894592013291, -0.10740456476798604, -0.014985449231401882],
    [0.9973431296794861, -0.059519868512043966, -0.02285154186080629],
    [0.999210283805767, -0.010005484778035506, -0.018806669388064745],
    [0.9988327719443519, 0.03865914657889342, -0.0036890676835223957],
    [0.9960248012457309, 0.08447250839867478, 0.02087590158536956],
    [0.9906861825329877, 0.1261438947481719, 0.05275805449015908],
    [0.982797490540791, 0.16316920230375553, 0.0896657748393162],
    [0.9724014763633637, 0.19574859718182602, 0.12947864696273448],
    [0.9595775003638258, 0.22458102727431453, 0.17050366370933095],
    [0.9444160952610356, 0.250595887162267, 0.2116105102349476],
    [0.9269992134517724, 0.2746909442880495, 0.25223200073519547],
    [0.9073889797387362, 0.2975364885513856, 0.29224793252680814],
    [0.8856248616097705, 0.31948188296785407, 0.33179642444424984],
    [0.8617270456027546, 0.34056931189173023, 0.37107007207108433],
    [0.835703094088533, 0.360629353976071, 0.41015220639928335],
    [0.8075557508948537, 0.37941222388457524, 0.44893242418187274],
    [0.7772916182043672, 0.3967024798279615, 0.4871152024217215],
    [0.7449324952165187, 0.41237481484988114, 0.5243078993149495],
    [0.7105324855268165, 0.426370920089645, 0.5601522002656664],
    [0.6742037832861633, 0.4386055418528866, 0.5944518853784293],
    [0.6361520526183954, 0.4488354162901014, 0.6272523465314274],
    [0.5967188487230254, 0.4565402000102843, 0.6588424757552626],
    [0.5564244889989491, 0.46086514637189036, 0.6896728968892016],
    [0.5160014095998301, 0.46066061244683526, 0.7202094111211096],
    [0.47640657777342454, 0.45462736108879775, 0.7507600608276475],
    [0.4388028397427063, 0.441546179095824, 0.7813230387048908],
    [0.40450336412199545, 0.4205440430163959, 0.8114984239457887],
    [0.37487998820812446, 0.39133442311780825, 0.8404906450621371],
    [0.3512439681218598, 0.35437076418297175, 0.8672050893497936],
    [0.33471462152407705, 0.3108698548364479, 0.8904178079577805],
    [0.3260958725587002, 0.26269143732852174, 0.9089784150350618],
];
