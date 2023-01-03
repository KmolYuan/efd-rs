#![doc(hidden)]

/// Epsilon for curve difference.
pub const EPS: f64 = 2e-14;

#[test]
fn efd2d() {
    use crate::*;
    use alloc::vec::Vec;
    let efd = Efd2::from_curve(closed_curve(PATH)).unwrap();
    // Test starting point
    let path = PATH
        .iter()
        .cycle()
        .skip(0)
        .take(PATH.len())
        .copied()
        .collect::<Vec<_>>();
    let efd_half = Efd2::from_curve(closed_curve(path)).unwrap();
    assert!(efd.l1_norm(&efd_half) < EPS);
    // Test transformation
    let trans = efd.as_trans();
    assert!((trans.trans()[0] - -1.248409055632365).abs() < f64::EPSILON);
    assert!((trans.trans()[1] - 55.26080122817753).abs() < f64::EPSILON);
    assert!((trans.rot().angle() - 0.6423416350347734).abs() < f64::EPSILON);
    assert!((trans.scale() - 48.16765830752243).abs() < f64::EPSILON);
    assert_eq!(efd.harmonic(), 6);
    // Test normalized
    let norm = efd.generate_norm(NORM.len());
    assert!(curve_diff(&norm, NORM) < EPS);
    // Test reconstruction
    let target = efd.generate(TARGET.len());
    assert!(curve_diff(&target, TARGET) < EPS);
}

#[test]
fn efd3d() {
    use crate::*;
    use alloc::vec::Vec;
    let efd = Efd3::from_curve(closed_curve(PATH3D)).unwrap();
    // Test starting point
    let path = PATH3D
        .iter()
        .cycle()
        .skip(PATH3D.len() / 2)
        .take(PATH3D.len())
        .copied()
        .collect::<Vec<_>>();
    let efd_half = Efd3::from_curve(closed_curve(path)).unwrap();
    assert!(efd.l1_norm(&efd_half) < EPS);
    // Test transformation
    let trans = efd.as_trans();
    assert!((trans.trans()[0] - 0.7239345388499508).abs() < f64::EPSILON);
    assert!((trans.trans()[1] - 0.09100107896533066).abs() < f64::EPSILON);
    assert!((trans.trans()[2] - 0.49979194975846675).abs() < f64::EPSILON);
    assert!((trans.rot()[0] - 0.45053605532930807).abs() < f64::EPSILON);
    assert!((trans.rot()[1] - 0.117379763019881).abs() < f64::EPSILON);
    assert!((trans.rot()[2] - 0.8778257767051408).abs() < f64::EPSILON);
    assert!((trans.rot()[3] - 0.11252181936726507).abs() < f64::EPSILON);
    assert!((trans.scale() - 0.5629099155595344).abs() < f64::EPSILON);
    assert_eq!(efd.harmonic(), 5);
    // Test normalized
    let norm = efd.generate_norm(NORM3D.len());
    assert!(curve_diff(&norm, NORM3D) < EPS);
    // Test reconstruction
    let target = efd.generate(NORM3D.len());
    assert!(curve_diff(&target, TARGET3D) < EPS);
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
    [31.419145699345854, 91.2805058483037],
    [17.418517196895003, 90.07914804660189],
    [4.244753253462852, 87.41682783854023],
    [-8.646084287386032, 82.23649438254617],
    [-19.957195505252518, 73.7627420984368],
    [-30.122220256970994, 65.14867465329937],
    [-40.139331074319415, 55.23886845955355],
    [-46.644407123619665, 42.92448766942674],
    [-45.22660855424976, 31.348934230012294],
    [-36.17817108831737, 21.749151088169825],
    [-23.6421413007904, 15.858740317938064],
    [-10.894937602526888, 17.11650068817552],
    [1.5512941624695569, 23.24490544911555],
    [12.50152606929542, 31.415007924947002],
    [19.65312082539083, 43.267887138455805],
    [25.875267554207756, 55.408580357818295],
    [36.1418456145505, 63.69207605293409],
    [45.40004894143898, 73.34595105201666],
    [43.52580541936121, 85.41974003908112],
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
    [1.050227461832115, -0.0005073577983769166, -0.13398960704180338],
    [1.0365663892279688, 0.11202479370538022, -0.12920943637029925],
    [0.9985387261042709, 0.21655220069241807, -0.11574941019431241],
    [0.9394554038975063, 0.3073436974666612, -0.09455778534642632],
    [0.863733361867743, 0.3813671367500579, -0.0671851817392291],
    [0.7759154108881284, 0.438339343791014, -0.03571874495201449],
    [0.6798484173878049, 0.48001984641513157, -0.0026029584195464543],
    [0.5782565710894993, 0.509086732275941, 0.029652912294861188],
    [0.4727797531695805, 0.5280536517106608, 0.05882350301588712],
    [0.36436421780410705, 0.5386232686166925, 0.08327143728386198],
    [0.2537675646004872, 0.5416552350991711, 0.1021001890760615],
    [0.14192136602594801, 0.5376479647436108, 0.1150899736697141],
    [0.029987509593099895, 0.5274088955086382, 0.12242763603083293],
    [-0.08089317297524969, 0.5125104263048473, 0.12432587406132825],
    [-0.1899997857408892, 0.4952305630104117, 0.12068425570438383],
    [-0.29732771050499324, 0.4779168420305763, 0.11094695343337445],
    [-0.4035560622076156, 0.4619900455495208, 0.0942521358349081],
    [-0.509591912040993, 0.44700346847489253, 0.06986229290499582],
    [-0.6157860594040715, 0.4302064401732634, 0.037750067063213016],
    [-0.721084040896925, 0.4069054308129039, -0.0008660543998251788],
    [-0.8224520696979825, 0.3716262258318725, -0.04325366589831209],
    [-0.914862351496601, 0.31976613309760765, -0.0853164085997142],
    [-0.9919483217674182, 0.2492092840149331, -0.12218473559876296],
    [-1.0472080520244234, 0.16134914577614629, -0.14912378986483363],
    [-1.0754290710400665, 0.06113845964570715, -0.16252034047084116],
    [-1.0739085698472086, -0.043892399867188056, -0.16067000822063982],
    [-1.0430901039300515, -0.14535863244199035, -0.14413574864716353],
    [-0.9864181347243078, -0.23594674277999228, -0.11557407833095504],
    [-0.9094620409216115, -0.3109456859570823, -0.07908542226522275],
    [-0.8185923210243373, -0.3689382982886084, -0.039285621847133946],
    [-0.7196216735656646, -0.41151028070986695, -0.0003703127754654048],
    [-0.6168086010325761, -0.44217204717142017, 0.034570838383659694],
    [-0.51247135140439, -0.4649371534144925, 0.06382925227831715],
    [-0.40723495749800476, -0.48307208616302194, 0.08692306174024181],
    [-0.30071926656711423, -0.4984086627007443, 0.10413193060987246],
    [-0.19234939740731738, -0.5113502550218882, 0.1159266221677882],
    [-0.08197344767544906, -0.521415338708166, 0.1225089679481081],
    [0.029904640931878994, -0.5279619706190825, 0.12361359527026737],
    [0.1422824834463385, -0.5307009881064348, 0.11861256976708236],
    [0.2540407436877669, -0.5297414982739989, 0.10685072665584962],
    [0.36429542723138525, -0.5251566971721802, 0.08806341693125617],
    [0.47249777254884195, -0.5163045693432788, 0.06271327817845074],
    [0.5781895426448104, -0.5012801296066139, 0.032127647739706694],
    [0.6804956585746555, -0.47685093446477667, -0.0015996305811739825],
    [0.7775782029574534, -0.43904240930582833, -0.03589365555442686],
    [0.8663241321693103, -0.3842678254085763, -0.06806776431453265],
    [0.9424705090742189, -0.31065049011125145, -0.0956648297072257],
    [1.001210754612375, -0.21906625433536148, -0.11669356670332842],
    [1.0381364540280147, -0.11349652372783343, -0.12974038020083856],
    [1.050227461832115, -0.000507357798373131, -0.13398960704180335],
];
#[rustfmt::skip]
pub const TARGET3D: &[[f64; 3]] = &[
    [0.32609587255870043, 0.2626914373285255, 0.9089784150350606],
    [0.3268547667440557, 0.2006447789107254, 0.9240999186073894],
    [0.3398215882907302, 0.13921695337641185, 0.9301168120003069],
    [0.3637961809992804, 0.08197292623730663, 0.9271597557272415],
    [0.39680771345482674, 0.031192039732756076, 0.9161104760767289],
    [0.43645616610909616, -0.01232208858649042, 0.8982095707207425],
    [0.4802943599000913, -0.048994779530759286, 0.8746377125118077],
    [0.5261572576225173, -0.07993658214616023, 0.8462307169734731],
    [0.5723660708442766, -0.10633878128718932, 0.8134198004416184],
    [0.6177765102934463, -0.1290440690228071, 0.7763847861045775],
    [0.6616891173038449, -0.1484342433534878, 0.7353146371821315],
    [0.7036787858103951, -0.1646289257531412, 0.6906264666721025],
    [0.7434178119499643, -0.177849205668205, 0.6430196609146271],
    [0.780557606565413, -0.18872607857971385, 0.5933251851639376],
    [0.8147034829972577, -0.1983568827156778, 0.5422155458468796],
    [0.8454769292789076, -0.20802515969636387, 0.48992179466411356],
    [0.8726254006970922, -0.21865513085192767, 0.4361227387946469],
    [0.8961230152799462, -0.23020697257087502, 0.3801155010610441],
    [0.9162116004938502, -0.24127568849119485, 0.3212643019638336],
    [0.9333567302768732, -0.24910710630401478, 0.25959917009704875],
    [0.9481272520815593, -0.25010367169471404, 0.19634963612958195],
    [0.9610362229234867, -0.24071164320979943, 0.13418919119778105],
    [0.9723953945363863, -0.21842896948297813, 0.07704447398810205],
    [0.9822299014492795, -0.18260940627856298, 0.029464817187614845],
    [0.9902777027149166, -0.134790574665085, -0.004297039299638206],
    [0.9960687066541616, -0.07842533372765628, -0.02123103126954462],
    [0.9990526836145024, -0.01809289840317313, -0.020256355420060213],
    [0.9987321492897538, 0.04156292871826528, -0.0024367536760497854],
    [0.994759957150851, 0.09683792878889043, 0.029338650699324176],
    [0.9869784488585697, 0.14561684000773917, 0.07110149689947359],
    [0.9753998836909388, 0.18750536173332832, 0.1187377590389131],
    [0.9601472913478506, 0.2234658807610645, 0.16882185533106503],
    [0.9413840245778821, 0.25514169716326196, 0.21911657103881638],
    [0.9192572216118706, 0.28413920490123334, 0.2686338929430592],
    [0.8938685026522837, 0.31151641448656264, 0.317309565348475],
    [0.8652712777036212, 0.3376142785219678, 0.3654713084481941],
    [0.8334849071157906, 0.36221390475414134, 0.41333219854869513],
    [0.7985154498844352, 0.3848713635892616, 0.4607018320065073],
    [0.760379937021587, 0.40522433842879696, 0.5069986129860544],
    [0.719140886729536, 0.4231012749415605, 0.5515129973616152],
    [0.6749633061383077, 0.43837466827130955, 0.5937674698044937],
    [0.6282025727551652, 0.4506357991854506, 0.6337844947493076],
    [0.5795177009932198, 0.4588690161895701, 0.6721198201085565],
    [0.5299852545716668, 0.46132493955124143, 0.7096239651598779],
    [0.4811730376437313, 0.4557224802116117, 0.7470149031767032],
    [0.43512852728222756, 0.43977619791959494, 0.7844303253213716],
    [0.39425005422900167, 0.41190248889172887, 0.8211438664657181],
    [0.361037852249039, 0.37186497854665507, 0.8555702392493594],
    [0.3377593287053868, 0.3211169401722231, 0.8855738080822104],
    [0.32609587255870026, 0.2626914373285234, 0.9089784150350613],
];
