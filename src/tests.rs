use ndarray::{arr2, Array2};

use crate::{efd_fitting, ElementWiseOpt};

thread_local! {
    static PATH: Array2<f64> = arr2(&[
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
    ]);
    static TARGET: Array2<f64> = arr2(&[
        [39.35863965131904, 83.29742588206082],
        [28.06426978660421, 90.17530230145958],
        [14.322369428735222, 89.31456476623697],
        [1.644141648731889, 84.80316243236997],
        [-10.594878257867837, 78.05523268450065],
        [-21.481011313772758, 69.6418738549989],
        [-31.51422384342044, 60.88040361398968],
        [-40.927190098600846, 50.42944632397454],
        [-45.28855318333383, 38.500182501760776],
        [-41.441192886353605, 27.955187648838773],
        [-32.07610003945156, 19.668006412715144],
        [-20.786321365878784, 15.481233369294323],
        [-9.16297124779308, 17.901961900785633],
        [1.8680500961438629, 24.16562879325819],
        [11.095844581723647, 32.082393132610164],
        [17.236398712404707, 43.47293538379268],
        [22.444136258469218, 55.055717188525996],
        [31.20682510176712, 62.82443436686037],
        [40.133214266242625, 71.61536010397093],
        [39.35863965131904, 83.29742588206082],
    ]);
}

#[test]
fn efd() {
    let ans = PATH.with(|path| efd_fitting(path, 20, None));
    TARGET.with(|target| {
        let err = (ans - target).abs().sum();
        assert!(err < 1e-12, "{}", err)
    });
}
