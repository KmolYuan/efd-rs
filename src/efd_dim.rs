use crate::*;
use alloc::vec;
use core::f64::consts::{PI, TAU};
use ndarray::{array, s, Array1, Array2, ArrayView1, Axis};
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

/// 2D EFD dimension type.
pub type D2 = [f64; 2];
/// 3D EFD dimension type.
pub type D3 = [f64; 3];
type CoeffsTerm<'a> = (ArrayView1<'a, f64>, Array1<f64>, Array1<f64>);

/// Trait for EFD dimension.
pub trait EfdDim {
    /// Transform type.
    type Trans: Trans;

    /// Generate coefficients and transform.
    fn from_curve_harmonic<'a, C>(
        curve: C,
        harmonic: usize,
    ) -> (Array2<f64>, Transform<Self::Trans>)
    where
        C: Into<CowCurve<'a, Self::Trans>>;

    /// Generate coordinates.
    fn generate_norm<'a, I>(iter: I) -> Curve<Self::Trans>
    where
        I: Iterator<Item = CoeffsTerm<'a>>;
}

impl EfdDim for D2 {
    type Trans = T2;

    fn from_curve_harmonic<'a, C>(
        curve: C,
        harmonic: usize,
    ) -> (Array2<f64>, Transform<Self::Trans>)
    where
        C: Into<CowCurve<'a, Self::Trans>>,
    {
        const DIM: usize = T2::DIM * 2;
        let curve = curve.into();
        let dxy = diff(ndarray::arr2(&curve), Some(Axis(0)));
        let dt = dxy.mapv(pow2).sum_axis(Axis(1)).mapv(f64::sqrt);
        let t = ndarray::concatenate![Axis(0), array![0.], cumsum(&dt, None)];
        let zt = *t.last().unwrap();
        debug_assert!(zt != 0.);
        let phi = &t * TAU / zt;
        let mut coeffs = Array2::zeros([harmonic, DIM]);
        for (i, mut c) in coeffs.axis_iter_mut(Axis(0)).enumerate() {
            let n = i as f64 + 1.;
            let t = 0.5 * zt / (n * n * PI * PI);
            let phi_n = &phi * n;
            let phi_n_front = phi_n.slice(s![..-1]);
            let phi_n_back = phi_n.slice(s![1..]);
            let cos_phi_n = (phi_n_back.mapv(f64::cos) - phi_n_front.mapv(f64::cos)) / &dt;
            let sin_phi_n = (phi_n_back.mapv(f64::sin) - phi_n_front.mapv(f64::sin)) / &dt;
            let s_cos = t * (&dxy * cos_phi_n.insert_axis(Axis(1)));
            let s_sin = t * (&dxy * sin_phi_n.insert_axis(Axis(1)));
            c[0] = s_cos.slice(s![.., 1]).sum();
            c[1] = s_sin.slice(s![.., 1]).sum();
            c[2] = s_cos.slice(s![.., 0]).sum();
            c[3] = s_sin.slice(s![.., 0]).sum();
        }
        let center = {
            let tdt = &t.slice(s![1..]) / &dt;
            let xi = cumsum(dxy.slice(s![.., 0]), None) - &dxy.slice(s![.., 0]) * &tdt;
            let c = diff(t.mapv(pow2), None) * 0.5 / &dt;
            let a0 = (&dxy.slice(s![.., 0]) * &c + xi * &dt).sum() / zt;
            let xi = cumsum(dxy.slice(s![.., 1]), None) - &dxy.slice(s![.., 1]) * &tdt;
            let c0 = (&dxy.slice(s![.., 1]) * c + xi * dt).sum() / zt;
            let [x, y] = curve.first().unwrap();
            [x + a0, y + c0]
        };
        // Shift angle
        let theta = {
            let c = &coeffs;
            let dy = 2. * (c[[0, 0]] * c[[0, 1]] + c[[0, 2]] * c[[0, 3]]);
            let dx = pow2(c[[0, 0]]) + pow2(c[[0, 2]]) - pow2(c[[0, 1]]) - pow2(c[[0, 3]]);
            dy.atan2(dx) * 0.5
        };
        for (i, mut c) in coeffs.axis_iter_mut(Axis(0)).enumerate() {
            let theta = na::Rotation2::new((i + 1) as f64 * theta);
            let m = na::matrix![c[0], c[1]; c[2], c[3]] * theta;
            for i in 0..DIM {
                c[i] = m[(i / 2, i % 2)];
            }
        }
        // The angle of semi-major axis
        let psi = {
            let psi = coeffs[[0, 2]].atan2(coeffs[[0, 0]]);
            if coeffs[[0, 2]] < 0. {
                let mut s = coeffs.slice_mut(s![..;2, ..]);
                s *= -1.;
                -psi + PI
            } else {
                -psi
            }
        };
        let rot = na::Rotation2::new(psi);
        for mut c in coeffs.axis_iter_mut(Axis(0)) {
            let m = rot * na::matrix![c[0], c[1]; c[2], c[3]];
            for i in 0..DIM {
                c[i] = m[(i / 2, i % 2)];
            }
        }
        let scale = coeffs[[0, 0]].hypot(coeffs[[0, 2]]);
        coeffs /= scale;
        let trans = Transform::new(center, psi, scale);
        (coeffs, trans)
    }

    fn generate_norm<'a, I>(iter: I) -> Curve<Self::Trans>
    where
        I: Iterator<Item = CoeffsTerm<'a>>,
    {
        iter.map(|(c, cos, sin)| {
            let x = &cos * c[2] + &sin * c[3];
            let y = &cos * c[0] + &sin * c[1];
            ndarray::stack![Axis(1), x, y]
        })
        .reduce(|a, b| a + b)
        .unwrap()
        .axis_iter(Axis(0))
        .map(|c| [c[0], c[1]])
        .collect()
    }
}

impl EfdDim for D3 {
    type Trans = T3;

    fn from_curve_harmonic<'a, C>(
        curve: C,
        harmonic: usize,
    ) -> (Array2<f64>, Transform<Self::Trans>)
    where
        C: Into<CowCurve<'a, Self::Trans>>,
    {
        const DIM: usize = T3::DIM * 2;
        let curve = curve.into();
        let dxyz = diff(ndarray::arr2(&curve), Some(Axis(0)));
        let dt = dxyz.mapv(pow2).sum_axis(Axis(1)).mapv(f64::sqrt);
        let t = ndarray::concatenate![Axis(0), array![0.], cumsum(&dt, None)];
        let zt = *t.last().unwrap();
        let phi = &t * TAU / zt;
        let mut coeffs = Array2::zeros([harmonic, DIM]);
        for (i, mut c) in coeffs.axis_iter_mut(Axis(0)).enumerate() {
            let n = i as f64 + 1.;
            let t = 0.5 * zt / (n * n * PI * PI);
            let phi_n = &phi * n;
            let phi_n_front = phi_n.slice(s![..-1]);
            let phi_n_back = phi_n.slice(s![1..]);
            let cos_phi_n = (phi_n_back.mapv(f64::cos) - phi_n_front.mapv(f64::cos)) / &dt;
            let sin_phi_n = (phi_n_back.mapv(f64::sin) - phi_n_front.mapv(f64::sin)) / &dt;
            let s_cos = t * (&dxyz * cos_phi_n.insert_axis(Axis(1)));
            let s_sin = t * (&dxyz * sin_phi_n.insert_axis(Axis(1)));
            c[0] = s_cos.slice(s![.., 2]).sum();
            c[1] = s_sin.slice(s![.., 2]).sum();
            c[2] = s_cos.slice(s![.., 1]).sum();
            c[3] = s_sin.slice(s![.., 1]).sum();
            c[4] = s_cos.slice(s![.., 0]).sum();
            c[5] = s_sin.slice(s![.., 0]).sum();
        }
        let center = {
            let tdt = &t.slice(s![1..]) / &dt;
            let xi = cumsum(dxyz.slice(s![.., 0]), None) - &dxyz.slice(s![.., 0]) * &tdt;
            let c = diff(t.mapv(pow2), None) * 0.5 / &dt;
            let a0 = (&dxyz.slice(s![.., 0]) * &c + xi * &dt).sum() / zt;
            let xi = cumsum(dxyz.slice(s![.., 1]), None) - &dxyz.slice(s![.., 1]) * &tdt;
            let c0 = (&dxyz.slice(s![.., 1]) * &c + xi * &dt).sum() / zt;
            let xi = cumsum(dxyz.slice(s![.., 2]), None) - &dxyz.slice(s![.., 2]) * &tdt;
            let e0 = (&dxyz.slice(s![.., 2]) * &c + xi * &dt).sum() / zt;
            let [x, y, z] = curve.first().unwrap();
            [x + a0, y + c0, z + e0]
        };
        // Shift angle
        let theta = {
            let c = &coeffs;
            let dy = 2. * (c[[0, 0]] * c[[0, 1]] + c[[0, 2]] * c[[0, 3]] + c[[0, 4]] * c[[0, 5]]);
            let dx = pow2(c[[0, 0]]) + pow2(c[[0, 2]]) + pow2(c[[0, 4]])
                - pow2(c[[0, 1]])
                - pow2(c[[0, 3]])
                - pow2(c[[0, 5]]);
            dy.atan2(dx) * 0.5
        };
        for (i, mut c) in coeffs.axis_iter_mut(Axis(0)).enumerate() {
            let theta = na::Rotation2::new((i + 1) as f64 * theta);
            let m = na::matrix![c[0], c[1]; c[2], c[3]; c[4], c[5]] * theta;
            for i in 0..DIM {
                c[i] = m[(i / 2, i % 2)];
            }
        }
        // The angle of semi-major axis
        let psi = {
            let u = na::Vector3::new(coeffs[[0, 0]], coeffs[[0, 2]], coeffs[[0, 4]]);
            let v = na::Vector3::new(coeffs[[0, 1]], coeffs[[0, 3]], coeffs[[0, 5]]);
            let uv = u.cross(&v);
            let g_axis = na::Vector3::x();
            let angle = uv.dot(&g_axis);
            let axis = na::Unit::new_normalize(uv.cross(&g_axis));
            na::Rotation3::from_axis_angle(&axis, angle).inverse()
        };
        for mut c in coeffs.axis_iter_mut(Axis(0)) {
            let m = psi * na::matrix![c[0], c[1]; c[2], c[3]; c[4], c[5]];
            for i in 0..DIM {
                c[i] = m[(i / 2, i % 2)];
            }
        }
        let (roll, pitch, yaw) = psi.euler_angles();
        let scale = (pow2(coeffs[[0, 0]]) + pow2(coeffs[[0, 2]]) + pow2(coeffs[[0, 4]])).sqrt();
        coeffs /= scale;
        let trans = Transform::new(center, [roll, pitch, yaw], scale);
        (coeffs, trans)
    }

    fn generate_norm<'a, I>(iter: I) -> Curve<Self::Trans>
    where
        I: Iterator<Item = CoeffsTerm<'a>>,
    {
        iter.map(|(c, cos, sin)| {
            let x = &cos * c[4] + &sin * c[5];
            let y = &cos * c[2] + &sin * c[3];
            let z = &cos * c[0] + &sin * c[1];
            ndarray::stack![Axis(1), x, y, z]
        })
        .reduce(|a, b| a + b)
        .unwrap()
        .axis_iter(Axis(0))
        .map(|c| [c[0], c[1], c[2]])
        .collect()
    }
}
