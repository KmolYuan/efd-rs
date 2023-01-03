use crate::*;
use alloc::vec;
use core::f64::consts::{PI, TAU};
use ndarray::{array, s, Array2, ArrayView1, Axis};
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

/// 2D EFD dimension type.
pub type D2 = [f64; 2];
/// 3D EFD dimension type.
pub type D3 = [f64; 3];

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

    /// Transform array slice to coordinate type.
    fn to_coord(a: ArrayView1<f64>) -> <Self::Trans as Trans>::Coord;
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
            for i in 0..DIM {
                let j = i / 2;
                c[i] = if i % 2 == 0 { &s_cos } else { &s_sin }
                    .slice(s![.., j])
                    .sum();
            }
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
        // Angle of starting point
        let theta = {
            let c = &coeffs;
            let dy = 2. * (c[[0, 0]] * c[[0, 1]] + c[[0, 2]] * c[[0, 3]]);
            let dx = pow2(c[[0, 0]]) + pow2(c[[0, 2]]) - pow2(c[[0, 1]]) - pow2(c[[0, 3]]);
            dy.atan2(dx) * 0.5
        };
        for (i, mut c) in coeffs.axis_iter_mut(Axis(0)).enumerate() {
            let theta = na::Rotation2::new((i + 1) as f64 * -theta);
            let m = theta * na::matrix![c[0], c[2]; c[1], c[3]];
            for i in 0..DIM {
                c[i] = m[(i % 2, i / 2)];
            }
        }
        // Normalize coefficients sign
        if harmonic > 1 && coeffs[[0, 0]] * coeffs[[1, 0]] < 0. {
            let mut s = coeffs.slice_mut(s![..;2, ..]);
            s *= -1.;
        }
        // Angle of semi-major axis
        let psi = {
            let u = na::Vector2::new(coeffs[[0, 0]], coeffs[[0, 2]]).normalize();
            let v = na::Vector2::new(coeffs[[0, 1]], coeffs[[0, 3]]).normalize();
            na::Rotation2::from_basis_unchecked(&[u, v])
        };
        let psi_inv = psi.inverse();
        for mut c in coeffs.axis_iter_mut(Axis(0)) {
            let m = psi_inv * na::matrix![c[0], c[1]; c[2], c[3]];
            for i in 0..DIM {
                c[i] = m[(i / 2, i % 2)];
            }
        }
        let scale = coeffs[[0, 0]].hypot(coeffs[[0, 2]]);
        coeffs /= scale;
        let trans = Transform::new(center, psi.into(), scale);
        (coeffs, trans)
    }

    fn to_coord(a: ArrayView1<f64>) -> <Self::Trans as Trans>::Coord {
        [a[0], a[1]]
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
            for i in 0..DIM {
                c[i] = if i % 2 == 0 { &s_cos } else { &s_sin }
                    .slice(s![.., i / 2])
                    .sum();
            }
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
        // Angle of starting point
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
            let theta = na::Rotation2::new((i + 1) as f64 * -theta);
            let m = theta * na::matrix![c[0], c[2], c[4]; c[1], c[3], c[5]];
            for i in 0..DIM {
                c[i] = m[(i % 2, i / 2)];
            }
        }
        // Normalize coefficients sign
        if harmonic > 1 && coeffs[[0, 0]] * coeffs[[1, 0]] < 0. {
            let mut s = coeffs.slice_mut(s![..;2, ..]);
            s *= -1.;
        }
        // Angle of semi-major axis
        let psi = {
            let u = na::Vector3::new(coeffs[[0, 0]], coeffs[[0, 2]], coeffs[[0, 4]]).normalize();
            let v = na::Vector3::new(coeffs[[0, 1]], coeffs[[0, 3]], coeffs[[0, 5]]).normalize();
            let uv = u.cross(&v);
            na::Rotation3::from_basis_unchecked(&[u, v, uv])
        };
        let psi_inv = psi.inverse();
        for mut c in coeffs.axis_iter_mut(Axis(0)) {
            let m = psi_inv * na::matrix![c[0], c[1]; c[2], c[3]; c[4], c[5]];
            for i in 0..DIM {
                c[i] = m[(i / 2, i % 2)];
            }
        }
        let scale = (pow2(coeffs[[0, 0]]) + pow2(coeffs[[0, 2]]) + pow2(coeffs[[0, 4]])).sqrt();
        coeffs /= scale;
        let trans = Transform::new(center, psi.into(), scale);
        (coeffs, trans)
    }

    fn to_coord(a: ArrayView1<f64>) -> <Self::Trans as Trans>::Coord {
        [a[0], a[1], a[2]]
    }
}
