use crate::*;
use alloc::{vec, vec::Vec};
use core::f64::consts::{PI, TAU};
use ndarray::{array, s, Array1, Array2, Axis};
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

/// Apply Nyquist Frequency on [`fourier_power`] with a custom threshold value.
///
/// The threshold must in [0, 1).
/// This function return none if the curve is less than 1.
///
/// ```
/// use efd::fourier_power3;
///
/// # let curve = efd::tests::PATH3D;
/// let harmonic = fourier_power3(curve, None);
/// # assert_eq!(harmonic, Some(5));
/// ```
pub fn fourier_power3<C, T>(curve: C, threshold: T) -> Option<usize>
where
    C: AsRef<[[f64; 3]]>,
    T: Into<Option<f64>>,
{
    let curve = curve.as_ref();
    (curve.len() > 1)
        .then_some(curve.len() / 2)
        .and_then(|nyq| Efd3::from_curve_harmonic(curve, nyq))
        .map(|efd| fourier_power(efd.coeffs(), threshold))
}

/// 3D EFD implementation.
#[derive(Clone, Debug)]
pub struct Efd3 {
    coeffs: Array2<f64>,
    trans: Transform3,
}

impl Efd3 {
    const DIM: usize = 6;

    /// Create object from a nx4 array with boundary check.
    pub fn try_from_coeffs(coeffs: Array2<f64>) -> Result<Self, EfdError<{ Self::DIM }>> {
        (coeffs.nrows() > 0 && coeffs.ncols() == Self::DIM && coeffs[[0, 0]] == 1.)
            .then(|| Self { coeffs, trans: Transform3::identity() })
            .ok_or(EfdError::<{ Self::DIM }>)
    }

    /// Calculate EFD coefficients from an existing discrete points.
    ///
    /// **The curve must be closed. (first == last)**
    ///
    /// Return none if the curve length is less than 1.
    pub fn from_curve<'a, C>(curve: C) -> Option<Self>
    where
        C: Into<CowCurve3<'a>>,
    {
        Self::from_curve_gate(curve, None)
    }

    /// Calculate EFD coefficients from an existing discrete points and Fourier
    /// power gate.
    ///
    /// **The curve must be closed. (first == last)**
    ///
    /// Return none if the curve length is less than 1.
    pub fn from_curve_gate<'a, C, T>(curve: C, threshold: T) -> Option<Self>
    where
        C: Into<CowCurve3<'a>>,
        T: Into<Option<f64>>,
    {
        let curve = curve.into();
        let harmonic = fourier_power3(&curve, threshold)?;
        Self::from_curve_harmonic(curve, harmonic)
    }

    /// Calculate EFD coefficients from an existing discrete points.
    ///
    /// **The curve must be closed. (first == last)**
    ///
    /// Return none if harmonic is zero or the curve length is less than 1.
    ///
    /// If the harmonic number is not given, it will be calculated with
    /// [`fourier_power`] function.
    pub fn from_curve_harmonic<'a, C, H>(curve: C, harmonic: H) -> Option<Self>
    where
        C: Into<CowCurve3<'a>>,
        H: Into<Option<usize>>,
    {
        let curve = curve.into().into_owned();
        let harmonic = harmonic.into().or_else(|| fourier_power3(&curve, None))?;
        assert!(harmonic > 0);
        if curve.len() < 2 {
            return None;
        }
        let dxyz = diff(ndarray::arr2(&curve), Some(Axis(0)));
        let dt = dxyz.mapv(pow2).sum_axis(Axis(1)).mapv(f64::sqrt);
        let t = ndarray::concatenate![Axis(0), array![0.], cumsum(&dt, None)];
        let zt = *t.last().unwrap();
        let phi = &t * TAU / zt;
        let mut coeffs = Array2::zeros([harmonic, Self::DIM]);
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
            for i in 0..Self::DIM {
                c[i] = m[(i / 2, i % 2)];
            }
        }
        // The angle of semi-major axis
        let [roll, pitch, yaw] = {
            let [xs, xc, ys, yc, zs, zc] = [
                coeffs[[0, 0]],
                coeffs[[0, 1]],
                coeffs[[0, 2]],
                coeffs[[0, 3]],
                coeffs[[0, 4]],
                coeffs[[0, 5]],
            ];
            let a2 = (pow2(xc) + pow2(yc) + pow2(zc)) * pow2(theta.cos())
                + (pow2(xs) + pow2(ys) + pow2(zs)) * pow2(theta.sin())
                - (xc * xs + yc * ys + zc * zs) * (theta * 2.).sin();
            let b2 = (pow2(xc) + pow2(yc) + pow2(zc)) * pow2(theta.sin())
                + (pow2(xs) + pow2(ys) + pow2(zs)) * pow2(theta.cos())
                + (xc * xs + yc * ys + zc * zs) * (theta * 2.).sin();
            let a = a2.sqrt();
            let b = b2.sqrt();
            let o21 = (yc * theta.cos() - ys * theta.sin()) / a;
            let o31 = (zc * theta.cos() - zs * theta.sin()) / a;
            let o22 = (yc * theta.sin() + ys * theta.cos()) / b;
            let o32 = (zc * theta.sin() + zs * theta.cos()) / b;
            let w = a * b / (xc * zs - xs * zc);
            let roll = (yc * zs - ys * zc).atan2(xc * zs - xs * zc);
            let pitch = (w * (o21 * o31 + o22 * o32)).acos();
            let yaw = (o32 / pitch.sin()).acos();
            // FIXME: Angle fixes
            // let roll = if w > 0. { roll } else { roll + PI };
            // let yaw = if o31 > 0. { yaw } else { -yaw };
            [-roll, -pitch, -yaw]
        };
        let psi = na::Rotation3::from_euler_angles(roll, pitch, yaw);
        for mut c in coeffs.axis_iter_mut(Axis(0)) {
            let m = psi * na::matrix![c[0], c[1]; c[2], c[3]; c[4], c[5]];
            for i in 0..Self::DIM {
                c[i] = m[(i / 2, i % 2)];
            }
        }
        let scale = coeffs[[0, 0]].abs();
        coeffs /= scale;
        let trans = Transform3::new(center, [roll, pitch, yaw], scale);
        Some(Self { coeffs, trans })
    }

    /// Builder method for adding transform type.
    pub fn trans(self, trans: Transform3) -> Self {
        Self { trans, ..self }
    }

    /// Consume self and return a raw array.
    pub fn into_inner(self) -> Array2<f64> {
        self.coeffs
    }

    /// Get the array view of the coefficients.
    pub fn coeffs(&self) -> ndarray::ArrayView2<f64> {
        self.coeffs.view()
    }

    /// Get the reference of transform type.
    pub fn as_trans(&self) -> &Transform3 {
        self
    }

    /// Get the mutable reference of transform type.
    pub fn as_trans_mut(&mut self) -> &mut Transform3 {
        self
    }

    /// Get the harmonic of the coefficients.
    pub fn harmonic(&self) -> usize {
        self.coeffs.nrows()
    }

    /// Square error.
    pub fn square_err(&self, rhs: &Self) -> f64 {
        (&self.coeffs - &rhs.coeffs).mapv(pow2).sum()
    }

    /// L1 norm error, aka Manhattan distance.
    pub fn l1_norm(&self, rhs: &Self) -> f64 {
        (&self.coeffs - &rhs.coeffs).mapv(f64::abs).sum()
    }

    /// L2 norm error, aka Euclidean distance.
    pub fn l2_norm(&self, rhs: &Self) -> f64 {
        (&self.coeffs - &rhs.coeffs).mapv(pow2).sum().sqrt()
    }

    /// Lp norm error, slower than [`Self::l1_norm()`] and [`Self::l2_norm()`].
    pub fn lp_norm(&self, rhs: &Self, p: i32) -> f64 {
        (&self.coeffs - &rhs.coeffs)
            .mapv(|x| x.abs().powi(p))
            .sum()
            .powf(1. / p as f64)
    }

    /// Reverse the order of described curve then return a mutable reference.
    pub fn reverse(&mut self) -> &mut Self {
        for i in (1..Self::DIM).step_by(2) {
            let mut s = self.coeffs.slice_mut(s![.., i]);
            s *= -1.;
        }
        self
    }

    /// Consume and return a reversed version of the coefficients. This method
    /// can avoid mutable require.
    ///
    /// Please clone the object if you want to do self-comparison.
    pub fn reversed(mut self) -> Self {
        self.reverse();
        self
    }

    /// Generate the normalized curve **without** transformation.
    ///
    /// The number of the points `n` must larger than 3.
    pub fn generate_norm(&self, n: usize) -> Vec<[f64; 3]> {
        assert!(n > 3, "n ({}) must larger than 3", n);
        let mut t = Array1::from_elem(n, 1. / (n - 1) as f64);
        t[0] = 0.;
        let t = cumsum(t, None) * TAU;
        self.coeffs
            .axis_iter(Axis(0))
            .enumerate()
            .map(|(i, c)| {
                let angle = &t * (i + 1) as f64;
                let cos = angle.mapv(f64::cos);
                let sin = angle.mapv(f64::sin);
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

    /// Generate the described curve from the coefficients.
    ///
    /// The number of the points `n` must given.
    pub fn generate(&self, n: usize) -> Vec<[f64; 3]> {
        self.trans.transform(&self.generate_norm(n))
    }
}

impl std::ops::Deref for Efd3 {
    type Target = Transform3;

    fn deref(&self) -> &Self::Target {
        &self.trans
    }
}

impl std::ops::DerefMut for Efd3 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.trans
    }
}
