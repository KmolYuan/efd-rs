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
    /// Create constant object from a nx4 array without boundary check.
    ///
    /// # Safety
    ///
    /// An invalid width may cause failure operation.
    pub unsafe fn from_coeffs_unchecked(coeffs: Array2<f64>) -> Self {
        Self { coeffs, trans: Transform3::identity() }
    }

    /// Create object from a nx4 array with boundary check.
    pub fn try_from_coeffs(coeffs: Array2<f64>) -> Result<Self, Efd3Error> {
        (coeffs.nrows() > 0 && coeffs.ncols() == 4 && coeffs[[0, 0]] == 1.)
            .then(|| Self { coeffs, trans: Transform3::identity() })
            .ok_or(Efd3Error(()))
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

    /// FIXME: Calculate EFD coefficients from an existing discrete points.
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
        // FIXME: Remove type annotation
        let curve: Vec<[f64; 3]> = curve.into().into_owned();
        let harmonic = harmonic.into().or_else(|| fourier_power3(&curve, None))?;
        assert!(harmonic > 0);
        if curve.len() < 2 {
            return None;
        }
        let dxyz = diff(ndarray::arr2(&curve), Some(Axis(0)));
        let dt = dxyz.mapv(pow2).sum_axis(Axis(1)).mapv(f64::sqrt);
        let t = ndarray::concatenate![Axis(0), array![0.], cumsum(&dt, None)];
        let zt = t.last().unwrap();
        let phi = &t * TAU / (zt + f64::EPSILON);
        let mut coeffs = Array2::zeros([harmonic, 6]);
        for (i, mut c) in coeffs.axis_iter_mut(Axis(0)).enumerate() {
            let n = i as f64 + 1.;
            let t = 0.5 * zt / (n * n * PI * PI);
            let phi_n = &phi * n;
            let phi_n_front = phi_n.slice(s![..-1]);
            let phi_n_back = phi_n.slice(s![1..]);
            let cos_phi_n = (phi_n_back.mapv(f64::cos) - phi_n_front.mapv(f64::cos)) / &dt;
            let sin_phi_n = (phi_n_back.mapv(f64::sin) - phi_n_front.mapv(f64::sin)) / &dt;
            c[0] = t * (&dxyz.slice(s![.., 1]) * &cos_phi_n).sum();
            c[1] = t * (&dxyz.slice(s![.., 1]) * &sin_phi_n).sum();
            c[2] = t * (&dxyz.slice(s![.., 0]) * &cos_phi_n).sum();
            c[3] = t * (&dxyz.slice(s![.., 0]) * &sin_phi_n).sum();
            c[4] = t * (&dxyz.slice(s![.., 2]) * &cos_phi_n).sum();
            c[5] = t * (&dxyz.slice(s![.., 2]) * &sin_phi_n).sum();
        }
        let center = {
            let tdt = &t.slice(s![1..]) / &dt;
            let xi = cumsum(dxyz.slice(s![.., 0]), None) - &dxyz.slice(s![.., 0]) * &tdt;
            let c = diff(t.mapv(pow2), None) * 0.5 / &dt;
            let a0 = (&dxyz.slice(s![.., 0]) * &c + xi * &dt).sum() / (zt + f64::EPSILON);
            let xi = cumsum(dxyz.slice(s![.., 1]), None) - &dxyz.slice(s![.., 1]) * &tdt;
            let c0 = (&dxyz.slice(s![.., 1]) * &c + xi * &dt).sum() / (zt + f64::EPSILON);
            let xi = cumsum(dxyz.slice(s![.., 2]), None) - &dxyz.slice(s![.., 2]) * &tdt;
            let e0 = (&dxyz.slice(s![.., 2]) * &c + xi * &dt).sum() / (zt + f64::EPSILON);
            let [x, y, z] = curve.first().unwrap();
            [x + a0, y + c0, z + e0]
        };
        // FIXME: Shift angle
        let theta1 = {
            let dy = 2. * (coeffs[[0, 0]] * coeffs[[0, 1]] + coeffs[[0, 2]] * coeffs[[0, 3]]);
            let dx = pow2(coeffs[[0, 0]]) - pow2(coeffs[[0, 1]]) + pow2(coeffs[[0, 2]])
                - pow2(coeffs[[0, 3]]);
            dy.atan2(dx) * 0.5
        };
        for (i, mut c) in coeffs.axis_iter_mut(Axis(0)).enumerate() {
            let angle = (i + 1) as f64 * theta1;
            let rot = array![[angle.cos(), -angle.sin()], [angle.sin(), angle.cos()]];
            let m = array![[c[0], c[1]], [c[2], c[3]], [c[4], c[5]]].dot(&rot);
            c.assign(&Array1::from_iter(m));
        }
        // FIXME: The angle of semi-major axis
        let psi = {
            let psi = coeffs[[0, 2]].atan2(coeffs[[0, 0]]);
            if psi > PI {
                let mut s = coeffs.slice_mut(s![..;2, ..]);
                s *= -1.;
                psi - PI
            } else {
                psi
            }
        };
        let rot = array![[psi.cos(), psi.sin()], [-psi.sin(), psi.cos()]];
        for mut c in coeffs.axis_iter_mut(Axis(0)) {
            let m = rot
                .t()
                .dot(&array![[c[0], c[1]], [c[2], c[3]], [c[4], c[5]]].t())
                .t()
                .to_owned();
            c.assign(&Array1::from_iter(m));
        }
        let scale = coeffs[[0, 0]].abs();
        coeffs /= scale;
        let trans = Transform3::new(center, [-psi, 0., 0.], scale);
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
        let mut s = self.coeffs.slice_mut(s![.., 1]);
        s *= -1.;
        let mut s = self.coeffs.slice_mut(s![.., 3]);
        s *= -1.;
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
