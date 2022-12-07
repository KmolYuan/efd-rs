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
/// use efd::fourier_power2;
///
/// # let curve = efd::tests::PATH;
/// let harmonic = fourier_power2(curve, None);
/// # assert_eq!(harmonic, Some(6));
/// ```
pub fn fourier_power2<C, T>(curve: C, threshold: T) -> Option<usize>
where
    C: AsRef<[[f64; 2]]>,
    T: Into<Option<f64>>,
{
    let curve = curve.as_ref();
    (curve.len() > 1)
        .then_some(curve.len() / 2)
        .and_then(|nyq| Efd2::from_curve_harmonic(curve, nyq))
        .map(|efd| fourier_power(efd.coeffs(), threshold))
}

/// Elliptical Fourier Descriptor coefficients.
/// Provide transformation between discrete points and coefficients.
///
/// # Transformation
///
/// The transformation of normalized coefficients.
///
/// Implements Kuhl and Giardina method of normalizing the coefficients
/// An, Bn, Cn, Dn. Performs 3 separate normalizations. First, it makes the
/// data location invariant by re-scaling the data to a common origin.
/// Secondly, the data is rotated with respect to the major axis. Thirdly,
/// the coefficients are normalized with regard to the absolute value of A₁.
///
/// Please see [`Transform`] for more information.
#[derive(Clone, Debug)]
pub struct Efd2 {
    coeffs: Array2<f64>,
    trans: Transform2,
}

impl Efd2 {
    const DIM: usize = 4;

    /// Create object from a nx4 array with boundary check.
    pub fn try_from_coeffs(coeffs: Array2<f64>) -> Result<Self, EfdError<{ Self::DIM }>> {
        (coeffs.nrows() > 0 && coeffs.ncols() == Self::DIM && coeffs[[0, 0]] == 1.)
            .then(|| Self { coeffs, trans: Transform2::identity() })
            .ok_or(EfdError::<{ Self::DIM }>)
    }

    /// Calculate EFD coefficients from an existing discrete points.
    ///
    /// **The curve must be closed. (first == last)**
    ///
    /// Return none if the curve length is less than 1.
    pub fn from_curve<'a, C>(curve: C) -> Option<Self>
    where
        C: Into<CowCurve2<'a>>,
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
        C: Into<CowCurve2<'a>>,
        T: Into<Option<f64>>,
    {
        let curve = curve.into();
        let harmonic = fourier_power2(&curve, threshold)?;
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
        C: Into<CowCurve2<'a>>,
        H: Into<Option<usize>>,
    {
        let curve = curve.into().into_owned();
        let harmonic = harmonic.into().or_else(|| fourier_power2(&curve, None))?;
        assert!(harmonic > 0);
        if curve.len() < 2 {
            return None;
        }
        let dxy = diff(ndarray::arr2(&curve), Some(Axis(0)));
        let dt = dxy.mapv(pow2).sum_axis(Axis(1)).mapv(f64::sqrt);
        let t = ndarray::concatenate![Axis(0), array![0.], cumsum(&dt, None)];
        let zt = *t.last().unwrap();
        debug_assert!(zt != 0.);
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
            for i in 0..Self::DIM {
                c[i] = m[(i / 2, i % 2)];
            }
        }
        // The angle of semi-major axis
        let psi = {
            let psi = coeffs[[0, 2]].atan2(coeffs[[0, 0]]);
            if psi > PI {
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
            for i in 0..Self::DIM {
                c[i] = m[(i / 2, i % 2)];
            }
        }
        let scale = coeffs[[0, 0]].abs();
        coeffs /= scale;
        let trans = Transform2::new(center, psi, scale);
        Some(Self { coeffs, trans })
    }

    /// Builder method for adding transform type.
    pub fn with_trans(self, trans: Transform2) -> Self {
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
    pub fn as_trans(&self) -> &Transform2 {
        self
    }

    /// Get the mutable reference of transform type.
    pub fn as_trans_mut(&mut self) -> &mut Transform2 {
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
    pub fn generate_norm(&self, n: usize) -> Vec<[f64; 2]> {
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

    /// Generate the described curve from the coefficients.
    ///
    /// The number of the points `n` must given.
    pub fn generate(&self, n: usize) -> Vec<[f64; 2]> {
        self.trans.transform(&self.generate_norm(n))
    }
}

impl std::ops::Deref for Efd2 {
    type Target = Transform2;

    fn deref(&self) -> &Self::Target {
        &self.trans
    }
}

impl std::ops::DerefMut for Efd2 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.trans
    }
}
