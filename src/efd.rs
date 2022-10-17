use crate::*;
use alloc::{vec, vec::Vec};
use core::f64::consts::{PI, TAU};
use ndarray::{array, s, Array, Array1, Array2, Axis, CowArray, Dimension};
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

/// Alias of the 2D EFD type.
pub type Efd2 = Efd;
type CowCurve<'a> = alloc::borrow::Cow<'a, [[f64; 2]]>;

#[inline(always)]
fn pow2(x: f64) -> f64 {
    x * x
}

/// Compute the total Fourier power and find the minimum number of harmonics
/// required to exceed the threshold fraction of the total power.
///
/// This function needs to use the full of coefficients,
/// and the threshold must in [0, 1).
///
/// ```
/// use efd::{fourier_power, Efd2};
///
/// # let curve = efd::tests::PATH;
/// // Nyquist Frequency
/// let nyq = curve.len() / 2;
/// let efd = Efd2::from_curve_harmonic(curve, nyq).unwrap();
/// // Use "None" for the default threshold (99.99%)
/// let harmonic = fourier_power(efd, None);
/// # assert_eq!(harmonic, 6);
/// ```
pub fn fourier_power<T>(efd: Efd, threshold: T) -> usize
where
    T: Into<Option<f64>>,
{
    let threshold = threshold.into().unwrap_or(0.9999);
    debug_assert!((0.0..1.).contains(&threshold));
    let lut = cumsum(efd.coeffs.mapv(pow2), None).sum_axis(Axis(1));
    let total_power = lut.last().unwrap();
    lut.iter()
        .enumerate()
        .find(|(_, power)| *power / total_power >= threshold)
        .map(|(i, _)| i + 1)
        .unwrap()
}

/// Apply Nyquist Frequency on [`fourier_power`] with 99.99% threshold value.
///
/// Return none if the curve is less than 1.
///
/// ```
/// use efd::fourier_power_nyq;
///
/// # let curve = efd::tests::PATH;
/// let harmonic = fourier_power_nyq(curve);
/// # assert_eq!(harmonic, Some(6));
/// ```
pub fn fourier_power_nyq<C>(curve: C) -> Option<usize>
where
    C: AsRef<[[f64; 2]]>,
{
    fourier_power_nyq_gate(curve, None)
}

/// Apply Nyquist Frequency on [`fourier_power`] with a custom threshold value.
///
/// The threshold must in [0, 1).
/// This function return none if the curve is less than 1.
pub fn fourier_power_nyq_gate<C, T>(curve: C, threshold: T) -> Option<usize>
where
    C: AsRef<[[f64; 2]]>,
    T: Into<Option<f64>>,
{
    let curve = curve.as_ref();
    (curve.len() > 1)
        .then_some(curve.len() / 2)
        .and_then(|nyq| Efd::from_curve_harmonic(curve, nyq))
        .map(|efd| fourier_power(efd, threshold))
}

/// Check the difference between two curves.
pub fn curve_diff<C1, C2>(a: C1, b: C2) -> f64
where
    C1: AsRef<[[f64; 2]]>,
    C2: AsRef<[[f64; 2]]>,
{
    a.as_ref()
        .iter()
        .zip(b.as_ref())
        .map(|(a, b)| (a[0] - b[0]).abs() + (a[1] - b[1]).abs())
        .sum()
}

fn diff<'a, D, A>(arr: A, axis: Option<Axis>) -> Array<f64, D>
where
    D: Dimension,
    A: Into<CowArray<'a, f64, D>>,
{
    let arr = arr.into();
    let axis = axis.unwrap_or_else(|| Axis(arr.ndim() - 1));
    let head = arr.slice_axis(axis, (..-1).into());
    let tail = arr.slice_axis(axis, (1..).into());
    &tail - &head
}

fn cumsum<'a, D, A>(arr: A, axis: Option<Axis>) -> Array<f64, D>
where
    D: Dimension + ndarray::RemoveAxis,
    A: Into<CowArray<'a, f64, D>>,
{
    let mut arr = arr.into().to_owned();
    let axis = axis.unwrap_or(Axis(0));
    arr.axis_iter_mut(axis).reduce(|prev, mut next| {
        next += &prev;
        next
    });
    arr
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
pub struct Efd {
    coeffs: Array2<f64>,
    trans: Transform,
}

impl Efd {
    /// Create constant object from a nx4 array without boundary check.
    ///
    /// # Safety
    ///
    /// An invalid width may cause failure operation.
    pub const unsafe fn from_coeffs_unchecked(coeffs: Array2<f64>) -> Self {
        Self { coeffs, trans: Transform::new() }
    }

    /// Create object from a nx4 array with boundary check.
    pub fn try_from_coeffs(coeffs: Array2<f64>) -> Result<Self, EfdError> {
        (coeffs.nrows() > 0 && coeffs.ncols() == 4 && coeffs[[0, 0]] == 1.)
            .then(|| Self { coeffs, trans: Transform::new() })
            .ok_or(EfdError(()))
    }

    /// Calculate EFD coefficients from an existing discrete points.
    ///
    /// **The curve must be closed. (first == last)**
    ///
    /// Return none if the curve length is less than 1.
    pub fn from_curve<'a, C>(curve: C) -> Option<Self>
    where
        C: Into<CowCurve<'a>>,
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
        C: Into<CowCurve<'a>>,
        T: Into<Option<f64>>,
    {
        let curve = curve.into();
        let harmonic = fourier_power_nyq_gate(&curve, threshold)?;
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
        C: Into<CowCurve<'a>>,
        H: Into<Option<usize>>,
    {
        let curve = curve.into().into_owned();
        let harmonic = harmonic.into().or_else(|| fourier_power_nyq(&curve))?;
        assert!(harmonic > 0);
        if curve.len() < 2 {
            return None;
        }
        let dxy = diff(ndarray::arr2(&curve), Some(Axis(0)));
        let dt = dxy.mapv(pow2).sum_axis(Axis(1)).mapv(f64::sqrt);
        let t = ndarray::concatenate![Axis(0), array![0.], cumsum(&dt, None)];
        let zt = t.last().unwrap();
        let phi = &t * TAU / (zt + f64::EPSILON);
        let mut coeffs = Array2::zeros([harmonic, 4]);
        for (i, mut c) in coeffs.axis_iter_mut(Axis(0)).enumerate() {
            let n = i as f64 + 1.;
            let t = 0.5 * zt / (n * n * PI * PI);
            let phi_n = &phi * n;
            let phi_n_front = phi_n.slice(s![..-1]);
            let phi_n_back = phi_n.slice(s![1..]);
            let cos_phi_n = (phi_n_back.mapv(f64::cos) - phi_n_front.mapv(f64::cos)) / &dt;
            let sin_phi_n = (phi_n_back.mapv(f64::sin) - phi_n_front.mapv(f64::sin)) / &dt;
            c[0] = t * (&dxy.slice(s![.., 1]) * &cos_phi_n).sum();
            c[1] = t * (&dxy.slice(s![.., 1]) * &sin_phi_n).sum();
            c[2] = t * (&dxy.slice(s![.., 0]) * &cos_phi_n).sum();
            c[3] = t * (&dxy.slice(s![.., 0]) * &sin_phi_n).sum();
        }
        let center = {
            let tdt = &t.slice(s![1..]) / &dt;
            let xi = cumsum(dxy.slice(s![.., 0]), None) - &dxy.slice(s![.., 0]) * &tdt;
            let c = diff(t.mapv(pow2), None) * 0.5 / &dt;
            let a0 = (&dxy.slice(s![.., 0]) * &c + xi * &dt).sum() / (zt + f64::EPSILON);
            let delta = cumsum(dxy.slice(s![.., 1]), None) - &dxy.slice(s![.., 1]) * &tdt;
            let c0 = (&dxy.slice(s![.., 1]) * c + delta * dt).sum() / (zt + f64::EPSILON);
            let [x, y] = curve.first().unwrap();
            [x + a0, y + c0]
        };
        // Shift angle
        let theta1 = {
            let dy = 2. * (coeffs[[0, 0]] * coeffs[[0, 1]] + coeffs[[0, 2]] * coeffs[[0, 3]]);
            let dx = pow2(coeffs[[0, 0]]) - pow2(coeffs[[0, 1]]) + pow2(coeffs[[0, 2]])
                - pow2(coeffs[[0, 3]]);
            dy.atan2(dx) * 0.5
        };
        for (i, mut c) in coeffs.axis_iter_mut(Axis(0)).enumerate() {
            let angle = (i + 1) as f64 * theta1;
            let rot = array![[angle.cos(), -angle.sin()], [angle.sin(), angle.cos()]];
            let m = array![[c[0], c[1]], [c[2], c[3]]].dot(&rot);
            c.assign(&Array1::from_iter(m));
        }
        // The angle of semi-major axis
        let psi = coeffs[[0, 2]].atan2(coeffs[[0, 0]]);
        let rot = array![[psi.cos(), psi.sin()], [-psi.sin(), psi.cos()]];
        for mut c in coeffs.axis_iter_mut(Axis(0)) {
            let m = rot.dot(&array![[c[0], c[1]], [c[2], c[3]]]);
            c.assign(&Array1::from_iter(m));
        }
        let scale = coeffs[[0, 0]].abs();
        coeffs /= scale;
        let trans = Transform { rot: -psi, scale, center };
        Some(Self { coeffs, trans })
    }

    /// Builder method for adding transform type.
    pub fn trans(self, trans: Transform) -> Self {
        Self { trans, ..self }
    }

    /// Consume self and return raw array.
    pub fn unwrap(self) -> Array2<f64> {
        self.coeffs
    }

    /// Get the array view of the coefficients.
    pub fn coeffs(&self) -> ndarray::ArrayView2<f64> {
        self.coeffs.view()
    }

    /// Get the reference of transform type.
    pub fn as_trans(&self) -> &Transform {
        self
    }

    /// Get the mutable reference of transform type.
    pub fn as_trans_mut(&mut self) -> &mut Transform {
        self
    }

    /// Get the harmonic of the coefficients.
    pub fn harmonic(&self) -> usize {
        self.coeffs.nrows()
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

impl std::ops::Deref for Efd {
    type Target = Transform;

    fn deref(&self) -> &Self::Target {
        &self.trans
    }
}

impl std::ops::DerefMut for Efd {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.trans
    }
}
