use crate::{CowCurve, Efd2Error, Geo2Info};
use alloc::{vec, vec::Vec};
use core::f64::consts::{PI, TAU};
use ndarray::{array, concatenate, s, Array, Array1, Array2, AsArray, Axis, Slice};
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

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
/// # use efd::tests::PATH;
///
/// # let curve = PATH;
/// // Nyquist Frequency
/// let nyq = curve.len() / 2;
/// let harmonic = fourier_power(Efd2::from_curve(curve, nyq).unwrap(), 0.9999);
/// # assert_eq!(harmonic, 6);
/// ```
pub fn fourier_power(efd: Efd2, threshold: f64) -> usize {
    debug_assert!((0.0..1.).contains(&threshold));
    let lut = cumsum(efd.coeffs.mapv(pow2).sum_axis(Axis(1)), None);
    let total_power = lut.last().unwrap();
    lut.iter()
        .enumerate()
        .find(|(_, power)| *power / total_power >= threshold)
        .map(|(i, _)| i + 1)
        .unwrap_or(lut.len())
}

/// A convenient function to apply Nyquist Frequency on [`fourier_power`]
/// function.
///
/// Return none if the curve is empty.
///
/// ```
/// use efd::fourier_power_nyq;
/// # use efd::tests::PATH;
///
/// # let curve = PATH;
/// let harmonic = fourier_power_nyq(curve);
/// # assert_eq!(harmonic, Some(6));
/// ```
pub fn fourier_power_nyq<'a, C>(curve: C) -> Option<usize>
where
    C: Into<CowCurve<'a>>,
{
    let curve = curve.into();
    if !curve.is_empty() {
        let nyq = curve.len() / 2;
        Some(fourier_power(Efd2::from_curve(curve, nyq)?, 0.9999))
    } else {
        None
    }
}

/// Check the difference between two curves.
pub fn curve_diff<'a, 'b, C1, C2>(a: C1, b: C2) -> f64
where
    C1: Into<CowCurve<'a>>,
    C2: Into<CowCurve<'a>>,
{
    a.into()
        .iter()
        .zip(b.into().iter())
        .map(|(a, b)| (a[0] - b[0]).abs() + (a[1] - b[1]).abs())
        .sum()
}

fn diff<'a, D, A>(arr: A, axis: Option<Axis>) -> Array<f64, D>
where
    D: ndarray::Dimension,
    A: AsArray<'a, f64, D>,
{
    let arr = arr.into();
    let axis = axis.unwrap_or_else(|| Axis(arr.ndim() - 1));
    let head = arr.slice_axis(axis, Slice::from(..-1));
    let tail = arr.slice_axis(axis, Slice::from(1..));
    &tail - &head
}

fn cumsum<D>(mut a: Array<f64, D>, axis: Option<Axis>) -> Array<f64, D>
where
    D: ndarray::Dimension + ndarray::RemoveAxis,
{
    let axis = axis.unwrap_or(Axis(0));
    a.axis_iter_mut(axis).reduce(|prev, mut next| {
        next += &prev;
        next
    });
    a
}

/// 2D Elliptical Fourier Descriptor coefficients.
/// Provide transformation between discrete points and coefficients.
///
/// # Geometry Information
///
/// The geometry information of normalized coefficients.
///
/// Implements Kuhl and Giardina method of normalizing the coefficients
/// An, Bn, Cn, Dn. Performs 3 separate normalizations. First, it makes the
/// data location invariant by re-scaling the data to a common origin.
/// Secondly, the data is rotated with respect to the major axis. Thirdly,
/// the coefficients are normalized with regard to the absolute value of A₁.
///
/// Please see [`Geo2Info`] for more information.
#[derive(Clone, Debug)]
pub struct Efd2 {
    coeffs: Array2<f64>,
    geo: Geo2Info,
}

impl Efd2 {
    /// Create constant object from a nx4 array without boundary check.
    ///
    /// # Safety
    ///
    /// An invalid width may cause failure operation.
    pub const unsafe fn from_coeffs_unchecked(coeffs: Array2<f64>) -> Self {
        Self { coeffs, geo: Geo2Info::new() }
    }

    /// Create object from a nx4 array with boundary check.
    pub fn try_from_coeffs(coeffs: Array2<f64>) -> Result<Self, Efd2Error> {
        (coeffs.ncols() == 4)
            .then(|| Self { coeffs, geo: Geo2Info::new() })
            .ok_or(Efd2Error)
    }

    /// Builder method for adding geometric information.
    pub fn with_geo(self, geo: Geo2Info) -> Self {
        Self { geo, ..self }
    }

    /// Calculate EFD coefficients from an existing discrete points.
    ///
    /// Return none if the curve is empty.
    ///
    /// If the harmonic number is not given, it will be calculated with
    /// [`fourier_power`] function.
    pub fn from_curve<'a, C, H>(curve: C, harmonic: H) -> Option<Self>
    where
        C: Into<CowCurve<'a>>,
        H: Into<Option<usize>>,
    {
        let curve = curve.into();
        let harmonic = harmonic
            .into()
            .or_else(|| fourier_power_nyq(curve.as_ref()))?;
        let dxy = diff(&ndarray::arr2(&curve), Some(Axis(0)));
        let dt = dxy.mapv(pow2).sum_axis(Axis(1)).mapv(f64::sqrt);
        let t = concatenate![Axis(0), array![0.], cumsum(dt.clone(), None)];
        let zt = t[t.len() - 1];
        let phi = &t * TAU / (zt + f64::EPSILON);
        let mut coeffs = Array2::zeros((harmonic, 4));
        for n in 0..harmonic {
            let n1 = n as f64 + 1.;
            let c = 0.5 * zt / (n1 * n1 * PI * PI);
            let phi_n = &phi * n1;
            let phi_n_front = phi_n.slice(s![..-1]);
            let phi_n_back = phi_n.slice(s![1..]);
            let cos_phi_n = (phi_n_back.mapv(f64::cos) - phi_n_front.mapv(f64::cos)) / &dt;
            let sin_phi_n = (phi_n_back.mapv(f64::sin) - phi_n_front.mapv(f64::sin)) / &dt;
            coeffs[[n, 0]] = c * (&dxy.slice(s![.., 1]) * &cos_phi_n).sum();
            coeffs[[n, 1]] = c * (&dxy.slice(s![.., 1]) * &sin_phi_n).sum();
            coeffs[[n, 2]] = c * (&dxy.slice(s![.., 0]) * &cos_phi_n).sum();
            coeffs[[n, 3]] = c * (&dxy.slice(s![.., 0]) * &sin_phi_n).sum();
        }
        let tdt = &t.slice(s![1..]) / &dt;
        let xi = cumsum(dxy.slice(s![.., 0]).to_owned(), None) - &dxy.slice(s![.., 0]) * &tdt;
        let c = diff(&t.mapv(pow2), None) * 0.5 / &dt;
        let a0 = (&dxy.slice(s![.., 0]) * &c + xi * &dt).sum() / (zt + f64::EPSILON);
        let delta = cumsum(dxy.slice(s![.., 1]).to_owned(), None) - &dxy.slice(s![.., 1]) * &tdt;
        let c0 = (&dxy.slice(s![.., 1]) * c + delta * dt).sum() / (zt + f64::EPSILON);
        let center = [curve[0][0] + a0, curve[0][1] + c0];
        // Shift angle
        let theta1 = {
            let dy = 2. * (coeffs[[0, 0]] * coeffs[[0, 1]] + coeffs[[0, 2]] * coeffs[[0, 3]]);
            let dx = coeffs[[0, 0]] * coeffs[[0, 0]] - coeffs[[0, 1]] * coeffs[[0, 1]]
                + coeffs[[0, 2]] * coeffs[[0, 2]]
                - coeffs[[0, 3]] * coeffs[[0, 3]];
            dy.atan2(dx) * 0.5
        };
        for n in 0..harmonic {
            let angle = (n + 1) as f64 * theta1;
            let rot = array![[angle.cos(), -angle.sin()], [angle.sin(), angle.cos()]];
            let m = array![
                [coeffs[[n, 0]], coeffs[[n, 1]]],
                [coeffs[[n, 2]], coeffs[[n, 3]]],
            ]
            .dot(&rot);
            coeffs[[n, 0]] = m[[0, 0]];
            coeffs[[n, 1]] = m[[0, 1]];
            coeffs[[n, 2]] = m[[1, 0]];
            coeffs[[n, 3]] = m[[1, 1]];
        }
        // The angle of semi-major axis
        let psi = coeffs[[0, 2]].atan2(coeffs[[0, 0]]);
        let rot = array![[psi.cos(), psi.sin()], [-psi.sin(), psi.cos()]];
        for n in 0..harmonic {
            let m = rot.dot(&array![
                [coeffs[[n, 0]], coeffs[[n, 1]]],
                [coeffs[[n, 2]], coeffs[[n, 3]]],
            ]);
            coeffs[[n, 0]] = m[[0, 0]];
            coeffs[[n, 1]] = m[[0, 1]];
            coeffs[[n, 2]] = m[[1, 0]];
            coeffs[[n, 3]] = m[[1, 1]];
        }
        let scale = coeffs[[0, 0]].abs();
        coeffs /= scale;
        let geo = Geo2Info { rot: -psi, scale, center };
        Some(Self { coeffs, geo })
    }

    /// Consume self and return raw array.
    pub fn unwrap(self) -> Array2<f64> {
        self.coeffs
    }

    /// Get the array view of the coefficients.
    pub fn coeffs(&self) -> ndarray::ArrayView2<f64> {
        self.coeffs.view()
    }

    /// Get the geometry information.
    pub fn geo(&self) -> &Geo2Info {
        self
    }

    /// Get the harmonic of the coefficients.
    pub fn harmonic(&self) -> usize {
        self.coeffs.nrows()
    }

    /// Manhattan distance.
    pub fn manhattan(&self, rhs: &Self) -> f64 {
        (&self.coeffs - &rhs.coeffs).mapv(f64::abs).sum()
    }

    /// Euclidean distance.
    pub fn euclidean(&self, rhs: &Self) -> f64 {
        (&self.coeffs - &rhs.coeffs).mapv(pow2).sum().sqrt()
    }

    /// Generate the normalized curve **without** geometry information.
    ///
    /// The number of the points `n` must lager than 3.
    pub fn generate_norm(&self, n: usize) -> Vec<[f64; 2]> {
        assert!(n > 3, "n ({}) must larger than 3", n);
        let mut t = vec![1. / (n - 1) as f64; n];
        t[0] = 0.;
        let t = cumsum(Array1::from(t), None);
        (0..self.harmonic())
            .fold(Array2::<f64>::zeros([n, 2]), |curve, n| {
                let angle = &t * (n + 1) as f64 * TAU;
                let cos = angle.mapv(f64::cos);
                let sin = angle.mapv(f64::sin);
                let x = &cos * self.coeffs[[n, 2]] + &sin * self.coeffs[[n, 3]];
                let y = &cos * self.coeffs[[n, 0]] + &sin * self.coeffs[[n, 1]];
                curve + ndarray::stack![Axis(1), x, y]
            })
            .axis_iter(Axis(0))
            .map(|c| [c[0], c[1]])
            .collect()
    }

    /// Generate the described curve from the coefficients.
    ///
    /// The number of the points `n` must given.
    pub fn generate(&self, n: usize) -> Vec<[f64; 2]> {
        self.geo.transform(&self.generate_norm(n))
    }
}

impl std::ops::Deref for Efd2 {
    type Target = Geo2Info;

    fn deref(&self) -> &Self::Target {
        &self.geo
    }
}
