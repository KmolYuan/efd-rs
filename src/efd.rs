use crate::*;
use core::{f64::consts::TAU, marker::PhantomData};
use ndarray::{s, Array1, Array2, Axis};
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

/// 2D EFD coefficients type.
pub type Efd2 = Efd<D2>;
/// 3D EFD coefficients type.
pub type Efd3 = Efd<D3>;

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
#[derive(Clone)]
pub struct Efd<D: EfdDim> {
    coeffs: Array2<f64>,
    trans: Transform<D::Trans>,
    _dim: PhantomData<D>,
}

impl<D: EfdDim> Efd<D> {
    /// Create object from a nx4 array with boundary check.
    pub fn try_from_coeffs(coeffs: Array2<f64>) -> Result<Self, EfdError<D>> {
        (coeffs.nrows() > 0 && coeffs.ncols() == D::Trans::DIM * 2)
            .then(|| Self {
                coeffs,
                trans: Transform::identity(),
                _dim: PhantomData,
            })
            .ok_or_else(EfdError::new)
    }

    /// Apply Nyquist Frequency on Fourier power analysis with a custom
    /// threshold value (default to 99.99%).
    ///
    /// This method returns the minimum harmonic number to keep the shape
    /// features.
    ///
    /// The threshold must in [0, 1). This function returns `None` if the curve
    /// length is less than 1.
    ///
    /// ```
    /// # let curve = efd::tests::PATH;
    /// let harmonic = efd::Efd2::gate(curve, None).unwrap();
    /// # assert_eq!(harmonic, 6);
    /// ```
    pub fn gate<'a, C, T>(curve: C, threshold: T) -> Option<usize>
    where
        C: Into<CowCurve<'a, D::Trans>>,
        T: Into<Option<f64>>,
    {
        let curve = curve.into();
        let threshold = threshold.into().unwrap_or(0.9999);
        assert!((0.0..1.).contains(&threshold));
        if curve.len() < 2 {
            return None;
        }
        // Nyquist Frequency
        let harmonic = curve.len() / 2;
        let (coeffs, _) = D::from_curve_harmonic(curve, harmonic);
        let lut = cumsum(coeffs.mapv(pow2), None).sum_axis(Axis(1));
        let total_power = lut.last().unwrap();
        let harmonic = lut
            .iter()
            .enumerate()
            .find(|(_, power)| *power / total_power >= threshold)
            .map(|(i, _)| i + 1)
            .unwrap_or(harmonic);
        Some(harmonic)
    }

    /// Calculate EFD coefficients from an existing discrete points.
    ///
    /// **The curve must be closed. (first == last)**
    ///
    /// Return none if the curve length is less than 1.
    pub fn from_curve<'a, C>(curve: C) -> Option<Self>
    where
        C: Into<CowCurve<'a, D::Trans>>,
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
        C: Into<CowCurve<'a, D::Trans>>,
        T: Into<Option<f64>>,
    {
        let curve = curve.into();
        let harmonic = Self::gate(curve.as_ref(), threshold)?;
        Self::from_curve_harmonic(curve, harmonic)
    }

    /// Calculate EFD coefficients from a series of existing discrete points.
    ///
    /// **The curve must be closed. (first == last)**
    ///
    /// Return none if harmonic number is zero or the curve is less than
    /// two points.
    ///
    /// If the harmonic number is not given, it will be calculated with
    /// [`Self::gate()`] function.
    ///
    /// See also [`closed_curve`].
    pub fn from_curve_harmonic<'a, C, H>(curve: C, harmonic: H) -> Option<Self>
    where
        C: Into<CowCurve<'a, D::Trans>>,
        H: Into<Option<usize>>,
    {
        let curve = curve.into();
        let harmonic = harmonic
            .into()
            .or_else(|| Self::gate(curve.as_ref(), None))?;
        assert!(harmonic > 0);
        if curve.len() < 2 {
            return None;
        }
        let (coeffs, trans) = D::from_curve_harmonic(curve, harmonic);
        Some(Self { coeffs, trans, _dim: PhantomData })
    }

    /// A builder method for changing transform type.
    pub fn with_trans(self, trans: Transform<D::Trans>) -> Self {
        Self { trans, ..self }
    }

    /// Consume self and return a raw array of the coefficients.
    pub fn into_inner(self) -> Array2<f64> {
        self.coeffs
    }

    /// Get the array view of the coefficients.
    pub fn coeffs(&self) -> ndarray::ArrayView2<f64> {
        self.coeffs.view()
    }

    /// Get the reference of transform type.
    pub fn as_trans(&self) -> &Transform<D::Trans> {
        &self.trans
    }

    /// Get the mutable reference of transform type.
    pub fn as_trans_mut(&mut self) -> &mut Transform<D::Trans> {
        &mut self.trans
    }

    /// Get the harmonic number of the coefficients.
    pub fn harmonic(&self) -> usize {
        self.coeffs.nrows()
    }

    /// Square error.
    ///
    /// # Panics
    ///
    /// When the harmonic number is different.
    pub fn square_err(&self, rhs: &Self) -> f64 {
        (&self.coeffs - &rhs.coeffs).mapv(pow2).sum()
    }

    /// L1 norm error, aka Manhattan distance.
    ///
    /// # Panics
    ///
    /// When the harmonic number is different.
    pub fn l1_norm(&self, rhs: &Self) -> f64 {
        (&self.coeffs - &rhs.coeffs).mapv(f64::abs).sum()
    }

    /// L2 norm error, aka Euclidean distance.
    ///
    /// # Panics
    ///
    /// When the harmonic number is different.
    pub fn l2_norm(&self, rhs: &Self) -> f64 {
        (&self.coeffs - &rhs.coeffs).mapv(pow2).sum().sqrt()
    }

    /// Lp norm error, slower than [`Self::l1_norm()`] and [`Self::l2_norm()`].
    ///
    /// # Panics
    ///
    /// When the harmonic number is different.
    pub fn lp_norm(&self, rhs: &Self, p: i32) -> f64 {
        (&self.coeffs - &rhs.coeffs)
            .mapv(|x| x.abs().powi(p))
            .sum()
            .powf(1. / p as f64)
    }

    /// Reverse the order of described curve then return a mutable reference.
    pub fn reverse(&mut self) -> &mut Self {
        let mut s = self.coeffs.slice_mut(s![.., 1..;2]);
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
    pub fn generate_norm(&self, n: usize) -> Curve<D::Trans> {
        assert!(n > 1, "n ({n}) must larger than 1");
        let mut t = Array1::from_elem(n, 1. / (n - 1) as f64);
        t[0] = 0.;
        let t = cumsum(t, None) * TAU;
        self.coeffs
            .axis_iter(Axis(0))
            .enumerate()
            .map(|(i, c)| {
                let lambda = &t * (i + 1) as f64;
                let cos = lambda.mapv(f64::cos);
                let sin = lambda.mapv(f64::sin);
                let mut path = Array2::zeros([t.len(), D::Trans::DIM]);
                for (i, mut s) in path.axis_iter_mut(Axis(1)).enumerate() {
                    s.assign(&(&cos * c[i * 2] + &sin * c[i * 2 + 1]));
                }
                path
            })
            .reduce(|a, b| a + b)
            .unwrap()
            .axis_iter(Axis(0))
            .map(D::to_coord)
            .collect()
    }

    /// Generate the described curve from the coefficients.
    ///
    /// The number of the points `n` must given.
    pub fn generate(&self, n: usize) -> Curve<D::Trans> {
        self.trans.transform(&self.generate_norm(n))
    }
}
