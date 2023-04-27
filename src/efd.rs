use crate::*;
use alloc::vec::Vec;
use core::{
    f64::consts::{PI, TAU},
    marker::PhantomData,
};
use ndarray::{s, Array1, Array2, Axis, Slice};

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
    /// Create object from a 2D array with boundary check.
    ///
    /// The array size is (harmonic) x (dimension x 2).
    ///
    /// The dimension is [`<<D as EfdDim>::Trans as Trans>::DIM`](Trans::DIM).
    pub fn try_from_coeffs(coeffs: Array2<f64>) -> Result<Self, EfdError<D>> {
        (coeffs.nrows() > 0 && coeffs.ncols() == D::Trans::DIM * 2)
            .then_some(Self {
                coeffs,
                trans: Transform::identity(),
                _dim: PhantomData,
            })
            .ok_or(EfdError::new())
    }

    /// Calculate EFD coefficients from an existing discrete points.
    ///
    /// # Panic
    ///
    /// Panic if the curve length is not greater than 1 in debug mode. This
    /// function check the lengths only. Please use [`valid_curve()`] to
    /// verify the curve if there has NaN input.
    #[must_use]
    pub fn from_curve<C>(curve: C, is_open: bool) -> Self
    where
        C: Curve<Coord<D>>,
    {
        Self::from_curve_gate(curve, is_open, None)
    }

    /// Calculate EFD coefficients from an existing discrete points and Fourier
    /// power threshold.
    ///
    /// If the threshold is not given, it will be calculated with Fourier power
    /// analysis.
    ///
    /// # Panic
    ///
    /// Panic if the `threshold` is not in `0.0..1.0` or the curve length is not
    /// greater than 1 in debug mode. This function check the lengths only.
    /// Please use [`valid_curve()`] to verify the curve if there has NaN
    /// input.
    #[must_use]
    pub fn from_curve_gate<C, T>(curve: C, is_open: bool, threshold: T) -> Self
    where
        C: Curve<Coord<D>>,
        Option<f64>: From<T>,
    {
        let curve = curve.as_curve();
        if curve.len() < 2 {
            panic!("Invalid curve! Please use `efd::valid_curve()` to verify.");
        }
        let threshold = Option::from(threshold).unwrap_or(0.9999);
        debug_assert!(
            (0.0..1.0).contains(&threshold),
            "threshold must in 0.0..1.0"
        );
        // Nyquist Frequency
        let harmonic = curve.len() / 2;
        let (mut coeffs, trans) = D::from_curve_harmonic(curve, harmonic, is_open);
        let lut = cumsum(coeffs.mapv(pow2), None).sum_axis(Axis(1));
        let total_power = lut.last().unwrap();
        let (harmonic, _) = lut
            .iter()
            .enumerate()
            .find(|(_, power)| *power / total_power >= threshold)
            .unwrap();
        coeffs.slice_axis_inplace(Axis(0), Slice::from(..=harmonic));
        Self { coeffs, trans, _dim: PhantomData }
    }

    /// Calculate EFD coefficients from a series of existing discrete points.
    ///
    /// If the harmonic number is not given, it will be calculated with Fourier
    /// power analysis.
    ///
    /// # Panic
    ///
    /// Panic if the specific harmonic is zero or the curve length is not
    /// greater than 1 in debug mode. This function check the lengths only.
    /// Please use [`valid_curve()`] to verify the curve if there has NaN
    /// input.
    #[must_use]
    pub fn from_curve_harmonic<C, H>(curve: C, is_open: bool, harmonic: H) -> Self
    where
        C: Curve<Coord<D>>,
        Option<usize>: From<H>,
    {
        if let Some(harmonic) = Option::from(harmonic) {
            debug_assert!(harmonic != 0, "harmonic must not be 0");
            let curve = curve.as_curve();
            if curve.len() < 2 {
                panic!("Invalid curve! Please use `efd::valid_curve()` to verify.");
            } else {
                let (coeffs, trans) = D::from_curve_harmonic(curve, harmonic, is_open);
                Self { coeffs, trans, _dim: PhantomData }
            }
        } else {
            Self::from_curve(curve, is_open)
        }
    }

    /// A builder method for changing transform type.
    #[must_use]
    pub fn with_trans(self, trans: Transform<D::Trans>) -> Self {
        Self { trans, ..self }
    }

    /// Consume self and return a raw array of the coefficients.
    #[must_use]
    pub fn into_inner(self) -> Array2<f64> {
        self.coeffs
    }

    /// Get the array view of the coefficients.
    #[must_use]
    pub fn coeffs(&self) -> ndarray::ArrayView2<f64> {
        self.coeffs.view()
    }

    /// Get the reference of transform type.
    #[must_use]
    pub fn as_trans(&self) -> &Transform<D::Trans> {
        &self.trans
    }

    /// Get the mutable reference of transform type.
    #[must_use]
    pub fn as_trans_mut(&mut self) -> &mut Transform<D::Trans> {
        &mut self.trans
    }

    /// Get the harmonic number of the coefficients.
    #[must_use]
    pub fn harmonic(&self) -> usize {
        self.coeffs.nrows()
    }

    /// Square error.
    ///
    /// The coefficients will paded automatically if harmonic number is
    /// different.
    #[must_use]
    pub fn square_err(&self, rhs: &Self) -> f64 {
        padding(self, rhs, |a, b| (a - b).mapv(pow2).sum())
    }

    /// L1 norm error, aka Manhattan distance.
    ///
    /// The coefficients will paded automatically if harmonic number is
    /// different.
    #[must_use]
    pub fn l1_norm(&self, rhs: &Self) -> f64 {
        padding(self, rhs, |a, b| (a - b).mapv(f64::abs).sum())
    }

    /// L2 norm error, aka Euclidean distance.
    ///
    /// The coefficients will paded automatically if harmonic number is
    /// different.
    #[must_use]
    pub fn l2_norm(&self, rhs: &Self) -> f64 {
        padding(self, rhs, |a, b| (a - b).mapv(pow2).sum().sqrt())
    }

    /// Lp norm error, slower than [`Self::l1_norm()`] and [`Self::l2_norm()`].
    ///
    /// The coefficients will paded automatically if harmonic number is
    /// different.
    #[must_use]
    pub fn lp_norm(&self, rhs: &Self, p: i32) -> f64 {
        padding(self, rhs, |a, b| {
            (a - b).mapv(|x| x.abs().powi(p)).sum().powf(1. / p as f64)
        })
    }

    /// Reverse the order of described curve then return a mutable reference.
    pub fn reverse_inplace(&mut self) {
        let mut s = self.coeffs.slice_mut(s![.., 1..;2]);
        s *= -1.;
    }

    /// Consume and return a reversed version of the coefficients. This method
    /// can avoid mutable require.
    ///
    /// Please clone the object if you want to do self-comparison.
    #[must_use]
    pub fn reversed(mut self) -> Self {
        self.reverse_inplace();
        self
    }

    /// Generate the described curve. (`theta=TAU`)
    ///
    /// # Panic
    ///
    /// The number of the points `n` must larger than 1.
    ///
    /// # See Also
    ///
    /// [`Efd::generate_half()`], [`Efd::generate_in()`],
    /// [`Efd::generate_norm_in()`]
    #[must_use]
    pub fn generate(&self, n: usize) -> Vec<Coord<D>> {
        self.generate_in(n, TAU)
    }

    /// Generate a half of the described curve. (`theta=PI`)
    ///
    /// # Panic
    ///
    /// The number of the points `n` must larger than 1.
    ///
    /// # See Also
    ///
    /// [`Efd::generate()`], [`Efd::generate_in()`], [`Efd::generate_norm_in()`]
    #[must_use]
    pub fn generate_half(&self, n: usize) -> Vec<Coord<D>> {
        self.generate_in(n, PI)
    }

    /// Generate the described curve in a specific angle `theta` (`0..=TAU`).
    ///
    /// # Panic
    ///
    /// The number of the points `n` must larger than 1.
    ///
    /// # See Also
    ///
    /// [`Efd::generate_half()`], [`Efd::generate_in()`],
    /// [`Efd::generate_norm_in()`]
    #[must_use]
    pub fn generate_in(&self, n: usize, theta: f64) -> Vec<Coord<D>> {
        let mut curve = self.generate_norm_in(n, theta);
        self.trans.transform_inplace(&mut curve);
        curve
    }

    /// Generate a normalized curve in a specific angle `theta` (`0..=TAU`).
    ///
    /// Normalized curve is **without** transformation.
    ///
    /// # Panic
    ///
    /// The number of the points `n` must larger than 1.
    ///
    /// # See Also
    ///
    /// [`Efd::generate()`], [`Efd::generate_half()`], [`Efd::generate_in()`]
    #[must_use]
    pub fn generate_norm_in(&self, n: usize, theta: f64) -> Vec<Coord<D>> {
        assert!(n > 1, "n ({n}) must larger than 1");
        let mut t = Array1::from_elem(n, 1. / (n - 1) as f64);
        t[0] = 0.;
        let t = cumsum(t, None) * theta;
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
}

fn padding<D, F>(a: &Efd<D>, b: &Efd<D>, f: F) -> f64
where
    D: EfdDim,
    F: Fn(&Array2<f64>, &Array2<f64>) -> f64,
{
    use core::cmp::Ordering::*;
    match a.harmonic().cmp(&b.harmonic()) {
        Equal => f(&a.coeffs, &b.coeffs),
        Greater => {
            let zeros = Array2::zeros([a.harmonic() - b.harmonic(), a.coeffs.ncols()]);
            f(&a.coeffs, &ndarray::concatenate![Axis(0), b.coeffs, zeros])
        }
        Less => {
            let zeros = Array2::zeros([b.harmonic() - a.harmonic(), a.coeffs.ncols()]);
            f(&ndarray::concatenate![Axis(0), a.coeffs, zeros], &b.coeffs)
        }
    }
}
