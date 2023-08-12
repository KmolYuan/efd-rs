use crate::*;
use alloc::vec::Vec;
use core::f64::consts::{PI, TAU};

/// 2D EFD coefficients type.
pub type Efd2 = Efd<D2>;
/// 3D EFD coefficients type.
pub type Efd3 = Efd<D3>;

/// Elliptical Fourier Descriptor coefficients.
/// Provide transformation between discrete points and coefficients.
///
/// Start with [`Efd::from_curve()`] and its related methods.
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
///
/// # Raw Coefficients
///
/// The coefficients is contained with `na::Matrix`, use
/// [`Efd::try_from_coeffs()`] to input the coefficients externally.
///
/// Use [`Efd::into_inner()`] to get the matrix of the coefficients.
#[derive(Clone)]
pub struct Efd<D: EfdDim> {
    coeffs: Coeff<D>,
    trans: Transform<D::Trans>,
}

impl<D: EfdDim> Efd<D> {
    /// Create object from a 2D array with boundary check and normalization.
    ///
    /// The array size is (harmonic) x (dimension x 2). The dimension is
    /// [`CoordHint::Dim`].
    ///
    /// Return none if the harmonic is zero.
    pub fn try_from_coeffs(mut coeffs: Coeff<D>) -> Option<Self> {
        (coeffs.ncols() != 0).then(|| {
            let trans = D::coeff_norm(&mut coeffs);
            Self { coeffs, trans }
        })
    }

    /// Create object from a 2D array with boundary check.
    ///
    /// The array size is (harmonic) x (dimension x 2). The dimension is
    /// [`CoordHint::Dim`].
    ///
    /// Return none if the harmonic is zero.
    pub fn try_from_coeffs_unnorm(coeffs: Coeff<D>) -> Option<Self> {
        (coeffs.ncols() != 0).then_some(Self { coeffs, trans: Transform::identity() })
    }

    /// Create object from a 2D array directly.
    ///
    /// The array size is (harmonic) x (dimension x 2). The dimension is
    /// [`CoordHint::Dim`].
    ///
    /// # Safety
    ///
    /// Other operations might panic if the harmonic is zero.
    pub fn from_coeffs_unchecked(coeffs: Coeff<D>) -> Self {
        Self { coeffs, trans: Transform::identity() }
    }

    /// Fully automated coefficient calculation.
    ///
    /// 1. The initial harmonic number is the same as the curve point.
    /// 1. Fourier Power Anaysis (FPA) uses 99.99% threshold.
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
        let len = curve.as_curve().len();
        Self::from_curve_harmonic(curve, is_open, if is_open { len * 2 } else { len })
            .fourier_power_anaysis(None)
    }

    /// Same as [`Efd::from_curve()`], but if your sampling points are large,
    /// use Nyquist Frequency as an initial harmonic number.
    ///
    /// Nyquist Frequency is half of the sample number.
    ///
    /// Please ensure the sampling points are generated from a known function
    /// and are more than enough. Otherwise, it will cause undersampling.
    #[must_use]
    pub fn from_curve_nyquist<C>(curve: C, is_open: bool) -> Self
    where
        C: Curve<Coord<D>>,
    {
        let len = curve.as_curve().len();
        Self::from_curve_harmonic(curve, is_open, if is_open { len } else { len / 2 })
            .fourier_power_anaysis(None)
    }

    /// Same as [`Efd::from_curve()`], but use a customized threshold in Fourier
    /// Power Anaysis (FPA).
    #[must_use]
    #[deprecated = "this method is rarely used"]
    pub fn from_curve_threshold<C, T>(curve: C, is_open: bool, threshold: T) -> Self
    where
        C: Curve<Coord<D>>,
        Option<f64>: From<T>,
    {
        let len = curve.as_curve().len();
        Self::from_curve_harmonic(curve, is_open, if is_open { len * 2 } else { len })
            .fourier_power_anaysis(threshold)
    }

    /// Manual coefficient calculation.
    ///
    /// 1. The initial harmonic is decide by user.
    /// 1. No harmonic reduced. Please call [`Efd::fourier_power_anaysis()`].
    ///
    /// # Panic
    ///
    /// Panic if the specific harmonic is zero or the curve length is not
    /// greater than 1 in the **debug mode**. This function check the lengths
    /// only. Please use [`valid_curve()`] to verify the curve if there has
    /// NaN input.
    #[must_use]
    pub fn from_curve_harmonic<C>(curve: C, is_open: bool, harmonic: usize) -> Self
    where
        C: Curve<Coord<D>>,
    {
        debug_assert!(harmonic != 0, "harmonic must not be 0");
        let curve = curve.as_curve();
        debug_assert!(curve.len() > 1, "the curve length must greater than 1");
        let (coeffs, trans) = D::get_coeff(curve, harmonic, is_open);
        Self { coeffs, trans }
    }

    /// Same as [`Efd::from_curve_harmonic()`] but without normalization.
    #[must_use]
    pub fn from_curve_harmonic_unnorm<C>(curve: C, is_open: bool, harmonic: usize) -> Self
    where
        C: Curve<Coord<D>>,
    {
        debug_assert!(harmonic != 0, "harmonic must not be 0");
        let curve = curve.as_curve();
        debug_assert!(curve.len() > 1, "the curve length must greater than 1");
        let (coeffs, trans) = D::get_coeff_unnorm(curve, harmonic, is_open);
        Self { coeffs, trans }
    }

    /// A builder method for changing transformation type.
    #[must_use]
    pub fn with_trans(self, trans: Transform<D::Trans>) -> Self {
        Self { trans, ..self }
    }

    /// Use Fourier Power Anaysis (FPA) to reduce the harmonic number.
    ///
    /// The coefficient memory will be saved but cannot be used twice due to
    /// undersampling.
    ///
    /// The default threshold is 99.99%.
    #[must_use]
    pub fn fourier_power_anaysis<T>(mut self, threshold: T) -> Self
    where
        Option<f64>: From<T>,
    {
        let threshold = Option::from(threshold).unwrap_or(0.9999);
        debug_assert!((0.0..1.0).contains(&threshold), "threshold must in 0..1");
        let mut lut = cumsum(self.coeffs.map(pow2)).row_sum();
        lut /= lut[lut.len() - 1];
        let harmonic = match lut
            .as_slice()
            .binary_search_by(|x| x.partial_cmp(&threshold).unwrap())
        {
            Ok(h) | Err(h) => h + 1,
        };
        self.coeffs.resize_horizontally_mut(harmonic, 0.);
        self
    }

    /// Force normalize the coefficients.
    ///
    /// If the coefficients are constructed by `*_unnorm` or `*_unchecked`
    /// methods, this method will normalize them.
    pub fn normalized(self) -> Self {
        let Self { mut coeffs, trans } = self;
        let trans_new = D::coeff_norm(&mut coeffs);
        Self { coeffs, trans: trans.apply(&trans_new) }
    }

    /// Consume self and return a raw array of the coefficients.
    #[must_use]
    pub fn into_inner(self) -> Coeff<D> {
        self.coeffs
    }

    /// Get a reference to the coefficients.
    #[must_use]
    pub fn coeffs(&self) -> &Coeff<D> {
        &self.coeffs
    }

    /// Get a view to the specific coefficients. (`0..self.harmonic()`)
    #[must_use]
    pub fn coeff(&self, harmonic: usize) -> CKernel<D> {
        CKernel::<D>::from_slice(self.coeffs.column(harmonic).data.into_slice())
    }

    /// Get an iterator over all the coefficients per harmonic.
    pub fn coeffs_iter(&self) -> impl Iterator<Item = CKernel<D>> {
        self.coeffs
            .column_iter()
            .map(|c| CKernel::<D>::from_slice(c.data.into_slice()))
    }

    /// Get a mutable iterator over all the coefficients per harmonic.
    pub fn coeffs_iter_mut(&mut self) -> impl Iterator<Item = CKernelMut<D>> {
        self.coeffs
            .column_iter_mut()
            .map(|c| CKernelMut::<D>::from_slice(c.data.into_slice_mut()))
    }

    /// Get the reference of transformation type.
    #[must_use]
    pub fn as_trans(&self) -> &Transform<D::Trans> {
        &self.trans
    }

    /// Get the mutable reference of transformation type.
    #[must_use]
    pub fn as_trans_mut(&mut self) -> &mut Transform<D::Trans> {
        &mut self.trans
    }

    /// Get the harmonic number of the coefficients.
    #[must_use]
    pub fn harmonic(&self) -> usize {
        self.coeffs.ncols()
    }

    /// Square error.
    ///
    /// The coefficients will paded automatically if harmonic number is
    /// different.
    #[must_use]
    pub fn square_err(&self, rhs: &Self) -> f64 {
        padding(self, rhs, |a, b| (a - b).map(pow2).sum())
    }

    /// L1 norm error, aka Manhattan distance.
    ///
    /// The coefficients will paded automatically if harmonic number is
    /// different.
    #[must_use]
    pub fn l1_norm(&self, rhs: &Self) -> f64 {
        padding(self, rhs, |a, b| (a - b).map(f64::abs).sum())
    }

    /// L2 norm error, aka Euclidean distance.
    ///
    /// The coefficients will paded automatically if harmonic number is
    /// different.
    #[must_use]
    pub fn l2_norm(&self, rhs: &Self) -> f64 {
        padding(self, rhs, |a, b| (a - b).map(pow2).sum().sqrt())
    }

    /// Lp norm error, slower than [`Self::l1_norm()`] and [`Self::l2_norm()`].
    ///
    /// The coefficients will paded automatically if harmonic number is
    /// different.
    #[must_use]
    pub fn lp_norm(&self, rhs: &Self, p: i32) -> f64 {
        padding(self, rhs, |a, b| {
            (a - b).map(|x| x.abs().powi(p)).sum().powf(1. / p as f64)
        })
    }

    /// Reverse the order of described curve then return a mutable reference.
    pub fn reverse_inplace(&mut self) {
        self.coeffs
            .row_iter_mut()
            .skip(D::Trans::dim())
            .for_each(|mut c| c *= -1.);
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
    #[must_use]
    pub fn generate(&self, n: usize) -> Vec<Coord<D>> {
        self.generate_in(n, TAU)
    }

    /// Generate a half of the described curve. (`theta=PI`)
    ///
    /// # Panic
    ///
    /// The number of the points `n` must larger than 1.
    #[must_use]
    pub fn generate_half(&self, n: usize) -> Vec<Coord<D>> {
        self.generate_in(n, PI)
    }

    /// Generate the described curve in a specific angle `theta` (`0..=TAU`).
    ///
    /// # Panic
    ///
    /// The number of the points `n` must larger than 1.
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
    #[must_use]
    pub fn generate_norm_in(&self, n: usize, theta: f64) -> Vec<Coord<D>> {
        assert!(n > 1, "n ({n}) must larger than 1");
        let t = na::Matrix1xX::from_fn(n, |_, i| i as f64 / (n - 1) as f64 * theta);
        self.coeffs
            .column_iter()
            .enumerate()
            .map(|(i, c)| {
                let t = &t * (i + 1) as f64;
                let t = na::Matrix2xX::from_rows(&[t.map(f64::cos), t.map(f64::sin)]);
                CKernel::<D>::from_slice(c.as_slice()) * t
            })
            .reduce(|a, b| a + b)
            .unwrap()
            .column_iter()
            .map(Coord::<D>::to_coord)
            .collect()
    }
}

impl<D: EfdDim> core::fmt::Debug for Efd<D>
where
    Transform<D::Trans>: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("Efd")
            .field("coeff", &CoeffFmt::<D>(&self.coeffs))
            .field("trans", &self.trans)
            .field("dim", &D::Trans::dim())
            .field("harmonic", &self.harmonic())
            .finish()
    }
}

struct CoeffFmt<'a, D: EfdDim>(&'a Coeff<D>);

impl<D: EfdDim> core::fmt::Debug for CoeffFmt<'_, D> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        let entries = self
            .0
            .column_iter()
            .map(|c| c.iter().copied().collect::<Vec<_>>());
        f.debug_list().entries(entries).finish()
    }
}

fn padding<D, F>(a: &Efd<D>, b: &Efd<D>, f: F) -> f64
where
    D: EfdDim,
    F: Fn(&Coeff<D>, &Coeff<D>) -> f64,
{
    use core::cmp::Ordering::*;
    match a.harmonic().cmp(&b.harmonic()) {
        Equal => f(&a.coeffs, &b.coeffs),
        Greater => {
            let b_coeffs = b.coeffs.clone().resize_horizontally(a.harmonic(), 0.);
            f(&a.coeffs, &b_coeffs)
        }
        Less => {
            let a_coeffs = a.coeffs.clone().resize_horizontally(b.harmonic(), 0.);
            f(&a_coeffs, &b.coeffs)
        }
    }
}
