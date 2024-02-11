use crate::{util::*, *};
use alloc::{format, vec::Vec};
use core::f64::consts::{PI, TAU};
#[cfg(not(feature = "std"))]
use num_traits::*;

/// Get the theta value `t` of each point coordinate and the normalized
/// geometric variables of the curve.
///
/// This function is faster than building [`Efd`] since it only calculates **one
/// harmonic**.
///
/// ```
/// use efd::get_target_pos;
///
/// let curve = [[0., 0.], [1., 0.], [1., 1.], [0., 1.]];
/// let (t, _geo) = get_target_pos(curve, true);
/// assert_eq!(t.len(), 4);
/// ```
///
/// See also [`Efd::from_curve_harmonic_and_get()`] if you want to get the
/// coefficients.
pub fn get_target_pos<C, const D: usize>(curve: C, is_open: bool) -> (Vec<f64>, GeoVar<Rot<D>, D>)
where
    C: Curve<D>,
    U<D>: EfdDim<D>,
{
    let (mut t, mut coeffs, geo) = U::get_coeff(curve.as_curve(), is_open, 1, None);
    let geo = geo * U::coeff_norm(&mut coeffs, Some(&mut t), None);
    (t, geo)
}

/// A 1D shape described by EFD.
pub type Efd1 = Efd<1>;
/// A 2D shape described by EFD.
pub type Efd2 = Efd<2>;
/// A 3D shape described by EFD.
pub type Efd3 = Efd<3>;

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
/// Please see [`GeoVar`] for more information.
///
/// # Raw Coefficients
///
/// The coefficients is contained with [`na::Matrix`], use
/// [`Efd::try_from_coeffs()`] to input the coefficients externally.
///
/// See also [`Efd::from_parts_unchecked()`] and [`Efd::into_inner()`] without
/// checking data.
pub struct Efd<const D: usize>
where
    U<D>: EfdDim<D>,
{
    coeffs: Coeffs<D>,
    geo: GeoVar<Rot<D>, D>,
}

impl<const D: usize> Efd<D>
where
    U<D>: EfdDim<D>,
{
    /// Create object from coefficients and geometric variables.
    ///
    /// Zero harmonic is allowed but meaningless. If the harmonic is zero, some
    /// operations will panic.
    ///
    /// ```
    /// use efd::{Efd2, GeoVar};
    /// let curve = Efd2::from_parts_unchecked(vec![], GeoVar::identity()).generate(20);
    /// assert_eq!(curve.len(), 0);
    /// ```
    ///
    /// See also [`Efd::into_inner()`].
    pub const fn from_parts_unchecked(coeffs: Coeffs<D>, geo: GeoVar<Rot<D>, D>) -> Self {
        Self { coeffs, geo }
    }

    /// Create object from a matrix with boundary check and normalization.
    ///
    /// The array size is (harmonic) x (dimension x 2). The dimension is `D`.
    ///
    /// Return none if the harmonic is zero.
    pub fn try_from_coeffs(mut coeffs: Coeffs<D>) -> Option<Self> {
        (!coeffs.is_empty()).then(|| Self {
            geo: U::coeff_norm(&mut coeffs, None, None),
            coeffs,
        })
    }

    /// Create object from a matrix with boundary check.
    ///
    /// The array size is (harmonic) x (dimension x 2). The dimension is `D`.
    ///
    /// Return none if the harmonic is zero.
    pub fn try_from_coeffs_unnorm(coeffs: Coeffs<D>) -> Option<Self> {
        (!coeffs.is_empty()).then_some(Self { coeffs, geo: GeoVar::identity() })
    }

    /// Fully automated coefficient calculation.
    ///
    /// 1. The initial harmonic number is the same as the curve point.
    /// 1. Fourier Power Anaysis (FPA) uses 99.99% threshold.
    ///
    /// # Tail End Closed
    ///
    /// If `curve.first() != curve.last()`, the curve will be automatically
    /// closed when `is_open` is false.
    ///
    /// # Open Curve Option
    ///
    /// The open curve option is for the curve that duplicated a reversed part
    /// of itself. For example,
    ///
    /// ```no_run
    /// # let curve_open = [];
    /// let efd = efd::Efd2::from_curve(curve_open, true);
    /// ```
    ///
    /// is equivalent to
    ///
    /// ```no_run
    /// # let curve_open = [];
    /// let curve_closed = curve_open
    ///     .iter()
    ///     .chain(curve_open.iter().rev().skip(1))
    ///     .cloned()
    ///     .collect::<Vec<_>>();
    /// let efd = efd::Efd2::from_curve(curve_closed, false);
    /// ```
    ///
    /// but not actually increase the data size.
    ///
    /// # Panics
    ///
    /// Panics if the curve length is not greater than 2 in debug mode. This
    /// function check the lengths only. Please use [`valid_curve()`] to
    /// verify the curve if there has NaN input.
    pub fn from_curve<C>(curve: C, is_open: bool) -> Self
    where
        C: Curve<D>,
    {
        let harmonic = harmonic!(is_open, curve);
        Self::from_curve_harmonic(curve, is_open, harmonic).fourier_power_anaysis(None)
    }

    /// Same as [`Efd::from_curve()`], but if your sampling points are large,
    /// use Nyquist Frequency as an initial harmonic number.
    ///
    /// Nyquist Frequency is half of the sample number.
    ///
    /// Please ensure the sampling points are generated from a known function
    /// and are more than enough. Otherwise, it will cause undersampling.
    ///
    /// See also [`harmonic_nyquist`].
    pub fn from_curve_nyquist<C>(curve: C, is_open: bool) -> Self
    where
        C: Curve<D>,
    {
        let harmonic = harmonic_nyquist!(is_open, curve);
        Self::from_curve_harmonic(curve, is_open, harmonic).fourier_power_anaysis(None)
    }

    /// Manual coefficient calculation.
    ///
    /// 1. The initial harmonic is decide by user.
    /// 1. No harmonic reduced. Please call [`Efd::fourier_power_anaysis()`].
    ///
    /// # Panics
    ///
    /// Panics if the specific harmonic is zero or the curve length is not
    /// greater than 2 in the **debug mode**. This function check the lengths
    /// only. Please use [`valid_curve()`] to verify the curve if there has
    /// NaN input.
    pub fn from_curve_harmonic<C>(curve: C, is_open: bool, harmonic: usize) -> Self
    where
        C: Curve<D>,
    {
        Self::from_curve_harmonic_and_get(curve, is_open, harmonic).0
    }

    /// Same as [`Efd::from_curve_harmonic()`] but get the the theta value of
    /// each point coordinate of the curve.
    ///
    /// See also [`get_target_pos()`] if you want to ignore the coefficients.
    ///
    /// # Panics
    ///
    /// Please see [`Efd::from_curve_harmonic()`].
    pub fn from_curve_harmonic_and_get<C>(
        curve: C,
        is_open: bool,
        harmonic: usize,
    ) -> (Self, Vec<f64>)
    where
        C: Curve<D>,
    {
        debug_assert!(harmonic != 0, "harmonic must not be 0");
        debug_assert!(curve.len() > 2, "the curve length must greater than 2");
        let (t, mut coeffs, geo) = U::get_coeff(curve.as_curve(), is_open, harmonic, None);
        let geo = geo * U::coeff_norm(&mut coeffs, None, None);
        (Self { coeffs, geo }, t)
    }

    /// Same as [`Efd::from_curve_harmonic()`] but without normalization.
    pub fn from_curve_harmonic_unnorm<C>(curve: C, is_open: bool, harmonic: usize) -> Self
    where
        C: Curve<D>,
    {
        debug_assert!(harmonic != 0, "harmonic must not be 0");
        debug_assert!(curve.len() > 2, "the curve length must greater than 2");
        let (_, coeffs, geo) = U::get_coeff(curve.as_curve(), is_open, harmonic, None);
        Self { coeffs, geo }
    }

    /// A builder method for changing geometric variables.
    pub fn with_geo(self, geo: GeoVar<Rot<D>, D>) -> Self {
        Self { geo, ..self }
    }

    /// A builder method using Fourier Power Anaysis (FPA) to reduce the
    /// harmonic number.
    ///
    /// The coefficient memory will be saved but cannot be used twice due to
    /// undersampling.
    ///
    /// The default threshold is 99.99%.
    ///
    /// # Panics
    ///
    /// Panics if the threshold is not in 0..1, or the harmonic is zero.
    pub fn fourier_power_anaysis<T>(mut self, threshold: T) -> Self
    where
        Option<f64>: From<T>,
    {
        let lut = self.coeffs.iter().map(|m| m.map(pow2).sum()).collect();
        self.set_harmonic(fourier_power_anaysis(lut, threshold));
        self
    }

    /// Set the harmonic number of the coefficients.
    ///
    /// # Panics
    ///
    /// Panics if the harmonic is zero or greater than the current harmonic.
    pub fn set_harmonic(&mut self, harmonic: usize) {
        let current = self.harmonic();
        assert!(
            (1..=current).contains(&harmonic),
            "harmonic must in 1..={current}"
        );
        self.coeffs.resize_with(harmonic, Kernel::zeros);
    }

    /// Force normalize the coefficients.
    ///
    /// If the coefficients are constructed by `*_unnorm` or `*_unchecked`
    /// methods, this method will normalize them.
    ///
    /// # Panics
    ///
    /// Panics if the harmonic is zero.
    pub fn normalized(self) -> Self {
        let Self { mut coeffs, geo } = self;
        let geo = geo * U::coeff_norm(&mut coeffs, None, None);
        Self { coeffs, geo }
    }

    /// Consume self and return the parts of this type.
    ///
    /// See also [`Efd::from_parts_unchecked()`].
    pub fn into_inner(self) -> (Coeffs<D>, GeoVar<Rot<D>, D>) {
        (self.coeffs, self.geo)
    }

    /// Get a reference to the coefficients.
    pub fn coeffs(&self) -> &[Kernel<D>] {
        &self.coeffs
    }

    /// Get a view to the specific coefficients. (`0..self.harmonic()`)
    pub fn coeff(&self, harmonic: usize) -> &Kernel<D> {
        &self.coeffs[harmonic]
    }

    /// Get an iterator over all the coefficients per harmonic.
    pub fn coeffs_iter(&self) -> impl Iterator<Item = &Kernel<D>> {
        self.coeffs.iter()
    }

    /// Get a mutable iterator over all the coefficients per harmonic.
    pub fn coeffs_iter_mut(&mut self) -> impl Iterator<Item = &mut Kernel<D>> {
        self.coeffs.iter_mut()
    }

    /// Get the reference of geometric variables.
    pub fn as_geo(&self) -> &GeoVar<Rot<D>, D> {
        &self.geo
    }

    /// Get the mutable reference of geometric variables.
    pub fn as_geo_mut(&mut self) -> &mut GeoVar<Rot<D>, D> {
        &mut self.geo
    }

    /// Check if the descibed curve is open.
    pub fn is_open(&self) -> bool {
        self.coeffs[0].column(1).sum() == 0.
    }

    /// Get the harmonic number of the coefficients.
    #[inline]
    pub fn harmonic(&self) -> usize {
        self.coeffs.len()
    }

    /// Check if the coefficients are valid.
    ///
    /// + The harmonic number must be greater than 0.
    /// + All the coefficients must not be `NaN` or zero.
    ///
    /// It is only helpful if this object is constructed by
    /// [`Efd::from_parts_unchecked()`].
    pub fn is_valid(&self) -> bool {
        !self.coeffs.is_empty()
            && !self
                .coeffs_iter()
                .any(|m| m.iter().all(|x| x.abs() < f64::EPSILON) || m.iter().any(|x| x.is_nan()))
    }

    /// Calculate the L1 distance between two coefficient set.
    ///
    /// For more distance methods, please see [`Distance`].
    pub fn distance(&self, rhs: &Self) -> f64 {
        self.l1_norm(rhs)
    }

    /// Reverse the order of described curve then return a mutable reference.
    pub fn reverse_inplace(&mut self) {
        for m in &mut self.coeffs {
            let mut m = m.column_mut(1);
            m *= -1.
        }
    }

    /// Consume and return a reversed version of the coefficients. This method
    /// can avoid mutable require.
    ///
    /// Please clone the object if you want to do self-comparison.
    pub fn reversed(mut self) -> Self {
        self.reverse_inplace();
        self
    }

    /// Generate (reconstruct) the described curve. (`theta=0~TAU`)
    ///
    /// # Panics
    ///
    /// Panics if the number of the points `n` is less than 2.
    pub fn generate(&self, n: usize) -> Vec<Coord<D>> {
        self.generate_in(n, TAU)
    }

    /// Generate (reconstruct) a half of the described curve. (`theta=0~PI`)
    ///
    /// # Panics
    ///
    /// Panics if the number of the points `n` is less than 2.
    pub fn generate_half(&self, n: usize) -> Vec<Coord<D>> {
        self.generate_in(n, PI)
    }

    fn generate_in(&self, n: usize, theta: f64) -> Vec<Coord<D>> {
        let mut curve = self.generate_norm_in(n, theta);
        self.geo.transform_inplace(&mut curve);
        curve
    }

    /// Generate (reconstruct) the described curve. (`theta=0~TAU`)
    ///
    /// Normalized curve is **without** transformation.
    ///
    /// # Panics
    ///
    /// Panics if the number of the points `n` is less than 2.
    pub fn generate_norm(&self, n: usize) -> Vec<Coord<D>> {
        self.generate_norm_in(n, TAU)
    }

    /// Generate (reconstruct) a half of the described curve. (`t=0~PI`)
    ///
    /// Normalized curve is **without** transformation.
    ///
    /// # Panics
    ///
    /// Panics if the number of the points `n` is less than 2.
    pub fn generate_norm_half(&self, n: usize) -> Vec<Coord<D>> {
        self.generate_norm_in(n, PI)
    }

    fn generate_norm_in(&self, n: usize, theta: f64) -> Vec<Coord<D>> {
        debug_assert!(n > 2, "n ({n}) must larger than 2");
        let iter = (0..n).map(|i| i as f64 / (n - 1) as f64 * theta);
        U::reconstruct(&self.coeffs, iter)
    }

    /// Generate (reconstruct) a described curve in a time series `t`.
    pub fn generate_by(&self, t: &[f64]) -> Vec<Coord<D>> {
        let mut curve = U::reconstruct(&self.coeffs, t.iter().copied());
        self.geo.transform_inplace(&mut curve);
        curve
    }

    /// Generate (reconstruct) a normalized curve in a time series `t`.
    ///
    /// Normalized curve is **without** transformation.
    pub fn generate_norm_by(&self, t: &[f64]) -> Vec<Coord<D>> {
        U::reconstruct(&self.coeffs, t.iter().copied())
    }

    /// Generate (reconstruct) a described curve in a normalized time series
    /// `t`.
    ///
    /// If the input angle is obtained from [`get_target_pos()`], the
    /// reconstruction must use this method.
    pub fn generate_by_t(&self, t: &[f64]) -> Vec<Coord<D>> {
        let mut curve = self.generate_norm_by_t(t);
        self.geo.transform_inplace(&mut curve);
        curve
    }

    /// Generate (reconstruct) a normalized curve in a normalized time series
    /// `t`.
    ///
    /// If the input angle is obtained from [`get_target_pos()`], the
    /// reconstruction must use this method.
    pub fn generate_norm_by_t(&self, t: &[f64]) -> Vec<Coord<D>> {
        if self.is_open() || self.harmonic() <= 1 {
            self.generate_norm_by(t)
        } else {
            U::reconstruct(&self.coeffs, t.iter().map(|t| t + PI))
        }
    }
}

impl<const D: usize> Clone for Efd<D>
where
    U<D>: EfdDim<D>,
{
    fn clone(&self) -> Self {
        Self { coeffs: self.coeffs.clone(), geo: self.geo.clone() }
    }
}

impl<const D: usize> core::fmt::Debug for Efd<D>
where
    U<D>: EfdDim<D>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        if self.is_valid() {
            f.debug_struct(&format!("Efd{D}"))
                .field("is_open", &self.is_open())
                .field("harmonic", &self.harmonic())
                .field("geo", &self.geo)
                .field("coeff", &CoeffFmt(&self.coeffs))
                .finish()
        } else {
            f.debug_struct(&format!("Efd{D}"))
                .field("is_valid", &false)
                .finish()
        }
    }
}

impl<const D: usize> core::fmt::Debug for PosedEfd<D>
where
    U<D>: EfdDim<D>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct(&format!("PosedEfd{D}"))
            .field("is_open", &self.is_open())
            .field("harmonic", &self.harmonic())
            .field("geo", &self.geo)
            .field("coeff", &CoeffFmt(&self.coeffs))
            .finish()
    }
}

struct CoeffFmt<'a, const D: usize>(&'a Coeffs<D>);

impl<const D: usize> core::fmt::Debug for CoeffFmt<'_, D> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        let entries = self.0.iter().map(|c| c.iter().copied().collect::<Vec<_>>());
        f.debug_list().entries(entries).finish()
    }
}

pub(crate) fn fourier_power_anaysis<T>(lut: Vec<f64>, threshold: T) -> usize
where
    Option<f64>: From<T>,
{
    let threshold = Option::from(threshold).unwrap_or(0.9999);
    assert!((0.0..1.0).contains(&threshold), "threshold must in 0..1");
    let lut = cumsum(na::Matrix1xX::from_vec(lut));
    let target = lut[lut.len() - 1] * threshold;
    match lut
        .as_slice()
        .binary_search_by(|x| x.partial_cmp(&target).unwrap())
    {
        Ok(h) | Err(h) => h + 1,
    }
}
