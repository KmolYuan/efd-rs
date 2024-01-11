use crate::{util::*, *};
use alloc::{format, vec::Vec};
use core::f64::consts::{PI, TAU};
#[cfg(not(feature = "std"))]
use num_traits::*;

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
/// The coefficients is contained with `na::Matrix`, use
/// [`Efd::try_from_coeffs()`] to input the coefficients externally.
///
/// Use [`Efd::into_inner()`] to get the matrix of the coefficients.
pub struct Efd<const D: usize>
where
    U<D>: EfdDim<D>,
    na::Const<D>: na::DimNameMul<na::U2>,
{
    coeffs: Coeffs<D>,
    geo: GeoVar<Rot<D>, D>,
}

impl<const D: usize> Efd<D>
where
    U<D>: EfdDim<D>,
    na::Const<D>: na::DimNameMul<na::U2>,
{
    /// Create object from a matrix with boundary check and normalization.
    ///
    /// The array size is (harmonic) x (dimension x 2). The dimension is `D`.
    ///
    /// Return none if the harmonic is zero.
    pub fn try_from_coeffs(mut coeffs: Coeffs<D>) -> Option<Self> {
        (coeffs.ncols() != 0).then(|| Self { geo: U::<D>::coeff_norm(&mut coeffs), coeffs })
    }

    /// Create object from a matrix with boundary check.
    ///
    /// The array size is (harmonic) x (dimension x 2). The dimension is `D`.
    ///
    /// Return none if the harmonic is zero.
    pub fn try_from_coeffs_unnorm(coeffs: Coeffs<D>) -> Option<Self> {
        (coeffs.ncols() != 0).then_some(Self { coeffs, geo: GeoVar::identity() })
    }

    /// Create object from a matrix directly.
    ///
    /// The array size is (harmonic) x (dimension x 2). The dimension is `D`.
    ///
    /// Zero harmonic is allowed but meaningless. If the harmonic is zero, some
    /// operations will panic.
    ///
    /// ```
    /// use efd::{Coeffs2, Efd2, GeoVar};
    /// let coeffs = Coeffs2::from_column_slice(&[]);
    /// let path = Efd2::from_parts_unchecked(coeffs, GeoVar::identity()).generate(20);
    /// assert_eq!(path.len(), 0);
    /// ```
    pub const fn from_parts_unchecked(coeffs: Coeffs<D>, geo: GeoVar<Rot<D>, D>) -> Self {
        Self { coeffs, geo }
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
    /// # let path_open = [];
    /// let efd = efd::Efd2::from_curve(path_open, true);
    /// ```
    ///
    /// is equivalent to
    ///
    /// ```no_run
    /// # let path_open = [];
    /// let path_closed = path_open
    ///     .iter()
    ///     .chain(path_open.iter().rev().skip(1))
    ///     .cloned()
    ///     .collect::<Vec<_>>();
    /// let efd = efd::Efd2::from_curve(path_closed, false);
    /// ```
    ///
    /// but not actually increase the data size.
    ///
    /// # Panics
    ///
    /// Panics if the curve length is not greater than 2 in debug mode. This
    /// function check the lengths only. Please use [`valid_curve()`] to
    /// verify the curve if there has NaN input.
    #[must_use]
    pub fn from_curve<C>(curve: C, is_open: bool) -> Self
    where
        C: Curve<Coord<D>>,
    {
        let len = curve.len();
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
        let len = curve.len();
        Self::from_curve_harmonic(curve, is_open, if is_open { len } else { len / 2 })
            .fourier_power_anaysis(None)
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
    #[must_use]
    pub fn from_curve_harmonic<C>(curve: C, is_open: bool, harmonic: usize) -> Self
    where
        C: Curve<Coord<D>>,
    {
        debug_assert!(harmonic != 0, "harmonic must not be 0");
        debug_assert!(curve.len() > 2, "the curve length must greater than 2");
        let curve = curve.as_curve();
        let [(mut coeffs, geo1)] = U::<D>::get_coeff([curve], is_open, harmonic);
        let geo2 = U::<D>::coeff_norm(&mut coeffs);
        Self { coeffs, geo: geo1 * geo2 }
    }

    /// Same as [`Efd::from_curve_harmonic()`] but without normalization.
    #[must_use]
    pub fn from_curve_harmonic_unnorm<C>(curve: C, is_open: bool, harmonic: usize) -> Self
    where
        C: Curve<Coord<D>>,
    {
        debug_assert!(harmonic != 0, "harmonic must not be 0");
        debug_assert!(curve.len() > 2, "the curve length must greater than 2");
        let curve = curve.as_curve();
        let [(coeffs, geo)] = U::<D>::get_coeff([curve], is_open, harmonic);
        Self { coeffs, geo }
    }

    /// A builder method for changing geometric variables.
    #[must_use]
    pub fn with_geo(self, geo: GeoVar<Rot<D>, D>) -> Self {
        Self { geo, ..self }
    }

    /// Use Fourier Power Anaysis (FPA) to reduce the harmonic number.
    ///
    /// The coefficient memory will be saved but cannot be used twice due to
    /// undersampling.
    ///
    /// The default threshold is 99.99%.
    ///
    /// # Panics
    ///
    /// Panics if the threshold is not in 0..1, or the harmonic is zero.
    #[must_use]
    pub fn fourier_power_anaysis<T>(mut self, threshold: T) -> Self
    where
        Option<f64>: From<T>,
    {
        let lut = self.coeffs.map(pow2).row_sum();
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
        self.coeffs.resize_horizontally_mut(harmonic, 0.);
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
        let geo_new = U::<D>::coeff_norm(&mut coeffs);
        Self { coeffs, geo: geo.apply(&geo_new) }
    }

    /// Consume self and return a raw array of the coefficients.
    #[must_use]
    pub fn into_inner(self) -> (Coeffs<D>, GeoVar<Rot<D>, D>) {
        (self.coeffs, self.geo)
    }

    /// Get a reference to the coefficients.
    #[must_use]
    pub fn coeffs(&self) -> &Coeffs<D> {
        &self.coeffs
    }

    /// Get a view to the specific coefficients. (`0..self.harmonic()`)
    #[must_use]
    pub fn coeff(&self, harmonic: usize) -> CKernel<D> {
        self.coeffs
            .column(harmonic)
            .reshape_generic(na::Const, na::U2)
    }

    /// Get an iterator over all the coefficients per harmonic.
    pub fn coeffs_iter(&self) -> impl Iterator<Item = CKernel<D>> {
        self.coeffs
            .column_iter()
            .map(|c| c.reshape_generic(na::Const, na::U2))
    }

    /// Get a mutable iterator over all the coefficients per harmonic.
    pub fn coeffs_iter_mut(&mut self) -> impl Iterator<Item = CKernelMut<D>> {
        self.coeffs
            .column_iter_mut()
            .map(|c| c.reshape_generic(na::Const, na::U2))
    }

    /// Get the reference of geometric variables.
    #[must_use]
    pub fn as_geo(&self) -> &GeoVar<Rot<D>, D> {
        &self.geo
    }

    /// Get the mutable reference of geometric variables.
    #[must_use]
    pub fn as_geo_mut(&mut self) -> &mut GeoVar<Rot<D>, D> {
        &mut self.geo
    }

    /// Check if the descibed curve is open.
    #[must_use]
    pub fn is_open(&self) -> bool {
        self.coeffs[(1, 0)] == 0.
    }

    /// Get the harmonic number of the coefficients.
    #[must_use]
    pub fn harmonic(&self) -> usize {
        self.coeffs.ncols()
    }

    /// Check if the coefficients are valid.
    ///
    /// + The harmonic number must be greater than 0.
    /// + All the coefficients must not be `NaN` or zero.
    ///
    /// It is only helpful if this object is constructed by
    /// [`Efd::from_parts_unchecked()`].
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.harmonic() > 0
            && !self
                .coeffs_iter()
                .any(|m| m.iter().all(|x| x.abs() < f64::EPSILON) || m.iter().any(|x| x.is_nan()))
    }

    /// Calculate the L1 distance between two coefficient set.
    ///
    /// For more distance methods, please see [`Distance`].
    #[must_use]
    pub fn distance(&self, rhs: &Self) -> f64 {
        self.l1_norm(rhs)
    }

    /// Reverse the order of described curve then return a mutable reference.
    pub fn reverse_inplace(&mut self) {
        self.coeffs
            .row_iter_mut()
            .skip(D)
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

    /// Generate (reconstruct) the described curve. (`theta=0~TAU`)
    ///
    /// # Panics
    ///
    /// Panics if the number of the points `n` is less than 2.
    #[must_use]
    pub fn generate(&self, n: usize) -> Vec<Coord<D>> {
        self.generate_in(n, TAU)
    }

    /// Generate (reconstruct) a half of the described curve. (`theta=0~PI`)
    ///
    /// # Panics
    ///
    /// Panics if the number of the points `n` is less than 2.
    #[must_use]
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
    #[must_use]
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
    #[must_use]
    pub fn generate_norm_half(&self, n: usize) -> Vec<Coord<D>> {
        self.generate_norm_in(n, PI)
    }

    fn generate_norm_in(&self, n: usize, theta: f64) -> Vec<Coord<D>> {
        debug_assert!(n > 2, "n ({n}) must larger than 2");
        let t = na::Matrix1xX::from_fn(n, |_, i| i as f64 / (n - 1) as f64 * theta);
        U::<D>::reconstruct(&self.coeffs, t)
    }

    /// Generate (reconstruct) a described curve in a series of time `t`.
    #[must_use]
    pub fn generate_by(&self, t: &[f64]) -> Vec<Coord<D>> {
        let mut curve = U::<D>::reconstruct(&self.coeffs, na::Matrix1xX::from_column_slice(t));
        self.geo.transform_inplace(&mut curve);
        curve
    }

    /// Generate (reconstruct) a normalized curve in a series of time `t`.
    ///
    /// Normalized curve is **without** transformation.
    #[must_use]
    pub fn generate_norm_by(&self, t: &[f64]) -> Vec<Coord<D>> {
        U::<D>::reconstruct(&self.coeffs, na::Matrix1xX::from_column_slice(t))
    }
}

impl<const D: usize> Clone for Efd<D>
where
    U<D>: EfdDim<D>,
    na::Const<D>: na::DimNameMul<na::U2>,
{
    fn clone(&self) -> Self {
        Self { coeffs: self.coeffs.clone(), geo: self.geo.clone() }
    }
}

impl<const D: usize> core::fmt::Debug for Efd<D>
where
    U<D>: EfdDim<D>,
    na::Const<D>: na::DimNameMul<na::U2>,
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
    na::Const<D>: na::DimNameMul<na::U2>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct(&format!("PosedEfd{D}"))
            .field("is_open", &self.is_open())
            .field("harmonic", &self.harmonic())
            .field("curve_geo", &self.curve_efd().geo)
            .field("curve_coeff", &CoeffFmt(&self.curve_efd().coeffs))
            .field("pose_geo", &self.pose_efd().geo)
            .field("pose_coeff", &CoeffFmt(&self.pose_efd().coeffs))
            .finish()
    }
}

struct CoeffFmt<'a, const D: usize>(&'a Coeffs<D>)
where
    na::Const<D>: na::DimNameMul<na::U2>;

impl<const D: usize> core::fmt::Debug for CoeffFmt<'_, D>
where
    na::Const<D>: na::DimNameMul<na::U2>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        let entries = self
            .0
            .column_iter()
            .map(|c| c.iter().copied().collect::<Vec<_>>());
        f.debug_list().entries(entries).finish()
    }
}

pub(crate) fn fourier_power_anaysis<T>(lut: na::Matrix1xX<f64>, threshold: T) -> usize
where
    Option<f64>: From<T>,
{
    let threshold = Option::from(threshold).unwrap_or(0.9999);
    assert!((0.0..1.0).contains(&threshold), "threshold must in 0..1");
    let lut = cumsum(lut);
    let target = lut[lut.len() - 1] * threshold;
    match lut
        .as_slice()
        .binary_search_by(|x| x.partial_cmp(&target).unwrap())
    {
        Ok(h) | Err(h) => h + 1,
    }
}
