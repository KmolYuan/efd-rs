//! Posed EFD (Elliptic Fourier Descriptor) is a special shape to describe the
//! pose of a curve. It is a combination of two curves, the first is the
//! original curve, and the second is the pose unit vectors.
//!
//! Please see the [`PosedEfd`] type for more information.
use crate::*;
use alloc::vec::Vec;
use core::{array, iter::zip};
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::*;

/// A 1D shape with a pose described by EFD.
pub type PosedEfd1 = PosedEfd<1>;
/// A 2D shape with a pose described by EFD.
pub type PosedEfd2 = PosedEfd<2>;
/// A 3D shape with a pose described by EFD.
pub type PosedEfd3 = PosedEfd<3>;

/// A global setting controls the posed EFD is using open curve as the signature
/// or not.
pub const IS_OPEN: bool = true;

/// Calculate the number of harmonics for posed EFD.
///
/// The number of harmonics is calculated by the minimum length of the curves.
/// And if the curve is open accroding to [`IS_OPEN`], the number is doubled.
///
/// ```
/// assert_eq!(efd::posed::harmonic(2, 3), 14);
/// ```
#[inline]
pub const fn harmonic(len1: usize, len2: usize) -> usize {
    let len = len1 + len2 + 2;
    if IS_OPEN {
        len * 2
    } else {
        len
    }
}

/// Transform 2D angles to unit vectors.
pub fn ang2vec(angles: &[f64]) -> Vec<Coord<2>> {
    angles.iter().map(|a| [a.cos(), a.sin()]).collect()
}

/// Get the path signature and its target position from a curve and its unit
/// vectors.
///
/// ```
/// use efd::posed::{ang2vec, path_signature};
/// # let curve = efd::tests::CURVE2D_POSE;
/// # let angles = efd::tests::ANGLE2D_POSE;
///
/// let (sig, t, geo) = path_signature(curve, ang2vec(angles), true);
/// ```
/// See also [`get_target_pos()`].
pub fn path_signature<C, V, const D: usize>(
    curve: C,
    vectors: V,
    is_open: bool,
) -> (Vec<Coord<D>>, Vec<f64>, GeoVar<Rot<D>, D>)
where
    U<D>: EfdDim<D>,
    C: Curve<D>,
    V: Curve<D>,
{
    let (sig, guide, geo1) = impl_path_signature(curve, vectors, is_open);
    let (mut t, mut coeffs, geo2) = U::get_coeff(&sig, IS_OPEN, 1, Some(&guide));
    // Only normalize the target position
    U::coeff_norm(&mut coeffs, Some(&mut t));
    (sig, t, geo1 * geo2)
}

fn impl_path_signature<C, V, const D: usize>(
    curve: C,
    vectors: V,
    is_open: bool,
) -> (Vec<Coord<D>>, Vec<f64>, GeoVar<Rot<D>, D>)
where
    U<D>: EfdDim<D>,
    C: Curve<D>,
    V: Curve<D>,
{
    // Get the length of the unit vectors
    let length = vectors.as_curve()[0].l2_norm();
    let (_, geo) = get_target_pos(curve.as_curve(), is_open);
    let geo_inv = geo.inverse();
    let mut sig = geo_inv.transform(curve);
    let dxyz = zip(&sig, &sig[1..])
        .map(|(a, b)| a.l2_err(b))
        .collect::<Vec<_>>();
    let mut guide = dxyz.clone();
    guide.reserve(dxyz.len() + 2);
    guide.push(length);
    guide.extend(dxyz.into_iter().rev());
    if !IS_OPEN {
        guide.push(length);
    }
    let vectors = geo_inv
        .only_rot()
        .with_scale(length.recip())
        .transform(vectors);
    for (i, v) in vectors.into_iter().enumerate().rev() {
        let p = &sig[i];
        sig.push(array::from_fn(|i| p[i] + length * v[i]));
    }
    (sig, guide, geo)
}

/// A shape with a pose described by EFD.
///
/// These are the same as [`Efd`] except that it has a pose, and the data are
/// always normalized and readonly.
///
/// Start with [`PosedEfd::from_series()`] and its related methods.
///
/// # Pose Representation
/// Pose is represented by an unit vector, which is rotated by the rotation
/// of the original shape.
#[derive(Clone)]
pub struct PosedEfd<const D: usize>
where
    U<D>: EfdDim<D>,
{
    efd: Efd<D>,
    is_open: bool,
}

impl PosedEfd2 {
    /// Calculate the coefficients from a curve and its angles from each point.
    pub fn from_angles<C>(curve: C, angles: &[f64], is_open: bool) -> Self
    where
        C: Curve<2>,
    {
        let harmonic = harmonic(curve.len(), angles.len());
        Self::from_angles_harmonic(curve, angles, is_open, harmonic).fourier_power_anaysis(None)
    }

    /// Calculate the coefficients from a curve and its angles from each point.
    ///
    /// The `harmonic` is the number of the coefficients to be calculated.
    pub fn from_angles_harmonic<C>(curve: C, angles: &[f64], is_open: bool, harmonic: usize) -> Self
    where
        C: Curve<2>,
    {
        Self::from_uvec_harmonic(curve, ang2vec(angles), is_open, harmonic)
    }
}

impl<const D: usize> PosedEfd<D>
where
    U<D>: EfdDim<D>,
{
    /// Create object from an [`Efd`] object.
    ///
    /// Posed EFD is a special shape to describe the pose, `efd` is only used to
    /// describe this motion signature.
    ///
    /// See also [`PosedEfd::into_inner()`].
    pub const fn from_efd(efd: Efd<D>, is_open: bool) -> Self {
        Self { efd, is_open }
    }

    /// Calculate the coefficients from two series of points.
    ///
    /// The second series is the pose series, the `curve2[i]` has the same time
    /// as `curve[i]`.
    pub fn from_series<C1, C2>(curve1: C1, curve2: C2, is_open: bool) -> Self
    where
        C1: Curve<D>,
        C2: Curve<D>,
    {
        let harmonic = harmonic(curve1.len(), curve2.len());
        Self::from_series_harmonic(curve1, curve2, is_open, harmonic).fourier_power_anaysis(None)
    }

    /// Calculate the coefficients from two series of points.
    ///
    /// The `harmonic` is the number of the coefficients to be calculated.
    pub fn from_series_harmonic<C1, C2>(
        curve1: C1,
        curve2: C2,
        is_open: bool,
        harmonic: usize,
    ) -> Self
    where
        C1: Curve<D>,
        C2: Curve<D>,
    {
        let vectors = zip(curve1.as_curve(), curve2.as_curve())
            .map(|(a, b)| array::from_fn(|i| b[i] - a[i]))
            .collect::<Vec<_>>();
        Self::from_uvec_harmonic(curve1, vectors, is_open, harmonic)
    }

    /// Calculate the coefficients from a curve and its unit vectors from each
    /// point.
    ///
    /// If the unit vectors is not normalized, the length of the first vector
    /// will be used as the scaling factor.
    pub fn from_uvec<C, V>(curve: C, vectors: V, is_open: bool) -> Self
    where
        C: Curve<D>,
        V: Curve<D>,
    {
        let harmonic = harmonic(curve.len(), vectors.len());
        Self::from_uvec_harmonic(curve, vectors, is_open, harmonic).fourier_power_anaysis(None)
    }

    /// Calculate the coefficients from a curve and its unit vectors from each
    /// point.
    ///
    /// If the unit vectors is not normalized, the length of the first vector
    /// will be used as the scaling factor.
    ///
    /// The `harmonic` is the number of the coefficients to be calculated.
    pub fn from_uvec_harmonic<C, V>(curve: C, vectors: V, is_open: bool, harmonic: usize) -> Self
    where
        C: Curve<D>,
        V: Curve<D>,
    {
        debug_assert!(harmonic != 0, "harmonic must not be 0");
        debug_assert!(curve.len() > 2, "the curve length must greater than 2");
        let (sig, guide, geo1) = impl_path_signature(curve, vectors, is_open);
        let (_, coeffs, geo2) = U::get_coeff(&sig, IS_OPEN, harmonic, Some(&guide));
        let efd = Efd::from_parts_unchecked(coeffs, geo1 * geo2);
        Self { efd, is_open }
    }

    /// Use Fourier Power Anaysis (FPA) to reduce the harmonic number.
    ///
    /// The default threshold is 99.99%.
    ///
    /// See also [`Efd::fourier_power_anaysis()`].
    ///
    /// # Panics
    ///
    /// Panics if the threshold is not in 0..1, or the harmonic is zero.
    pub fn fourier_power_anaysis<T>(mut self, threshold: T) -> Self
    where
        Option<f64>: From<T>,
    {
        self.efd = self.efd.fourier_power_anaysis(threshold);
        self
    }

    /// Check if the descibed curve is open.
    ///
    /// Unlike [`Efd::is_open()`], this method is not the `is_open` of the
    /// coefficients.
    pub const fn is_open(&self) -> bool {
        self.is_open
    }

    /// Consume self and return the parts of this type. The first is the curve
    /// coefficients, and the second is the pose coefficients.
    ///
    /// See also [`PosedEfd::from_efd()`].
    pub fn into_inner(self) -> Efd<D> {
        self.efd
    }
}

impl<const D: usize> core::ops::Deref for PosedEfd<D>
where
    U<D>: EfdDim<D>,
{
    type Target = Efd<D>;

    fn deref(&self) -> &Self::Target {
        &self.efd
    }
}
