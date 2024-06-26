use crate::*;
use alloc::{format, vec::Vec};

/// 1D geometric type.
pub type GeoVar1 = GeoVar<na::Rotation<f64, 1>, 1>;
/// 2D geometric type.
pub type GeoVar2 = GeoVar<na::UnitComplex<f64>, 2>;
/// 3D geometric type.
pub type GeoVar3 = GeoVar<na::UnitQuaternion<f64>, 3>;

type Sim<R, const D: usize> = na::Similarity<f64, R, D>;

/// Rotation hint for [`GeoVar`].
pub trait RotHint<const D: usize>:
    na::AbstractRotation<f64, D> + Sync + Send + core::fmt::Debug
{
    /// Get the rotation matrix.
    fn matrix(self) -> na::SMatrix<f64, D, D>;
}

impl<const D: usize> RotHint<D> for na::Rotation<f64, D> {
    fn matrix(self) -> na::SMatrix<f64, D, D> {
        self.into_inner()
    }
}

impl RotHint<2> for na::UnitComplex<f64> {
    fn matrix(self) -> na::SMatrix<f64, 2, 2> {
        self.to_rotation_matrix().into_inner()
    }
}

impl RotHint<3> for na::UnitQuaternion<f64> {
    fn matrix(self) -> na::SMatrix<f64, 3, 3> {
        self.to_rotation_matrix().into_inner()
    }
}

/// Geometric variables.
///
/// This type record the information of raw coefficients. You can merge two
/// instance with `*`/`*=` operator.
/// ```
/// use efd::{tests::*, Efd2};
/// # use efd::Curve as _;
/// # let curve1 = CURVE2D;
/// # let curve2 = CURVE2D;
///
/// let a = Efd2::from_curve(curve1, false);
/// let b = Efd2::from_curve(curve2, false);
/// let geo = b.as_geo() * a.as_geo().inverse();
/// assert!(curve_diff(&geo.transform(curve1), curve2) < EPS);
/// ```
#[derive(Clone)]
pub struct GeoVar<R: RotHint<D>, const D: usize> {
    inner: Sim<R, D>,
}

impl<R: RotHint<D>, const D: usize> Default for GeoVar<R, D> {
    fn default() -> Self {
        Self::identity()
    }
}

impl<R, const D: usize> core::fmt::Debug for GeoVar<R, D>
where
    R: RotHint<D>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct(&format!("GeoVar{D}"))
            .field("translation", &self.trans())
            .field("rotation", self.rot())
            .field("scale", &self.scale())
            .finish()
    }
}

impl<R, const D: usize> GeoVar<R, D>
where
    R: RotHint<D>,
{
    /// Create a new identity instance.
    pub fn identity() -> Self {
        Self { inner: Sim::identity() }
    }

    /// Create a new instance.
    pub fn new(trans: [f64; D], rot: R, scale: f64) -> Self {
        Self { inner: Sim::from_parts(trans.into(), rot, scale) }
    }

    /// Create a new instance from translation.
    pub fn from_trans(trans: [f64; D]) -> Self {
        Self::identity().with_trans(trans)
    }

    /// Create a new instance from rotation.
    pub fn from_rot(rot: R) -> Self {
        Self::identity().with_rot(rot)
    }

    /// Create a new instance from rotation.
    pub fn from_scale(scale: f64) -> Self {
        Self::identity().with_scale(scale)
    }

    /// Create a new instance from translation.
    pub fn only_trans(self) -> Self {
        Self::from_trans(*self.trans())
    }

    /// Create a new instance from rotation.
    pub fn only_rot(self) -> Self {
        Self::from_rot(self.inner.isometry.rotation)
    }

    /// Create a new instance from rotation and scaling.
    pub fn only_rot_scale(self) -> Self {
        Self::from_scale(self.inner.scaling())
    }

    /// With the translate property.
    pub fn with_trans(mut self, trans: [f64; D]) -> Self {
        self.set_trans(trans);
        self
    }

    /// With the rotation property.
    pub fn with_rot(mut self, rot: R) -> Self {
        self.set_rot(rot);
        self
    }

    /// With the scaling property.
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.inner.set_scaling(scale);
        self
    }

    /// Merge inverse `self` and `rhs`. (`rhs * self^T`)
    ///
    /// It can be used on a not normalized contour `a` transforming to `b`.
    /// ```
    /// use efd::{tests::*, Efd2};
    /// # use efd::Curve as _;
    /// # let curve1 = CURVE2D;
    /// # let curve2 = CURVE2D;
    ///
    /// let a = Efd2::from_curve(curve1, false);
    /// let b = Efd2::from_curve(curve2, false);
    /// let geo = a.as_geo().to(b.as_geo());
    /// assert!(curve_diff(&geo.transform(curve1), curve2) < EPS);
    /// ```
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub fn to(&self, rhs: &Self) -> Self {
        rhs * self.inverse()
    }

    /// Inverse `self`. (`self^T`/`self^-1`)
    /// ```
    /// use efd::{tests::*, Efd2};
    /// # use efd::Curve as _;
    /// # let curve = CURVE2D;
    ///
    /// let efd = Efd2::from_curve(curve, false);
    /// let curve = efd.recon(curve.len());
    /// let curve_norm = efd.recon_norm(curve.len());
    /// let curve = efd.as_geo().inverse().transform(curve);
    /// # assert!(curve_diff(&curve, &curve_norm) < EPS);
    /// ```
    #[must_use = "this returns the result of the operation, without modifying the original"]
    #[inline]
    pub fn inverse(&self) -> Self {
        Self { inner: self.inner.inverse() }
    }

    /// Get the translate property.
    #[inline]
    pub fn trans(&self) -> &[f64; D] {
        &self.inner.isometry.translation.vector.data.0[0]
    }

    /// Get the rotation property.
    #[inline]
    pub fn rot(&self) -> &R {
        &self.inner.isometry.rotation
    }

    /// Get the scaling property.
    #[inline]
    pub fn scale(&self) -> f64 {
        self.inner.scaling()
    }

    /// Set the translate property.
    #[inline]
    pub fn set_trans(&mut self, trans: [f64; D]) {
        self.inner.isometry.translation = trans.into();
    }

    /// Set the rotation property.
    #[inline]
    pub fn set_rot(&mut self, rot: R) {
        self.inner.isometry.rotation = rot;
    }

    /// Set the scaling property.
    #[inline]
    pub fn set_scale(&mut self, scale: f64) {
        self.inner.set_scaling(scale);
    }

    /// Transform a point.
    ///
    /// Please see [`GeoVar::transform()`] for more information.
    #[must_use = "The transformed point is returned as a new value"]
    #[inline]
    pub fn transform_pt(&self, p: [f64; D]) -> [f64; D] {
        self.inner.transform_point(&na::Point::from(p)).into()
    }

    /// Transform a contour with this information.
    ///
    /// This function rotates first, then translates.
    /// ```
    /// use efd::{tests::*, *};
    /// # let curve = CURVE2D;
    /// let efd = Efd2::from_curve(curve, false);
    /// // Normalize the curve
    /// let curve_norm = efd.as_geo().inverse().transform(&curve);
    /// ```
    #[must_use = "The transformed point is returned as a new value"]
    pub fn transform<C>(&self, curve: C) -> Vec<[f64; D]>
    where
        C: Curve<D>,
    {
        curve
            .to_curve()
            .into_iter()
            .map(|c| self.transform_pt(c))
            .collect()
    }

    /// Transform a contour in-placed with this information.
    pub fn transform_inplace<C>(&self, mut curve: C)
    where
        C: AsMut<[[f64; D]]>,
    {
        for c in curve.as_mut() {
            *c = self.transform_pt(*c);
        }
    }

    /// Transform an iterator contour and return a new iterator.
    pub fn transform_iter<'a, C>(&'a self, curve: C) -> impl Iterator<Item = [f64; D]> + 'a
    where
        C: IntoIterator<Item = [f64; D]> + 'a,
    {
        curve.into_iter().map(|c| self.transform_pt(c))
    }
}

macro_rules! impl_mul {
    ($ty:ty) => {
        impl_mul!(@$ty, $ty, &$ty, $ty, $ty, &$ty, &$ty, &$ty);
    };
    (@$($ty1:ty, $ty2:ty),+) => {$(
        impl<R: RotHint<D>, const D: usize> core::ops::Mul<$ty2> for $ty1 {
            type Output = GeoVar<R, D>;
            fn mul(self, rhs: $ty2) -> Self::Output {
                GeoVar { inner: &self.inner * &rhs.inner }
            }
        }
    )+};
}

impl_mul!(GeoVar<R, D>);
impl<R: RotHint<D>, const D: usize> core::ops::MulAssign<Self> for GeoVar<R, D> {
    fn mul_assign(&mut self, rhs: Self) {
        *self *= &rhs;
    }
}
impl<R: RotHint<D>, const D: usize> core::ops::MulAssign<&Self> for GeoVar<R, D> {
    fn mul_assign(&mut self, rhs: &Self) {
        self.inner *= &rhs.inner;
    }
}
