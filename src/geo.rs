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
pub trait RotHint<const D: usize>: na::AbstractRotation<f64, D> + core::fmt::Debug {
    /// Get the rotation matrix.
    fn matrix(self) -> na::SMatrix<f64, D, D>;
    /// Mirror the rotation matrix in-place.
    ///
    /// This function is used to multiply negative one to all components.
    fn mirror_inplace(&mut self);
}

impl<const D: usize> RotHint<D> for na::Rotation<f64, D> {
    fn matrix(self) -> na::SMatrix<f64, D, D> {
        self.into_inner()
    }

    fn mirror_inplace(&mut self) {
        *self.matrix_mut_unchecked() *= -1.;
    }
}

impl RotHint<2> for na::UnitComplex<f64> {
    fn matrix(self) -> na::SMatrix<f64, 2, 2> {
        self.to_rotation_matrix().into_inner()
    }

    fn mirror_inplace(&mut self) {
        *self.as_mut_unchecked() *= -1.;
    }
}

impl RotHint<3> for na::UnitQuaternion<f64> {
    fn matrix(self) -> na::SMatrix<f64, 3, 3> {
        self.to_rotation_matrix().into_inner()
    }

    fn mirror_inplace(&mut self) {
        *self.as_mut_unchecked() *= -1.;
    }
}

/// Geometric variables.
///
/// This type record the information of raw coefficients.
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
    /// Create with identity matrix.
    pub fn identity() -> Self {
        Self { inner: Sim::identity() }
    }

    /// Create a new instance.
    pub fn new(trans: Coord<D>, rot: R, scale: f64) -> Self {
        Self { inner: Sim::from_parts(trans.into(), rot, scale) }
    }

    /// Create a new instance from translation.
    pub fn from_trans(trans: Coord<D>) -> Self {
        Self::new(trans, R::identity(), 1.)
    }

    /// Create a new instance from rotation.
    pub fn from_rot(rot: R) -> Self {
        Self::new([0.; D], rot, 1.)
    }

    /// Create a new instance from translation.
    pub fn only_trans(self) -> Self {
        Self::from_trans(self.trans())
    }

    /// Create a new instance from rotation.
    pub fn only_rot(self) -> Self {
        Self::from_rot(self.inner.isometry.rotation)
    }

    /// Transform a point.
    ///
    /// Please see [`GeoVar::transform()`] for more information.
    #[must_use = "The transformed point is returned as a new value"]
    pub fn transform_pt(&self, p: Coord<D>) -> Coord<D> {
        self.inner.transform_point(&na::Point::from(p)).into()
    }

    /// Transform a contour with this information.
    ///
    /// This function rotates first, then translates.
    ///
    /// ```
    /// use efd::{tests::*, *};
    /// # let curve = CURVE2D;
    /// let efd = Efd2::from_curve(curve, false);
    /// // Normalize the curve
    /// let curve_norm = efd.as_geo().inverse().transform(&curve);
    /// ```
    #[must_use = "The transformed point is returned as a new value"]
    pub fn transform<C>(&self, curve: C) -> Vec<Coord<D>>
    where
        C: AsRef<[Coord<D>]>,
    {
        curve
            .as_ref()
            .iter()
            .map(|c| self.transform_pt(*c))
            .collect()
    }

    /// Transform a contour in-placed with this information.
    pub fn transform_inplace<C>(&self, mut curve: C)
    where
        C: AsMut<[Coord<D>]>,
    {
        for c in curve.as_mut() {
            *c = self.transform_pt(*c);
        }
    }

    /// Transform an iterator contour.
    pub fn transform_iter<'a, C>(&'a self, curve: C) -> impl Iterator<Item = Coord<D>> + 'a
    where
        C: IntoIterator<Item = Coord<D>> + 'a,
    {
        curve.into_iter().map(|c| self.transform_pt(c))
    }

    /// Merge inverse `self` and `rhs` matrices. (`rhs * self^T`)
    ///
    /// It can be used on a not normalized contour `a` transforming to `b`.
    ///
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

    /// Merge two matrices. (`rhs * self`)
    ///
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
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub fn apply(&self, rhs: &Self) -> Self {
        rhs * self
    }

    /// Inverse matrices. (`self^T`)
    ///
    /// ```
    /// use efd::{tests::*, Efd2};
    /// # use efd::Curve as _;
    /// # let curve = CURVE2D;
    ///
    /// let efd = Efd2::from_curve(curve, false);
    /// let curve = efd.generate(curve.len());
    /// let curve_norm = efd.generate_norm(curve.len());
    /// let curve = efd.as_geo().inverse().transform(curve);
    /// # assert!(curve_diff(&curve, &curve_norm) < EPS);
    /// ```
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub fn inverse(&self) -> Self {
        Self { inner: self.inner.inverse() }
    }

    /// Get the translate property.
    pub fn trans(&self) -> Coord<D> {
        self.inner.isometry.translation.vector.data.0[0]
    }

    /// Get the rotation property.
    pub fn rot(&self) -> &R {
        &self.inner.isometry.rotation
    }

    /// Get the scaling property.
    pub fn scale(&self) -> f64 {
        self.inner.scaling()
    }
}

macro_rules! impl_mul {
    ($ty1:ty, $ty2:ty) => {
        impl<R: RotHint<D>, const D: usize> core::ops::Mul<$ty2> for $ty1 {
            type Output = GeoVar<R, D>;
            fn mul(self, rhs: $ty2) -> Self::Output {
                GeoVar { inner: &self.inner * &rhs.inner }
            }
        }
    };
}

impl_mul!(GeoVar<R, D>, Self);
impl_mul!(&GeoVar<R, D>, Self);
impl_mul!(GeoVar<R, D>, &Self);
impl_mul!(&GeoVar<R, D>, GeoVar<R, D>);
