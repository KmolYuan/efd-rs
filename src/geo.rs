//! Geometric types, the geometric invariant.
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
/// This type record the information of raw coefficients.
#[derive(Clone)]
pub struct GeoVar<R: RotHint<D>, const D: usize> {
    inner: Sim<R, D>,
}

impl<R: RotHint<D>, const D: usize> Default for GeoVar<R, D> {
    fn default() -> Self {
        Self { inner: Sim::identity() }
    }
}

impl<R, const D: usize> core::fmt::Debug for GeoVar<R, D>
where
    R: core::fmt::Debug + RotHint<D>,
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
    /// Create a new transform type.
    #[must_use]
    pub fn new(trans: Coord<D>, rot: R, scale: f64) -> Self {
        Self { inner: Sim::from_parts(trans.into(), rot, scale) }
    }

    /// Create with identity matrix.
    #[must_use]
    pub fn identity() -> Self {
        Self::default()
    }

    /// Transform a point.
    ///
    /// Please see [`GeoVar::transform()`] for more information.
    #[must_use]
    pub fn transform_pt(&self, p: Coord<D>) -> Coord<D> {
        self.inner.transform_point(&na::Point::from(p)).into()
    }

    /// Transform a contour with this information.
    ///
    /// This function rotates first, then translates.
    ///
    /// ```
    /// use efd::{tests::*, *};
    /// # let target = TARGET;
    /// # let efd = Efd2::from_curve(PATH, false);
    /// # let path = efd.generate_norm(target.len());
    /// let path1 = efd.as_geo().transform(&path);
    /// # let geo = efd.as_geo();
    /// let path1_inv = geo.inverse().transform(&path1);
    /// # assert!(curve_diff(&path1, TARGET) < EPS);
    /// # assert!(curve_diff(&path1_inv, &path) < EPS);
    /// ```
    #[must_use]
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
        curve
            .as_mut()
            .iter_mut()
            .for_each(|c| *c = self.transform_pt(*c));
    }

    /// Transform an iterator contour.
    pub fn transform_iter<'a, C>(&'a self, curve: C) -> impl Iterator<Item = Coord<D>> + 'a
    where
        C: IntoIterator<Item = Coord<D>> + 'a,
    {
        curve.into_iter().map(|c| self.transform_pt(c))
    }

    /// Merge inverse `self` and `rhs` matrices.
    ///
    /// It can be used on a not normalized contour `a` transforming to `b`.
    ///
    /// ```
    /// use efd::{tests::*, Efd2};
    /// # use efd::Curve as _;
    /// # let path1 = PATH;
    /// # let path2 = PATH;
    ///
    /// let a = Efd2::from_curve(path1, false);
    /// let b = Efd2::from_curve(path2, false);
    /// let geo = a.as_geo().to(b.as_geo());
    /// assert!(curve_diff(&geo.transform(path1), path2) < EPS);
    /// ```
    #[must_use]
    pub fn to(&self, rhs: &Self) -> Self {
        self.inverse().apply(rhs)
    }

    /// Merge two matrices.
    ///
    /// Same as `rhs * self`.
    ///
    /// ```
    /// use efd::{tests::*, Efd2};
    /// # use efd::Curve as _;
    /// # let path1 = PATH;
    /// # let path2 = PATH;
    ///
    /// let a = Efd2::from_curve(path1, false);
    /// let b = Efd2::from_curve(path2, false);
    /// let geo = b.as_geo() * a.as_geo().inverse();
    /// assert!(curve_diff(&geo.transform(path1), path2) < EPS);
    /// ```
    #[must_use]
    pub fn apply(&self, rhs: &Self) -> Self {
        Self { inner: &rhs.inner * &self.inner }
    }

    /// Inverse matrices.
    ///
    /// ```
    /// use efd::{tests::*, Efd2};
    /// # use efd::Curve as _;
    /// # let path = PATH;
    ///
    /// let efd = Efd2::from_curve(path, false);
    /// let path = efd.generate(path.len());
    /// let path_norm = efd.generate_norm(path.len());
    /// let path = efd.as_geo().inverse().transform(path);
    /// # assert!(curve_diff(&path, &path_norm) < EPS);
    /// ```
    #[must_use]
    pub fn inverse(&self) -> Self {
        Self { inner: self.inner.inverse() }
    }

    /// Translate property.
    #[must_use]
    pub fn trans(&self) -> Coord<D> {
        self.inner.isometry.translation.vector.data.0[0]
    }

    /// Rotation property.
    #[must_use]
    pub fn rot(&self) -> &R {
        &self.inner.isometry.rotation
    }

    /// Scaling property.
    #[must_use]
    pub fn scale(&self) -> f64 {
        self.inner.scaling()
    }
}

macro_rules! impl_mul {
    ($ty1:ty, $ty2:ty) => {
        impl<R, const D: usize> core::ops::Mul<$ty2> for $ty1
        where
            R: RotHint<D>,
        {
            type Output = GeoVar<R, D>;
            #[must_use]
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
