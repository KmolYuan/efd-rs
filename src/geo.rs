//! Geometric types, the geometric invariant.
use crate::*;
use alloc::vec::Vec;

/// 2D transformation inner type.
pub type T2 = na::Similarity2<f64>;
/// 3D transformation inner type.
pub type T3 = na::Similarity3<f64>;
/// 2D geometric type.
pub type GeoVar2 = GeoVar<T2>;
/// 3D geometric type.
pub type GeoVar3 = GeoVar<T3>;

type Sim<R, const DIM: usize> = na::Similarity<f64, R, DIM>;

/// A trait used in inner type of [`GeoVar`].
pub trait Transform {
    /// Coordinate/Translation type.
    type Coord: CoordHint;
    /// Rotation angle type.
    type Rot: Clone + Default + 'static;

    /// Default identity state.
    fn identity() -> Self;
    /// Creation from three properties.
    fn new(trans: Self::Coord, rot: Self::Rot, scale: f64) -> Self;
    /// Transform a point.
    fn transform(&self, p: &Self::Coord) -> Self::Coord;
    /// Merge two matrices.
    fn apply(&self, rhs: &Self) -> Self;
    /// Inverse matrices.
    fn inverse(&self) -> Self;
    /// Translation property.
    fn trans(&self) -> Self::Coord;
    /// Rotation property.
    fn rot(&self) -> &Self::Rot;
    /// Scaling property.
    fn scale(&self) -> f64;

    /// The value of the dimension.
    fn dim() -> usize {
        <<Self::Coord as CoordHint>::Dim as na::DimName>::dim()
    }
}

/// Hint for transforming coordinate type to matrix.
pub trait CoordHint: Clone + core::fmt::Debug + Default + PartialEq + Sized + 'static {
    /// Dimension. Is a constant width.
    type Dim: na::DimNameMul<na::U2>;
    /// Flaten type.
    type Flat: Iterator<Item = f64>;
    /// Mutable flaten type.
    type FlatMut<'a>: Iterator<Item = &'a mut f64>;

    /// Transform array slice to coordinate type.
    fn to_coord(c: na::MatrixView<f64, Self::Dim, na::U1>) -> Self;
    /// Flaten method.
    fn flat(self) -> Self::Flat;
    /// Mutable flaten method.
    fn flat_mut(&mut self) -> Self::FlatMut<'_>;
}

impl<const DIM: usize> CoordHint for [f64; DIM]
where
    na::Const<DIM>: na::DimNameMul<na::U2>,
    [f64; DIM]: Default,
{
    type Dim = na::Const<DIM>;
    type Flat = <Self as IntoIterator>::IntoIter;
    type FlatMut<'a> = <&'a mut Self as IntoIterator>::IntoIter;

    fn to_coord(c: na::MatrixView<f64, Self::Dim, na::U1>) -> Self {
        c.as_slice().try_into().unwrap()
    }

    fn flat(self) -> Self::Flat {
        self.into_iter()
    }

    fn flat_mut(&mut self) -> Self::FlatMut<'_> {
        self.iter_mut()
    }
}

impl<R, const DIM: usize> Transform for Sim<R, DIM>
where
    R: na::AbstractRotation<f64, DIM> + Default + 'static,
    na::Const<DIM>: na::DimNameMul<na::U2>,
    [f64; DIM]: Default,
{
    type Coord = [f64; DIM];
    type Rot = R;

    fn identity() -> Self {
        Self::identity()
    }

    fn new(trans: Self::Coord, rot: Self::Rot, scale: f64) -> Self {
        let trans = na::Translation { vector: na::SVector::from(trans) };
        Self::from_parts(trans, rot, scale)
    }

    fn transform(&self, p: &Self::Coord) -> Self::Coord {
        let p = self * na::Point::from(*p);
        CoordHint::to_coord(na::MatrixView::from(&p.coords))
    }

    fn apply(&self, rhs: &Self) -> Self {
        rhs * self
    }

    fn inverse(&self) -> Self {
        self.inverse()
    }

    fn trans(&self) -> Self::Coord {
        let view = na::MatrixView::from(&self.isometry.translation.vector);
        CoordHint::to_coord(view)
    }

    fn rot(&self) -> &Self::Rot {
        &self.isometry.rotation
    }

    fn scale(&self) -> f64 {
        self.scaling()
    }
}

/// Geometric variables.
///
/// This type record the information of raw coefficients.
#[derive(Clone)]
pub struct GeoVar<T: Transform> {
    inner: T,
}

impl<T: Transform> Default for GeoVar<T> {
    fn default() -> Self {
        Self::identity()
    }
}

impl<T: Transform> core::fmt::Debug for GeoVar<T>
where
    T::Coord: core::fmt::Debug,
    T::Rot: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct(&alloc::format!("Transform{}", T::dim()))
            .field("translation", &self.trans())
            .field("rotation", self.rot())
            .field("scale", &self.scale())
            .finish()
    }
}

impl<T: Transform> GeoVar<T> {
    /// Create a new transform type.
    #[must_use]
    pub fn new(trans: T::Coord, rot: T::Rot, scale: f64) -> Self {
        Self { inner: T::new(trans, rot, scale) }
    }

    /// Create with identity matrix.
    #[must_use]
    pub fn identity() -> Self {
        Self { inner: T::identity() }
    }

    /// Transform a point.
    ///
    /// Please see [`Self::transform()`] for more information.
    #[must_use]
    pub fn transform_pt(&self, p: &T::Coord) -> T::Coord {
        self.inner.transform(p)
    }

    /// Transform a contour with this information.
    ///
    /// This function rotates first, then translates.
    ///
    /// ```
    /// use efd::{tests::*, *};
    /// # let target = TARGET;
    /// # let efd = Efd2::from_curve(PATH, false);
    /// # let path = efd.generate_norm_in(target.len(), std::f64::consts::TAU);
    /// let path1 = efd.as_geo().transform(&path);
    /// # let geo = efd.as_geo();
    /// let path1_inv = geo.inverse().transform(&path1);
    /// # assert!(curve_diff(&path1, TARGET) < EPS);
    /// # assert!(curve_diff(&path1_inv, &path) < EPS);
    /// ```
    #[must_use]
    pub fn transform<C>(&self, curve: C) -> Vec<T::Coord>
    where
        C: AsRef<[T::Coord]>,
    {
        curve
            .as_ref()
            .iter()
            .map(|c| self.transform_pt(c))
            .collect()
    }

    /// Transform a contour in-placed with this information.
    pub fn transform_inplace<C>(&self, mut curve: C)
    where
        C: AsMut<[T::Coord]>,
    {
        curve
            .as_mut()
            .iter_mut()
            .for_each(|c| *c = self.transform_pt(c));
    }

    /// Transform an iterator contour.
    pub fn transform_iter<'a, C>(&'a self, curve: C) -> impl Iterator<Item = T::Coord> + 'a
    where
        C: IntoIterator<Item = T::Coord> + 'a,
    {
        curve.into_iter().map(|c| self.transform_pt(&c))
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
        Self { inner: self.inner.apply(&rhs.inner) }
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
    /// let path_norm = efd.generate_norm_in(path.len(), std::f64::consts::TAU);
    /// let path = efd.as_geo().inverse().transform(path);
    /// # assert!(curve_diff(&path, &path_norm) < EPS);
    /// ```
    #[must_use]
    pub fn inverse(&self) -> Self {
        Self { inner: self.inner.inverse() }
    }

    /// Translate property.
    #[must_use]
    pub fn trans(&self) -> T::Coord {
        self.inner.trans()
    }

    /// Rotation property.
    #[must_use]
    pub fn rot(&self) -> &T::Rot {
        self.inner.rot()
    }

    /// Scaling property.
    #[must_use]
    pub fn scale(&self) -> f64 {
        self.inner.scale()
    }
}

macro_rules! impl_mul {
    ($ty1:ty, $ty2:ty) => {
        impl<T: Transform> core::ops::Mul<$ty2> for $ty1 {
            type Output = GeoVar<T>;
            #[must_use]
            fn mul(self, rhs: $ty2) -> Self::Output {
                rhs.apply(&self)
            }
        }
    };
}

impl_mul!(GeoVar<T>, Self);
impl_mul!(&GeoVar<T>, Self);
impl_mul!(GeoVar<T>, &Self);
impl_mul!(&GeoVar<T>, GeoVar<T>);
