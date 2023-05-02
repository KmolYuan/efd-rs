use crate::*;
use alloc::vec::Vec;

/// 2D transformation inner type.
pub type T2 = na::Similarity2<f64>;
/// 3D transformation inner type.
pub type T3 = na::Similarity3<f64>;
/// 2D transformation type.
pub type Transform2 = Transform<T2>;
/// 3D transformation type.
pub type Transform3 = Transform<T3>;

type Coord<T> = <T as Trans>::Coord;
type Rot<T> = <T as Trans>::Rot;

type Sim<R, const DIM: usize> = na::Similarity<f64, R, DIM>;

/// A trait used in inner type of [`Transform`].
pub trait Trans {
    /// Coordinate/Translation type.
    type Coord: CoordHint;
    /// Rotation angle type.
    type Rot: Clone + 'static;

    /// Default identity state.
    fn identity() -> Self;
    /// Creation from three properties.
    fn new(trans: Self::Coord, rot: Self::Rot, scale: f64) -> Self;
    /// Transform a point.
    fn transform(&self, p: &Self::Coord) -> Self::Coord;

    /// The value of the dimension.
    fn dim() -> usize {
        <<Self::Coord as CoordHint>::Dim as na::DimName>::dim()
    }
}

/// Hint for transforming coordinate type to matrix.
pub trait CoordHint: Clone + PartialEq + Sized + 'static {
    /// Dimension. Is a constant width.
    type Dim: na::base::DimName;
    /// Coefficient number per harmonic. Is a constant width.
    type CDim: na::base::DimName;
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

macro_rules! impl_hint {
    ($ty:ty, $dim:ident, $cdim:ident, |$c:ident| $self:expr) => {
        impl CoordHint for $ty {
            type Dim = na::$dim;
            type CDim = na::$cdim;
            type Flat = <Self as IntoIterator>::IntoIter;
            type FlatMut<'a> = <&'a mut Self as IntoIterator>::IntoIter;

            fn to_coord($c: na::MatrixView<f64, Self::Dim, na::U1>) -> Self {
                $self
            }

            fn flat(self) -> Self::Flat {
                self.into_iter()
            }

            fn flat_mut(&mut self) -> Self::FlatMut<'_> {
                self.iter_mut()
            }
        }
    };
}

impl_hint!([f64; 2], U2, U4, |c| [c[0], c[1]]);
impl_hint!([f64; 3], U3, U6, |c| [c[0], c[1], c[2]]);

impl Trans for T2 {
    type Coord = [f64; 2];
    type Rot = na::UnitComplex<f64>;

    fn identity() -> Self {
        Self::identity()
    }

    fn new(trans: Self::Coord, rot: Self::Rot, scale: f64) -> Self {
        let trans = na::Translation2::new(trans[0], trans[1]);
        Self::from_parts(trans, rot, scale)
    }

    fn transform(&self, p: &Self::Coord) -> Self::Coord {
        let p = self * na::Point2::new(p[0], p[1]);
        [p.x, p.y]
    }
}

impl Trans for T3 {
    type Coord = [f64; 3];
    type Rot = na::UnitQuaternion<f64>;

    fn identity() -> Self {
        Self::identity()
    }

    fn new([x, y, z]: Self::Coord, rot: Self::Rot, scale: f64) -> Self {
        Self::from_parts(na::Translation3::new(x, y, z), rot, scale)
    }

    fn transform(&self, p: &Self::Coord) -> Self::Coord {
        let p = self * na::Point3::new(p[0], p[1], p[2]);
        [p.x, p.y, p.z]
    }
}

/// Transform type.
///
/// This type record the information of raw coefficients.
#[derive(Clone)]
pub struct Transform<T: Trans> {
    inner: T,
}

impl<R, const DIM: usize> Default for Transform<Sim<R, DIM>>
where
    Sim<R, DIM>: Trans<Rot = R>,
    Coord<Sim<R, DIM>>: CoordHint<Dim = na::Const<DIM>>,
    Rot<Sim<R, DIM>>: na::AbstractRotation<f64, DIM>,
{
    fn default() -> Self {
        Self::identity()
    }
}

impl<R, const DIM: usize> core::fmt::Debug for Transform<Sim<R, DIM>>
where
    Sim<R, DIM>: Trans<Rot = R>,
    Coord<Sim<R, DIM>>: CoordHint<Dim = na::Const<DIM>> + core::fmt::Debug,
    Rot<Sim<R, DIM>>: na::AbstractRotation<f64, DIM> + core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct(&alloc::format!("Transform{}", Sim::dim()))
            .field("translation", &self.trans())
            .field("rotation", &self.rot())
            .field("scale", &self.scale())
            .finish()
    }
}

impl<T: Trans> Transform<T> {
    /// Create a new transform type.
    #[must_use]
    pub fn new(trans: Coord<T>, rot: Rot<T>, scale: f64) -> Self {
        Self { inner: T::new(trans, rot, scale) }
    }

    /// Create without transform.
    #[must_use]
    pub fn identity() -> Self {
        Self { inner: T::identity() }
    }

    /// Transform a point.
    ///
    /// Please see [`Self::transform()`] for more information.
    #[must_use]
    pub fn transform_pt(&self, p: &Coord<T>) -> Coord<T> {
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
    /// let path1 = efd.as_trans().transform(&path);
    /// # let trans = efd.as_trans();
    /// let path1_inv = trans.inverse().transform(&path1);
    /// # assert!(curve_diff(&path1, TARGET) < EPS);
    /// # assert!(curve_diff(&path1_inv, &path) < EPS);
    /// ```
    #[must_use]
    pub fn transform<C>(&self, curve: C) -> Vec<Coord<T>>
    where
        C: AsRef<[Coord<T>]>,
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
        C: AsMut<[Coord<T>]>,
    {
        curve
            .as_mut()
            .iter_mut()
            .for_each(|c| *c = self.transform_pt(c));
    }

    /// Transform an iterator contour.
    pub fn transform_iter<'a, C>(&'a self, curve: C) -> impl Iterator<Item = Coord<T>> + 'a
    where
        C: IntoIterator<Item = Coord<T>> + 'a,
    {
        curve.into_iter().map(|c| self.transform_pt(&c))
    }
}

impl<R, const DIM: usize> Transform<Sim<R, DIM>>
where
    Sim<R, DIM>: Trans<Rot = R>,
    Coord<Sim<R, DIM>>: CoordHint<Dim = na::Const<DIM>>,
    Rot<Sim<R, DIM>>: na::AbstractRotation<f64, DIM>,
{
    /// Translate property.
    #[must_use]
    pub fn trans(&self) -> Coord<Sim<R, DIM>> {
        let view = na::MatrixView::from(&self.inner.isometry.translation.vector);
        CoordHint::to_coord(view)
    }

    /// Rotation property.
    #[must_use]
    pub fn rot(&self) -> &Rot<Sim<R, DIM>> {
        &self.inner.isometry.rotation
    }

    /// Scaling property.
    #[must_use]
    pub fn scale(&self) -> f64 {
        self.inner.scaling()
    }

    /// An operator on two transformation matrices.
    ///
    /// It can be used on a not normalized contour `a` transforming to `b`.
    ///
    /// ```
    /// use efd::{curve_diff, tests::*, Efd2};
    /// # use efd::Curve as _;
    /// # let path1 = PATH;
    /// # let path2 = PATH;
    ///
    /// let a = Efd2::from_curve(path1, false);
    /// let b = Efd2::from_curve(path2, false);
    /// let trans = a.as_trans().to(b.as_trans());
    /// assert!(curve_diff(&trans.transform(path1), path2) < EPS);
    /// ```
    #[must_use]
    pub fn to(&self, rhs: &Self) -> Self {
        self.inverse().apply(rhs)
    }

    /// Merge two transformation matrices.
    ///
    /// Same as `rhs * self`.
    ///
    /// ```
    /// use efd::{curve_diff, tests::*, Efd2};
    /// # use efd::Curve as _;
    /// # let path1 = PATH;
    /// # let path2 = PATH;
    ///
    /// let a = Efd2::from_curve(path1, false);
    /// let b = Efd2::from_curve(path2, false);
    /// let trans = b.as_trans() * a.as_trans().inverse();
    /// assert!(curve_diff(&trans.transform(path1), path2) < EPS);
    /// ```
    #[must_use]
    pub fn apply(&self, rhs: &Self) -> Self {
        Self { inner: &rhs.inner * &self.inner }
    }

    /// Inverse the operation of this information.
    ///
    /// ```
    /// use efd::{curve_diff, tests::*, Efd2};
    /// # use efd::Curve as _;
    /// # let path = PATH;
    ///
    /// let efd = Efd2::from_curve(path, false);
    /// let path = efd.generate(path.len());
    /// let path_norm = efd.generate_norm_in(path.len(), std::f64::consts::TAU);
    /// let path = efd.as_trans().inverse().transform(path);
    /// # assert!(curve_diff(&path, &path_norm) < EPS);
    /// ```
    #[must_use]
    pub fn inverse(&self) -> Self {
        Self { inner: self.inner.inverse() }
    }
}

macro_rules! impl_mul {
    ($ty1:ty, $ty2:ty) => {
        impl<R, const DIM: usize> core::ops::Mul<$ty2> for $ty1
        where
            Sim<R, DIM>: Trans<Rot = R>,
            Coord<Sim<R, DIM>>: CoordHint<Dim = na::Const<DIM>>,
            Rot<Sim<R, DIM>>: na::AbstractRotation<f64, DIM>,
        {
            type Output = Transform<Sim<R, DIM>>;
            #[must_use]
            fn mul(self, rhs: $ty2) -> Self::Output {
                self.apply(&rhs)
            }
        }
    };
}

impl_mul!(Transform<Sim<R, DIM>>, Self);
impl_mul!(&Transform<Sim<R, DIM>>, Self);
impl_mul!(Transform<Sim<R, DIM>>, &Self);
impl_mul!(&Transform<Sim<R, DIM>>, Transform<Sim<R, DIM>>);
