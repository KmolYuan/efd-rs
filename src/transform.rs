use alloc::vec::Vec;

/// 2D transformation inner type.
pub type T2 = na::Similarity2<f64>;
/// 3D transformation inner type.
pub type T3 = na::Similarity3<f64>;
/// 2D transformation type.
pub type Transform2 = Transform<T2>;
/// 3D transformation type.
pub type Transform3 = Transform<T3>;

/// A trait used in inner type of [`Transform`].
pub trait Trans {
    /// Dimension hint. It might be 2 or 3.
    const DIM: usize;
    /// Coordinate/Translation type.
    type Coord: Clone + PartialEq + 'static;
    /// Rotation angle type.
    type Rot: Clone + 'static;
    /// Scaling factor type.
    type Scale: Clone + 'static;
    /// Default identity state.
    fn identity() -> Self;
    /// Creation from three properties.
    fn new(trans: Self::Coord, rot: Self::Rot, scale: Self::Scale) -> Self;
    /// Transform a point.
    fn transform(&self, p: &Self::Coord) -> Self::Coord;
    /// Get the translate property.
    fn trans(&self) -> Self::Coord;
    /// Get the rotation property.
    fn rot(&self) -> Self::Rot;
    /// Get the scaling property.
    fn scale(&self) -> Self::Scale;
    /// A inverted transform.
    fn inverse(&self) -> Self;
    /// Merge two transform.
    fn apply(&self, rhs: &Self) -> Self;
}

impl Trans for T2 {
    const DIM: usize = 2;
    type Coord = [f64; 2];
    type Rot = na::UnitComplex<f64>;
    type Scale = f64;

    fn identity() -> Self {
        Self::identity()
    }

    fn new(trans: Self::Coord, rot: Self::Rot, scale: Self::Scale) -> Self {
        let trans = na::Translation2::new(trans[0], trans[1]);
        Self::from_parts(trans, rot, scale)
    }

    fn transform(&self, p: &Self::Coord) -> Self::Coord {
        let p = self.transform_point(&na::Point2::new(p[0], p[1]));
        [p.x, p.y]
    }

    fn trans(&self) -> Self::Coord {
        let t = self.isometry.translation;
        [t.x, t.y]
    }

    fn rot(&self) -> Self::Rot {
        self.isometry.rotation
    }

    fn scale(&self) -> Self::Scale {
        self.scaling()
    }

    fn inverse(&self) -> Self {
        self.inverse()
    }

    fn apply(&self, rhs: &Self) -> Self {
        rhs * self
    }
}

impl Trans for T3 {
    const DIM: usize = 3;
    type Coord = [f64; 3];
    type Rot = na::UnitQuaternion<f64>;
    type Scale = f64;

    fn identity() -> Self {
        Self::identity()
    }

    fn new([x, y, z]: Self::Coord, rot: Self::Rot, scale: Self::Scale) -> Self {
        Self::from_parts(na::Translation3::new(x, y, z), rot, scale)
    }

    fn transform(&self, p: &Self::Coord) -> Self::Coord {
        let p = self.transform_point(&na::Point3::new(p[0], p[1], p[2]));
        [p.x, p.y, p.z]
    }

    fn trans(&self) -> Self::Coord {
        let t = &self.isometry.translation;
        [t.x, t.y, t.z]
    }

    fn rot(&self) -> Self::Rot {
        self.isometry.rotation
    }

    fn scale(&self) -> Self::Scale {
        self.scaling()
    }

    fn inverse(&self) -> Self {
        self.inverse()
    }

    fn apply(&self, rhs: &Self) -> Self {
        rhs * self
    }
}

/// Transform type.
///
/// This type record the information of raw coefficients.
#[derive(Clone)]
pub struct Transform<T: Trans> {
    inner: T,
}

impl<T: Trans> core::fmt::Debug for Transform<T>
where
    T::Coord: core::fmt::Debug,
    T::Rot: core::fmt::Debug,
    T::Scale: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct(&alloc::format!("Transform{}", T::DIM))
            .field("translation", &self.inner.trans())
            .field("rotation", &self.inner.rot())
            .field("scale", &self.inner.scale())
            .finish()
    }
}

impl<T: Trans> Default for Transform<T> {
    fn default() -> Self {
        Self::identity()
    }
}

impl<T: Trans> Transform<T> {
    /// Create without transform.
    #[must_use]
    pub fn identity() -> Self {
        Self { inner: T::identity() }
    }

    /// Create a new transform type.
    #[must_use]
    pub fn new(trans: T::Coord, rot: T::Rot, scale: T::Scale) -> Self {
        Self { inner: T::new(trans, rot, scale) }
    }

    /// Translate property.
    #[must_use]
    pub fn trans(&self) -> T::Coord {
        self.inner.trans()
    }

    /// Rotation property.
    #[must_use]
    pub fn rot(&self) -> T::Rot {
        self.inner.rot()
    }

    /// Scaling property.
    #[must_use]
    pub fn scale(&self) -> T::Scale {
        self.inner.scale()
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
        Self { inner: self.inner.apply(&rhs.inner) }
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
    /// let path1 = efd.as_trans().transform(&path);
    /// # let trans = efd.as_trans();
    /// let path1_inv = trans.inverse().transform(&path1);
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
}

macro_rules! impl_mul {
    ($ty1:ty, $ty2:ty) => {
        impl<T: Trans> core::ops::Mul<$ty2> for $ty1 {
            type Output = Transform<T>;
            #[must_use]
            fn mul(self, rhs: $ty2) -> Self::Output {
                Transform { inner: rhs.inner.apply(&self.inner) }
            }
        }
    };
}

impl_mul!(Transform<T>, Self);
impl_mul!(&Transform<T>, Self);
impl_mul!(Transform<T>, &Self);
impl_mul!(&Transform<T>, Transform<T>);
