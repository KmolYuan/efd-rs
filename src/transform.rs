use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

/// 2D transformation type.
pub type Transform2 = Transform<na::Similarity2<f64>>;
/// 3D transformation type.
pub type Transform3 = Transform<na::Similarity3<f64>>;

/// A trait used in inner type of [`Transform`].
pub trait TransTrait {
    /// Dimension hint.
    const DIM: usize;
    /// Coordinate/Translation type.
    type Coord;
    /// Rotation angle type.
    type Rot;
    /// Scaling factor type.
    type Scale;
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

impl TransTrait for na::Similarity2<f64> {
    const DIM: usize = 2;
    type Coord = [f64; 2];
    type Rot = f64;
    type Scale = f64;

    fn identity() -> Self {
        Self::identity()
    }

    fn new(trans: Self::Coord, rot: Self::Rot, scale: Self::Scale) -> Self {
        let trans = na::Vector2::new(trans[0], trans[1]);
        Self::new(trans, rot, scale)
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
        self.isometry.rotation.angle()
    }

    fn scale(&self) -> Self::Scale {
        self.scaling()
    }

    fn inverse(&self) -> Self {
        self.inverse()
    }

    fn apply(&self, rhs: &Self) -> Self {
        self * rhs
    }
}

impl TransTrait for na::Similarity3<f64> {
    const DIM: usize = 3;
    type Coord = [f64; 3];
    type Rot = [f64; 3];
    type Scale = f64;

    fn identity() -> Self {
        Self::identity()
    }

    fn new([x, y, z]: Self::Coord, [a, b, c]: Self::Rot, scale: Self::Scale) -> Self {
        let trans = na::Vector3::new(x, y, z);
        let rot = na::UnitQuaternion::from_euler_angles(a, b, c).scaled_axis();
        Self::new(trans, rot, scale)
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
        let (r, p, y) = self.isometry.rotation.euler_angles();
        [r, p, y]
    }

    fn scale(&self) -> Self::Scale {
        self.scaling()
    }

    fn inverse(&self) -> Self {
        self.inverse()
    }

    fn apply(&self, rhs: &Self) -> Self {
        self * rhs
    }
}

/// Transform type.
///
/// This type record the information of raw coefficients.
#[derive(Clone)]
pub struct Transform<T: TransTrait> {
    inner: T,
}

impl<T: TransTrait> core::fmt::Debug for Transform<T>
where
    T::Coord: core::fmt::Debug,
    T::Rot: core::fmt::Debug,
    T::Scale: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct(&format!("Transform{}", T::DIM))
            .field("translation", &self.inner.trans())
            .field("rotation", &self.inner.rot())
            .field("scale", &self.inner.scale())
            .finish()
    }
}

impl<T: TransTrait> Default for Transform<T> {
    fn default() -> Self {
        Self::identity()
    }
}

impl<T: TransTrait> Transform<T> {
    /// Create without transform.
    pub fn identity() -> Self {
        Self { inner: T::identity() }
    }

    /// Create a new transform type.
    pub fn new(trans: T::Coord, rot: T::Rot, scale: T::Scale) -> Self {
        Self { inner: T::new(trans, rot, scale) }
    }

    /// Translate property.
    pub fn trans(&self) -> T::Coord {
        self.inner.trans()
    }

    /// Rotation property.
    pub fn rot(&self) -> T::Rot {
        self.inner.rot()
    }

    /// Scaling property.
    pub fn scale(&self) -> T::Scale {
        self.inner.scale()
    }

    /// An operator on two transformation matrices.
    ///
    /// It can be used on a not normalized contour `a` transforming to `b`.
    ///
    /// ```
    /// use efd::{curve_diff, Efd2};
    /// # let path1 = efd::tests::PATH;
    /// # let path2 = efd::tests::PATH;
    ///
    /// let a = Efd2::from_curve_gate(path1, None).unwrap();
    /// let b = Efd2::from_curve_gate(path2, None).unwrap();
    /// assert!(curve_diff(&a.to(&b).transform(path1), path2) < 1e-12);
    /// ```
    pub fn to(&self, rhs: &Self) -> Self {
        self.inverse().apply(rhs)
    }

    /// Merge two transformation matrices.
    pub fn apply(&self, rhs: &Self) -> Self {
        Self { inner: self.inner.apply(&rhs.inner) }
    }

    /// Inverse the operation of this information.
    pub fn inverse(&self) -> Self {
        Self { inner: self.inner.inverse() }
    }

    /// Transform a contour with this information.
    ///
    /// This function rotates first, then translates.
    ///
    /// ```
    /// # use efd::{curve_diff, tests::{PATH, TARGET}, Efd2};
    /// # let target = TARGET;
    /// # let efd = Efd2::from_curve_gate(PATH, None).unwrap();
    /// # let path = efd.generate_norm(target.len());
    /// let path1 = efd.transform(&path);
    /// # let trans = &efd;
    /// let path1_inv = trans.inverse().transform(&path1);
    /// # assert!(curve_diff(&path1, TARGET) < 1e-12);
    /// # assert!(curve_diff(&path1_inv, &path) < 1e-12);
    /// ```
    pub fn transform<C>(&self, curve: C) -> Vec<T::Coord>
    where
        C: AsRef<[T::Coord]>,
    {
        curve
            .as_ref()
            .iter()
            .map(|c| self.inner.transform(c))
            .collect()
    }
}