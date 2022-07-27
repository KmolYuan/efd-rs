use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

/// Geometric information.
///
/// This type record the information of raw coefficients.
///
/// Since [`Efd`](crate::Efd) implemented `Deref` for this type,
/// the methods are totally shared.
#[derive(Clone, Debug)]
pub struct GeoInfo {
    /// Angle of the semi-major axis,
    /// the rotation angle of the first ellipse.
    pub rot: f64,
    /// Scaling factor.
    pub scale: f64,
    /// Center of the first ellipse.
    /// The "DC" component / bias terms of the Fourier series.
    pub center: [f64; 2],
}

impl Default for GeoInfo {
    fn default() -> Self {
        Self::new()
    }
}

impl GeoInfo {
    /// Create a non-offset info.
    pub const fn new() -> Self {
        Self { rot: 0., scale: 1., center: [0.; 2] }
    }

    /// Create information from two vectors.
    pub fn from_vector(start: [f64; 2], end: [f64; 2]) -> Self {
        let dx = end[0] - start[0];
        let dy = end[1] - start[1];
        Self {
            rot: dy.atan2(dx),
            scale: dx.hypot(dy),
            center: start,
        }
    }

    /// An chain operator on two information.
    ///
    /// It can be used on a not normalized contour `a` transforming to another geometry `b`.
    ///
    /// ```
    /// use efd::{curve_diff, Efd};
    /// # use efd::tests::PATH;
    /// # let path1 = PATH;
    /// # let path2 = PATH;
    ///
    /// let a = Efd::from_curve(path1, None);
    /// let b = Efd::from_curve(path2, None);
    /// assert!(curve_diff(&a.to(&b).transform(path1), path2) < 1e-12);
    /// ```
    pub fn to(&self, rhs: &Self) -> Self {
        let rot = rhs.rot - self.rot;
        let scale = rhs.scale / self.scale;
        let center_a = self.center[1].atan2(self.center[0]) + rot;
        let d = self.center[1].hypot(self.center[0]) * scale;
        GeoInfo {
            rot,
            scale,
            center: [
                rhs.center[0] - d * center_a.cos(),
                rhs.center[1] - d * center_a.sin(),
            ],
        }
    }

    /// Transform a contour with this information.
    ///
    /// This function rotates first, then translates.
    ///
    /// ```
    /// # use efd::{curve_diff, tests::{PATH, TARGET}, Efd};
    /// # let path = PATH;
    /// # let target = TARGET;
    /// # let efd = Efd::from_curve(path, None);
    /// # let path = efd.generate_norm(target.len());
    /// let path1 = efd.transform(&path);
    /// # let geo = efd.geo;
    /// let path2 = geo.transform(&path);
    /// # assert!(curve_diff(&path1, TARGET) < 1e-12);
    /// # assert!(curve_diff(&path2, TARGET) < 1e-12);
    /// ```
    pub fn transform(&self, curve: &[[f64; 2]]) -> Vec<[f64; 2]> {
        self.transform_iter(curve.iter().copied()).collect()
    }

    /// Transform an object that can turn into iterator.
    pub fn transform_iter<'a, I>(&'a self, iter: I) -> impl Iterator<Item = [f64; 2]> + 'a
    where
        I: IntoIterator<Item = [f64; 2]> + 'a,
    {
        iter.into_iter().map(move |[x, y]| {
            let dx = x * self.scale;
            let dy = y * self.scale;
            let ca = self.rot.cos();
            let sa = self.rot.sin();
            let x = self.center[0] + dx * ca - dy * sa;
            let y = self.center[1] + dx * sa + dy * ca;
            [x, y]
        })
    }
}
