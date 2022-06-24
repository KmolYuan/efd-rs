use crate::math::{atan2, cos, hypot, sin};
use alloc::vec::Vec;

/// Geometric information.
///
/// This type record the information of raw coefficients.
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
        Self { rot: 0.0, scale: 1., center: [0., 0.] }
    }
}

impl GeoInfo {
    /// Create information from a vector.
    pub fn from_vector(start: [f64; 2], end: [f64; 2]) -> Self {
        let dx = end[0] - start[0];
        let dy = end[1] - start[1];
        Self {
            rot: atan2(dy, dx),
            scale: hypot(dx, dy),
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
    ///
    /// The `Efd` type can called with [`Efd::to`](crate::Efd::to).
    pub fn to(&self, rhs: &Self) -> Self {
        let rot = rhs.rot - self.rot;
        let scale = rhs.scale / self.scale;
        let center_a = atan2(self.center[1], self.center[0]) + rot;
        let d = hypot(self.center[1], self.center[0]) * scale;
        GeoInfo {
            rot,
            scale,
            center: [
                rhs.center[0] - d * cos(center_a),
                rhs.center[1] - d * sin(center_a),
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
    /// # let path = efd.generate(target.len());
    /// # let geo = efd.geo;
    /// let path_new = geo.transform(&path);
    /// # assert!(curve_diff(&path_new, TARGET) < 1e-12);
    /// ```
    ///
    /// The `Efd` type can called with [`Efd::transform`](crate::Efd::transform).
    pub fn transform(&self, curve: &[[f64; 2]]) -> Vec<[f64; 2]> {
        let mut out = curve.to_vec();
        for c in out.iter_mut() {
            let dx = c[0] * self.scale;
            let dy = c[1] * self.scale;
            let ca = cos(self.rot);
            let sa = sin(self.rot);
            c[0] = self.center[0] + dx * ca - dy * sa;
            c[1] = self.center[1] + dx * sa + dy * ca;
        }
        out
    }
}
