use crate::math::{atan2, copysign, cos, hypot, sin};
use std::f64::consts::FRAC_2_PI;

/// Geometric information.
///
/// This type record the information of raw coefficients.
#[derive(Clone, Debug)]
pub struct GeoInfo {
    /// Angle of the semi-major axis,
    /// the rotation angle of the first ellipse.
    pub rotation: f64,
    /// Scaling factor.
    pub scale: f64,
    /// Center of the first ellipse.
    /// The "DC" component / bias terms of the Fourier series.
    pub center: (f64, f64),
}

impl Default for GeoInfo {
    fn default() -> Self {
        Self {
            rotation: 0.0,
            scale: 1.,
            center: (0.0, 0.0),
        }
    }
}

impl GeoInfo {
    /// An chain operator on two information.
    ///
    /// It can be used on a not normalized contour `a` transforming to another geometry `b`.
    ///
    /// **The starting angle will not change.**
    ///
    /// ```
    /// use efd::Efd;
    /// # use efd::tests::PATH;
    /// # let path1 = PATH;
    /// # let path2 = PATH;
    ///
    /// let a = Efd::from_curve(path1, None).geo;
    /// let b = Efd::from_curve(path2, None).geo;
    /// let c = a.to(&b);
    /// ```
    pub fn to(&self, rhs: &Self) -> Self {
        let mut a = self.rotation - rhs.rotation;
        if sin(a) < 0. {
            a += copysign(FRAC_2_PI, cos(a));
        }
        let scale = rhs.scale / self.scale;
        let center_a = atan2(self.center.1, self.center.0) + a;
        let d = hypot(self.center.1, self.center.0) * scale;
        GeoInfo {
            rotation: a,
            scale,
            center: (
                rhs.center.0 - d * cos(center_a),
                rhs.center.1 - d * sin(center_a),
            ),
        }
    }

    /// Transform a contour with this information.
    ///
    /// ```
    /// use efd::Efd;
    /// # use efd::tests::{PATH, TARGET};
    /// # let path = PATH;
    /// # let target = TARGET;
    /// let efd = Efd::from_curve(path, None);
    /// let path = efd.generate(target.len());
    /// let path_new = efd.geo.transform(&path);
    /// # assert_eq!(path_new, TARGET);
    /// ```
    pub fn transform(&self, curve: &[[f64; 2]]) -> Vec<[f64; 2]> {
        let mut out = Vec::with_capacity(curve.len());
        for c in curve {
            let angle = self.rotation;
            let dx = c[0] * self.scale;
            let dy = c[1] * self.scale;
            let x = self.center.0 + dx * cos(angle) - dy * sin(angle);
            let y = self.center.1 + dx * sin(angle) + dy * cos(angle);
            out.push([x, y]);
        }
        out
    }
}
