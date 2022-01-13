use crate::math::{atan2, copysign, cos, hypot, sin};
use std::f64::consts::FRAC_2_PI;

/// Geometric information.
///
/// This type record the information of raw coefficients.
#[derive(Clone, Debug)]
pub struct GeoInfo {
    /// Angle of the semi-major axis,
    /// the rotation angle of the first ellipse.
    pub semi_major_axis_angle: f64,
    /// Shift angle between each ellipse.
    /// This is also the starting angle.
    pub starting_angle: f64,
    /// Scaling factor.
    pub scale: f64,
    /// Center of the first ellipse.
    /// The "DC" component / bias terms of the Fourier series.
    pub center: (f64, f64),
}

impl Default for GeoInfo {
    fn default() -> Self {
        Self {
            semi_major_axis_angle: 0.0,
            starting_angle: 0.0,
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
        let mut a = self.semi_major_axis_angle - rhs.semi_major_axis_angle;
        if sin(a) < 0. {
            a += copysign(FRAC_2_PI, cos(a));
        }
        let scale = rhs.scale / self.scale;
        let center_a = atan2(self.center.1, self.center.0) + a;
        let d = hypot(self.center.1, self.center.0) * scale;
        GeoInfo {
            semi_major_axis_angle: a,
            starting_angle: self.starting_angle, // Keep original
            scale,
            center: (
                rhs.center.0 - d * cos(center_a),
                rhs.center.1 - d * sin(center_a),
            ),
        }
    }
}
