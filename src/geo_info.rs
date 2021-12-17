use std::f64::consts::FRAC_2_PI;

/// Geometric information.
///
/// This type record the information of raw coefficients.
#[derive(Clone)]
pub struct GeoInfo {
    /// Angle of the semi-major axis,
    /// the rotation angle of the first ellipse
    pub semi_major_axis_angle: f64,
    /// Shift angle between each ellipse
    pub shift_angle: f64,
    /// Scaling factor
    pub scale: f64,
    /// Center of the first ellipse
    pub locus: (f64, f64),
}

impl Default for GeoInfo {
    fn default() -> Self {
        Self {
            semi_major_axis_angle: 0.0,
            shift_angle: 0.0,
            scale: 1.,
            locus: (0.0, 0.0),
        }
    }
}

impl GeoInfo {
    /// An chain operator on two information.
    ///
    /// It can be used on a not normalized contour `a` transforming to another geometry `b`.
    ///
    /// ```
    /// use efd::Efd;
    /// # use efd::tests::PATH;
    /// # let path1 = PATH;
    /// # let path2 = PATH;
    ///
    /// let a = Efd::from_curve(path1, None).normalize();
    /// let b = Efd::from_curve(path2, None).normalize();
    /// let c = a.to(&b);
    /// ```
    pub fn to(&self, rhs: &Self) -> Self {
        let mut a = self.semi_major_axis_angle - rhs.semi_major_axis_angle;
        if a.sin() < 0. {
            a += FRAC_2_PI.copysign(a.cos());
        }
        let scale = rhs.scale / self.scale;
        let locus_a = self.locus.1.atan2(self.locus.0) + a;
        let d = self.locus.1.hypot(self.locus.0) * scale;
        GeoInfo {
            semi_major_axis_angle: a,
            shift_angle: self.shift_angle, // Keep original
            scale,
            locus: (
                rhs.locus.0 - d * locus_a.cos(),
                rhs.locus.1 - d * locus_a.sin(),
            ),
        }
    }
}
