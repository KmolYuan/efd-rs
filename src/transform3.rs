use alloc::vec::Vec;

/// 3D transform type.
#[derive(Clone, Debug, PartialEq)]
pub struct Transform3 {
    /// Angle of the semi-major axis,
    /// the rotation angle of the first ellipse.
    ///
    /// `[XY, YZ, XZ]`
    pub rot: [f64; 3],
    /// Scaling factor.
    pub scale: f64,
    /// Center of the first ellipse.
    /// The "DC" component / bias terms of the Fourier series.
    pub center: [f64; 3],
}

impl Default for Transform3 {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(unused_variables)]
impl Transform3 {
    /// Create without transform.
    pub const fn new() -> Self {
        Self { rot: [0.; 3], scale: 1., center: [0.; 3] }
    }

    /// TODO: Create from two vectors.
    pub fn from_vector(start: [f64; 3], end: [f64; 3]) -> Self {
        todo!()
    }

    /// TODO: Transform a contour with this information.
    ///
    /// This function rotates first, then translates.
    ///
    /// ```
    /// # use efd::{curve_diff, tests::{PATH, TARGET}, Efd2};
    /// # let path = PATH;
    /// # let target = TARGET;
    /// # let efd = Efd2::from_curve_gate(path, None).unwrap();
    /// # let path = efd.generate_norm(target.len());
    /// let path1 = efd.transform(&path);
    /// # let trans = &efd;
    /// let path2 = trans.transform(&path);
    /// # assert!(curve_diff(&path1, TARGET) < 1e-12);
    /// # assert!(curve_diff(&path2, TARGET) < 1e-12);
    /// ```
    pub fn transform<C>(&self, curve: C) -> Vec<[f64; 3]>
    where
        C: AsRef<[[f64; 3]]>,
    {
        curve.as_ref().iter().map(|[x, y, z]| todo!()).collect()
    }
}
