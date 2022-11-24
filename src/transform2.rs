use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

/// Transform type.
///
/// This type record the information of raw coefficients.
///
/// Since [`Efd2`](crate::Efd2) implemented `Deref` for this type,
/// the methods are totally shared.
#[derive(Clone, Debug, PartialEq)]
pub struct Transform2 {
    /// Angle of the semi-major axis,
    /// the rotation angle of the first ellipse.
    pub rot: f64,
    /// Scaling factor.
    pub scale: f64,
    /// Center of the first ellipse.
    /// The "DC" component / bias terms of the Fourier series.
    pub center: [f64; 2],
}

impl Default for Transform2 {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform2 {
    /// Create without transform.
    pub const fn new() -> Self {
        Self { rot: 0., scale: 1., center: [0.; 2] }
    }

    /// Create from two vectors.
    pub fn from_vector(start: [f64; 2], end: [f64; 2]) -> Self {
        let dx = end[0] - start[0];
        let dy = end[1] - start[1];
        Self {
            rot: dy.atan2(dx),
            scale: dx.hypot(dy),
            center: start,
        }
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
        let rot = rhs.rot - self.rot;
        let scale = rhs.scale / self.scale;
        let center_a = self.center[1].atan2(self.center[0]) + rot;
        let d = self.center[1].hypot(self.center[0]) * scale;
        Self {
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
    pub fn transform<C>(&self, curve: C) -> Vec<[f64; 2]>
    where
        C: AsRef<[[f64; 2]]>,
    {
        curve
            .as_ref()
            .iter()
            .map(|[x, y]| {
                let dx = x * self.scale;
                let dy = y * self.scale;
                let ca = self.rot.cos();
                let sa = self.rot.sin();
                let x = self.center[0] + dx * ca - dy * sa;
                let y = self.center[1] + dx * sa + dy * ca;
                [x, y]
            })
            .collect()
    }
}
