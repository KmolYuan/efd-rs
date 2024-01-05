use crate::*;
use alloc::vec::Vec;
use core::f64::consts::{PI, TAU};
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::*;

/// A 2D shape with a pose described by EFD.
pub type PosedEfd2 = PosedEfd<2>;
/// A 3D shape with a pose described by EFD.
pub type PosedEfd3 = PosedEfd<3>;

/// A shape with a pose described by EFD.
///
/// These are the same as [`Efd`] except that it has a pose, and the data are
/// always normalized and readonly.
///
/// # Pose Representation
/// Pose is represented by an unit vector, which is rotated by the rotation
/// of the original shape.
#[derive(Clone)]
pub struct PosedEfd<const D: usize>
where
    U<D>: EfdDim<D>,
    na::Const<D>: na::DimNameMul<na::U2>,
{
    efd: Efd<D>,
    pose: Coeff<D>,
}

impl<const D: usize> PosedEfd<D>
where
    U<D>: EfdDim<D>,
    na::Const<D>: na::DimNameMul<na::U2>,
{
    // TODO

    /// Consume self and return a raw array of the coefficients.
    /// The first is the EFD coefficients, and the second is the pose
    /// coefficients.
    #[must_use]
    pub fn into_inner(self) -> (Coeff<D>, Coeff<D>) {
        (self.efd.into_inner(), self.pose)
    }

    /// Calculate the L1 distance between two coefficient set.
    ///
    /// For more distance methods, please see [`Distance`].
    #[must_use]
    pub fn distance(&self, rhs: &Self) -> f64 {
        self.l1_norm(rhs)
    }

    /// Get a reference to the pose coefficients.
    #[must_use]
    pub fn pose_coeffs(&self) -> &Coeff<D> {
        &self.pose
    }

    /// Get a view to the specific pose coefficients. (`0..self.harmonic()`)
    #[must_use]
    pub fn pose_coeff(&self, harmonic: usize) -> CKernel<D> {
        CKernel::<D>::from_slice(self.pose.column(harmonic).data.into_slice())
    }

    /// Get an iterator over all the pose coefficients per harmonic.
    pub fn coeffs_iter(&self) -> impl Iterator<Item = CKernel<D>> {
        self.pose
            .column_iter()
            .map(|c| CKernel::<D>::from_slice(c.data.into_slice()))
    }

    /// Generate (reconstruct) the pose series.
    #[must_use]
    pub fn generate_pose(&self, n: usize) -> Vec<Coord<D>> {
        self.generate_pose_in(n, TAU)
    }

    /// Generate (reconstruct) a half of the pose series. (`theta=PI`)
    ///
    /// # Panics
    ///
    /// Panics if the number of the points `n` is less than 2.
    #[must_use]
    pub fn generate_pose_half(&self, n: usize) -> Vec<Coord<D>> {
        self.generate_in(n, PI)
    }

    /// Generate (reconstruct) a pose series in a specific angle `theta`
    /// (`0..=TAU`).
    ///
    /// Normalized curve is **without** transformation.
    ///
    /// # Panics
    ///
    /// Panics if the number of the points `n` is less than 2.
    #[must_use]
    pub fn generate_pose_in(&self, n: usize, theta: f64) -> Vec<Coord<D>> {
        assert!(n > 1, "n ({n}) must larger than 1");
        let t = na::Matrix1xX::from_fn(n, |_, i| i as f64 / (n - 1) as f64 * theta);
        let mut curve = self
            .pose
            .column_iter()
            .enumerate()
            .map(|(i, c)| {
                let t = &t * (i + 1) as f64;
                let t = na::Matrix2xX::from_rows(&[t.map(f64::cos), t.map(f64::sin)]);
                CKernel::<D>::from_slice(c.as_slice()) * t
            })
            .reduce(|a, b| a + b)
            .unwrap_or_else(|| MatrixRxX::<D>::from_vec(Vec::new()))
            .column_iter()
            .map(|row| core::array::from_fn(|i| row[i]))
            .collect::<Vec<_>>();
        self.as_geo().transform_inplace(&mut curve);
        curve
    }
}

impl<const D: usize> core::ops::Deref for PosedEfd<D>
where
    U<D>: EfdDim<D>,
    na::Const<D>: na::DimNameMul<na::U2>,
{
    type Target = Efd<D>;

    fn deref(&self) -> &Self::Target {
        &self.efd
    }
}
