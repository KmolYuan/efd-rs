use crate::{util::*, *};
use alloc::vec::Vec;
use core::f64::consts::{PI, TAU};
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::*;

/// A 2D shape with a pose described by EFD.
pub type PosedEfd2 = PosedEfd<2>;
/// A 3D shape with a pose described by EFD.
pub type PosedEfd3 = PosedEfd<3>;

/// Unit vector type of the posed EFD.
pub type UVector<const D: usize> =
    na::Unit<na::Vector<f64, na::Const<D>, na::ArrayStorage<f64, D, 1>>>;

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
    /// Create a new [`PosedEfd`] from a [`Efd`] and a pose coefficients.
    pub fn from_parts_unchecked(efd: Efd<D>, pose: Coeff<D>) -> Self {
        Self { efd, pose }
    }

    /// Calculate the coefficients from two series of points.
    ///
    /// The second series is the pose series, the `curve2[i]` has the same time
    /// as `curve[i]`.
    pub fn from_series<C1, C2>(curve: C1, curve2: C2, is_open: bool) -> Self
    where
        C1: Curve<Coord<D>>,
        C2: Curve<Coord<D>>,
    {
        let len = curve.len().min(curve2.len());
        Self::from_series_harmonic(curve, curve2, is_open, if is_open { len } else { len / 2 })
            .fourier_power_anaysis(None)
    }

    /// Calculate the coefficients from two series of points.
    ///
    /// The `harmonic` is the number of the coefficients to be calculated.
    pub fn from_series_harmonic<C1, C2>(
        curve: C1,
        curve2: C2,
        is_open: bool,
        harmonic: usize,
    ) -> Self
    where
        C1: Curve<Coord<D>>,
        C2: Curve<Coord<D>>,
    {
        let curve = curve.as_curve();
        let vectors = curve
            .iter()
            .zip(curve2.as_curve())
            .map(|(a, b)| na::Point::from(*b) - na::Point::from(*a))
            .map(UVector::new_normalize)
            .collect::<Vec<_>>();
        Self::from_vectors_harmonic(curve, vectors, is_open, harmonic)
    }

    /// Calculate the coefficients from a curve and its vector of each point.
    pub fn from_vectors<C, V>(curve: C, vectors: V, is_open: bool) -> Self
    where
        C: Curve<Coord<D>>,
        V: Curve<UVector<D>>,
    {
        let len = curve.len().min(vectors.len());
        Self::from_vectors_harmonic(curve, vectors, is_open, if is_open { len } else { len / 2 })
            .fourier_power_anaysis(None)
    }

    /// Calculate the coefficients from a curve and its vector of each point.
    ///
    /// The `harmonic` is the number of the coefficients to be calculated.
    #[allow(unused_variables)]
    pub fn from_vectors_harmonic<C, V>(curve: C, vectors: V, is_open: bool, harmonic: usize) -> Self
    where
        C: Curve<Coord<D>>,
        V: Curve<UVector<D>>,
    {
        todo!() // TODO
    }

    /// Use Fourier Power Anaysis (FPA) to reduce the harmonic number.
    ///
    /// The default threshold is 99.99%.
    ///
    /// See also [`Efd::fourier_power_anaysis()`].
    ///
    /// # Panics
    ///
    /// Panics if the threshold is not in 0..1, or the harmonic is zero.
    #[must_use]
    pub fn fourier_power_anaysis<T>(mut self, threshold: T) -> Self
    where
        Option<f64>: From<T>,
    {
        let lut = (self.coeffs().map(pow2) + self.pose.map(pow2)).row_sum();
        self.set_harmonic(fourier_power_anaysis(lut, threshold));
        self
    }

    /// Set the harmonic number of the coefficients.
    ///
    /// # Panics
    ///
    /// Panics if the harmonic is zero or greater than the current harmonic.
    pub fn set_harmonic(&mut self, harmonic: usize) {
        let current = self.harmonic();
        assert!(
            (1..current).contains(&harmonic),
            "harmonic must in 1..={current}"
        );
        self.efd.set_harmonic(harmonic);
        self.pose.resize_horizontally_mut(harmonic, 0.);
    }

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
        let mut curve = U::<D>::reconstruct(&self.pose, n, theta);
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
