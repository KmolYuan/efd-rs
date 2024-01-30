use crate::{util::*, *};
use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::*;

/// A 1D shape with a pose described by EFD.
pub type PosedEfd1 = PosedEfd<1>;
/// A 2D shape with a pose described by EFD.
pub type PosedEfd2 = PosedEfd<2>;
/// A 3D shape with a pose described by EFD.
pub type PosedEfd3 = PosedEfd<3>;

type Vector<const D: usize> = na::Vector<f64, na::Const<D>, na::ArrayStorage<f64, D, 1>>;

fn uvec<V, const D: usize>(v: V) -> Coord<D>
where
    Vector<D>: From<V>,
{
    Vector::from(v).normalize().data.0[0]
}

/// A shape with a pose described by EFD.
///
/// These are the same as [`Efd`] except that it has a pose, and the data are
/// always normalized and readonly.
///
/// Start with [`PosedEfd::from_series()`] and its related methods.
///
/// # Pose Representation
/// Pose is represented by an unit vector, which is rotated by the rotation
/// of the original shape.
#[derive(Clone)]
pub struct PosedEfd<const D: usize>
where
    U<D>: EfdDim<D>,
{
    curve: Efd<D>,
    pose: Efd<D>,
}

impl PosedEfd2 {
    /// Calculate the coefficients from a curve and its angles from each point.
    pub fn from_angles<C>(curve: C, angles: &[f64], is_open: bool) -> Self
    where
        C: Curve<2>,
    {
        let harmonic = harmonic!(is_open, curve, angles);
        Self::from_angles_harmonic(curve, angles, is_open, harmonic).fourier_power_anaysis(None)
    }

    /// Calculate the coefficients from a curve and its angles from each point.
    ///
    /// The `harmonic` is the number of the coefficients to be calculated.
    pub fn from_angles_harmonic<C>(curve: C, angles: &[f64], is_open: bool, harmonic: usize) -> Self
    where
        C: Curve<2>,
    {
        let vectors = angles
            .iter()
            .map(|a| uvec([a.cos(), a.sin()]))
            .collect::<Vec<_>>();
        Self::from_uvec_harmonic_unchecked(curve, vectors, is_open, harmonic)
    }
}

impl<const D: usize> PosedEfd<D>
where
    U<D>: EfdDim<D>,
{
    /// Create a new [`PosedEfd`] from two [`Efd`]s. (`curve` and `pose`)
    ///
    /// See also [`PosedEfd::into_inner()`].
    pub const fn from_parts_unchecked(curve: Efd<D>, pose: Efd<D>) -> Self {
        Self { curve, pose }
    }

    /// Calculate the coefficients from two series of points.
    ///
    /// The second series is the pose series, the `curve2[i]` has the same time
    /// as `curve[i]`.
    pub fn from_series<C1, C2>(curve1: C1, curve2: C2, is_open: bool) -> Self
    where
        C1: Curve<D>,
        C2: Curve<D>,
    {
        let harmonic = harmonic!(is_open, curve1, curve2);
        Self::from_series_harmonic(curve1, curve2, is_open, harmonic).fourier_power_anaysis(None)
    }

    /// Calculate the coefficients from two series of points.
    ///
    /// The `harmonic` is the number of the coefficients to be calculated.
    pub fn from_series_harmonic<C1, C2>(
        curve1: C1,
        curve2: C2,
        is_open: bool,
        harmonic: usize,
    ) -> Self
    where
        C1: Curve<D>,
        C2: Curve<D>,
    {
        let vectors = core::iter::zip(curve1.as_curve(), curve2.as_curve())
            .map(|(a, b)| uvec(na::Point::from(*b) - na::Point::from(*a)))
            .collect::<Vec<_>>();
        Self::from_uvec_harmonic_unchecked(curve1, vectors, is_open, harmonic)
    }

    /// Calculate the coefficients from a curve and its unit vectors from each
    /// point.
    ///
    /// See also [`PosedEfd::from_uvec_unchecked()`] if you want to skip the
    /// unit vector calculation.
    pub fn from_uvec<C, V>(curve: C, vectors: V, is_open: bool) -> Self
    where
        C: Curve<D>,
        V: Curve<D>,
    {
        let harmonic = harmonic!(is_open, curve, vectors);
        Self::from_uvec_harmonic(curve, vectors, is_open, harmonic)
    }

    /// Calculate the coefficients from a curve and its unit vectors from each
    /// point.
    ///
    /// See also [`PosedEfd::from_uvec_harmonic_unchecked()`] if you want to
    /// skip the unit vector calculation.
    pub fn from_uvec_harmonic<C, V>(curve: C, vectors: V, is_open: bool, harmonic: usize) -> Self
    where
        C: Curve<D>,
        V: Curve<D>,
    {
        let vectors = vectors.to_curve().into_iter().map(uvec).collect::<Vec<_>>();
        Self::from_uvec_harmonic_unchecked(curve, vectors, is_open, harmonic)
    }

    /// Calculate the coefficients from a curve and its unit vectors from each
    /// point.
    pub fn from_uvec_unchecked<C, V>(curve: C, vectors: V, is_open: bool) -> Self
    where
        C: Curve<D>,
        V: Curve<D>,
    {
        let harmonic = harmonic!(is_open, curve, vectors);
        Self::from_uvec_harmonic_unchecked(curve, vectors, is_open, harmonic)
            .fourier_power_anaysis(None)
    }

    /// Calculate the coefficients from a curve and its unit vectors from each
    /// point.
    ///
    /// The `harmonic` is the number of the coefficients to be calculated.
    pub fn from_uvec_harmonic_unchecked<C, V>(
        curve: C,
        vectors: V,
        is_open: bool,
        harmonic: usize,
    ) -> Self
    where
        C: Curve<D>,
        V: Curve<D>,
    {
        debug_assert!(harmonic != 0, "harmonic must not be 0");
        debug_assert!(curve.len() > 2, "the curve length must greater than 2");
        let (_, [(mut curve, geo1), (mut pose, pose_geo)]) =
            U::get_coeff([curve.as_curve(), vectors.as_curve()], is_open, harmonic);
        let geo2 = U::coeff_norm(&mut curve, None, None);
        let curve = Efd::from_parts_unchecked(curve, geo1 * &geo2);
        let geo2 = U::coeff_norm(&mut pose, None, Some(geo2.rot()));
        let pose = Efd::from_parts_unchecked(pose, pose_geo * geo2);
        Self { curve, pose }
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
    pub fn fourier_power_anaysis<T>(mut self, threshold: T) -> Self
    where
        Option<f64>: From<T>,
    {
        let lut = core::iter::zip(self.curve.coeffs(), self.pose.coeffs())
            .map(|(m1, m2)| m1.map(pow2).sum() + m2.map(pow2).sum())
            .collect();
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
            (1..=current).contains(&harmonic),
            "harmonic ({harmonic}) must in 1..={current}"
        );
        self.curve.set_harmonic(harmonic);
        self.pose.set_harmonic(harmonic);
    }

    /// Consume self and return the parts of this type. The first is the curve
    /// coefficients, and the second is the pose coefficients.
    ///
    /// See also [`PosedEfd::from_parts_unchecked()`].
    pub fn into_inner(self) -> (Efd<D>, Efd<D>) {
        (self.curve, self.pose)
    }

    /// Check if the described curve is open.
    pub fn is_open(&self) -> bool {
        self.curve.is_open()
    }

    /// Get the harmonic number of the coefficients.
    pub fn harmonic(&self) -> usize {
        self.curve.harmonic()
    }

    /// Check if the coefficients are valid.
    ///
    /// It is only helpful if this object is constructed by
    /// [`PosedEfd::from_parts_unchecked()`].
    pub fn is_valid(&self) -> bool {
        self.curve.is_valid() && self.pose.is_valid()
    }

    /// Calculate the L1 distance between two coefficient set.
    ///
    /// For more distance methods, please see [`Distance`].
    #[must_use = "this returns the result of the operation, without modifying the original"]
    pub fn distance(&self, rhs: &Self) -> f64 {
        self.l1_norm(rhs)
    }

    /// Get the reference of the curve coefficients.
    pub fn curve_efd(&self) -> &Efd<D> {
        &self.curve
    }

    /// Get the reference of the posed coefficients.
    pub fn pose_efd(&self) -> &Efd<D> {
        &self.pose
    }

    /// Obtain the curve and pose for visualization.
    ///
    /// The `len` is the length of the pose vector.
    pub fn generate(&self, n: usize, len: f64) -> (Vec<Coord<D>>, Vec<Coord<D>>) {
        generate_pair(self.curve.generate(n), self.pose.generate(n), len)
    }

    /// Obtain the curve and pose for visualization in half range.
    ///
    /// The `len` is the length of the pose vector.
    pub fn generate_half(&self, n: usize, len: f64) -> (Vec<Coord<D>>, Vec<Coord<D>>) {
        generate_pair(self.curve.generate_half(n), self.pose.generate_half(n), len)
    }

    /// Obtain the curve and pose for visualization from a series of time `t`.
    pub fn generate_by(&self, t: &[f64], len: f64) -> (Vec<Coord<D>>, Vec<Coord<D>>) {
        generate_pair(self.curve.generate_by(t), self.pose.generate_by(t), len)
    }
}

fn generate_pair<const D: usize>(
    curve: Vec<Coord<D>>,
    pose: Vec<Coord<D>>,
    len: f64,
) -> (Vec<Coord<D>>, Vec<Coord<D>>) {
    let pose = core::iter::zip(&curve, pose)
        .map(|(p, v)| na::Point::from(*p) + na::Vector::from(v) * len)
        .map(|p| p.coords.data.0[0])
        .collect();
    (curve, pose)
}
