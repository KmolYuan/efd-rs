//! Posed EFD (Elliptic Fourier Descriptor) is a special shape to describe the
//! pose of a curve. It is a combination of two curves, the first is the
//! original curve, and the second is the pose unit vectors.
//!
//! Please see the [`PosedEfd`] type for more information.
use crate::*;
use alloc::vec::Vec;
use core::{array, iter::zip};
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::*;

/// A 1D shape with a pose described by EFD.
pub type PosedEfd1 = PosedEfd<1>;
/// A 2D shape with a pose described by EFD.
pub type PosedEfd2 = PosedEfd<2>;
/// A 3D shape with a pose described by EFD.
pub type PosedEfd3 = PosedEfd<3>;

/// Transform 2D angles to unit vectors.
pub fn ang2vec(angles: &[f64]) -> Vec<[f64; 2]> {
    angles.iter().map(|a| [a.cos(), a.sin()]).collect()
}

/// Motion signature with the target position.
///
/// Used to present an "original motion". Can be compared with [`PosedEfd`] by
/// [`PosedEfd::err_sig()`].
pub struct MotionSig<const D: usize>
where
    U<D>: EfdDim<D>,
{
    /// Normalized curve
    pub curve: Vec<[f64; D]>,
    /// Normalized unit vectors
    pub vectors: Vec<[f64; D]>,
    t: Vec<f64>,
    geo: GeoVar<Rot<D>, D>,
}

impl<const D: usize> MotionSig<D>
where
    U<D>: EfdDim<D>,
{
    /// Get the path signature and its target position from a curve and its unit
    /// vectors.
    ///
    /// ```
    /// use efd::posed::{ang2vec, MotionSig};
    /// # let curve = efd::tests::CURVE2D_POSE;
    /// # let angles = efd::tests::ANGLE2D_POSE;
    ///
    /// let sig = MotionSig::new(curve, ang2vec(angles));
    /// ```
    ///
    /// See also [`PathSig`].
    pub fn new<C, V>(curve: C, vectors: V) -> Self
    where
        C: Curve<D>,
        V: Curve<D>,
    {
        let PathSig { curve, t, geo } = PathSig::new(curve.as_curve(), true);
        let mut vectors = geo.inverse().only_rot().transform(vectors);
        for (p, v) in zip(&curve, &mut vectors) {
            *v = array::from_fn(|i| p[i] + v[i]);
        }
        Self { curve, vectors, t, geo }
    }

    /// Get the reference of normalized time parameters.
    pub fn as_t(&self) -> &[f64] {
        &self.t
    }

    /// Get the reference of geometric variables.
    pub fn as_geo(&self) -> &GeoVar<Rot<D>, D> {
        &self.geo
    }
}

/// An open-curve shape with a pose described by EFD.
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
    pub fn from_angles<C>(curve: C, angles: &[f64]) -> Self
    where
        C: Curve<2>,
    {
        let harmonic = harmonic(false, curve.len());
        Self::from_angles_harmonic(curve, angles, harmonic).fourier_power_anaysis(None)
    }

    /// Calculate the coefficients from a curve and its angles from each point.
    ///
    /// The `harmonic` is the number of the coefficients to be calculated.
    pub fn from_angles_harmonic<C>(curve: C, angles: &[f64], harmonic: usize) -> Self
    where
        C: Curve<2>,
    {
        Self::from_uvec_harmonic(curve, ang2vec(angles), harmonic)
    }
}

impl<const D: usize> PosedEfd<D>
where
    U<D>: EfdDim<D>,
{
    /// Create object from an [`Efd`] object.
    ///
    /// Posed EFD is a special shape to describe the pose, `efd` is only used to
    /// describe this motion signature.
    ///
    /// See also [`PosedEfd::into_inner()`].
    pub const fn from_parts_unchecked(efd: Efd<D>, posed: Efd<D>) -> Self {
        Self { curve: efd, pose: posed }
    }

    /// Calculate the coefficients from two series of points.
    ///
    /// The second series is the pose series, the `curve2[i]` has the same time
    /// as `curve[i]`.
    pub fn from_series<C1, C2>(curve_p: C1, curve_q: C2) -> Self
    where
        C1: Curve<D>,
        C2: Curve<D>,
    {
        let harmonic = harmonic(true, curve_p.len());
        Self::from_series_harmonic(curve_p, curve_q, harmonic).fourier_power_anaysis(None)
    }

    /// Calculate the coefficients from two series of points.
    ///
    /// The `harmonic` is the number of the coefficients to be calculated.
    pub fn from_series_harmonic<C1, C2>(curve_p: C1, curve_q: C2, harmonic: usize) -> Self
    where
        C1: Curve<D>,
        C2: Curve<D>,
    {
        let vectors = zip(curve_p.as_curve(), curve_q.as_curve())
            .map(|(p, q)| {
                let norm = p.l2_err(q);
                array::from_fn(|i| (q[i] - p[i]) / norm)
            })
            .collect::<Vec<_>>();
        Self::from_uvec_harmonic(curve_p, vectors, harmonic)
    }

    /// Calculate the coefficients from a curve and its unit vectors from each
    /// point.
    ///
    /// If the unit vectors is not normalized, the length of the first vector
    /// will be used as the scaling factor.
    pub fn from_uvec<C, V>(curve: C, vectors: V) -> Self
    where
        C: Curve<D>,
        V: Curve<D>,
    {
        let harmonic = harmonic(true, curve.len());
        Self::from_uvec_harmonic(curve, vectors, harmonic).fourier_power_anaysis(None)
    }

    /// Calculate the coefficients from a curve and its unit vectors from each
    /// point.
    ///
    /// If the unit vectors is not normalized, the length of the first vector
    /// will be used as the scaling factor.
    ///
    /// The `harmonic` is the number of the coefficients to be calculated.
    pub fn from_uvec_harmonic<C, V>(curve: C, vectors: V, harmonic: usize) -> Self
    where
        C: Curve<D>,
        V: Curve<D>,
    {
        debug_assert!(harmonic != 0, "harmonic must not be 0");
        debug_assert!(curve.len() > 2, "the curve length must greater than 2");
        debug_assert!(
            curve.len() == vectors.len(),
            "the curve length must be equal to the vectors length"
        );
        let curve = curve.as_curve();
        let guide = zip(curve, &curve[1..])
            .map(|(a, b)| a.l2_err(b))
            .collect::<Vec<_>>();
        let (_, mut coeffs, geo) = U::get_coeff(curve, true, harmonic, Some(&guide));
        let geo = geo * U::norm_coeff(&mut coeffs, None);
        let geo_inv = geo.inverse();
        let p_norm = geo_inv.transform(curve);
        let mut q_norm = geo_inv.only_rot().transform(vectors);
        for (p, q) in zip(p_norm, &mut q_norm) {
            *q = array::from_fn(|i| p[i] + q[i]);
        }
        let curve = Efd::from_parts_unchecked(coeffs, geo);
        let (_, mut coeffs, q_trans) = U::get_coeff(&q_norm, true, harmonic, Some(&guide));
        U::norm_zeta(&mut coeffs, None);
        let pose = Efd::from_parts_unchecked(coeffs, q_trans);
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
        Option<f64>: From<T> + Clone,
    {
        let threshold = Option::from(threshold).unwrap_or(0.999999);
        let [harmonic1, harmonic2] = [&self.curve, &self.pose].map(|efd| {
            let lut = efd.coeffs_iter().map(|m| m.map(util::pow2).sum()).collect();
            fourier_power_anaysis(lut, Some(threshold))
        });
        let harmonic = harmonic1.max(harmonic2);
        self.curve.set_harmonic(harmonic);
        self.pose.set_harmonic(harmonic);
        self
    }

    /// Consume self and return the parts of this type. The first is the curve
    /// coefficients, and the second is the pose coefficients.
    ///
    /// See also [`PosedEfd::from_parts_unchecked()`].
    pub fn into_inner(self) -> (Efd<D>, Efd<D>) {
        (self.curve, self.pose)
    }

    /// Get the reference of the curve coefficients.
    pub fn as_curve(&self) -> &Efd<D> {
        &self.curve
    }

    /// Get the reference of the pose coefficients.
    pub fn as_pose(&self) -> &Efd<D> {
        &self.pose
    }

    /// Get the reference of geometric variables.
    pub fn as_geo(&self) -> &GeoVar<Rot<D>, D> {
        self.curve.as_geo()
    }

    /// Get the harmonic number of the coefficients.
    #[inline]
    pub fn harmonic(&self) -> usize {
        self.curve.harmonic()
    }

    /// Calculate the error between two [`PosedEfd`].
    pub fn err(&self, rhs: &Self) -> f64 {
        (2. * self.curve.err(&rhs.curve))
            .max(self.pose.err(&rhs.pose))
            .max((self.pose.as_geo().trans()).l2_err(&rhs.pose.as_geo().trans()))
    }

    /// Calculate the error from a [`MotionSig`].
    pub fn err_sig(&self, sig: &MotionSig<D>) -> f64 {
        let curve =
            zip(self.curve.recon_norm_by(&sig.t), &sig.curve).map(|(a, b)| 2. * a.l2_err(b));
        let pose = zip(self.pose.recon_by(&sig.t), &sig.vectors).map(|(a, b)| a.l2_err(b));
        curve.chain(pose).fold(0., f64::max)
    }
}
