use crate::*;
use alloc::vec::Vec;
use core::{array, iter::zip};
#[cfg(not(feature = "std"))]
#[allow(unused_imports)]
use num_traits::*;

const IS_OPEN: bool = false;

/// A 1D shape with a pose described by EFD.
pub type PosedEfd1 = PosedEfd<1>;
/// A 2D shape with a pose described by EFD.
pub type PosedEfd2 = PosedEfd<2>;
/// A 3D shape with a pose described by EFD.
pub type PosedEfd3 = PosedEfd<3>;

fn uvec<const D: usize>(v: [f64; D]) -> Coord<D> {
    let norm = v.l2_norm(&[0.; D]);
    v.map(|x| x / norm)
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
    efd: Efd<D>,
    is_open: bool,
}

impl PosedEfd2 {
    /// Calculate the coefficients from a curve and its angles from each point.
    pub fn from_angles<C>(curve: C, angles: &[f64], is_open: bool) -> Self
    where
        C: Curve<2>,
    {
        let harmonic = harmonic!(IS_OPEN, curve, angles);
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
            .map(|a| [a.cos(), a.sin()])
            .collect::<Vec<_>>();
        Self::from_uvec_harmonic_unchecked(curve, vectors, is_open, harmonic)
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
    pub const fn from_efd(efd: Efd<D>, is_open: bool) -> Self {
        Self { efd, is_open }
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
        let harmonic = harmonic!(IS_OPEN, curve1, curve2);
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
        let vectors = zip(curve1.as_curve(), curve2.as_curve())
            .map(|(a, b)| uvec(array::from_fn(|i| b[i] - a[i])))
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
        let harmonic = harmonic!(IS_OPEN, curve, vectors);
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
        let harmonic = harmonic!(IS_OPEN, curve, vectors);
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
        let (_, geo1) = get_target_pos(curve.as_curve(), is_open);
        let geo_inv = geo1.inverse();
        let mut curve = geo_inv.transform(curve);
        // A constant length to define unit vectors
        const LENGTH: f64 = 1.;
        let vectors = zip(&curve, geo_inv.only_rot().transform(vectors))
            .map(|(p, v)| array::from_fn(|i| p[i] + LENGTH * v[i]))
            .rev()
            .collect::<Vec<_>>();
        let rev_guide = curve
            .iter()
            .rev()
            .map(|p| array::from_fn(|i| p[i] + vectors[0][i]));
        let mut guide = curve.clone();
        guide.extend(rev_guide);
        curve.extend(vectors);
        let (_, coeffs, geo2) = U::get_coeff(&curve, IS_OPEN, harmonic, Some(&guide));
        let efd = Efd::from_parts_unchecked(coeffs, geo1 * geo2);
        Self { efd, is_open }
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
        self.efd = self.efd.fourier_power_anaysis(threshold);
        self
    }

    /// Check if the descibed curve is open.
    ///
    /// Unlike [`Efd::is_open()`], this method is not the `is_open` of the
    /// coefficients.
    pub const fn is_open(&self) -> bool {
        self.is_open
    }

    /// Consume self and return the parts of this type. The first is the curve
    /// coefficients, and the second is the pose coefficients.
    ///
    /// See also [`PosedEfd::from_efd()`].
    pub fn into_inner(self) -> Efd<D> {
        self.efd
    }
}

impl<const D: usize> core::ops::Deref for PosedEfd<D>
where
    U<D>: EfdDim<D>,
{
    type Target = Efd<D>;

    fn deref(&self) -> &Self::Target {
        &self.efd
    }
}
