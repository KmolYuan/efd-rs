use ndarray::{arr2, Array, ArrayView2, Axis, CowArray, Dimension, FixedInitializer};
#[cfg(not(feature = "std"))]
use num_traits::Float as _;

pub(crate) type CowCurve<'a, A> = alloc::borrow::Cow<'a, [A]>;
pub(crate) type CowCurve2<'a> = CowCurve<'a, [f64; 2]>;
pub(crate) type CowCurve3<'a> = CowCurve<'a, [f64; 3]>;

#[inline(always)]
pub(crate) fn pow2(x: f64) -> f64 {
    x * x
}

pub(crate) fn diff<'a, D, A>(arr: A, axis: Option<Axis>) -> Array<f64, D>
where
    D: Dimension,
    A: Into<CowArray<'a, f64, D>>,
{
    let arr = arr.into();
    let axis = axis.unwrap_or_else(|| Axis(arr.ndim() - 1));
    let head = arr.slice_axis(axis, (..-1).into());
    let tail = arr.slice_axis(axis, (1..).into());
    &tail - &head
}

pub(crate) fn cumsum<'a, D, A>(arr: A, axis: Option<Axis>) -> Array<f64, D>
where
    D: Dimension + ndarray::RemoveAxis,
    A: Into<CowArray<'a, f64, D>>,
{
    let mut arr = arr.into().to_owned();
    let axis = axis.unwrap_or(Axis(0));
    arr.axis_iter_mut(axis).reduce(|prev, mut next| {
        next += &prev;
        next
    });
    arr
}

/// Check the difference between two curves.
pub fn curve_diff<A, B>(a: &[A], b: &[B]) -> f64
where
    A: FixedInitializer<Elem = f64> + Clone,
    B: FixedInitializer<Elem = f64> + Clone,
{
    let a = arr2(a);
    let b = arr2(b);
    a.axis_iter(Axis(0))
        .zip(b.axis_iter(Axis(0)))
        .map(|(a, b)| (&a - &b).mapv(f64::abs).sum())
        .sum()
}

/// Compute the total Fourier power and find the minimum number of harmonics
/// required to exceed the threshold fraction of the total power.
///
/// This function needs to use the full of coefficients,
/// and the threshold must in [0, 1).
///
/// ```
/// use efd::{fourier_power, Efd2};
///
/// # let curve = efd::tests::PATH;
/// // Nyquist Frequency
/// let nyq = curve.len() / 2;
/// let efd = Efd2::from_curve_harmonic(curve, nyq).unwrap();
/// // Use "None" for the default threshold (99.99%)
/// let harmonic = fourier_power(efd.coeffs(), None);
/// # assert_eq!(harmonic, 6);
/// ```
pub fn fourier_power<T>(coeffs: ArrayView2<f64>, threshold: T) -> usize
where
    T: Into<Option<f64>>,
{
    let threshold = threshold.into().unwrap_or(0.9999);
    debug_assert!((0.0..1.).contains(&threshold));
    let lut = cumsum(coeffs.mapv(pow2), None).sum_axis(Axis(1));
    let total_power = lut.last().unwrap();
    lut.iter()
        .enumerate()
        .find(|(_, power)| *power / total_power >= threshold)
        .map(|(i, _)| i + 1)
        .unwrap()
}

/// Close the curve
pub fn closed_curve<'a, A, C>(curve: C) -> Vec<A>
where
    A: Clone + 'a,
    C: Into<CowCurve<'a, A>>,
{
    let mut c = curve.into().into_owned();
    c.push(c[0].clone());
    c
}
