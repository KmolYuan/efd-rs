//! This crate implements Elliptical Fourier Descriptor (EFD) curve fitting
//! by using `ndarray` to handle the 2D arrays.
//!
//! Reference: Kuhl, FP and Giardina, CR (1982). Elliptic Fourier features of
//! a closed contour. Computer graphics and image processing, 18(3), 236-258.
//!
//! The following is an example. The contours are always Nx2 array in the functions.
//! ```
//! use std::f64::consts::TAU;
//! use ndarray::{Array1, Axis, stack};
//! use efd::{efd_fitting, ElementWiseOpt};
//!
//! fn main() {
//!     const N: usize = 10;
//!     let circle = stack!(Axis(1),
//!                         Array1::linspace(0., TAU, N).cos(),
//!                         Array1::linspace(0., TAU, N).sin());
//!     assert_eq!(circle.shape(), &[10, 2]);
//!     let new_curve = efd_fitting(&circle, 20, None);
//!     assert_eq!(new_curve.shape(), &[20, 2]);
//! }
//! ```
//!
//! Arrays have "owned" and "view" two data types, all functions are compatible.
//!
pub use crate::element_opt::*;
use ndarray::{array, concatenate, s, stack, Array1, Array2, AsArray, Axis, Ix2};
use std::f64::consts::{PI, TAU};

mod element_opt;
#[cfg(test)]
mod tests;

/// Curve fitting using Elliptical Fourier Descriptor.
///
/// Giving the contour and the number of output path (`n`).
/// The `harmonic` is the number of harmonic terms.
/// Use `Option::None` to auto detect the number of harmonics.
pub fn efd_fitting<'a, A>(contour: A, n: usize, harmonic: Option<usize>) -> Array2<f64>
where
    A: AsArray<'a, f64, Ix2>,
{
    assert!(n > 3, "n must larger than 3, current is {}", n);
    let contour = contour.into();
    let harmonic = harmonic.unwrap_or(fourier_power(
        &calculate_efd(contour, nyquist(contour)),
        nyquist(contour),
        1.,
    ));
    let coeffs = calculate_efd(contour, harmonic);
    let (coeffs, rot) = normalize_efd(&coeffs, false);
    let locus = locus(contour);
    let contour = inverse_transform(&coeffs, locus, n, harmonic);
    rotate_contour(&contour, -rot, locus)
}

/// Returns the maximum number of harmonics that can be computed for a given
/// contour, the Nyquist Frequency.
pub fn nyquist<'a, A>(zx: A) -> usize
where
    A: AsArray<'a, f64, Ix2>,
{
    zx.into().nrows() / 2
}

fn cumsum<'a, A>(a: A) -> Array1<f64>
where
    A: AsArray<'a, f64>,
{
    let a = a.into();
    let mut out = Array1::zeros(a.len());
    for (i, &v) in a.iter().enumerate() {
        out[i] = v;
        if i > 0 {
            out[i] += out[i - 1];
        }
    }
    out
}

/// Compute the total Fourier power and find the minimum number of harmonics
/// required to exceed the threshold fraction of the total power.
///
/// This function needs to use the full of coefficients,
/// and the threshold usually used as 1.
pub fn fourier_power<'a, A>(coeffs: A, nyq: usize, threshold: f64) -> usize
where
    A: AsArray<'a, f64, Ix2>,
{
    let coeffs = coeffs.into();
    let total_power = 0.5 * coeffs.square().sum();
    let mut power = 0.;
    for i in 0..nyq as usize {
        power += 0.5 * coeffs.slice(s![i, ..]).square().sum();
        if power / total_power >= threshold {
            return i + 1;
        }
    }
    nyq
}

/// Compute the Elliptical Fourier Descriptors for a polygon.
pub fn calculate_efd<'a, A>(contour: A, harmonic: usize) -> Array2<f64>
where
    A: AsArray<'a, f64, Ix2>,
{
    let contour = contour.into();
    let dxy = diff(contour, Some(Axis(0)));
    let dt = dxy.square().sum_axis(Axis(1)).sqrt();
    let t = concatenate!(Axis(0), array![0.], cumsum(&dt));
    let zt = t[t.len() - 1];
    let phi = &t * TAU / (zt + 1e-20);
    let mut coeffs = Array2::zeros((harmonic, 4));
    for n in 1..(harmonic + 1) {
        let c = zt / (2. * (n * n) as f64 * PI * PI);
        let phi_n = &phi * n as f64;
        let cos_phi_n = (phi_n.slice(s![1..]).cos() - phi_n.slice(s![..-1]).cos()) / &dt;
        let sin_phi_n = (phi_n.slice(s![1..]).sin() - phi_n.slice(s![..-1]).sin()) / &dt;
        coeffs[[n - 1, 0]] = c * (&dxy.slice(s![.., 1]) * &cos_phi_n).sum();
        coeffs[[n - 1, 1]] = c * (&dxy.slice(s![.., 1]) * &sin_phi_n).sum();
        coeffs[[n - 1, 2]] = c * (&dxy.slice(s![.., 0]) * &cos_phi_n).sum();
        coeffs[[n - 1, 3]] = c * (&dxy.slice(s![.., 0]) * &sin_phi_n).sum();
    }
    coeffs
}

/// Normalize the Elliptical Fourier Descriptor coefficients for a polygon.
///
/// If `norm` optional is true, normalize all coefficients by first one.
pub fn normalize_efd<'a, A>(coeffs: A, norm: bool) -> (Array2<f64>, f64)
where
    A: AsArray<'a, f64, Ix2>,
{
    let coeffs = coeffs.into();
    let theta1 = f64::atan2(
        2. * (coeffs[[0, 0]] * coeffs[[0, 1]] + coeffs[[0, 2]] * coeffs[[0, 3]]),
        coeffs[[0, 0]] * coeffs[[0, 0]] - coeffs[[0, 1]] * coeffs[[0, 1]]
            + coeffs[[0, 2]] * coeffs[[0, 2]]
            - coeffs[[0, 3]] * coeffs[[0, 3]],
    ) * 0.5;
    let mut coeffs = coeffs.to_owned().clone();
    for n in 0..coeffs.nrows() {
        let angle = (n + 1) as f64 * theta1;
        let m = array![
            [coeffs[[n, 0]], coeffs[[n, 1]]],
            [coeffs[[n, 2]], coeffs[[n, 3]]],
        ]
        .dot(&array![
            [angle.cos(), -angle.sin()],
            [angle.sin(), angle.cos()],
        ]);
        coeffs
            .slice_mut(s![n, ..])
            .assign(&Array1::from_iter(m.iter().cloned()));
    }
    let psi1 = f64::atan2(coeffs[[0, 2]], coeffs[[0, 0]]);
    let psi2 = array![[psi1.cos(), psi1.sin()], [-psi1.sin(), psi1.cos()]];
    for n in 0..coeffs.nrows() {
        let m = psi2.dot(&array![
            [coeffs[[n, 0]], coeffs[[n, 1]]],
            [coeffs[[n, 2]], coeffs[[n, 3]]],
        ]);
        coeffs
            .slice_mut(s![n, ..])
            .assign(&Array1::from_iter(m.iter().cloned()));
    }
    if norm {
        coeffs /= coeffs[[0, 0]].abs();
    }
    (coeffs, psi1)
}

/// Compute the dc coefficients, used as the locus when calling [`inverse_transform`].
pub fn locus<'a, A>(contour: A) -> (f64, f64)
where
    A: AsArray<'a, f64, Ix2>,
{
    let contour = contour.into();
    let dxy = diff(contour, Some(Axis(0)));
    let dt = dxy.square().sum_axis(Axis(1)).sqrt();
    let t = concatenate!(Axis(0), array![0.], cumsum(&dt));
    let zt = t[t.len() - 1];
    let xi = cumsum(&dxy.slice(s![.., 0])) - &dxy.slice(s![.., 0]) / &dt * t.slice(s![1..]);
    let c = diff(&t.square(), None) / (&dt * 2.);
    let a0 = (&dxy.slice(s![.., 0]) * &c + xi * &dt).sum() / (zt + 1e-20);
    let delta = cumsum(&dxy.slice(s![.., 1])) - &dxy.slice(s![.., 1]) / &dt * t.slice(s![1..]);
    let c0 = (&dxy.slice(s![.., 1]) * &c + delta * &dt).sum() / (zt + 1e-20);
    (contour[[0, 0]] + a0, contour[[0, 1]] + c0)
}

/// Perform an inverse fourier transform to convert the coefficients back into
/// spatial coordinates.
pub fn inverse_transform<'a, A>(
    coeffs: A,
    locus: (f64, f64),
    n: usize,
    harmonic: usize,
) -> Array2<f64>
where
    A: AsArray<'a, f64, Ix2>,
{
    let coeffs = coeffs.into();
    let t = Array1::linspace(0., 1., n);
    let mut contour0 = Array1::ones(n);
    let mut contour1 = Array1::ones(n);
    contour0 *= locus.0;
    contour1 *= locus.1;
    for n in 0..harmonic {
        let angle = &t * (n + 1) as f64 * TAU;
        let cos = angle.cos();
        let sin = angle.sin();
        contour0 += &(&cos * coeffs[[n, 2]]);
        contour0 += &(&sin * coeffs[[n, 3]]);
        contour1 += &(&cos * coeffs[[n, 0]]);
        contour1 += &(&sin * coeffs[[n, 1]]);
    }
    stack!(Axis(1), contour0, contour1)
}

/// Rotates a contour about a point by a given amount expressed in degrees.
pub fn rotate_contour<'a, A>(contour: A, angle: f64, (cpx, cpy): (f64, f64)) -> Array2<f64>
where
    A: AsArray<'a, f64, Ix2>,
{
    let contour = contour.into();
    let mut out = Array2::zeros(contour.dim());
    for i in 0..contour.nrows() {
        let dx = contour[[i, 0]] - cpx;
        let dy = contour[[i, 1]] - cpy;
        out[[i, 0]] = cpx + dx * angle.cos() - dy * angle.sin();
        out[[i, 1]] = cpy + dx * angle.sin() + dy * angle.cos();
    }
    out
}
