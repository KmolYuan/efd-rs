extern crate ndarray;

use ndarray::{array, Array, Array1, Array2, Axis, concatenate, s};

/// Curve fitting using Elliptical Fourier Descriptor.
pub fn efd_fitting(contour: &Array2<f64>, n: usize) -> Array2<f64> {
    let harmonic = fourier_power(
        &calculate_efd(contour, nyquist(contour)),
        nyquist(contour),
        1. - 1e-4,
    );
    let (coeffs, rot) = normalize_efd(&calculate_efd(contour, harmonic));
    let locus_v = locus(contour);
    let contour = inverse_transform(
        &coeffs,
        locus_v,
        if n < 3 { contour.nrows() } else { n },
        harmonic,
    );
    rotate_contour(&contour, -rot, locus_v)
}

/// Returns the maximum number of harmonics that can be computed for a given
/// contour, the Nyquist Frequency.
fn nyquist(zx: &Array2<f64>) -> usize {
    zx.nrows() / 2
}

fn square(v: f64) -> f64 {
    v * v
}

fn diff1(a: &Array1<f64>) -> Array1<f64> {
    let len = a.len() - 1;
    let mut out = Array1::zeros(len);
    for i in 0..len {
        out[i] = a[i + 1] - a[i];
    }
    out
}

fn diff2(a: &Array2<f64>) -> Array2<f64> {
    let len = a.nrows() - 1;
    let mut out = Array2::zeros((len, 2));
    for i in 0..len {
        for j in 0..2 {
            out[[i, j]] = a[[i + 1, j]] - a[[i, j]];
        }
    }
    out
}

fn cumsum(a: &Array1<f64>) -> Array1<f64> {
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
pub fn fourier_power(coeffs: &Array2<f64>, nyq: usize, threshold: f64) -> usize {
    let mut total_power = 0.;
    let mut current_power = 0.;
    for i in 0..nyq as usize {
        total_power += 0.5 * coeffs.slice(s![i, ..]).mapv(square).sum();
    }
    for i in 0..nyq as usize {
        current_power += 0.5 * coeffs.slice(s![i, ..]).mapv(square).sum();
        if current_power / total_power > threshold {
            return i + 1;
        }
    }
    nyq
}

/// Compute the Elliptical Fourier Descriptors for a polygon.
pub fn calculate_efd(contour: &Array2<f64>, harmonic: usize) -> Array2<f64> {
    let dxy = diff2(contour);
    let dt = dxy.mapv(square).sum_axis(Axis(0)).mapv(f64::sqrt);
    let t = concatenate(Axis(0), &[array![0.].view(), cumsum(&dt).view()]).unwrap();
    let zt = t[t.len() - 1];
    let phi = t.mapv(|v| v * std::f64::consts::TAU / zt + 1e-20);
    let mut coeffs = Array2::zeros((harmonic, 4));
    for n in 1..(harmonic + 1) {
        let c = zt / (square(n as f64) * std::f64::consts::TAU);
        let phi_n = phi.mapv(|v| v * n as f64);
        let cos_phi_n = phi_n.slice(s![1..]).mapv(f64::cos) - phi_n.slice(s![..-1]).mapv(f64::cos);
        let sin_phi_n = phi_n.slice(s![1..]).mapv(f64::sin) - phi_n.slice(s![..-1]).mapv(f64::sin);
        coeffs[[n - 1, 0]] = (dxy.slice(s![.., 1]).into_owned() / dt.view() * cos_phi_n.view()).sum();
        coeffs[[n - 1, 1]] = (dxy.slice(s![.., 1]).into_owned() / dt.view() * sin_phi_n.view()).sum();
        coeffs[[n - 1, 2]] = (dxy.slice(s![.., 0]).into_owned() / dt.view() * cos_phi_n.view()).sum();
        coeffs[[n - 1, 3]] = (dxy.slice(s![.., 0]).into_owned() / dt.view() * sin_phi_n.view()).sum();
        coeffs.slice_mut(s![n - 1, ..]).mapv_inplace(|v| v * c);
    }
    coeffs
}

/// Normalize the Elliptical Fourier Descriptor coefficients for a polygon.
pub fn normalize_efd(coeffs: &Array2<f64>) -> (Array2<f64>, f64) {
    let theta1 = 0.5 * f64::atan2(
        2. * (coeffs[[0, 0]] * coeffs[[0, 1]] + coeffs[[0, 2]] * coeffs[[0, 3]]),
        coeffs[[0, 0]] * coeffs[[0, 0]] - coeffs[[0, 1]] * coeffs[[0, 1]]
            + coeffs[[0, 2]] * coeffs[[0, 2]] - coeffs[[0, 3]] * coeffs[[0, 3]],
    );
    let mut coeffs = coeffs.clone();
    for n in 0..coeffs.nrows() {
        let angle = (n + 1) as f64 * theta1;
        let m = array![
            [coeffs[[n, 0]], coeffs[[n, 1]]],
            [coeffs[[n, 2]], coeffs[[n, 3]]]
        ].dot(&array![
            [angle.cos(), -angle.sin()],
            [angle.sin(), angle.cos()],
        ]);
        coeffs.slice_mut(s![n, ..]).assign(&Array::from_iter(m.iter().cloned()));
    }
    let psi1 = coeffs[[0, 2]].atan2(coeffs[[0, 0]]);
    let psi2 = array![
        [psi1.cos(), psi1.sin()],
        [-psi1.sin(), psi1.cos()],
    ];
    for n in 0..coeffs.nrows() {
        let m = psi2.dot(&array![
            [coeffs[[n, 0]], coeffs[[n, 1]]],
            [coeffs[[n, 2]], coeffs[[n, 3]]],
        ]);
        coeffs.slice_mut(s![n, ..]).assign(&Array::from_iter(m.iter().cloned()));
    }
    (coeffs, psi1)
}

/// Compute the dc coefficients, used as the locus when calling [inverse_transform](fn.inverse_transform.html).
pub fn locus(contour: &Array2<f64>) -> (f64, f64) {
    let dxy = diff2(contour);
    let dt = dxy.mapv(square).sum_axis(Axis(0)).mapv(f64::sqrt);
    let t = concatenate(Axis(0), &[array![0.].view(), cumsum(&dt).view()]).unwrap();
    let zt = t[t.len() - 1];
    let diffs = diff1(&t.mapv(square));
    let xi = cumsum(&dxy.slice(s![.., 0]).into_owned())
        - dxy.slice(s![.., 1]).into_owned() / dt.view() * t.slice(s![1..]);
    let a0 = (dxy.slice(s![.., 0]).into_owned() / dt.mapv(|v| v * 2.)
        * diffs.view() + xi * dt.view()).sum() / (zt + 1e-20);
    let delta = cumsum(&dxy.slice(s![.., 1]).into_owned())
        - dxy.slice(s![.., 1]).into_owned() / dt.view() * t.slice(s![1..]);
    let c0 = (dxy.slice(s![.., 1]).into_owned() / dt.mapv(|v| v * 2.)
        * diffs.view() + delta * dt.view()).sum() / (zt + 1e-20);
    (contour[[0, 0]] + a0, contour[[0, 1]] + c0)
}

/// Perform an inverse fourier transform to convert the coefficients back into
/// spatial coordinates.
pub fn inverse_transform(
    coeffs: &Array2<f64>,
    locus_v: (f64, f64),
    n: usize,
    harmonic: usize,
) -> Array2<f64> {
    let t = Array1::linspace(0., 1., n);
    let mut contour = Array2::ones((n, 2));
    contour.slice_mut(s![.., 0]).mapv_inplace(|v| v * locus_v.0);
    contour.slice_mut(s![.., 1]).mapv_inplace(|v| v * locus_v.1);
    for n in 0..harmonic {
        let angle = t.mapv(|v| v * (n + 1) as f64 * std::f64::consts::TAU);
        let cos = angle.mapv(f64::cos);
        let sin = angle.mapv(f64::sin);
        let mut contour0 = contour.slice_mut(s![.., 0]);
        contour0 += &cos.mapv(|v| v * coeffs[[n, 2]]);
        contour0 += &sin.mapv(|v| v * coeffs[[n, 3]]);
        let mut contour1 = contour.slice_mut(s![.., 1]);
        contour1 += &cos.mapv(|v| v * coeffs[[n, 0]]);
        contour1 += &sin.mapv(|v| v * coeffs[[n, 1]]);
    }
    contour
}

/// Rotates a contour about a point by a given amount expressed in degrees.
pub fn rotate_contour(
    contour: &Array2<f64>,
    angle: f64,
    (cpx, cpy): (f64, f64),
) -> Array2<f64> {
    let mut out = Array2::zeros(contour.raw_dim());
    for i in 0..contour.nrows() {
        let dx = contour[[i, 0]] - cpx;
        let dy = contour[[i, 1]] - cpy;
        out[[i, 0]] = cpx + dx * angle.cos() - dy * angle.sin();
        out[[i, 1]] = cpy + dx * angle.sin() - dy * angle.cos();
    }
    out
}

#[cfg(test)]
mod tests {
    // TODO: unit test here
}
