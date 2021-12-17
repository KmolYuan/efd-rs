use crate::GeoInfo;
use ndarray::{
    arr2, array, concatenate, s, Array, Array1, Array2, AsArray, Axis, Dimension, Slice, Zip,
};
use std::{
    f64::consts::{PI, TAU},
    ops::Sub,
};

fn diff<'a, A, D, V>(arr: V, axis: Option<Axis>) -> Array<A, D>
where
    A: Sub<Output = A> + Clone + 'static,
    D: Dimension,
    V: AsArray<'a, A, D>,
{
    let arr = arr.into();
    let axis = axis.unwrap_or_else(|| Axis(arr.ndim() - 1));
    let head = arr.slice_axis(axis, Slice::from(..-1));
    let tail = arr.slice_axis(axis, Slice::from(1..));
    &tail - &head
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
///
/// ```
/// use efd::{fourier_power, Efd};
/// # use efd::tests::PATH;
/// # let curve = PATH;
/// // Nyquist Frequency
/// let nyq = curve.len() / 2;
/// let harmonic = fourier_power(Efd::from_curve(curve, Some(nyq)), nyq, 1.);
/// # assert_eq!(harmonic, 6);
/// ```
pub fn fourier_power(efd: Efd, nyq: usize, threshold: f64) -> usize {
    let total_power = 0.5 * efd.c.mapv(|v| v * v).sum();
    let mut power = 0.;
    for i in 0..nyq {
        power += 0.5 * efd.c.slice(s![i, ..]).mapv(|v| v * v).sum();
        if power / total_power >= threshold {
            return i + 1;
        }
    }
    nyq
}

/// Elliptical Fourier Descriptor coefficients.
/// Provide transformation between discrete points and coefficients.
#[derive(Clone)]
pub struct Efd {
    /// Coefficients
    pub c: Array2<f64>,
    /// Center of the first ellipse
    pub locus: (f64, f64),
}

impl Efd {
    /// Calculate EFD coefficients from an existing discrete points.
    ///
    /// If the harmonic number is not given, it will be calculated with [`fourier_power`] function.
    pub fn from_curve(curve: &[[f64; 2]], harmonic: Option<usize>) -> Self {
        let harmonic = match harmonic {
            Some(h) => h,
            None => {
                // Nyquist Frequency
                let nyq = curve.len() / 2;
                fourier_power(Self::from_curve(curve, Some(nyq)), nyq, 1.)
            }
        };
        let dxy = diff(&arr2(curve), Some(Axis(0)));
        let dt = dxy.mapv(|v| v * v).sum_axis(Axis(1)).mapv(f64::sqrt);
        let t = concatenate!(Axis(0), array![0.], cumsum(&dt));
        let zt = t[t.len() - 1];
        let phi = &t * TAU / (zt + 1e-20);
        let mut coeffs = Array2::zeros((harmonic, 4));
        for n in 0..harmonic {
            let n1 = n as f64 + 1.;
            let c = 0.5 * zt / (n1 * n1 * PI * PI);
            let phi_n = &phi * n1;
            let phi_n_front = phi_n.slice(s![..-1]);
            let phi_n_back = phi_n.slice(s![1..]);
            let cos_phi_n = (phi_n_back.mapv(f64::cos) - phi_n_front.mapv(f64::cos)) / &dt;
            let sin_phi_n = (phi_n_back.mapv(f64::sin) - phi_n_front.mapv(f64::sin)) / &dt;
            coeffs[[n, 0]] = c * (&dxy.slice(s![.., 1]) * &cos_phi_n).sum();
            coeffs[[n, 1]] = c * (&dxy.slice(s![.., 1]) * &sin_phi_n).sum();
            coeffs[[n, 2]] = c * (&dxy.slice(s![.., 0]) * &cos_phi_n).sum();
            coeffs[[n, 3]] = c * (&dxy.slice(s![.., 0]) * &sin_phi_n).sum();
        }
        let tdt = &t.slice(s![1..]) / &dt;
        let xi = cumsum(dxy.slice(s![.., 0])) - &dxy.slice(s![.., 0]) * &tdt;
        let c = diff(&t.mapv(|v| v * v), None) * 0.5 / &dt;
        let a0 = (&dxy.slice(s![.., 0]) * &c + xi * &dt).sum() / (zt + 1e-20);
        let delta = cumsum(dxy.slice(s![.., 1])) - &dxy.slice(s![.., 1]) * &tdt;
        let c0 = (&dxy.slice(s![.., 1]) * c + delta * dt).sum() / (zt + 1e-20);
        Self {
            c: coeffs,
            locus: (curve[0][0] + a0, curve[0][1] + c0),
        }
    }

    /// Normalize the coefficients and get the geometry information.
    ///
    /// **Locus will loss with this operation.**
    ///
    /// Implements Kuhl and Giardina method of normalizing the coefficients
    /// An, Bn, Cn, Dn. Performs 3 separate normalizations. First, it makes the
    /// data location invariant by re-scaling the data to a common origin.
    /// Secondly, the data is rotated with respect to the major axis. Thirdly,
    /// the coefficients are normalized with regard to the absolute value of A₁.
    /// This code is adapted from the pyefd module.
    pub fn normalize(&mut self) -> GeoInfo {
        // Shift angle
        let theta1 =
            (2. * (self.c[[0, 0]] * self.c[[0, 1]] + self.c[[0, 2]] * self.c[[0, 3]])).atan2(
                self.c[[0, 0]] * self.c[[0, 0]] - self.c[[0, 1]] * self.c[[0, 1]]
                    + self.c[[0, 2]] * self.c[[0, 2]]
                    - self.c[[0, 3]] * self.c[[0, 3]],
            ) * 0.5;
        for n in 0..self.c.nrows() {
            let angle = (n + 1) as f64 * theta1;
            let rot = array![[angle.cos(), -angle.sin()], [angle.sin(), angle.cos()]];
            let m = array![
                [self.c[[n, 0]], self.c[[n, 1]]],
                [self.c[[n, 2]], self.c[[n, 3]]],
            ]
            .dot(&rot);
            self.c
                .slice_mut(s![n, ..])
                .assign(&Array1::from_iter(m.iter().cloned()));
        }
        // The angle of semi-major axis
        let psi = self.c[[0, 2]].atan2(self.c[[0, 0]]);
        let rot = array![[psi.cos(), psi.sin()], [-psi.sin(), psi.cos()]];
        for n in 0..self.c.nrows() {
            let m = rot.dot(&array![
                [self.c[[n, 0]], self.c[[n, 1]]],
                [self.c[[n, 2]], self.c[[n, 3]]],
            ]);
            self.c
                .slice_mut(s![n, ..])
                .assign(&Array1::from_iter(m.iter().cloned()));
        }
        let scale = self.c[[0, 0]].abs();
        self.c /= scale;
        let locus = self.locus;
        self.locus = (0., 0.);
        GeoInfo {
            semi_major_axis_angle: psi,
            shift_angle: theta1,
            scale,
            locus,
        }
    }

    /// Generate the described curve from the coefficients with specific point number.
    pub fn generate(&self, n: usize) -> Vec<[f64; 2]> {
        assert!(n > 3, "n must larger than 3, current is {}", n);
        let t = Array1::linspace(0., 1., n);
        let mut curve = vec![[0., 0.]; n];
        for n in 0..self.c.nrows() {
            let angle = &t * (n + 1) as f64 * TAU;
            let cos = angle.mapv(f64::cos);
            let sin = angle.mapv(f64::sin);
            let x = &cos * self.c[[n, 2]] + &sin * self.c[[n, 3]];
            let y = &cos * self.c[[n, 0]] + &sin * self.c[[n, 1]];
            Zip::from(&mut curve).and(&x).and(&y).for_each(|c, x, y| {
                c[0] = x + self.locus.0;
                c[1] = y + self.locus.1;
            });
        }
        curve
    }
}
