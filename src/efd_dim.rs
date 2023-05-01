use crate::*;
use core::f64::consts::{PI, TAU};

/// 2D EFD dimension type.
pub enum D2 {}
/// 3D EFD dimension type.
pub enum D3 {}
/// 2D Coefficient type.
pub type Coeff2 = Coeff<D2>;
/// 3D Coefficient type.
pub type Coeff3 = Coeff<D3>;
/// Coefficient type.
pub type Coeff<D> = na::OMatrix<f64, CDim<D>, na::Dyn>;
/// Coordinate view used in the conversion method.
pub type CoordView<'a, D> = na::MatrixView<'a, f64, Dim<D>, na::U1>;
/// Alias for the dimension.
pub type Dim<D> = <<<D as EfdDim>::Trans as Trans>::Coord as CoordHint>::Dim;
/// Alias for the coefficient number.
pub type CDim<D> = <<<D as EfdDim>::Trans as Trans>::Coord as CoordHint>::CDim;
/// Alias for the rotation type.
pub type Rot<D> = <<D as EfdDim>::Trans as Trans>::Rot;

pub(crate) type CKernel<'a, D> = na::MatrixView<'a, f64, na::U2, Dim<D>>;
pub(crate) type CKernelMut<'a, D> = na::MatrixViewMut<'a, f64, na::U2, Dim<D>>;

/// Trait for EFD dimension.
pub trait EfdDim {
    /// Transform type.
    type Trans: Trans;

    /// Generate coefficients and transform.
    fn from_curve_harmonic(
        curve: &[Coord<Self>],
        harmonic: usize,
        is_open: bool,
    ) -> (Coeff<Self>, Transform<Self::Trans>);
}

impl EfdDim for D2 {
    type Trans = T2;

    fn from_curve_harmonic(
        curve: &[Coord<Self>],
        harmonic: usize,
        is_open: bool,
    ) -> (Coeff<Self>, Transform<Self::Trans>) {
        let (mut coeffs, center) = get_coeff_center(curve, harmonic, is_open);
        // Angle of starting point
        let theta = {
            let c = CKernel::<Self>::from_slice(coeffs.column(0).data.into_slice());
            let dy = 2. * c.row_product().sum();
            let dx = c.map(pow2).column_sum();
            0.5 * dy.atan2(dx[0] - dx[1])
        };
        for (i, mut c) in coeffs.column_iter_mut().enumerate() {
            let theta = na::Rotation2::new((i + 1) as f64 * -theta);
            let mut m = CKernelMut::<Self>::from_slice(c.as_mut_slice());
            m.copy_from(&(theta * &m));
        }
        // Normalize coefficients sign
        if harmonic > 1 && coeffs[(0, 0)] * coeffs[(0, 1)] < 0. {
            coeffs
                .column_iter_mut()
                .step_by(2)
                .for_each(|mut s| s *= -1.);
        }
        // Angle of semi-major axis
        // Case in 2D are equivalent to:
        //
        // let u = na::Vector3::new(coeffs[[0, 0]], coeffs[[0, 2]], 0.).normalize();
        // na::UnitComplex::new(u.dot(&na::Vector3::x()).acos())
        let psi = na::UnitComplex::new(coeffs[(2, 0)].atan2(coeffs[(0, 0)]));
        let psi_inv = psi.to_rotation_matrix();
        for mut c in coeffs.column_iter_mut() {
            let mut m = CKernelMut::<Self>::from_slice(c.as_mut_slice());
            m.copy_from(&(&m * psi_inv));
        }
        let scale = {
            let c = CKernel::<Self>::from_slice(coeffs.column(0).data.into_slice());
            c.row(0).map(pow2).sum().sqrt()
        };
        coeffs /= scale;
        let trans = Transform::new(center, psi, scale);
        (coeffs, trans)
    }
}

impl EfdDim for D3 {
    type Trans = T3;

    fn from_curve_harmonic(
        curve: &[Coord<Self>],
        harmonic: usize,
        is_open: bool,
    ) -> (Coeff<Self>, Transform<Self::Trans>) {
        let (mut coeffs, center) = get_coeff_center(curve, harmonic, is_open);
        // Angle of starting point
        let theta = {
            let c = CKernel::<Self>::from_slice(coeffs.column(0).data.into_slice());
            let dy = 2. * c.row_product().sum();
            let dx = c.map(pow2).column_sum();
            0.5 * dy.atan2(dx[0] - dx[1])
        };
        for (i, mut c) in coeffs.column_iter_mut().enumerate() {
            let theta = na::Rotation2::new((i + 1) as f64 * -theta);
            let mut m = CKernelMut::<Self>::from_slice(c.as_mut_slice());
            m.copy_from(&(theta * &m));
        }
        // Normalize coefficients sign
        if harmonic > 1 && coeffs[(0, 0)] * coeffs[(0, 1)] < 0. {
            coeffs
                .column_iter_mut()
                .step_by(2)
                .for_each(|mut s| s *= -1.);
        }
        // Angle of semi-major axis
        let psi = {
            let c = CKernel::<Self>::from_slice(coeffs.column(0).data.into_slice());
            let u = c.row(0).transpose().normalize();
            let v = c.row(1).transpose().normalize();
            na::Rotation3::from_basis_unchecked(&[u, v, u.cross(&v)])
        };
        let psi_inv = psi;
        for mut c in coeffs.column_iter_mut() {
            let mut m = CKernelMut::<Self>::from_slice(c.as_mut_slice());
            m.copy_from(&(&m * psi_inv));
        }
        let scale = {
            let c = CKernel::<Self>::from_slice(coeffs.column(0).data.into_slice());
            c.row(0).map(pow2).sum().sqrt()
        };
        coeffs /= scale;
        let trans = Transform::new(center, psi.into(), scale);
        (coeffs, trans)
    }
}

fn get_coeff_center<A: CoordHint>(
    curve: &[A],
    harmonic: usize,
    is_open: bool,
) -> (MatrixRxX<A::CDim>, A) {
    let curve_arr = if is_open {
        to_mat(curve)
    } else {
        to_mat(curve.closed_lin())
    };
    let dxyz = diff(curve_arr);
    let dt = dxyz.map(pow2).row_sum().map(f64::sqrt);
    let t = cumsum(dt.clone()).insert_column(0, 0.);
    let zt = t[t.len() - 1] * if is_open { 2. } else { 1. };
    let scalar = zt / (PI * PI) * if is_open { 1. } else { 0.5 };
    let phi = &t * TAU / zt;
    let mut coeffs = MatrixRxX::<A::CDim>::zeros(harmonic);
    for (i, mut c) in coeffs.column_iter_mut().enumerate() {
        let n = i as f64 + 1.;
        let phi = &phi * n;
        let phi_front = phi.columns_range(..phi.len() - 1);
        let phi_back = phi.columns_range(1..);
        let scalar = scalar / (n * n);
        let cos_phi = (phi_back.map(f64::cos) - phi_front.map(f64::cos)).component_div(&dt);
        dxyz.row_iter()
            .zip(c.iter_mut().step_by(2))
            .for_each(|(d, c)| *c = scalar * (d.component_mul(&cos_phi)).sum());
        if !is_open {
            let sin_phi = (phi_back.map(f64::sin) - phi_front.map(f64::sin)).component_div(&dt);
            dxyz.row_iter()
                .zip(c.iter_mut().skip(1).step_by(2))
                .for_each(|(d, c)| *c = scalar * (d.component_mul(&sin_phi)).sum());
        }
    }
    let tdt = t.columns_range(1..).component_div(&dt);
    let c = 0.5 * diff(t.map(pow2)).component_div(&dt);
    let mut center = curve[0].clone();
    dxyz.row_iter()
        .zip(center.flat_mut())
        .for_each(|(dxyz, oxyz)| {
            let xi = cumsum(dxyz) - dxyz.component_mul(&tdt);
            *oxyz += (dxyz.component_mul(&c) + xi.component_mul(&dt)).sum() / zt
                * if is_open { 2. } else { 1. };
        });
    (coeffs, center)
}
