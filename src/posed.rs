use crate::*;

/// A 2D shape with a pose described by EFD.
pub type PosedEfd2 = PosedEfd<D2, D2>;
/// A 3D shape with a pose described by EFD.
pub type PosedEfd3 = PosedEfd<D3, D2>;

/// A shape with a pose described by EFD.
pub struct PosedEfd<D1: EfdDim, D2: EfdDim> {
    efd: Efd<D1>,
    pose: Coeff<D2>,
}

impl<D1: EfdDim, D2: EfdDim> PosedEfd<D1, D2> {
    // TODO
}

impl<D1: EfdDim, D2: EfdDim> Clone for PosedEfd<D1, D2> {
    fn clone(&self) -> Self {
        Self { efd: self.efd.clone(), pose: self.pose.clone() }
    }
}

impl<D1: EfdDim, D2: EfdDim> core::ops::Deref for PosedEfd<D1, D2> {
    type Target = Efd<D1>;

    fn deref(&self) -> &Self::Target {
        &self.efd
    }
}
