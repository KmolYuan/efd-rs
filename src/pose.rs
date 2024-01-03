use crate::*;

/// A 2D shape with a pose described by EFD.
pub type PosedEfd2 = PosedEfd<D2>;
/// A 3D shape with a pose described by EFD.
pub type PosedEfd3 = PosedEfd<D3>;

/// A shape with a pose described by EFD.
pub struct PosedEfd<D: EfdDim> {
    efd: Efd<D>,
    pose: Coeff<D>,
}

impl<D: EfdDim> PosedEfd<D> {
    // TODO
}

impl<D: EfdDim> Clone for PosedEfd<D> {
    fn clone(&self) -> Self {
        Self { efd: self.efd.clone(), pose: self.pose.clone() }
    }
}

impl<D: EfdDim> core::ops::Deref for PosedEfd<D> {
    type Target = Efd<D>;

    fn deref(&self) -> &Self::Target {
        &self.efd
    }
}
