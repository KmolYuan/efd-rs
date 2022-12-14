use crate::{EfdDim, Trans};
use alloc::format;
use core::marker::PhantomData;

/// An error type for EFD coefficients.
/// Raised when the input array width is not correct to its dimension.
pub struct EfdError<D: EfdDim> {
    _marker: PhantomData<D>,
}

impl<D: EfdDim> EfdError<D> {
    pub(crate) fn new() -> Self {
        Self { _marker: PhantomData }
    }
}

impl<D: EfdDim> core::fmt::Debug for EfdError<D> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct(&format!("EfdError{}", <D::Trans as Trans>::DIM))
            .finish()
    }
}

impl<D: EfdDim> core::fmt::Display for EfdError<D> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "input array width must be {}", D::Trans::DIM * 2)
    }
}

#[cfg(feature = "std")]
impl<D: EfdDim> std::error::Error for EfdError<D> {}
