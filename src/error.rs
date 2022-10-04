/// Alias of the 2D error type.
pub type Efd2Error = EfdError;

/// An error type for EFD coefficients.
/// Raised when the input array width is not correct to its dimension.
#[derive(Debug)]
pub struct EfdError(pub(crate) ());

impl core::fmt::Display for EfdError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "input array width must be 4")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for EfdError {}
