/// An error type for 2D EFD coefficients.
/// Raised when the input array width is not 4.
#[derive(Debug)]
pub struct Efd2Error;

impl core::fmt::Display for Efd2Error {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "input array width must be 4")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Efd2Error {}
