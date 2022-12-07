/// An error type for EFD coefficients.
/// Raised when the input array width is not correct to its dimension.
#[derive(Debug)]
pub struct EfdError<const DIM: usize>;

impl<const DIM: usize> core::fmt::Display for EfdError<DIM> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "input array width must be {DIM}")
    }
}

#[cfg(feature = "std")]
impl<const DIM: usize> std::error::Error for EfdError<DIM> {}
