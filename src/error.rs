/// An error type is raised when the input array width is not 4.
#[derive(Debug)]
pub struct EfdError;

impl core::fmt::Display for EfdError {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "input array width must be 4")
    }
}
