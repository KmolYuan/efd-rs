macro_rules! impl_err {
    ($(struct $name:ident { $d:literal, $w:literal })+) => {$(
        /// An error type for
        #[doc = $d]
        /// EFD coefficients.
        /// Raised when the input array width is not correct to its dimension.
        #[derive(Debug)]
        pub struct $name(pub(crate) ());

        impl core::fmt::Display for $name {
            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                write!(f, concat!["input array width must be ", $w])
            }
        }

        #[cfg(feature = "std")]
        impl std::error::Error for $name {}
    )+};
}

impl_err! {
    struct Efd2Error { "2D", "4" }
    struct Efd3Error { "3D", "6" }
}
