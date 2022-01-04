#[cfg(feature = "libm")]
pub(crate) use libm::{atan2, copysign, cos, fabs as abs, hypot, sin, sqrt};

#[cfg(not(feature = "libm"))]
macro_rules! std_math {
    ($(fn $name:ident($($v:ident),+))+) => {$(
        #[inline(always)]
        pub(crate) fn $name($($v: f64),+) -> f64 {
            f64::$name($($v),+)
        }
    )+};
}

#[cfg(not(feature = "libm"))]
std_math! {
    fn atan2(a, b)
    fn copysign(a, b)
    fn cos(v)
    fn abs(v)
    fn hypot(a, b)
    fn sin(v)
    fn sqrt(v)
}
