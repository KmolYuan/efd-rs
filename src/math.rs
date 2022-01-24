macro_rules! std_math {
    (@fn $name:ident($($v:ident),+) as $old_name:ident) => {
        #[cfg(feature = "libm")]
        { use libm::$old_name as $name; $name($($v),+) }
        #[cfg(not(feature = "libm"))]
        { f64::$name($($v),+) }
    };
    (@fn $name:ident($($v:ident),+)) => {
        #[cfg(feature = "libm")]
        { use libm::$name; $name($($v),+) }
        #[cfg(not(feature = "libm"))]
        { f64::$name($($v),+) }
    };
    ($(fn $name:ident($($v:ident),+) $(as $old_name:ident)?)+) => {$(
        #[inline(always)]
        pub(crate) fn $name($($v: f64),+) -> f64 {
            std_math!{@fn $name($($v),+) $(as $old_name)?}
        }
    )+};
}

std_math! {
    fn atan2(a, b)
    fn cos(v)
    fn abs(v) as fabs
    fn hypot(a, b)
    fn sin(v)
    fn sqrt(v)
}

#[inline(always)]
pub(crate) fn pow2(v: f64) -> f64 {
    v * v
}
