use ndarray::ScalarOperand;
use num_traits::{Float as NumFloat, FloatConst, NumAssignOps};

/// A float number type used in EFD.
pub trait Float: NumFloat + FloatConst + NumAssignOps + ScalarOperand + 'static {
    #[doc(hidden)]
    #[inline]
    fn two() -> Self {
        Self::from(2.).unwrap()
    }
    #[doc(hidden)]
    #[inline]
    fn half() -> Self {
        Self::from(0.5).unwrap()
    }
    #[doc(hidden)]
    #[inline]
    fn pow2(self) -> Self {
        self * self
    }
}

impl<T> Float for T where T: NumFloat + FloatConst + NumAssignOps + ScalarOperand + 'static {}
