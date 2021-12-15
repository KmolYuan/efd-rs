use ndarray::{Array, ArrayBase, Data, Dimension, RawData};
use num_traits::Float;

/// Implement several element-wise operations for [`ndarray::ArrayBase`]s.
pub(crate) trait ElementWiseOpt {
    /// Output type.
    type Out;
    /// Square function.
    fn square(&self) -> Self::Out;
    /// Square root function.
    fn sqrt(&self) -> Self::Out;
    /// Sine function.
    fn sin(&self) -> Self::Out;
    /// Cosine function.
    fn cos(&self) -> Self::Out;
    /// Absolute function.
    fn abs(&self) -> Self::Out;
}

impl<A, S, D> ElementWiseOpt for ArrayBase<S, D>
where
    A: Float,
    S: RawData<Elem = A> + Data,
    D: Dimension,
{
    type Out = Array<A, D>;
    fn square(&self) -> Self::Out {
        self.mapv(|v| v * v)
    }
    fn sqrt(&self) -> Self::Out {
        self.mapv(A::sqrt)
    }
    fn sin(&self) -> Self::Out {
        self.mapv(A::sin)
    }
    fn cos(&self) -> Self::Out {
        self.mapv(A::cos)
    }
    fn abs(&self) -> Self::Out {
        self.mapv(A::abs)
    }
}
