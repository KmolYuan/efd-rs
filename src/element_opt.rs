use ndarray::{Array, ArrayBase, AsArray, Axis, Data, Dimension, RawData, ScalarOperand, Slice};
use num_traits::{Float, NumOps};

/// Implement several element-wise operations for [`ndarray::ArrayBase`]s.
pub trait ElementWiseOpt {
    type Out;
    fn square(&self) -> Self::Out;
    fn sqrt(&self) -> Self::Out;
    fn sin(&self) -> Self::Out;
    fn cos(&self) -> Self::Out;
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

/// Calculate the n-th discrete difference along the given axis.
/// Same as NumPy version.
///
/// Reference from <https://github.com/rust-ndarray/ndarray/issues/787>.
pub fn diff<'a, A: 'a, D, V>(arr: V, axis: Option<Axis>) -> Array<A, D>
where
    A: NumOps + ScalarOperand,
    D: Dimension,
    V: AsArray<'a, A, D>,
{
    let view = arr.into();
    let axis = axis.unwrap_or(Axis(view.ndim() - 1));
    let head = view.slice_axis(axis, Slice::from(..-1));
    let tail = view.slice_axis(axis, Slice::from(1..));
    &tail - &head
}
