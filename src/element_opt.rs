use ndarray::{Array, ArrayBase, Data, Dimension, RawData};

pub(crate) trait ElementWiseOpt {
    type Out;
    fn square(&self) -> Self::Out;
    fn sin(&self) -> Self::Out;
    fn cos(&self) -> Self::Out;
    fn abs(&self) -> Self::Out;
}

impl<S, D> ElementWiseOpt for ArrayBase<S, D>
    where S: RawData<Elem=f64> + Data,
          D: Dimension {
    type Out = Array<f64, D>;
    fn square(&self) -> Self::Out { self.mapv(|v| v * v) }
    fn sin(&self) -> Self::Out { self.mapv(f64::sin) }
    fn cos(&self) -> Self::Out { self.mapv(f64::cos) }
    fn abs(&self) -> Self::Out { self.mapv(f64::abs) }
}
