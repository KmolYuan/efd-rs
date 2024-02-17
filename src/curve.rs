use alloc::{borrow::Cow, vec::Vec};

/// Coordinate type of the dimension `D`.
pub type Coord<const D: usize> = [f64; D];

pub(crate) type MatrixRxX<const D: usize> = na::OMatrix<f64, na::Const<D>, na::Dyn>;

pub(crate) fn to_mat<C, const D: usize>(curve: C) -> MatrixRxX<D>
where
    C: Curve<D>,
{
    MatrixRxX::from_iterator(curve.len(), curve.as_curve().iter().flatten().copied())
}

/// Copy-on-write curve type.
///
/// Instead of using [`Cow<Coord<D>>`](std::borrow::Cow), this is a trait, which
/// does not require any conversion.
pub trait Curve<const D: usize>: Sized {
    /// Move or copy curve type into the owned type [`Vec`].
    fn to_curve(self) -> Vec<Coord<D>>;

    /// Elements view.
    fn as_curve(&self) -> &[Coord<D>];

    /// Length of the curve.
    fn len(&self) -> usize {
        self.as_curve().len()
    }

    /// Check if the curve is empty.
    fn is_empty(&self) -> bool {
        self.as_curve().is_empty()
    }

    /// Close the curve by the first element.
    ///
    /// # Panics
    ///
    /// Panics if the curve is empty.
    fn closed_lin(self) -> Vec<Coord<D>> {
        let mut c = self.to_curve();
        c.push(c[0]);
        c
    }

    /// Remove the last element.
    fn popped_last(self) -> Vec<Coord<D>> {
        let mut curve = self.to_curve();
        curve.pop();
        curve
    }

    /// Check if a curve's first and end points are the same.
    fn is_closed(&self) -> bool {
        let curve = self.as_curve();
        curve.first() == curve.last()
    }
}

impl<const D: usize> Curve<D> for Vec<Coord<D>> {
    fn to_curve(self) -> Vec<Coord<D>> {
        self
    }

    fn as_curve(&self) -> &[Coord<D>] {
        self
    }
}

macro_rules! impl_slice {
    () => {
        fn to_curve(self) -> Vec<Coord<D>> {
            self.to_vec()
        }

        fn as_curve(&self) -> &[Coord<D>] {
            self
        }
    };
}

impl<const D: usize, const N: usize> Curve<D> for [Coord<D>; N] {
    impl_slice!();
}

impl<const D: usize> Curve<D> for &[Coord<D>] {
    impl_slice!();
}

impl<const D: usize> Curve<D> for Cow<'_, [Coord<D>]> {
    impl_slice!();
}

impl<const D: usize, T: Curve<D> + Clone> Curve<D> for &T {
    fn to_curve(self) -> Vec<Coord<D>> {
        self.clone().to_curve()
    }

    fn as_curve(&self) -> &[Coord<D>] {
        (*self).as_curve()
    }
}
