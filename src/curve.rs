//! Curve trait and its implementations.
use crate::{CoordHint, EfdDim, Trans};
use alloc::vec::Vec;

/// Alias for evaluate `EfdDim::Trans::Coord` from `D`.
pub type Coord<D> = <<D as EfdDim>::Trans as Trans>::Coord;

pub(crate) type MatrixRxX<R> = na::OMatrix<f64, R, na::Dyn>;

pub(crate) fn to_mat<A, C>(curve: C) -> MatrixRxX<A::Dim>
where
    A: CoordHint,
    C: Curve<A>,
{
    let curve = curve.to_curve();
    MatrixRxX::<A::Dim>::from_iterator(curve.len(), curve.into_iter().flat_map(A::flat))
}

/// Copy-on-write curve type.
pub trait Curve<A: Clone>: Sized {
    /// Move or copy curve type into owned type. (`Vec`)
    #[must_use]
    fn to_curve(self) -> Vec<A>;

    /// Elements view.
    #[must_use]
    fn as_curve(&self) -> &[A];

    /// Close the curve by the first element.
    ///
    /// Panic with empty curve.
    #[must_use]
    fn closed_lin(self) -> Vec<A> {
        let mut c = self.to_curve();
        c.push(c[0].clone());
        c
    }

    /// Remove the last element.
    #[must_use]
    fn pop_last(self) -> Vec<A> {
        let mut curve = self.to_curve();
        curve.pop();
        curve
    }

    /// Check if a curve's first and end points are the same.
    #[must_use]
    fn is_closed(&self) -> bool
    where
        A: PartialEq,
    {
        let curve = self.as_curve();
        match (curve.first(), curve.last()) {
            (Some(a), Some(b)) => a == b,
            _ => false,
        }
    }
}

impl<A: Clone> Curve<A> for Vec<A> {
    fn to_curve(self) -> Vec<A> {
        self
    }

    fn as_curve(&self) -> &[A] {
        self
    }
}

macro_rules! impl_slice {
    () => {
        fn to_curve(self) -> Vec<A> {
            self.to_vec()
        }

        fn as_curve(&self) -> &[A] {
            self
        }
    };
}

impl<A: Clone, const N: usize> Curve<A> for [A; N] {
    impl_slice!();
}

impl<A: Clone> Curve<A> for &[A] {
    impl_slice!();
}

impl<A: Clone> Curve<A> for alloc::borrow::Cow<'_, [A]> {
    impl_slice!();
}

impl<A: Clone, T: Curve<A> + Clone> Curve<A> for &T {
    fn to_curve(self) -> Vec<A> {
        self.clone().to_curve()
    }

    fn as_curve(&self) -> &[A] {
        (*self).as_curve()
    }
}
