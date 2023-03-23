use crate::{EfdDim, Trans};
use alloc::vec::Vec;

/// Alias for evaluate `EfdDim::Trans::Coord` from `D`.
pub type Coord<D> = <<D as EfdDim>::Trans as Trans>::Coord;

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

    /// Close the open curve with its direction-inverted part.
    ///
    /// Panic with empty curve.
    #[must_use]
    fn closed_rev(self) -> Vec<A> {
        let mut curve = self.to_curve();
        let curve2 = curve.iter().rev().skip(1).cloned().collect::<Vec<_>>();
        curve.extend(curve2);
        curve
    }

    /// Check if a curve's first and end points are the same.
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

impl<A: Clone> Curve<A> for &Vec<A> {
    fn to_curve(self) -> Vec<A> {
        self.clone()
    }

    fn as_curve(&self) -> &[A] {
        self
    }
}

impl<A: Clone, const N: usize> Curve<A> for [A; N] {
    fn to_curve(self) -> Vec<A> {
        self.to_vec()
    }

    fn as_curve(&self) -> &[A] {
        self
    }
}

impl<A: Clone, const N: usize> Curve<A> for &[A; N] {
    fn to_curve(self) -> Vec<A> {
        self.to_vec()
    }

    fn as_curve(&self) -> &[A] {
        self.as_slice()
    }
}

impl<A: Clone> Curve<A> for &[A] {
    fn to_curve(self) -> Vec<A> {
        self.to_vec()
    }

    fn as_curve(&self) -> &[A] {
        self
    }
}
