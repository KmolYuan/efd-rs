# EFD Rust Library

[![dependency status](https://deps.rs/repo/github/KmolYuan/efd-rs/status.svg)](https://deps.rs/crate/efd/)
[![documentation](https://docs.rs/efd/badge.svg)](https://docs.rs/efd)

Elliptical Fourier Descriptor (EFD) implementation in Rust. This crate implements 2D/3D EFD and its related functions.

Keyword Alias:

+ Elliptical Fourier Analysis (EFA)
+ Elliptical Fourier Function (EFF)

Reference: Kuhl, FP and Giardina, CR (1982). Elliptic Fourier features of a closed contour. Computer graphics and image processing, 18(3), 236-258.

This is an unofficial implementation.

```bibtex
@article{kuhl1982elliptic,
  title={Elliptic Fourier features of a closed contour},
  author={Kuhl, Frank P and Giardina, Charles R},
  journal={Computer graphics and image processing},
  volume={18},
  number={3},
  pages={236--258},
  year={1982},
  publisher={Elsevier}
}
```

Example of re-describing a new curve:

```rust
let curve = vec![
    [0., 0.],
    [1., 1.],
    [2., 2.],
    [3., 3.],
    [2., 2.],
    [1., 1.],
];
let described_curve = efd::Efd2::from_curve(curve, false).unwrap().generate(20);
```
