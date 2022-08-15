# efd

[![dependency status](https://deps.rs/repo/github/KmolYuan/efd-rs/status.svg)](https://deps.rs/crate/efd/)
[![documentation](https://docs.rs/efd/badge.svg)](https://docs.rs/efd)

A light, highly generic library for Elliptical Fourier Descriptor (EFD). This crate implements EFD and its related functions.

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

Simple usage of resampling circle:

```rust
use efd::Efd;

let curve = vec![[0.; 2], [1.; 2], [2.; 2], [3.; 2]];
let new_curve = Efd::from_curve(&curve, None).generate(20);
```
