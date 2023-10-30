# EFD Rust Library

[![dependency status](https://deps.rs/repo/github/KmolYuan/efd-rs/status.svg)](https://deps.rs/crate/efd/)
[![documentation](https://docs.rs/efd/badge.svg)](https://docs.rs/efd)

Elliptical Fourier Descriptor (EFD) implementation in Rust. This crate implements 2D/3D EFD and its related functions.

Keyword Alias:

+ Elliptical Fourier Analysis (EFA)
+ Elliptical Fourier Function (EFF)

This is an unofficial implementation.

Example of re-describing a new closed curve:

```rust
let curve = vec![
    [0., 0.],
    [1., 1.],
    [2., 2.],
    [3., 3.],
    [2., 2.],
    [1., 1.],
];
assert!(efd::util::valid_curve(&curve).is_some());
let described_curve = efd::Efd2::from_curve(curve, false).generate(20);
```

## Bibliography

```plain
Kuhl, FP and Giardina, CR (1982). Elliptic Fourier features of a closed contour. Computer graphics and image processing, 18(3), 236-258.
```

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

## Example Images

<div>
<img width="40%" alt="2d" src="https://raw.githubusercontent.com/KmolYuan/efd-rs/main/img/2d.svg"/>
<img width="40%" alt="3d" src="https://raw.githubusercontent.com/KmolYuan/efd-rs/main/img/3d.svg"/>
</div>
