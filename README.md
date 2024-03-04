# EFD Rust Library

[![dependency status](https://deps.rs/repo/github/KmolYuan/efd-rs/status.svg)](https://deps.rs/crate/efd/)
[![documentation](https://docs.rs/efd/badge.svg)](https://docs.rs/efd)

Elliptical Fourier Descriptor (EFD) implementation in Rust. This crate implements 1D/2D/3D EFD and its related functions.

Keyword Alias:

+ Elliptical Fourier Analysis (EFA)
+ Elliptical Fourier Function (EFF)

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
let described_curve = efd::Efd2::from_curve(curve, false).recon(20);
```

The harmonic number can be set with `efd::Efd::from_curve_harmonic()` method. The following figures show the reconstruction of a 2D closed curve with 1-4 harmonics.

<div>
<img width="20%" alt="1h" src="https://raw.githubusercontent.com/KmolYuan/efd-rs/main/img/2dh1.svg"/>
<img width="20%" alt="2h" src="https://raw.githubusercontent.com/KmolYuan/efd-rs/main/img/2dh2.svg"/>
<img width="20%" alt="3h" src="https://raw.githubusercontent.com/KmolYuan/efd-rs/main/img/2dh3.svg"/>
<img width="20%" alt="4h" src="https://raw.githubusercontent.com/KmolYuan/efd-rs/main/img/2dh4.svg"/>
</div>

## Example Images

2D and 3D closed curve:

<div>
<img width="40%" alt="2d" src="https://raw.githubusercontent.com/KmolYuan/efd-rs/main/img/2d.svg"/>
<img width="40%" alt="3d" src="https://raw.githubusercontent.com/KmolYuan/efd-rs/main/img/3d.svg"/>
</div>

2D and 3D open curve:

<div>
<img width="40%" alt="2d" src="https://raw.githubusercontent.com/KmolYuan/efd-rs/main/img/2d_open.svg"/>
<img width="40%" alt="3d" src="https://raw.githubusercontent.com/KmolYuan/efd-rs/main/img/3d_open.svg"/>
</div>

Posed EFD combined a curve with a pose (unit vectors) to describe the orientation of each point.

2D open curve and its full reconstruction:

<div>
<img width="40%" alt="posed" src="https://raw.githubusercontent.com/KmolYuan/efd-rs/main/img/posed.svg"/>
<img width="40%" alt="posed-full" src="https://raw.githubusercontent.com/KmolYuan/efd-rs/main/img/posed-full.svg"/>
</div>

## Citations

### Original

+ Kuhl, FP and Giardina, CR (1982). Elliptic Fourier features of a closed contour. Computer graphics and image processing, 18(3), 236-258. <https://doi.org/10.1016/0146-664X(82)90034-X>

### My Applications

+ Chang, Y., Chang, JL., Lee, JJ. (2024). Atlas-Based Path Synthesis of Planar Four-Bar Linkages Using Elliptical Fourier Descriptors. In: Okada, M. (eds) Advances in Mechanism and Machine Science. IFToMM WC 2023. Mechanisms and Machine Science, vol 149. Springer, Cham. <https://doi.org/10.1007/978-3-031-45709-8_20>
