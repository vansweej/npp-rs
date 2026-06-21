//! Golden-image correctness test for `Mean` on `CudaImage<u8>`.
//!
//! Proves that the macro-generated two-call dance produces correct per-channel
//! means, not just correct geometry (C12 minimum mitigation).
//!
//! # Manual procedure to pin the golden reference
//!
//! 1. Run on a GPU host: `nix develop . --command cargo test --features gpu --test golden_mean`
//! 2. The test will print the captured output and panic ("golden reference not yet pinned").
//! 3. Copy the printed bytes into `EXPECTED` below.
//! 4. Re-run to confirm the assertion passes.

#![cfg(feature = "gpu")]

use cudarc::driver::CudaDevice;
use npp_rs::cuda::default_cuda_device;
use npp_rs::image::CudaImage;
use npp_rs::imageops::Mean;
use std::sync::Arc;

const W: u32 = 12;
const H: u32 = 4;

/// Input: procedurally generated 3-channel u8 gradient (12x4).
fn make_input() -> Vec<u8> {
    let mut data = Vec::with_capacity((W * H * 3) as usize);
    for y in 0..H {
        for x in 0..W {
            data.push((x * 21) as u8); // R: x-gradient
            data.push((y * 64) as u8); // G: y-gradient
            data.push(128); // B: constant
        }
    }
    data
}

/// Golden per-channel mean for the above input.
/// Generated on NVIDIA GPU. Bit-exact integer arithmetic (no floating point).
///
/// NOTE: Set this to `&[]` initially, run on GPU, copy the printed values here.
const EXPECTED: &[f64] = &[
    // Golden data will be pinned after first GPU test run
];

const EPSILON: f64 = 1e-12;

#[test]
fn test_golden_mean_u8_c3() {
    let device: Arc<CudaDevice> = default_cuda_device().expect("CUDA device init");

    // 3-channel source (RGB)
    let src = CudaImage::from_host(device.clone(), 3, W, H, &make_input()).expect("src allocation");

    let result = src.mean().expect("mean");

    if EXPECTED.is_empty() {
        eprintln!("=== Golden reference NOT pinned for Mean u8 C3 ===");
        eprintln!("Captured output ({}) values: {:?}", result.len(), result);
        panic!("golden reference not yet pinned for Mean u8 C3");
    }

    assert_eq!(
        result.len(),
        EXPECTED.len(),
        "Mean output length mismatch: expected {}, got {}",
        EXPECTED.len(),
        result.len()
    );

    for (i, (got, expected)) in result.iter().zip(EXPECTED.iter()).enumerate() {
        assert!(
            (got - expected).abs() < EPSILON,
            "Mean channel {i}: got {got}, expected {expected}",
        );
    }
}
